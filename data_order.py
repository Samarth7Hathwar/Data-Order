# Importing the required libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torchvision.transforms as transforms
from torchvision import datasets
from timm import create_model
from sklearn.cluster import KMeans
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import RandAugment
from timm.data.mixup import Mixup
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR


# Paths to the training and validation datasets
data_path = '/ds/images/imagenet'
train_dir = f"{data_path}/train"
val_dir = f"{data_path}/val_folders"

# Setting up transformations for data augmentation and normalization
transform = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),        #Apply RandAugment
    transforms.RandomResizedCrop((224, 224)),   # Randomly crop and resize images to a fixed size
    transforms.RandomHorizontalFlip(),         # Add random horizontal flipping for variation
    transforms.ToTensor(),                     # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with standard ImageNet mean
                         std=[0.229, 0.224, 0.225])   # and standard deviation
])

# Checking if the dataset directories exist
assert os.path.exists(train_dir), f"Training directory not found: {train_dir}"
assert os.path.exists(val_dir), f"Validation directory not found: {val_dir}"

# Distributed initialization
dist.init_process_group(backend='nccl')  # Initialize distributed training
local_rank = int(os.getenv('LOCAL_RANK', 0))  # Get the local rank of the process
device = torch.device(f'cuda:{local_rank}')

# Loading the datasets using ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

# Creating DataLoaders with distributed samplers
train_sampler = DistributedSampler(train_data, shuffle=True)
val_sampler = DistributedSampler(val_data, shuffle=False)

# Creating DataLoaders for efficient data loading
train_loader = DataLoader(train_data, batch_size=512, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=512, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=True)

augmentation_fn = Mixup(
    mixup_alpha=0.2,  # Probability for MixUp
    cutmix_alpha=1.0,  # Probability for CutMix
    prob=1.0,  # Always apply MixUp or CutMix
    switch_prob=0.5,  # 50% chance to switch between MixUp and CutMix
    mode='batch'  # Apply augmentation to the entire batch
)

# Define a wrapper for the NFNet model
class NFNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super(NFNetWrapper, self).__init__()
        self.model = create_model('nfnet_f0', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# Function to train the model for one epoch
def train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn=None):
    net.train()  # Set the model to training mode
    total_loss, correct, total = 0, 0, 0

    # Loop through the training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the GPU
        optimizer.zero_grad()
        
        # Apply MixUp or CutMix augmentation
        if augmentation_fn is not None:
            images, labels = augmentation_fn(images, labels)            
                  
        # Forward pass with AMP (Automatic Mixed Precision)
        # I have used AMP to help speed up computations and reduce memory usage without affevting model accuracy
        with torch.amp.autocast('cuda'):
            outputs = net(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        # Track loss and accuracy
        total_loss += loss.item()
        _, preds = outputs.max(1)  # Get predictions
        correct += (preds == labels.argmax(dim=1)).sum().item() if augmentation_fn else (preds == labels).sum().item()
        total += labels.size(0)

    # Return average loss and accuracy for the epoch
    return total_loss / len(train_loader), 100 * correct / total


# Function to validate the model
def validate(net, val_loader, criterion, device):
    net.eval()  # Set the model to evaluation mode
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # AMP for validation
            with torch.amp.autocast('cuda'):
                outputs = net(images)
                loss = criterion(outputs, labels)

            # Track loss and accuracy
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Return average validation loss and accuracy
    return total_loss / len(val_loader), 100 * correct / total

"""
# Function to incrementally train the model
def train_incrementally(net, train_data, val_loader, order, criterion, optimizer, scaler, device, num_epochs=1, step_size=10, augmentation_fn=None):
    net = DataParallel(net)  # Enable multi-GPU training
    all_acc = []  # Store accuracies for each step

    # Loop through subsets of classes
    for step in range(2, len(train_data.classes) + 2, step_size):
        current_classes = order[:step]  # Get the current subset of classes
        train_indices = [i for i, (_, label) in enumerate(train_data.samples) if label in current_classes]
        train_subset = Subset(train_data, train_indices)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=8)

        # Train for the specified number of epochs
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn=None)

        # Validate after training
        val_loss, val_acc = validate(net, val_loader, criterion, device)
        all_acc.append(val_acc)
        print(f"Training with {len(current_classes)} classes: Accuracy after training: {val_acc:.2f}%")

    return all_acc
"""


# Function to generate a dissimilar class order using clustering
def get_dissimilar_order(train_data):
    print("Generating dissimilar order using ResNet embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the fully connected layer
    resnet.eval()

    class_features = []
    for class_idx in range(len(train_data.classes)):
        indices = [i for i, (_, label) in enumerate(train_data.samples) if label == class_idx]
        subset = Subset(train_data, indices[:10])  # Use the first 10 images of each class
        loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=8)

        embeddings = []
        for images, _ in loader:
            images = images.to(device)
            with torch.no_grad():
                embedding = resnet(images).squeeze(-1).squeeze(-1)  # Flatten spatial dimensions
                embeddings.append(embedding.cpu().numpy())  # Shape: (batch_size, num_features)

        # Concatenate embeddings into a single array and average along the first axis
        embeddings = np.concatenate(embeddings, axis=0)  # Shape: (num_samples, num_features)
        class_mean_embedding = np.mean(embeddings, axis=0)  # Shape: (num_features,)
        class_features.append(class_mean_embedding)

    # Convert class_features to a 2D array (num_classes, num_features)
    class_features = np.array(class_features)  # Shape: (num_classes, num_features)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(class_features)
    
    # Return the order of classes based on their cluster labels
    return sorted(range(len(train_data.classes)), key=lambda x: kmeans.labels_[x])

# Custom Linear Warm-Up + Cosine Decay Scheduler
def linear_warmup_and_cosine_decay(epoch):
    if epoch < 5:  # Warm-up phase
        return epoch / 5  # Linearly scale the learning rate
    else:  # Cosine annealing phase
        return 0.5 * (1 + np.cos((epoch - 5) / (num_epochs - 5) * np.pi))

# Set up the device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device(f"cuda:{local_rank}")
net = NFNetWrapper(num_classes=len(train_data.classes)).to(device)
net = DDP(net, device_ids=[local_rank], output_device=local_rank)  # Use DDP
#net = torch.compile(net)  # Optimize the model with Torch Compile

num_epochs = 120  # Set the number of epochs to 120
criterion = nn.CrossEntropyLoss().to(device)
batch_size = 512
max_lr = 0.1 * (batch_size / 256)

optimizer = optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

#Added Warmup period and using scheduled learning rate
lr_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_and_cosine_decay)
scaler = torch.cuda.amp.GradScaler()

# Generate class orders
"""
alphabetical_order = list(range(len(train_data.classes)))
random_order = np.random.permutation(len(train_data.classes)).tolist()
"""
# Generate a dissimilar class order
dissimilar_order = get_dissimilar_order(train_data)
"""
# Train with Alphabetical Order
print("\nTraining with Alphabetical Order")
train_incrementally(net, train_data, val_loader, alphabetical_order, criterion, optimizer, scaler, device)

# Train with Random Order
print("\nTraining with Random Order")
net = NFNetWrapper(num_classes=len(train_data.classes)).to(device)
net = torch.compile(net)
train_incrementally(net, train_data, val_loader, random_order, criterion, optimizer, scaler, device)
"""
"""
# Train with the dissimilar order
print("\nTraining with Dissimilar Order")
#net = torch.compile(net)
train_incrementally(net, train_data, val_loader, dissimilar_order, criterion, optimizer, scaler, device, num_epochs=90, augmentation_fn=augmentation_fn)
"""

# Train the model normally with the dissimilar order

if local_rank == 0:
    print("\nStarting normal training with Dissimilar Order...")

for epoch in range(num_epochs):
    # Training step
    train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn=augmentation_fn)
    
    # Validation step
    val_loss, val_acc = validate(net, val_loader, criterion, device)
    
    # Epoch-level learning rate scheduler step
    lr_scheduler.step()
    
    if local_rank == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    torch.cuda.empty_cache()
    
              
# Destroy process group after training
dist.destroy_process_group()
