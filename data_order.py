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
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


# Paths to the training and validation datasets
data_path = '/ds/images/imagenet'
train_dir = f"{data_path}/train"
val_dir = f"{data_path}/val_folders"

# Setting up transformations for data augmentation and normalization
transform = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),        # Apply RandAugment
    transforms.RandomResizedCrop((224, 224)),   # Randomly crop and resize images to a fixed size
    transforms.RandomHorizontalFlip(),         # Add random horizontal flipping for variation
    transforms.ToTensor(),                     # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with standard ImageNet mean
                         std=[0.229, 0.224, 0.225])   # and standard deviation
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),  # Standard center crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Checking if the dataset directories exist
assert os.path.exists(train_dir), f"Training directory not found: {train_dir}"
assert os.path.exists(val_dir), f"Validation directory not found: {val_dir}"


# Distributed setup
def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{rank}")  # Use rank to assign GPU
    return device

def cleanup():
    dist.destroy_process_group()


# Initialize distributed training
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = setup(local_rank, world_size)
print(f"Local Rank: {local_rank}, World Size: {world_size}, Device: {device}")

# Loading the datasets using ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

# Creating DataLoaders with distributed samplers
train_sampler = DistributedSampler(train_data, shuffle=True)
val_sampler = DistributedSampler(val_data, shuffle=False)

# Batch size per GPU
batch_size = 512

# Creating DataLoaders for efficient data loading
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=True)

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

        # Check for NaN values in the input data
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("NaN or inf values detected in the input data!")
            continue

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

        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

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
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Check for NaN values in the input data
            if torch.isnan(images).any() or torch.isinf(images).any():
                print("NaN or inf values detected in the validation data!")
                continue

            # AMP for validation
            with torch.amp.autocast('cuda'):
                outputs = net(images)
                loss = criterion(outputs, labels)

            # Track loss and accuracy
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Reduce across all GPUs
    total_loss_tensor = torch.tensor(total_loss, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    total_tensor = torch.tensor(total, device=device)

    # Perform all-reduce operation to sum values across all GPUs
    dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Only GPU 0 should print the final validation accuracy
    if dist.get_rank() == 0:
        total_loss = total_loss_tensor.item() / dist.get_world_size()
        val_acc = 100 * correct_tensor.item() / total_tensor.item()
        return total_loss, val_acc
    else:
        return None, None  # Other GPUs return None


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


# Model setup with DDP
net = NFNetWrapper(num_classes=len(train_data.classes)).to(device)
net = DDP(net, device_ids=[local_rank], output_device=local_rank)  # Use DDP
net = torch.compile(net)  # Optimize the model with Torch Compile

# Learning rate scaling
base_lr = 0.01  # Reduced base learning rate to prevent instability
effective_batch_size = batch_size * world_size  # Total batch size across all GPUs
max_lr = base_lr * (effective_batch_size / 256)  # Scale learning rate

num_epochs = 120  # Set the number of epochs to 120
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

warmup_epochs = 5
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
scaler = torch.cuda.amp.GradScaler()

# Generate a dissimilar class order
#dissimilar_order = get_dissimilar_order(train_data)

# Train the model normally with the dissimilar order
if local_rank == 0:
    print("\nStarting normal training")

for epoch in range(num_epochs):
    # Training step
    train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn=augmentation_fn)

    # Validation step
    val_loss, val_acc = validate(net, val_loader, criterion, device)

    # Step the scheduler
    scheduler.step()

    if val_loss is not None and val_acc is not None:
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    torch.cuda.empty_cache()

# Destroy process group after training
cleanup()
