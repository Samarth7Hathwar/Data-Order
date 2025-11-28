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
    RandAugment(num_ops=2, magnitude=7),        #Apply RandAugment
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

batch_size = 256

# Creating DataLoaders for efficient data loading
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=True)

augmentation_fn = Mixup(
    mixup_alpha=0.2,  # Probability for MixUp
    cutmix_alpha=1.0,  # Probability for CutMix
    prob=0.5,  # Always apply MixUp or CutMix
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
        scaler.unscale_(optimizer)             # Unscale gradients to fp32
        #clip_grad_norm_(net.parameters(), max_norm=1.0)
        clip_grad_value_(net.parameters(), clip_value=0.5)  # Instead of clip_grad_norm_
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

# Function to incrementally train the model
def train_incrementally(net, train_data, val_loader, order, criterion, optimizer, scaler, device, step_size=10, augmentation_fn=None):
    all_acc = []  # Store accuracies for each step
    total_epoch_counter = 0  # Track total epochs across all steps

    # Loop through subsets of classes
    for step in range(step_size, len(train_data.classes) + 1, step_size):
        
        if step <= 200:
            current_epochs = 5
        else:
            current_epochs = 2
        
        # Get the current subset of classes
        current_classes = order[:step]
        if dist.get_rank() == 0:
            print(f"\nIncremental Step {step}/{len(train_data.classes)}: Training with {len(current_classes)} classes.")

        # Filter training samples to include only current classes
        train_indices = [i for i, (_, label) in enumerate(train_data.samples) if label in current_classes]
        train_subset = Subset(train_data, train_indices)

        # Create a DistributedSampler for the subset
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=8, pin_memory=True, drop_last=True)

        # Train for the specified number of epochs
        for epoch in range(current_epochs):
            train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn)
            val_loss, val_acc = validate(net, val_loader, criterion, device)
            
            # Update scheduler after every epoch
            scheduler.step()  
            total_epoch_counter += 1

            # Print with total epoch count
            if dist.get_rank() == 0:
                print(f"Total Epoch {total_epoch_counter}/{num_epochs}, "
                      f"Step {step}/{len(train_data.classes)}, Epoch {epoch + 1}/{current_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Store validation accuracy for the current step
        all_acc.append(val_acc)

    return all_acc


# Set up the device and model
# Distributed initialization
local_rank = int(os.getenv('LOCAL_RANK', 0))  # Get local rank
torch.cuda.set_device(local_rank)  # Map local rank to GPU
device = torch.device(f'cuda:{local_rank}')

# Model setup with DDP
net = NFNetWrapper(num_classes=len(train_data.classes)).to(device)
net = DDP(net, device_ids=[local_rank], output_device=local_rank)  # Use DDP
#net = torch.compile(net)  # Optimize the model with Torch Compile

num_epochs = 260  # Set the number of epochs to 120
criterion = nn.CrossEntropyLoss().to(device)
world_size = dist.get_world_size()
effective_batch_size = batch_size * world_size
max_lr = 0.01 * (effective_batch_size / 256)
optimizer = optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=5e-4)

warmup_epochs = 50
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
scaler = torch.cuda.amp.GradScaler()

# Generate class orders
alphabetical_order = list(range(len(train_data.classes)))
random_order = np.random.permutation(len(train_data.classes)).tolist()

# Train with Random Order
print("\nTraining with Random Order")
# Use the pre-defined `net` (already wrapped in DDP) and optimizer
incremental_acc_random = train_incrementally(
    net, train_data, val_loader, alphabetical_order, criterion, optimizer, scaler, device, step_size=10, augmentation_fn=None
)

# Print final validation accuracy for each step
if local_rank == 0:
    print(f"Validation Accuracies at each step with Random Order: {incremental_acc_random}")

              
# Destroy process group after training
dist.destroy_process_group()
