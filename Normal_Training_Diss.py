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
import torch.nn.functional as F
from random import shuffle
import random

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

val_transform = transforms.Compose([
    transforms.Resize(256),                    # Resize shorter side to 256
    transforms.CenterCrop(224),                # Center crop to 224x224
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
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

batch_size = 128

# For each class, I compute the centroid of features obtained from a few samples.
def compute_class_centroids(dataset, feature_model, device, num_samples=10):
    centroids = {label: [] for label in range(len(dataset.classes))}
    feature_model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            if len(centroids[label]) < num_samples:
                image = image.unsqueeze(0).to(device)
                feat = feature_model(image)
                centroids[label].append(feat.squeeze(0).cpu())
            # Stop early if all classes have enough samples
            if all(len(centroids[l]) >= num_samples for l in centroids):
                break
    # Average features to get a centroid per class
    for label in centroids:
        centroids[label] = torch.stack(centroids[label], dim=0).mean(dim=0)
    return centroids

# A matrix of Euclidean distances between class centroids is computed.
def compute_dissimilarity_matrix(centroids):

    num_classes = len(centroids)
    distances = torch.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            a = centroids[i]
            b = centroids[j]
            distances[i, j] = torch.dist(a, b, p=2)
    return distances

# Generating class order by starting from an arbitrary class repeatedly choosing 
# the next class that is most dissimilar (Farthest from current one)
def generate_class_order(dist_matrix):

    num_classes = dist_matrix.shape[0]
    remaining = list(range(num_classes))
    order = []
    current = remaining.pop(0)  # Start with the first class
    order.append(current)
    while remaining:
        next_class = max(remaining, key=lambda x: dist_matrix[current, x])
        order.append(next_class)
        remaining.remove(next_class)
        current = next_class
    return order

# I am grouping indices by class, shuffle within each group and then interleave
# them according to the provided class order 
def create_distance_based_order_indices(dataset, class_order):

    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        class_indices.setdefault(label, []).append(idx)
    # Shuffle indices within each class for some additional randomness
    for label in class_indices:
        shuffle(class_indices[label])
    dissimilar_order = []
    # Interleave samples according to the computed class order
    while any(class_indices[label] for label in class_order):
        for label in class_order:
            if class_indices[label]:
                dissimilar_order.append(class_indices[label].pop())
    return dissimilar_order
    
feature_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_model.fc = nn.Identity()  # Remove the classification head
feature_model.to(device)

# Compute centroids for each class using a few samples
centroids = compute_class_centroids(train_data, feature_model, device, num_samples=10)

# Compute the dissimilarity matrix between class centroids
dist_matrix = compute_dissimilarity_matrix(centroids)

# Generate an ordering of classes that maximizes dissimilarity between consecutive classes
class_order = generate_class_order(dist_matrix)

# Create dataset indices based on the computed class order
indices = create_distance_based_order_indices(train_data, class_order)


# Create a Subset and DataLoader with the dissimilar order
train_subset = Subset(train_data, indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, drop_last=True)

# Validation DataLoader remains using DistributedSampler
val_sampler = DistributedSampler(val_data, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler,
                        num_workers=8, pin_memory=True, drop_last=True)

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
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # AMP for validation
            with torch.amp.autocast('cuda'):
                outputs = net(images)
                loss = criterion(outputs, labels)
            
            batch_size_val = labels.size(0)
            total_loss += loss.item() * batch_size_val  # weight loss by batch size
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += batch_size_val

    # Create tensors from the local sums
    total_loss_tensor = torch.tensor(total_loss, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    total_tensor = torch.tensor(total, device=device)

    # Aggregate the sums across all GPUs
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    global_loss = total_loss_tensor.item() / total_tensor.item()
    global_acc = 100.0 * correct_tensor.item() / total_tensor.item()
    return global_loss, global_acc


# Set up the device and model
# Distributed initialization
local_rank = int(os.getenv('LOCAL_RANK', 0))  # Get local rank
torch.cuda.set_device(local_rank)  # Map local rank to GPU
device = torch.device(f'cuda:{local_rank}')

# Model setup with DDP
net = NFNetWrapper(num_classes=len(train_data.classes)).to(device)
net = DDP(net, device_ids=[local_rank], output_device=local_rank)  # Use DDP
net = torch.compile(net)  # Optimize the model with Torch Compile
effective_batch_size =  batch_size * dist.get_world_size()

num_epochs = 120  # Set the number of epochs to 120
criterion = nn.CrossEntropyLoss().to(device)
max_lr = 0.1 * (effective_batch_size / 256)
optimizer = optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

warmup_epochs = 5
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
scaler = torch.cuda.amp.GradScaler()



if local_rank == 0:
    print("\nStarting normal training with dissimilar order...")

for epoch in range(num_epochs):
    # Training step
    train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn=augmentation_fn)
    
    # Validation step
    val_loss, val_acc = validate(net, val_loader, criterion, device)
    
    # Step the scheduler
    scheduler.step()
    
    if local_rank == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    torch.cuda.empty_cache()
    
              
# Destroy process group after training
dist.destroy_process_group()
