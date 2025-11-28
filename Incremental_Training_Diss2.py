# Importing the required libraries
import os
import copy
import random 
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
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm.loss import SoftTargetCrossEntropy
from PIL import Image

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
            if all(len(centroids[l]) >= num_samples for l in centroids):
                break
    for label in centroids:
        centroids[label] = torch.stack(centroids[label], dim=0).mean(dim=0)
    return centroids

def compute_dissimilarity_matrix(centroids):
    num_classes = len(centroids)
    distances = torch.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            distances[i, j] = torch.dist(centroids[i], centroids[j], p=2)
    return distances

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

def create_distance_based_order_indices(dataset, class_order):
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        class_indices.setdefault(label, []).append(idx)
    for label in class_indices:
        shuffle(class_indices[label])
    dissimilar_order = []
    while any(class_indices[label] for label in class_order):
        for label in class_order:
            if class_indices[label]:
                dissimilar_order.append(class_indices[label].pop())
    return dissimilar_order

augmentation_fn = Mixup(
    mixup_alpha=0.2,  # Probability for MixUp
    cutmix_alpha=1.0,  # Probability for CutMix
    prob=1.0,  # Always apply MixUp or CutMix
    switch_prob=0.5,  # 50% chance to switch between MixUp and CutMix
    mode='batch',  # Apply augmentation to the entire batch
    num_classes=len(train_data.classes),
    label_smoothing=0.0
)

# Define a wrapper for the NFNet model
class NFNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super(NFNetWrapper, self).__init__()
        self.model = create_model('nfnet_f0', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# This handles both soft (MixUp) and standard integer labels in the same forward pass.
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.soft_ce = SoftTargetCrossEntropy()
        self.hard_ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # If targets is 1D (integer labels), use normal CE
        # If targets is 2D (soft labels from MixUp/CutMix), use soft CE
        if targets.ndim == 1:
            if targets.numel() == outputs.size(0) * outputs.size(1):
                targets = targets.view_as(outputs)
                return self.soft_ce(outputs, targets)
            else:
                return self.hard_ce(outputs, targets)
        else:
            return self.soft_ce(outputs, targets)
            
def compute_ewc_loss(model, old_params, lambda_reg):
    ewc_loss = 0.0
    for param, old_param in zip(model.parameters(), old_params):
        ewc_loss += ((param - old_param)**2).sum()
    return lambda_reg * ewc_loss

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, new_indices, replay_buffer):
        self.train_data = train_data
        self.new_indices = new_indices  # indices for new samples (with integer labels)
        self.replay_buffer = replay_buffer  # list of tuples (image, soft_label)
        self.num_classes = len(train_data.classes)  # We'll use this for one-hot

    def __len__(self):
        return len(self.new_indices) + len(self.replay_buffer)

    def __getitem__(self, idx):
        if idx < len(self.new_indices):
            # New sample -> integer label -> convert to one-hot
            image, label_int = self.train_data[self.new_indices[idx]]
            oh_label = torch.zeros(self.num_classes, dtype=torch.float32)
            oh_label[label_int] = 1.0
            return image, oh_label
        else:
            # For replay samples, return stored image and soft label (already shape [num_classes])
            return self.replay_buffer[idx - len(self.new_indices)]
      
# Function to train the model for one epoch
def train_epoch(net, train_loader, criterion, optimizer, scaler, device, augmentation_fn=None, ewc_old_params=None, lambda_reg=0.001):
    net.train()  # Set the model to training mode
    total_loss, correct, total = 0, 0, 0
    
    # Initialize sample counter for this epoch
    samples_this_epoch = 0

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
            if ewc_old_params is not None:
                loss = loss + compute_ewc_loss(net, ewc_old_params, lambda_reg)

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
        # If using soft labels (replay exemplars), use argmax for comparison
        """
        if augmentation_fn:
            correct += (preds == labels.argmax(dim=1)).sum().item()
        else:
            correct += (preds == labels).sum().item()
        """
        true_labels = labels if labels.ndim == 1 else labels.argmax(dim=1)
        correct += (preds == true_labels).sum().item()
        total += labels.size(0)
        
        # Accumulate the number of samples processed in this epoch
        samples_this_epoch += labels.size(0)


    # Return average loss and accuracy for the epoch
    return total_loss / len(train_loader), 100 * correct / total, samples_this_epoch


# Function to validate the model
def validate(net, val_loader, criterion, device):
    net.eval()  # Set the model to evaluation mode
    total_loss, correct, total = 0.0, 0, 0

    # We'll also convert val labels to one-hot in-line
    if hasattr(val_loader.dataset, 'dataset'):
        num_classes = len(val_loader.dataset.dataset.classes)
    else:
        num_classes = len(val_loader.dataset.classes)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            # Convert integer labels -> one-hot
            # We'll create a batch of shape [batch_size, num_classes].
            one_hot_labels = torch.zeros(labels.size(0), num_classes, device=device)
            for i in range(labels.size(0)):
                one_hot_labels[i, labels[i]] = 1.0
            labels = one_hot_labels
            
            # AMP for validation
            with torch.amp.autocast('cuda'):
                outputs = net(images)
                loss = criterion(outputs, labels)
            
            batch_size_val = labels.size(0)
            total_loss += loss.item() * batch_size_val
            # For accuracy: compare argmax
            preds = outputs.argmax(dim=1)
            true  = labels.argmax(dim=1)
            correct += (preds == true).sum().item()
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

# Function to incrementally train the model
def train_incrementally(net, train_data, val_data, order, criterion, optimizer, scaler, device, step_size=50, augmentation_fn=None):
    all_acc = []  # Store accuracies for each step
    incremental_epoch_counter = 0  # Track total epochs across all steps
    old_classes = set()  # Keep track of classes seen so far
    replay_buffer = []  # List to store replay exemplars: (image, soft_label)
    exemplar_per_class = 100 
    lambda_reg = 0.001  # Regularization strength
    old_params = None  # For EWC regularization
    
    normal_epoch_samples = len(train_data) / dist.get_world_size()
    samples_so_far = 0  

    # Loop through subsets of classes
    for step in range(step_size, len(train_data.classes) + 1, step_size):
        current_classes = set(order[:step])
        new_classes = current_classes - old_classes
        current_epochs = 12
        
        if dist.get_rank() == 0:
            print(f"\nIncremental Step {step}/{len(train_data.classes)}: Training with {len(current_classes)} classes.")
            print(f"New classes in this step: {new_classes}")

        # Create combined dataset
        all_data_indices = [i for i, (_, label) in enumerate(train_data.samples) if label in current_classes]
        combined_dataset = CombinedDataset(train_data, all_data_indices, replay_buffer)
        
        # Create distributed sampler and loader
        train_sampler_local = DistributedSampler(combined_dataset, shuffle=True)
        train_loader_step = DataLoader(combined_dataset, batch_size=batch_size, 
                                     sampler=train_sampler_local, num_workers=8, 
                                     pin_memory=True, drop_last=True)

        # Create teacher model
        teacher_model = copy.deepcopy(net.module)
        teacher_model.eval()
        teacher_model.to(device)

        # Training phase
        
        for epoch in range(current_epochs):
            incremental_epoch_counter += 1
            train_loss, train_acc, epoch_samples = train_epoch(
                net, train_loader_step, criterion, optimizer, scaler, device, 
                augmentation_fn, old_params, lambda_reg
            )
            
            # Validation
            current_val_loader = DataLoader(val_data, batch_size=batch_size, 
                                shuffle=False, num_workers=8, 
                                pin_memory=True)
            val_loss, val_acc = validate(net, current_val_loader, criterion, device)

            # Update tracking
            samples_so_far += epoch_samples
            
            while samples_so_far >= normal_epoch_samples:
                scheduler.step()
                samples_so_far -= normal_epoch_samples
                
            if dist.get_rank() == 0:
                print(f"Total Epoch {incremental_epoch_counter}/{num_epochs}, "
                      f"Step {step}/{len(train_data.classes)}, Epoch {epoch+1}/{current_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Exemplar collection
        if dist.get_rank() == 0:
            print(f"Samples trained in this step: {samples_so_far}")
            
        for cls in new_classes:
            cls_indices = [i for i, (_, label) in enumerate(train_data.samples) 
                          if label == cls]
            if len(cls_indices) > exemplar_per_class:
                # Synchronized index selection
                if dist.get_rank() == 0:
                    sampled_indices = random.sample(cls_indices, exemplar_per_class)
                else:
                    sampled_indices = [0] * exemplar_per_class
                
                sampled_indices = torch.tensor(sampled_indices, device=device)
                dist.broadcast(sampled_indices, src=0)
                sampled_indices = sampled_indices.cpu().tolist()
            else:
                sampled_indices = cls_indices

            # Store exemplars with validation transforms
            for idx in sampled_indices:
                img_path, _ = train_data.samples[idx]
                image = Image.open(img_path).convert('RGB')
                image = val_transform(image).to(device).unsqueeze(0)
                
                with torch.no_grad():
                    teacher_output = teacher_model(image)
                    soft_label = torch.softmax(teacher_output, dim=1).squeeze(0)
                replay_buffer.append((image.squeeze(0).cpu(), soft_label.cpu()))

        # Update state
        old_classes = current_classes.copy()
        old_params = [p.detach().clone() for p in net.parameters()]
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
net = torch.compile(net)  # Optimize the model with Torch Compile

num_epochs = 240 
criterion = CombinedLoss().to(device)
world_size = dist.get_world_size()
effective_batch_size = batch_size * world_size

max_lr = 0.01 * (effective_batch_size / 256)  
optimizer = optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=5e-4)

warmup_epochs = 15
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
scaler = torch.cuda.amp.GradScaler()

# Generate class orders
alphabetical_order = list(range(len(train_data.classes)))
random_order = np.random.permutation(len(train_data.classes)).tolist()
feature_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_model.fc = nn.Identity()  # Remove the classification head
feature_model.to(device)

# Compute centroids and distance matrix
centroids = compute_class_centroids(train_data, feature_model, device, num_samples=10)
dist_matrix = compute_dissimilarity_matrix(centroids)
# Generate dissimilar class order
dissimilar_order = generate_class_order(dist_matrix)





print("\nTraining with Dissimilar Order")

incremental_acc_alpha = train_incrementally(
    net, train_data, val_data, dissimilar_order, criterion, optimizer, scaler, device, step_size=50, augmentation_fn=None
)

# Print final validation accuracy for each step
if local_rank == 0:
    print(f"Validation Accuracies at each step: {incremental_acc_alpha}")


# Destroy process group after training
dist.destroy_process_group()
