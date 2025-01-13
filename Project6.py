# Importing the required libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets
from timm import create_model
from sklearn.cluster import KMeans
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.parallel import DataParallel

# Paths to the training and validation datasets
data_path = '/ds/images/imagenet'
train_dir = f"{data_path}/train"
val_dir = f"{data_path}/val_folders"

# Setting up transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),  # Randomly crop and resize images to a fixed size
    transforms.RandomHorizontalFlip(),         # Add random horizontal flipping for variation
    transforms.ToTensor(),                     # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with standard ImageNet mean
                         std=[0.229, 0.224, 0.225])   # and standard deviation
])

# Checking if the dataset directories exist
assert os.path.exists(train_dir), f"Training directory not found: {train_dir}"
assert os.path.exists(val_dir), f"Validation directory not found: {val_dir}"

# Loading the datasets using ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

# Creating DataLoaders for efficient data loading
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=30)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=30)

# Define a wrapper for the NFNet model
class NFNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super(NFNetWrapper, self).__init__()
        self.model = create_model('nfnet_f0', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# Function to train the model for one epoch
def train_epoch(net, train_loader, criterion, optimizer, scaler, device):
    net.train()  # Set the model to training mode
    total_loss, correct, total = 0, 0, 0

    # Loop through the training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the GPU
        optimizer.zero_grad()

        # Forward pass with AMP (Automatic Mixed Precision)
        # I have used AMP to help speed up computations and reduce memory usage without affevting model accuracy
        with torch.cuda.amp.autocast():
            outputs = net(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track loss and accuracy
        total_loss += loss.item()
        _, preds = outputs.max(1)  # Get predictions
        correct += (preds == labels).sum().item()
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
def train_incrementally(net, train_data, val_loader, order, criterion, optimizer, scaler, device, step_size=10, num_epochs=1):
    net = DataParallel(net)  # Enable multi-GPU training
    all_acc = []  # Store accuracies for each step

    # Loop through subsets of classes
    for step in range(2, len(train_data.classes) + 2, step_size):
        current_classes = order[:step]  # Get the current subset of classes
        train_indices = [i for i, (_, label) in enumerate(train_data.samples) if label in current_classes]
        train_subset = Subset(train_data, train_indices)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=30)

        # Train for the specified number of epochs
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, scaler, device)

        # Validate after training
        val_loss, val_acc = validate(net, val_loader, criterion, device)
        all_acc.append(val_acc)
        print(f"Training with {len(current_classes)} classes: Accuracy after training: {val_acc:.2f}%")

    return all_acc

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
        loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=30)

        embeddings = []
        for images, _ in loader:
            images = images.to(device)
            with torch.no_grad():
                embedding = resnet(images).mean(dim=0).cpu().numpy()
                embeddings.append(embedding)
        class_features.append(np.mean(embeddings, axis=0))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(class_features)
    return sorted(range(len(train_data.classes)), key=lambda x: kmeans.labels_[x])

# Set up the device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = NFNetWrapper(num_classes=len(train_data.classes)).to(device)
net = torch.compile(net)  # Optimize the model with Torch Compile

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
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

# Train with the dissimilar order
print("\nTraining with Dissimilar Order")
net = torch.compile(net)
train_incrementally(net, train_data, val_loader, dissimilar_order, criterion, optimizer, scaler, device)
