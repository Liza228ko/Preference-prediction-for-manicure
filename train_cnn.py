import os
import sys

# Add external libs to path
sys.path.append('/media/liza/B779-017B/ai/python_libs')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

def train_model(data_dir, num_epochs=10, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset (Only Yes and No)
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    # Filter only 'yes' and 'no' classes
    wanted_classes = ['no', 'yes']
    class_to_idx = {cls: i for i, cls in enumerate(wanted_classes)}
    
    indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
               if full_dataset.classes[label] in wanted_classes]
    
    # Create a mapping for the new labels (0 for no, 1 for yes)
    orig_to_new = {full_dataset.class_to_idx[cls]: i for i, cls in enumerate(wanted_classes)}
    
    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, orig_to_new[y]
            
        def __len__(self):
            return len(self.subset)

    # Use raw images (no transform yet) to create the subset
    base_subset = Subset(datasets.ImageFolder(data_dir), indices)
    
    # Split
    train_size = int(0.8 * len(base_subset))
    val_size = len(base_subset) - train_size
    train_indices, val_indices = random_split(range(len(base_subset)), [train_size, val_size])
    
    train_dataset = FilteredDataset(Subset(base_subset, train_indices), transform=data_transforms)
    val_dataset = FilteredDataset(Subset(base_subset, val_indices), transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Balanced Dataset: {len(base_subset)} images ({train_size} train, {val_size} val)")

    # 3. Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    # 4. Training
    print("\nTraining Deep Learning Model (CNN)...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/train_size:.4f}")

    # 5. Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nCNN Evaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=wanted_classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wanted_classes, yticklabels=wanted_classes)
    plt.savefig('cnn_confusion_matrix.png')
    print("Done! saved to 'cnn_confusion_matrix.png'")

if __name__ == "__main__":
    train_model('dataset', num_epochs=8)
