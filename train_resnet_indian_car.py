import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from tqdm import tqdm
import json
import numpy as np

#Dataset path
data_dir = "car_dataset"  

#Parameters
num_epochs = 15
batch_size = 32
learning_rate = 0.001
val_split = 0.2
random_seed = 42

#Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
class_names = full_dataset.classes

with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Saved class names to class_names.json")

total_len = len(full_dataset)
val_len = int(total_len * val_split)
train_len = total_len - val_len
torch.manual_seed(random_seed)
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct = 0.0, 0

    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training", unit="batch")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        running_loss += loss.item() * inputs.size(0)
        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / train_len
    train_acc = correct / train_len * 100

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_loss += loss.item() * inputs.size(0)

    val_loss /= val_len
    val_acc = val_correct / val_len * 100

    print(f" Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Loss: {train_loss:.4f}/{val_loss:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), "resnet_indian_car.pth")
        best_val_acc = val_acc
        print(" Best model saved!")

print("Training finished.")
