import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from model import TomatoCNN
from datetime import datetime
import json
import numpy as np

# Paths
data_dir = "data"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Safe Image Loader
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, target
        except Exception:
            print(f"Skipped corrupted image: {path}")
            return self.__getitem__((index + 1) % len(self.samples))

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset & Class Labels
dataset = SafeImageFolder(data_dir, transform=train_transform)
num_classes = len(dataset.classes)
class_names = dataset.classes

# Save class names for app use
with open(os.path.join(model_dir, "class_names.json"), "w") as f:
    json.dump(class_names, f)

# Train/Validation Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TomatoCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
best_loss = float("inf")
for epoch in range(10):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            _, predicted = torch.max(outputs, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    accuracy = 100 * correct / total

    print(f"[INFO] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.2f}%")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Best model saved to {model_path}")

# Final Evaluation: Confusion Matrix & Report
print("\n[INFO] Generating final evaluation on validation set...")
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for val_images, val_labels in val_loader:
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        outputs = model(val_images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(val_labels.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, f"confusion_matrix_{timestamp}.png"))
plt.close()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names)
with open(os.path.join(model_dir, f"classification_report_{timestamp}.txt"), "w") as f:
    f.write(report)

print("[INFO] Evaluation complete. Confusion matrix and report saved.")
