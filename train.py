import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import os

# Hyperparameters
batch_size = 16
epochs = 5
learning_rate = 0.001

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define tasks and datasets
tasks = {
    "face_recognition": "datasets/face_recognition",
    "mask_detection": "datasets/face_mask_detection",
    "lighting": "datasets/lighting_conditions",
    "crowd_density": "datasets/crowd_density",
    "suspicious_object": "datasets/suspicious_objects",
    "animal_intrusion": "datasets/animal_intrusion",
    "motion_detection": "datasets/motion_detection"
}

# Train a model for each task
for task, dataset_path in tasks.items():
    if not os.path.exists(dataset_path):
        print(f"Skipping {task}: Dataset folder not found.")
        continue

    print(f"\n🔄 Training model for: {task}")

    # Load dataset
    train_dataset = ImageFolder(dataset_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - {task} Loss: {running_loss/len(train_loader)}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{task}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved: {model_path}")
