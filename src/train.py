import torch
from model import EmotionCNN
from dataset import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders("data", batch_size=16)

print("Class mapping:", train_loader.dataset.classes)

model = EmotionCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("----------------------------")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/emotion_model.pth")

print("Model saved successfully.")