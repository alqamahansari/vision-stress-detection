from torchvision import datasets, transforms
from collections import Counter

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="data/train", transform=transform)

labels = [label for _, label in dataset]
counts = Counter(labels)

print("Class distribution:")
for cls, count in counts.items():
    print(dataset.classes[cls], count)