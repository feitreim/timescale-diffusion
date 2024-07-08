import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from blocks.unet import UNet

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# Hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 10

# MNIST Dataset and DataLoader
transform = v2.Compose(
    [v2.ToTensor(), v2.Resize([32, 32]), v2.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./datasets", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Model (replace this with your own model)
class ClassificationUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            depth=4,
            down_layers=["ResDown", 'AttnDown', 'AttnDown'],
            up_layers=['AttnUp', 'AttnUp', 'ResUp'],
            in_dims=3,
            h_dims=512,
            out_dims=3,
            kernel_size=3,
            padding=1,
            e_dims=64
        )
        self.ff = nn.Linear(10 * 32 * 32, 10)

    def forward(self, x):
        x = self.model(x)
        x = self.ff(x.flatten(1))
        return x


model = ClassificationUNet()
summary(model, input_size=(batch_size, 1, 32, 32))
model = torch.compile(model, fullgraph=True, mode="max-autotune")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
            )

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the 10000 test images: {100 * correct / total}%")
