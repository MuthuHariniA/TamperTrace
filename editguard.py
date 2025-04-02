import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader import get_data_loader

class EditGuard(nn.Module):
    def __init__(self):
        super(EditGuard, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Reduce feature map size
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Match actual feature map size
        self.fc2 = nn.Linear(256, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # Reduce size dynamically
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def get_model():
    return EditGuard()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    dataloader = get_data_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    print("Model structure:", model)

    for epoch in range(5):  # Train for 5 epochs
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            print("Image shape:", images.shape)  # Debugging

            optimizer.zero_grad()
            outputs = model(images)
            print("Output shape:", outputs.shape)  # Debugging

            loss = loss_fn(outputs[:, 1], labels.float())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

