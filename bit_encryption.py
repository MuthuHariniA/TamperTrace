import torch
import torch.nn as nn
import torch.optim as optim

class BitEncryptionModule(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super(BitEncryptionModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def encrypt_watermark(watermark_tensor, model):
    """Encrypts a watermark tensor using the BitEncryptionModule."""
    with torch.no_grad():
        encrypted_tensor = model(watermark_tensor)
    return encrypted_tensor
