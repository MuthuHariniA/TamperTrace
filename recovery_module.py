import torch
import torch.nn as nn

class RecoveryModule(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super(RecoveryModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def recover_watermark(encrypted_tensor, model):
    """Recovers a watermark tensor using the RecoveryModule."""
    with torch.no_grad():
        recovered_tensor = model(encrypted_tensor)
    return recovered_tensor
