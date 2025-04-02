import torch.nn as nn
import torch


class TransformerBlock(nn.Module):
    """Transformer Block to capture non-local dependencies."""
    def __init__(self, channels):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm([channels, 32, 32])  # Assuming input size is 32x32
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Flatten spatial dimensions
        x = self.relu(self.linear1(self.norm(x)))
        x = self.linear2(x)
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)  # Reshape back
        return x
