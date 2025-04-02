import torch
import torch.nn as nn

class MaskExtractor(nn.Module):
    def __init__(self):
        super(MaskExtractor, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
