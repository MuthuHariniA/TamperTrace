import torch
import torch.nn as nn
import torch.nn.functional as F
from ibsn_model.residual_block import ResidualBlock
from ibsn_model.transformer_block import TransformerBlock  


class IBSN(nn.Module):
    """United Image-bit Steganography Network (IBSN)."""
    def __init__(self, num_res_blocks=5, num_trans_blocks=5, num_prompts=3, channels=64):
        super(IBSN, self).__init__()

        # Residual & Transformer Blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])
        self.trans_blocks = nn.ModuleList([TransformerBlock(channels) for _ in range(num_trans_blocks)])

        # Learnable Degradation Prompts
        self.prompts = nn.Parameter(torch.randn(num_prompts, channels, 1, 1))

        # Global Average Pooling & Softmax Weight Calculation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(channels, num_prompts, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, Irec):
        """Forward pass of IBSN model."""
        x = Irec
        for res_block in self.res_blocks:
            x = res_block(x)

        for trans_block in self.trans_blocks:
            x = trans_block(x)

        # Compute dynamic degradation weights
        gap_features = self.gap(x)
        weights = F.softmax(self.conv1x1(gap_features), dim=1)  # Shape: (B, num_prompts, 1, 1)

        # Apply learned degradation prompts
        prompt_features = sum(w * p for w, p in zip(weights.split(1, dim=1), self.prompts))

        # Upsample and fuse the features
        prompt_features = F.interpolate(prompt_features, size=x.shape[2:], mode="bilinear", align_corners=False)
        enhanced_features = self.conv3x3(prompt_features + x)

        return enhanced_features
