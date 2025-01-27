import torch
import torch.nn as nn
import torch.nn.functional as F
from fourier_layers import FourierLayer


class FNOBlock(nn.Module):
    """
    一个FNOBlock包含若干个FourierLayer(示例n_layers=1或2等),
    并在入口/出口用1x1卷积来变换通道数.
    """
    def __init__(self, in_channels=3, out_channels=1, width=32, modes=16, n_layers=1):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, width, 1)
        layers = []
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            layers.append(FourierLayer(width, modes, is_last=is_last))
        self.fno_layers = nn.Sequential(*layers)
        self.out_proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.in_proj(x)      # => (B, width, H, W)
        x = self.fno_layers(x)   # => (B, width, H, W)
        x = self.out_proj(x)     # => (B, out_channels, H, W)
        return x