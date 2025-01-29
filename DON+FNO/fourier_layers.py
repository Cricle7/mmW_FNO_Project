import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierConv2d(nn.Module):
    """
    做一次 2D rFFT -> 频域截断 -> 复数乘法 -> iFFT
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y, 2) * (in_channel, out_channel, x, y, 2) -> (batch, out_channel, x, y, 2)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        x_ft = torch.view_as_real(x_ft)  # => (batch, in_channels, H, W_half, 2)

        batchsize, in_channels, height, width_half, _ = x_ft.shape
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width_half,
            2,
            device=x.device,
            dtype=x.dtype,
        )

        # 正频率
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # 负频率
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FourierLayer(nn.Module):
    def __init__(self, features_, wavenumber, is_last=False):
        super().__init__()
        self.W = nn.Conv2d(features_, features_, 1)
        self.fourier_conv = FourierConv2d(features_, features_, *wavenumber)
        if not is_last:
            self.act = F.gelu
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2)