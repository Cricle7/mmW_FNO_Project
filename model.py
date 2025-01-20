import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 如果你之前把 FC_nn, LayerNorm 等放到 utils.py，也可 from utils import FC_nn, LayerNorm
# 下面仅演示核心 FNO

class fourier_conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        in_channels:  输入通道数
        out_channels: 输出通道数
        modes1, modes2: 傅里叶模式截断数量
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        # 实部+虚部用 2 表示
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y, 2) * (in_channel, out_channel, x, y, 2)
        # -> (batch, out_channel, x, y, 2)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x: (batch, in_channels, h, w)
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])
        x_ft = torch.view_as_real(x_ft)  # -> (batch, channel, h, w//2+1, 2)

        batchsize = x.shape[0]
        _, _, height, width_half, _ = x_ft.shape

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width_half,
            2,
            device=x.device,
            dtype=x.dtype
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 逆变换
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Fourier_layer(nn.Module):
    def __init__(self, channels, modes, is_last=False):
        super().__init__()
        self.fourier_conv = fourier_conv_2d(channels, channels, modes, modes)
        self.w = nn.Conv2d(channels, channels, 1)
        self.is_last = is_last

    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.w(x)
        if not self.is_last:
            return F.gelu(x1 + x2)
        else:
            return x1 + x2


class FNO(pl.LightningModule):
    def __init__(self,
                 modes=16,
                 width=32,
                 num_layers=4,
                 lr=1e-3,
                 step_size=20,
                 gamma=0.5,
                 weight_decay=1e-5,
                 eta_min=1e-5):
        super().__init__()
        """
        modes:        傅里叶模式数
        width:        每层通道数
        num_layers:   Fourier_layer 的层数
        """
        self.save_hyperparameters()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        # 假设输入是 2 通道 (实部 + 虚部)
        self.input_channels = 2
        # 假设输出是 1 通道 (灰度图)；如果你的目标图像是 1 通道
        self.output_channels = 1

        # 升维：把 (2) -> (width)
        self.fc0 = nn.Linear(self.input_channels, self.width)

        # 构建若干个 Fourier_layer
        layers = []
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            layers.append(Fourier_layer(self.width, self.modes, is_last=is_last))
        self.fno_layers = nn.Sequential(*layers)

        # 降维：把 (width) -> (1)
        self.fc1 = nn.Linear(self.width, self.output_channels)

        # 损失函数
        self.loss_func = nn.MSELoss()

        # 优化相关
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min

    def forward(self, x):
        """
        x: (batch, H, W, 2)  -> 先转成 (batch, H, W, C), C=2
        我们需要 (batch, C, H, W) 做 2D 卷积 => permute
        """
        # [batch, H, W, 2] => [batch, H*W, 2]
        batchsize, height, width, _ = x.shape
        x = x.view(batchsize, -1, self.input_channels)
        # 过 fc0 => [batch, H*W, width]
        x = self.fc0(x)  # 升维
        # reshape => [batch, width, H, W]
        x = x.view(batchsize, height, width, self.width).permute(0, 3, 1, 2)

        # 经过多层 Fourier_layer
        x = self.fno_layers(x)

        # x: [batch, width, H, W] => [batch, H, W, width]
        x = x.permute(0, 2, 3, 1).contiguous()

        # 降维 => [batch, H*W, 1]
        x = x.view(batchsize*height*width, self.width)
        x = self.fc1(x)
        # 再 reshape 回图像 => [batch, H, W, 1]
        x = x.view(batchsize, height, width, self.output_channels)

        return x

    def training_step(self, batch, batch_idx):
        # 训练的单步
        input_data, target_img = batch  # [batch, H, W, 2], [batch, H, W, 1]
        pred = self(input_data)        # [batch, H, W, 1]
        loss = self.loss_func(pred, target_img)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证的单步
        input_data, target_img = batch
        pred = self(input_data)
        loss = self.loss_func(pred, target_img)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.step_size, eta_min=self.eta_min)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }