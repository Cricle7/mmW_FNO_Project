import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from deeponet import DeepONetMulti
from fno_blocks import FNOBlock


class DeepONetFNOModel(pl.LightningModule):
    """
    组合: DeepONet + FNO + FNO + FNO

    架构思路:
    1) DeepONet: 输入 (函数采样 + 坐标grid), 输出 (batch, NxNy, 1)
    2) reshape => (batch, 1, Nx, Ny)
    3) 第1个 FNOBlock => (batch, 1, Nx, Ny)
    4) 第2个 FNOBlock => (batch, 1, Nx, Ny)
    5) 第3个 FNOBlock => (batch, 1, Nx, Ny)
    """

    def __init__(
        self,
        deeponet_cfg,
        fno_width=32,
        fno_modes=16,
        Nx=10,
        Ny=10,
        in_channels_for_fno=3,
        **kwargs
    ):
        """
        deeponet_cfg: dict, 用于实例化 DeepONetMulti 的参数
        Nx, Ny: 空间离散大小
        in_channels_for_fno=3 表示: 1 通道(DeepONet输出) + 2 通道(坐标) => 3通道
        """
        super().__init__()
        self.save_hyperparameters()

        # 1) 初始化DeepONet
        self.deeponet = DeepONetMulti(**deeponet_cfg)

        # 2) 初始化三个 FNOBlock
        self.fno1 = FNOBlock(
            in_channels=in_channels_for_fno,
            out_channels=1,
            width=fno_width,
            modes=fno_modes,
            n_layers=1,  # 您可根据需要加深
        )
        self.fno2 = FNOBlock(
            in_channels=1 + 2,  # 因为会再次拼坐标(1 通道输出 + 2D坐标)
            out_channels=1,
            width=fno_width,
            modes=fno_modes,
            n_layers=1,
        )
        self.fno3 = FNOBlock(
            in_channels=1 + 2,
            out_channels=1,
            width=fno_width,
            modes=fno_modes,
            n_layers=1,
        )

        self.Nx = Nx
        self.Ny = Ny

        # 这里的损失函数可视需求替换
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        x: (B, Nx, Ny, 1) 表示函数采样, 
        注: 也可传 (B, 1, Nx, Ny), 看您如何组织.
        """
        B = x.shape[0]
        # 先用 DeepONet 编码 => out_deeponet: (B, Nx*Ny, 1)
        # DeepONet 需要 trunk 输入(坐标)在 self.deeponet.grid 中, 需要 branch 输入 = x.view(B, -1)
        # 这里调用前, 确保 self.deeponet.grid = (1, Nx*Ny, 2)
        x_branch = x.view(B, -1)
        if self.deeponet.grid is None:
            raise ValueError("Must provide grid to self.deeponet for trunk input")

        y_var = self.deeponet.grid.repeat(B, 1, 1)  # (B, Nx*Ny, 2)
        out_deeponet = self.deeponet([x_branch], y_var)  # => (B, Nx*Ny, 1)

        # reshape => (B, Nx, Ny, 1)
        out_dponet_2d = out_deeponet.view(B, self.Nx, self.Ny, 1)

        # 现在我们想将(输出 + 坐标) 作为FNO的输入
        # => (B, Nx, Ny, 3) = concat( out_dponet_2d, grid[...,0], grid[...,1] )
        # 其中 grid_2d: (1, Nx, Ny, 2) => repeat到 (B, Nx, Ny, 2)
        grid_2d = self.deeponet.grid.view(1, self.Nx, self.Ny, 2).repeat(B, 1, 1, 1)
        # 拼通道
        fno_in = torch.cat([out_dponet_2d, grid_2d], dim=-1)  # (B, Nx, Ny, 3)

        # 调整为 (B, 3, Nx, Ny)
        fno_in = fno_in.permute(0, 3, 1, 2).contiguous()
        # 第1个FNO
        fno_out1 = self.fno1(fno_in)  # => (B, 1, Nx, Ny)

        # 拼坐标 => (B, (1+2), Nx, Ny)
        fno_in2 = torch.cat([fno_out1, grid_2d.permute(0, 3, 1, 2)], dim=1)
        fno_out2 = self.fno2(fno_in2)  # => (B, 1, Nx, Ny)

        # 第3个FNO
        fno_in3 = torch.cat([fno_out2, grid_2d.permute(0, 3, 1, 2)], dim=1)
        fno_out3 = self.fno3(fno_in3)  # => (B, 1, Nx, Ny)

        return fno_out3  # 最终输出

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x: (B, Nx, Ny, 1), y: (B, Nx, Ny, 1)
        pred = self.forward(x)  # => (B, 1, Nx, Ny)
        loss = self.criterion(pred, y.permute(0, 3, 1, 2))  # y也调成 (B,1,Nx,Ny)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        val_loss = self.criterion(pred, y.permute(0, 3, 1, 2))
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # 简单示例: 使用AdamW, 或自行替换
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return [optimizer], [scheduler]