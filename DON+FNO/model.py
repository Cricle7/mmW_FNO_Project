import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from basic_layers import *
class DeepONetFNO_LightningModel(pl.LightningModule):
    def __init__(self,
                 deepo_net: nn.Module,
                 fno_net: nn.Module,
                 Nx=32, Ny=32,
                 lr=1e-3,
                 weight_decay=1e-6):
        """
        deepo_net: DeepONet 对象
        fno_net  : FNO 对象
        Nx, Ny   : 将 DeepONet 的输出重新映射到 Nx x Ny 的网格上(示例)
        """
        super().__init__()
        self.deepo_net = deepo_net
        self.fno_net = fno_net
        self.Nx = Nx
        self.Ny = Ny

        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, func_vals, coords_1d, src=None):
        """
        func_vals : (B, branch_in_dim)        - DeepONet 分支输入
        coords_1d : (B, Nx*Ny, 2)            - DeepONet 主干输入 (二维坐标)
        src       : (B, Nx, Ny) 或 (B, Nx, Ny, 2) (可选，若要做复数乘法)
        返回      : FNO 预测结果
        """
        # 1. 使用 DeepONet 得到 Nx*Ny 个坐标点上的函数值
        B, total_points, _ = coords_1d.shape
        # out_deepo: (B, Nx*Ny)
        out_deepo = self.deepo_net(func_vals, coords_1d)  # (B, Nx*Ny)

        # 2. 将 (B, Nx*Ny) reshape 为 (B, Nx, Ny)
        out_deepo_2d = out_deepo.view(B, self.Nx, self.Ny)
        # 3. 如果需要在 FNO 中拼接坐标，可以获取 get_grid2D:
        grid = get_grid2D(out_deepo_2d.shape, out_deepo_2d.device)  # (B, Nx, Ny, 2)

        # 4. 构造 FNO 的输入 (B, Nx, Ny, C)
        #    这里简单示例：将 DeepONet 的函数值当作一维通道 + grid坐标 => dim_input = 3
        #    如果 fno_net 初始化时 with_grid=True, dim_input=1+2=3
        x_fno_in = torch.cat([out_deepo_2d.unsqueeze(-1), grid], dim=-1)
        # x_fno_in: (B, Nx, Ny, 3)

        # 5. 传入 FNO
        out_fno = self.fno_net(x_fno_in, src=src)
        return out_fno

    def training_step(self, batch, batch_idx):
        """
        假设 batch = (func_vals, coords_1d, src, y_true)
        """
        func_vals, coords_1d, src, y_true = batch
        pred = self(func_vals, coords_1d, src)
        # 如果 pred 是复数，视情况取实部/模等
        if torch.is_complex(pred):
            pred = pred.real  # 这里仅示例取实部

        loss = self.criterion(pred.view(pred.size(0), -1),
                              y_true.view(y_true.size(0), -1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        func_vals, coords_1d, src, y_true = batch
        pred = self(func_vals, coords_1d, src)
        if torch.is_complex(pred):
            pred = pred.real
        val_loss = self.criterion(pred.view(pred.size(0), -1),
                                  y_true.view(y_true.size(0), -1))
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 也可使用余弦退火、StepLR 等
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # 按验证集 loss 调整学习率
            },
        }