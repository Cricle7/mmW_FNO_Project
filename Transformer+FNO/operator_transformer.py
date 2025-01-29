import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

import math

from encoder_decoder import SpatialEncoder2D, PointWiseDecoder2DSimple


def create_grid(res_h, res_w):
    """
    在 [0,1] x [0,1] 上创建一个 res_h*res_w 的网格点，
    返回形状 [res_h*res_w, 2]
    """
    x_coords = torch.linspace(0, 1, steps=res_w)
    y_coords = torch.linspace(0, 1, steps=res_h)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [res_h, res_w]
    # 最终形状 [res_h*res_w, 2], 先(y坐标, x坐标)
    grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
    return grid  # [res_h*res_w, 2]


class OperatorTransformer(pl.LightningModule):
    """
    LightningModule封装
    """
    def __init__(self,
                 input_channels=2,   # 对应输入的通道数(实部+虚部=2)
                 in_emb_dim=96,
                 out_seq_emb_dim=256,
                 heads=4,
                 depth=6,
                 res=64,
                 latent_channels=256,
                 out_channels=1,  # 预测1通道(灰度图)
                 scale=0.5,
                 learning_rate=1e-3,
                 step_size=100,
                 gamma=0.5,
                 weight_decay=1e-5,
                 eta_min=1e-5):
        super().__init__()

        self.save_hyperparameters()

        self.res = res

        # 构造网格 [1, res*res, 2]
        grid = create_grid(res, res)
        self.register_buffer('grid', grid.unsqueeze(0), persistent=False)

        self.encoder = SpatialEncoder2D(
            input_channels=input_channels + 2,  # +2是把坐标(x,y)也拼进输入做编码 (可选)
            in_emb_dim=in_emb_dim,
            out_seq_emb_dim=out_seq_emb_dim,
            heads=heads,
            depth=depth,
            res=res,
            use_ln=True
        )

        self.decoder = PointWiseDecoder2DSimple(
            latent_channels=latent_channels,
            out_channels=out_channels,
            res=res,
            scale=scale
        )

        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min

        # loss
        self.criterion = nn.MSELoss()
        self.criterion_val = nn.MSELoss()

    def forward(self, x):
        """
        x: [b, 2, h, w]  => (实部, 虚部)
        返回: [b, 1, h, w]
        """
        b, c, h, w = x.shape
        assert h == self.res and w == self.res, "输入分辨率需与 self.res 一致"

        # 先把输入转为 [b, h, w, c]
        x = rearrange(x, 'b c h w -> b h w c')

        # 再 flatten => [b, n, c], n = h*w
        x = rearrange(x, 'b hh ww cc -> b (hh ww) cc')

        # 获取对应坐标 => [1, n, 2] => 扩展到 [b, n, 2]
        pos = self.grid.repeat(b, 1, 1)  # [b, h*w, 2]

        # 在通道维度上拼接坐标
        # x: [b, n, (c+2)]
        x = torch.cat([x, pos], dim=-1)

        # 走 Encoder
        z = self.encoder(x, input_pos=pos)  # [b, n, out_seq_emb_dim=256] or latent_channels

        # 走 Decoder：query_pos = 同样的网格坐标
        out = self.decoder(z, query_pos=pos, context_pos=pos)  # [b, n, out_channels]

        # 最终把 out reshape 回 [b, out_channels, h, w]
        out = rearrange(out, 'b (hh ww) c -> b c hh ww', hh=h, ww=w)
        return out

    def training_step(self, batch, batch_idx):
        """
        batch: (input_data, target_img)
          input_data: [b,2,H,W]
          target_img: [b,1,H,W]
        """
        x, y = batch
        out = self(x)  # [b,1,H,W]
        loss = self.criterion(out, y)
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        val_loss = self.criterion_val(out, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.step_size,
            eta_min=self.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }