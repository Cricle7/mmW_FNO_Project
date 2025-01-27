import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F

from basic_layers import FcnSingle


class DeepONetMulti(pl.LightningModule):
    """
    将多个branch net和多个trunk net组合, 形成DeepONet.
    在 forward 时, 分别将branch输入(如函数采样)和 trunk输入(坐标y)进行映射, 最终乘和再求和得到结果.
    """
    def __init__(
        self,
        input_dim=2,
        operator_dims=[10*10],  # 比如用 10x10 => 100 作为 branch 的输入维度
        output_dim=1,
        planes_branch=[64]*3,
        planes_trunk=[64]*3,
        activation="gelu",
        learning_rate=1e-3,
        step_size=100,
        gamma=0.5,
        weight_decay=1e-5,
        eta_min=1e-4,
        grid=None
    ):
        super(DeepONetMulti, self).__init__()
        self.save_hyperparameters()
        self.branches = nn.ModuleList()
        for dim in operator_dims:
            self.branches.append( FcnSingle([dim] + planes_branch, activation=activation))
        self.trunks = nn.ModuleList()
        for _ in range(output_dim):
            self.trunks.append( FcnSingle([input_dim] + planes_trunk, activation=activation))

        self.output_dim = output_dim
        self.grid = grid  # 若需要在内部使用

        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min

        self.criterion = nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, u_vars, y_var):
        """
        u_vars: List[Tensor], 比如 [batch_size, operator_dims_i]
        y_var:  (batch_size, Nxy, input_dim)
        """
        # 1) 先做分支
        #   每个branch输出 => (batch_size, branch_out_dim=planes_branch[-1])
        B = 1.0
        for u_var, branch_net in zip(u_vars, self.branches):
            branch_out = branch_net(u_var)  # => (B, hidden_dim)
            B = B * branch_out  # 逐个相乘(也可以相加, 视算法而定)

        # 2) 主干
        # trunk有 self.output_dim 个 => each trunk => (B, Nxy, trunk_hidden_dim)
        outs = []
        for trunk_net in self.trunks:
            T = trunk_net(y_var)  # => (B, Nxy, hidden_dim末层=planes_trunk[-1])
            # 这里假设 trunk_net 最后一层输出也是 (B, Nxy, hidden_dim=planes_trunk[-1]),
            # 而 branch_out 是 (B, planes_branch[-1]) => 需要匹配???
            # 简单起见, 若 planes_branch[-1] = planes_trunk[-1], 则可做点乘后 sum
            # 下面示例：先unsqueeze在 dim=1, 让维度对齐
            # B: (B, hidden_dim), T: (B, Nxy, hidden_dim)
            # => B.unsqueeze(1): (B, 1, hidden_dim)
            # => multiply => (B, Nxy, hidden_dim) => sum over last dim => (B, Nxy)
            BT = B.unsqueeze(1) * T
            out_val = torch.sum(BT, dim=-1)
            outs.append(out_val)  # (B, Nxy)

        # 最后将多个 trunk 输出拼成 (B, Nxy, output_dim)
        out_var = torch.stack(outs, dim=-1)
        return out_var

    def training_step(self, batch, batch_idx):
        x, y = batch  # x, y: 形状需与 forward 对上
        B = x.shape[0]
        # 假设 x: (B, res, res, 1)，则 flatten => (B, res*res)
        # trunk输入 y_var: (B, res*res, input_dim=2)
        # branch输入 u_var: (B, res*res)
        # 下面只是示例，可根据实际数据组织
        x_branch = x.view(B, -1)  # (B, operator_dim)  e.g. 100
        # trunk: grid => (1, res*res, 2) => repeat到(B, res*res, 2)
        # 这里假设 self.grid.shape = (1, NxNy, 2)
        if self.grid is not None:
            y_var = self.grid.repeat(B, 1, 1)
        else:
            raise ValueError("self.grid is None, please provide a grid for trunk input")

        # forward
        u_vars = [x_branch]  # 如果有多个 branch，可以多加
        out = self(u_vars, y_var)  # => (B, NxNy, output_dim)
        # y: (B, res, res, 1) => (B, res*res, 1)
        loss = self.criterion(
            out.view(B, -1), y.view(B, -1)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B = x.shape[0]
        if self.grid is None:
            raise ValueError("self.grid is None, please provide a grid for trunk input")

        x_branch = x.view(B, -1)
        y_var = self.grid.repeat(B, 1, 1)

        u_vars = [x_branch]
        out = self(u_vars, y_var)
        val_loss = self.criterion(out.view(B, -1), y.view(B, -1))
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.step_size, eta_min=self.hparams.eta_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}