import torch
import torch.nn as nn
import pytorch_lightning as pl
from deeponet import DeepONetMulti
from fno import FNO

def get_grid2D(Nx, Ny, device):
    gridx = torch.linspace(0, 1, Nx, device=device)
    gridy = torch.linspace(0, 1, Ny, device=device)
    x, y = torch.meshgrid(gridx, gridy, indexing='ij')  # (Nx, Ny)
    x = x.reshape(-1, 1)  # => (Nx*Ny, 1)
    y = y.reshape(-1, 1)  # => (Nx*Ny, 1)
    grid = torch.cat([x, y], dim=-1)    # => (Nx*Ny, 2)
    # 加个 batch 维度 => (1, Nx*Ny, 2)
    grid = grid.unsqueeze(0)
    return grid
class DeepONetFNO(pl.LightningModule):
    """
    示例：同时初始化 DeepONetMulti 与 FNO，并在 forward 中做一次简单串联。
    假设:
    - DataLoader 每个 batch 返回 (input_data, target_img)
      其中:
        input_data.shape = (B, 2, 64, 64)     # [实部, 虚部] 2 通道
        target_img.shape = (B, 1, 64, 64)    # 单通道灰度图
    - 在这里, 我们演示一种“DeepONet -> FNO”的简单串联:
      1) 用 DeepONet 将 input_data 展平后做 branch 输入, trunk 输入为网格
      2) DeepONet 输出一个 (B, Nx*Ny, 1) 的场, reshape 为 (B, Nx, Ny, 1)
      3) 作为 FNO 的 sos；将原始 input_data 的 (实+虚) 拼成 (B, Nx, Ny, 2) 作为 src
      4) 通过 FNO 得到最终预测
      5) 与 target_img 做 MSE
    """

    def __init__(
        self,
        deeponet_config: dict,
        fno_config: dict,
        Nx: int = 64,
        Ny: int = 64,
    ):
        """
        deeponet_config, fno_config: 用于初始化 DeepONetMulti 和 FNO 的超参数字典
        Nx, Ny: 对应图像 / 网格大小
        """
        super().__init__()
        # 保存超参（可选）
        self.save_hyperparameters(ignore=["deeponet_config", "fno_config"])

        # 1) 初始化 DeepONetMulti
        self.deeponet = DeepONetMulti(**deeponet_config)

        # 2) 初始化 FNO
        self.fno = FNO(**fno_config)

        # 均方误差
        self.criterion = nn.MSELoss()

        # Nx, Ny 用来生成 trunk 网格或 reshape
        self.Nx = Nx
        self.Ny = Ny

    def forward(self, input_data):
        """
        本例中不在 forward 内部直接算 loss，只演示一次“DeepONet -> FNO”的前向流程。
        input_data: (B, 2, Nx, Ny)
        返回: pred, 形状可能为 (B, Nx, Ny, 2) 或 (B, Nx, Ny) 等
        """
        B = input_data.shape[0]
        device = input_data.device

        # ============ 1) 处理 DeepONet 的输入 ============
        # DeepONetMulti 的 branch 输入一般是 [B, branch_dim]，这里简单将 (2, Nx, Ny) 全部 flatten
        # 假设 branch_in = 2*Nx*Ny
        branch_in = input_data.reshape(B, -1)  # => (B, 2*Nx*Ny)

        # trunk 输入需要 (B, Nx*Ny, input_dim)，假设 input_dim=2 (x,y 坐标)
        # 先生成 (1, Nx*Ny, 2)，再 repeat(B,1,1)
        grid = get_grid2D(self.Nx, self.Ny, device)    # => (1, Nx*Ny, 2)
        trunk_in = grid.repeat(B, 1, 1)                # => (B, Nx*Ny, 2)

        # ============ 2) DeepONet 前向 ============
        # 这里我们只示范一个 branch，即 u_vars=[branch_in]
        # DeepONetMulti.forward(u_vars, y_var) -> (B, Nx*Ny, output_dim)
        out_deeponet = self.deeponet([branch_in], trunk_in)  # => (B, Nx*Ny, output_dim=1)

        # reshape 成 (B, Nx, Ny, 1)，作为 FNO 中的 sos
        sos = out_deeponet.view(B, self.Nx, self.Ny, -1)  # => (B, Nx, Ny, 1)

        # ============ 3) 准备 FNO 的 src ============
        # 假设将 input_data 两个通道当作实部 + 虚部 => (B, Nx, Ny, 2)
        src = input_data.permute(0, 2, 3, 1).contiguous()  # => (B, Nx, Ny, 2)

        # ============ 4) FNO 前向 ============
        # fno.forward(sos, src) => (B, Nx, Ny) 或 (B, Nx, Ny, 2) (视 add_term 而定)
        pred = self.fno(sos, src)
        return pred

    def training_step(self, batch, batch_idx):
        """
        一个 batch 通常是 (input_data, target_img)
        input_data: (B, 2, 64, 64)
        target_img: (B, 1, 64, 64)
        """
        input_data, target_img = batch
        # 前向得到预测
        pred = self(input_data)
        B = pred.shape[0]

        # 如果 pred 是复数形式或者 (B, Nx, Ny, 2)，可根据需求取实部
        # 下面仅示例如果是 (B, Nx, Ny, 2) 的情况，取第 0 通道作为“预测”
        if pred.ndim == 4 and pred.shape[-1] == 2:
            # 例如仅取实部 => pred[..., 0]
            pred = pred[..., 0].unsqueeze(1)  # => (B, 1, Nx, Ny)
        else:
            # 若 pred 本身就是 (B, Nx, Ny)，则加个 channel 维度
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)     # => (B, 1, Nx, Ny)

        # 与 target_img 对比
        # 注意 target_img.shape = (B, 1, 64, 64)
        loss = self.criterion(pred, target_img)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data, target_img = batch
        pred = self(input_data)

        # 若 pred 最后维度=2，示例只取第0维度
        if pred.ndim == 4 and pred.shape[-1] == 2:
            pred = pred[..., 0].unsqueeze(1)
        else:
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)

        val_loss = self.criterion(pred, target_img)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        这里示例: 用一个 AdamW 对 deeponet 与 FNO 的全部参数做优化
        也可以分两个优化器分别训练
        """
        params = list(self.deeponet.parameters()) + list(self.fno.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }