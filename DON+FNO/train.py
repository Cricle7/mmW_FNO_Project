# train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from data_module import MyDataModule
from model import DeepONetFNOModel

def main():
    # 假设数据路径:
    train_data_path = "train_data.mat"
    test_data_path = "test_data.mat"

    # 这里仅示例, 请替换成实际数值
    ntrain = 1000
    ntest = 200
    res = 10

    # 1) 初始化 DataModule
    dm = MyDataModule(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        ntrain=ntrain,
        ntest=ntest,
        res=res,
        batch_size=8
    )
    dm.setup()

    # 2) 构造坐标网格 => (1, Nx*Ny, 2)
    Nx, Ny = res, res
    gridx = torch.linspace(0, 1, Nx)
    gridy = torch.linspace(0, 1, Ny)
    # 生成mesh
    gx, gy = torch.meshgrid(gridx, gridy, indexing="ij")  # => (Nx, Ny)
    grid = torch.stack([gx, gy], dim=-1).view(1, Nx * Ny, 2)

    # 3) 配置DeepONet
    deeponet_cfg = {
        "input_dim": 2,
        "operator_dims": [Nx * Ny],  # branch的输入维度(函数采样后flat)
        "output_dim": 1,
        "planes_branch": [64, 64],
        "planes_trunk": [64, 64],
        "grid": grid,   # 作为trunk的坐标
        "learning_rate": 1e-3,
        # ...
    }

    # 4) 实例化 DeepONetFNOModel
    model = DeepONetFNOModel(
        deeponet_cfg=deeponet_cfg,
        fno_width=32,
        fno_modes=16,
        Nx=Nx,
        Ny=Ny,
        in_channels_for_fno=3  # 1通道(DeepONet输出) + 2通道(坐标)
    )

    # 5) 定义回调
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="deeponetfno-{epoch:02d}-{val_loss:.4f}"
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    # 6) Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_cb, lr_monitor_cb]
    )

    trainer.fit(model, dm)

    # (可选) 测试
    trainer.test(model, dm)

if __name__ == "__main__":
    main()