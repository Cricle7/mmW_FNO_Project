import pytorch_lightning as pl
from data_loader import DataModuleWrapper
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# 从您刚写好的 deeponet_fno.py 中导入 DeepONetFNO
from deeponet_fno import DeepONetFNO

def main():
    # 1) 准备数据索引与 DataModuleWrapper
    root_dir = "../32to64DataSet"
    total_imgs = 10841
    all_indices = list(range(1, total_imgs))
    train_indices, val_indices = train_test_split(all_indices,
                                                  test_size=0.2,
                                                  random_state=42)

    dm = DataModuleWrapper(
        root_dir=root_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=256,  # 原先给的batch_size
        num_workers=24
    )

    # 2) 设置 DeepONet 与 FNO 的配置字典 (根据您的需求进行调整)
    #    以下仅作示例, 核心是将2通道(实+虚)视为 branch 输入, output_dim=1 => Nx×Ny 的幅度
    #    若需要更复杂的结构, 可在此修改超参数
    deeponet_config = {
        "input_dim": 2,               # trunk输入坐标维度(如x,y)；若不在 trunk 用坐标，可无视
        "operator_dims": [64 * 64 * 2],# branch输入维度; 把 (2,64,64) flatten => 2*64*64=8192
        "output_dim": 1,              # 输出维度
        "planes_branch": [64, 64],
        "planes_trunk": [64, 64],
        "activation": "gelu"
        # 这里不一定需要传 learning_rate / step_size 等,
        # 因为在 deeponet_fno.py 里会统一定义 optimizer
    }

    # 对应 fno.py 中 FNO 的一些参数名称和用法, 请根据实际修改:
    fno_config = {
        "wavenumber": [16, 16, 16, 16],  # 对应原先的 modes=16, 4层 => [16,16,16,16]
        "features_": 32,                # 对应原先的 width=32
        "padding": 9,                   # 如果需要, 保持和旧版本一致
        "dim_input": 1,                 # 我们示例: DeepONet 输出是 (Nx,Ny,1), FNO会加4->共5
        "with_grid": True,
        "add_term": True,
        # 下列为 FNO 优化相关, 也可在 deeponet_fno.py 里统一配置
        "learning_rate": 1e-3,
        "step_size": 100,
        "gamma": 0.5,
        "weight_decay": 1e-5,
        "eta_min": 1e-5
    }

    # 3) 初始化 DeepONet + FNO 的整合模型
    #    Nx=64, Ny=64 表示输入分辨率
    model = DeepONetFNO(
        deeponet_config=deeponet_config,
        fno_config=fno_config,
        Nx=64,
        Ny=64
    )

    # 4) 回调函数
    #    (1) 保存模型: 每20个epoch保存一次, 保留 val_loss 最优
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath='checkpoints',
        filename='model_epoch_{epoch:02d}_{val_loss:.2f}',
        every_n_epochs=20
    )

    # (2) 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # (3) 提前停止: 如果 val_loss 长时间不提升
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=100,   # 可根据需要调整
        mode='min'
    )

    # 5) 创建 Trainer 并开始训练
    trainer = pl.Trainer(
        max_epochs=900,
        accelerator="cpu",  # 若有GPU可切换: accelerator="gpu", devices=1
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        # log_every_n_steps=10,
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()