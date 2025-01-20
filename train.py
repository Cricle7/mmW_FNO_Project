import pytorch_lightning as pl
from data_loader import DataModuleWrapper
from model import FNO


def main():
    root_dir = r"D:\Project\python\inversion\32to64数据集\32to64DataSet"

    # 例如，训练集为 1~1000，验证集为 1001~1200
    train_indices = list(range(1, 1001))
    val_indices   = list(range(1001, 1201))

    dm = DataModuleWrapper(
        root_dir=root_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=4,       # 根据需要设置
        num_workers=0       # Windows 上如果报错，也可以先用 0
    )

    # 初始化模型
    model = FNO(
        modes=16,
        width=32,
        num_layers=4,
        lr=1e-3,
        step_size=20,
        gamma=0.5,
        weight_decay=1e-5,
        eta_min=1e-5
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",  # 没有 GPU 就用 cpu
        devices=1
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
    