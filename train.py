import pytorch_lightning as pl
from data_loader import DataModuleWrapper
from model import FNO
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

def main():
    root_dir = "./32to64DataSet"
    total_imgs = 10841
    all_indices = list(range(1, total_imgs))
    train_indices, val_indices = train_test_split(all_indices,
                                                  test_size=0.2,
                                                  random_state=42)

    dm = DataModuleWrapper(
        root_dir=root_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=256,  # 尝试相对大一些的batch_size，但也要看显存情况
        num_workers=24
    )

    model = FNO(
        modes=16,
        width=32,
        num_layers=4,
        lr=1e-3,
        weight_decay=1e-5,
        eta_min=1e-5
        # 若使用 ReduceLROnPlateau，可将 step_size/gamma 暂时忽略
    )

    # 回调1: 每20个epoch保存一次模型, 仅保留val_loss最小的那个权重
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath='checkpoints',
        filename='model_epoch_{epoch:02d}_{val_loss:.2f}',
        every_n_epochs=20
    )

    # 回调2: 监控学习率(方便观察ReduceLROnPlateau何时降低lr)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 回调3(可选): 若val_loss长时间不提升，可以提前停止
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=100,   # 可根据需要调整
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=900,
        accelerator="cpu",  
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        # log_every_n_steps=10  # 根据需要打印日志
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()