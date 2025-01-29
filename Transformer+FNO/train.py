import pytorch_lightning as pl
from data_loader import DataModuleWrapper
from operator_transformer import OperatorTransformer
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


def main():
    root_dir = "../32to64DataSet"
    total_imgs = 10841
    all_indices = list(range(1, total_imgs))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.2, random_state=42
    )

    dm = DataModuleWrapper(
        root_dir=root_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=256,  # 可根据实际显存/速度需求调整
        num_workers=24
    )

    # 使用 OperatorTransformer 替换原先的 FNO
    model = OperatorTransformer(
        input_channels=2,   # 输入通道(实部+虚部)
        in_emb_dim=96,
        out_seq_emb_dim=256,
        heads=4,
        depth=6,
        res=64,  # 跟 DataModule 里 resize 的尺寸对应
        latent_channels=256,
        out_channels=1,
        scale=0.5,
        learning_rate=1e-3,
        step_size=100,
        gamma=0.5,
        weight_decay=1e-5,
        eta_min=1e-5
    )

    # 回调1: 每20个epoch保存一次模型, 仅保留 val_loss 最小的
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath='checkpoints',
        filename='model_epoch_{epoch:02d}_{val_loss:.4f}',
        every_n_epochs=20
    )
    # 回调2: 监控学习率
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # 回调3: 早停
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=100,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=900,
        accelerator="gpu",  # 如果有 GPU, 可换成 "gpu", devices=1(或其他数量)
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()