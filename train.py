import pytorch_lightning as pl
from data_loader import DataModuleWrapper
from model import FNO
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    root_dir = "./32to64DataSet"
    total_imgs = 10841
    all_indices = list(range(1,total_imgs))
    train_indices, val_indices = train_test_split(all_indices,
                                                test_size=0.2,
                                                random_state=42)    # 例如，训练集为 1~1000，验证集为 1001~1200
    dm = DataModuleWrapper(
        root_dir=root_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=1024,       # 根据需要设置
        num_workers=8       # Windows 上如果报错，也可以先用 0
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

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',            # 监控验证损失
        save_top_k=1,                  # 只保留最好的模型
        mode='min',                    # 模式为最小化（即验证损失最小）
        dirpath='checkpoints',         # 模型保存路径
        filename='model_epoch_{epoch:02d}_{val_loss:.2f}',  # 模型文件命名规则
        every_n_epochs=20             # 每20轮保存一次
    )

    trainer = pl.Trainer(
        max_epochs=900,
        accelerator="gpu",  # 没有 GPU 就用 cpu
        devices=1
    )
    

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
    
