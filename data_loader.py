import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision.transforms.functional import resize
from PIL import Image
from scipy.io import loadmat


class MmwDataset(Dataset):
    def __init__(self, root_dir, index_list):
        super().__init__()
        self.root_dir = root_dir
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        idx = self.index_list[i]
        # 1) 读取可见函数的实部/虚部 (形状通常是 [H, W])
        real_path = os.path.join(self.root_dir, "VisibilityFunction", "real", f"{idx}.mat")
        real_data = loadmat(real_path)["miuI"]  # 修改为实际的键名
        imag_path = os.path.join(self.root_dir, "VisibilityFunction", "imaginary", f"{idx}.mat")
        imag_data = loadmat(imag_path)["miuQ"]  # 修改为实际的键名

        # 2) 拼成 2 通道 => [2, H, W] (channel-first)
        input_data = np.stack([real_data, imag_data], axis=0).astype(np.float32)

        # 3) 读取目标图像 (灰度) => [H, W]
        target_path = os.path.join(self.root_dir, "TargetScene", f"{idx}.bmp")
        target_img = Image.open(target_path).convert("L")
        target_img = np.array(target_img, dtype=np.float32)
        # 加一个通道维 => [1, H, W]
        target_img = np.expand_dims(target_img, axis=0)

        # 4) 转成 torch.Tensor => [2,H,W], [1,H,W]
        input_data = torch.from_numpy(input_data)
        target_img = torch.from_numpy(target_img)

        # 5) 统一缩放到 32×32 (你也可以换其他尺寸)
        #    现在 input_data.shape = [2,H,W], target_img.shape = [1,H,W]
        #    torchvision.transforms.functional.resize 支持 3D 张量 [C,H,W] 进行插值
        input_data = resize(input_data, [32, 32])  
        target_img = resize(target_img, [32, 32])  

        return input_data, target_img


class DataModuleWrapper(LightningDataModule):
    """
    LightningDataModule 封装数据集，方便 Trainer.fit(model, data_module=dm) 自动加载
    """
    def __init__(self, root_dir, train_indices, val_indices, batch_size=4, num_workers=0):
        super().__init__()
        self.root_dir = root_dir
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Lightning 在不同阶段会自动调用 setup().
        这里分别初始化 train_dataset 和 val_dataset
        """
        self.train_dataset = MmwDataset(self.root_dir, self.train_indices)
        self.val_dataset   = MmwDataset(self.root_dir, self.val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )