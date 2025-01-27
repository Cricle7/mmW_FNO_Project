# data_module.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from scipy.io import loadmat


class MyMatDataset(Dataset):
    def __init__(self, x_data, y_data):
        """
        x_data, y_data 均是 numpy 或 torch.tensor, shape例如:
          x_data: (N, Nx, Ny, 1)
          y_data: (N, Nx, Ny, 1)
        """
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path,
        test_data_path,
        ntrain,
        ntest,
        res=10,
        batch_size=8,
        use_cuda=False
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.ntrain = ntrain
        self.ntest = ntest
        self.res = res
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def setup(self, stage=None):
        train_data = loadmat(self.train_data_path)
        x_train = train_data["coeff"][: self.ntrain, : self.res, : self.res]  # (ntrain, res, res)
        y_train = train_data["sol"][: self.ntrain, : self.res, : self.res]

        test_data = loadmat(self.test_data_path)
        x_test = test_data["coeff"][-self.ntest :, : self.res, : self.res]
        y_test = test_data["sol"][-self.ntest :, : self.res, : self.res]

        # reshape => (N, res, res, 1)
        x_train = x_train[..., None]
        y_train = y_train[..., None]
        x_test = x_test[..., None]
        y_test = y_test[..., None]

        # 转成torch
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)
        x_test = torch.FloatTensor(x_test)
        y_test = torch.FloatTensor(y_test)

        self.train_dataset = MyMatDataset(x_train, y_train)
        self.test_dataset = MyMatDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # 也可分出单独验证集，这里简化用test做验证
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)