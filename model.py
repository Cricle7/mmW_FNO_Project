import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class fourier_conv_2d(nn.Module):
    """
    做一次 2D Fourier Transform + 截断 + 卷积 的层
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        in_channels:  输入通道数
        out_channels: 输出通道数
        modes1, modes2: 在频域截断的模式数
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        # weights1、weights2 分别对应正向频率和负向频率上的可学习参数(复数=2个实数)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y, 2) * (in_channel, out_channel, x, y, 2)
        # -> (batch, out_channel, x, y, 2)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        # 1) 先做 2D FFT
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])           # -> complex64
        x_ft = torch.view_as_real(x_ft)                   # -> (batch, channel, H, W//2+1, 2)

        batchsize = x_ft.shape[0]
        _, _, height, width_half, _ = x_ft.shape

        # 2) 构造输出在频域的张量
        out_ft = torch.zeros(batchsize,
                             self.out_channels,
                             height,
                             width_half,
                             2,
                             device=x.device,
                             dtype=x.dtype)

        # 截断并做复数乘法(正频率部分)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # 负频率部分
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 3) IFFT 回到时域
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Fourier_layer(nn.Module):
    """
    把 fourier_conv_2d + 1x1 卷积 做一个残差块
    """
    def __init__(self, channels, modes, is_last=False):
        super().__init__()
        self.fourier_conv = fourier_conv_2d(channels, channels, modes, modes)
        self.w = nn.Conv2d(channels, channels, kernel_size=1)
        self.is_last = is_last

    def forward(self, x):
        # x: (batch, channels, H, W)
        x1 = self.fourier_conv(x)
        x2 = self.w(x)
        out = x1 + x2
        if not self.is_last:
            out = F.gelu(out)
        return out


class FNO(pl.LightningModule):
    def __init__(self,
                 modes=16,
                 width=32,
                 num_layers=4,
                 lr=1e-3,
                 step_size=20,
                 gamma=0.5,
                 weight_decay=1e-5,
                 eta_min=1e-5):
        super().__init__()
        """
        modes:      傅里叶模式数 (频域截断)
        width:      每层的通道数
        num_layers: 堆叠 Fourier_layer 的层数
        """
        self.save_hyperparameters()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        # 你的输入是(实+虚)=2通道
        self.input_channels = 2
        # 你的输出是1通道(灰度)
        self.output_channels = 1

        # 第一个 1×1 卷积，把 2 通道 -> width
        self.fc0 = nn.Conv2d(self.input_channels, self.width, kernel_size=1)

        # 构建若干个 Fourier_layer
        layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layers.append(Fourier_layer(self.width, self.modes, is_last=is_last))
        self.fno_layers = nn.Sequential(*layers)

        # 最后 1×1 卷积，把 width -> 1 通道
        self.fc1 = nn.Conv2d(self.width, self.output_channels, kernel_size=1)

        # 损失函数
        self.loss_func = nn.MSELoss()

        # 优化相关
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min

    def forward(self, x):
        """
        x: (batch, 2, 32, 32) —— channel first
        """
        # 2) 升维 => (batch, width, H, W)
        x = self.fc0(x)

        # 3) 依次通过若干 Fourier_layer => 仍是 (batch, width, H, W)
        x = self.fno_layers(x)

        # 4) 降维 => (batch, 1, H, W)
        x = self.fc1(x)

        return x

    def training_step(self, batch, batch_idx):
        # batch = (input_data, target_img)
        # input_data: (batch, H, W, 2), target_img: (batch, H, W, 1)
        input_data, target_img = batch
        pred = self(input_data)  # -> (batch, H, W, 1)
        loss = self.loss_func(pred, target_img)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data, target_img = batch
        pred = self(input_data)
        loss = self.loss_func(pred, target_img)
        psnr_values = []
        ssim_values = []
        pred_np = pred.detach().cpu().numpy()  # 将预测结果转为 NumPy 数组
        tgt_np = target_img.detach().cpu().numpy()  # 将目标图像转为 NumPy 数组

        # 计算整个 batch 的 PSNR 和 SSIM
        for i in range(pred_np.shape[0]):
            psnr_val = psnr(tgt_np[i, 0], pred_np[i, 0], data_range=1.0)
            ssim_val = ssim(tgt_np[i, 0], pred_np[i, 0], data_range=1.0)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_psnr', avg_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim', avg_ssim, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.step_size, eta_min=self.eta_min)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
