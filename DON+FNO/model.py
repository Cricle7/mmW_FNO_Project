import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 若想用余弦退火，请取消注释下面两行
# from torch.optim.lr_scheduler import CosineAnnealingLR

# 评价指标
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class fourier_conv_2d(nn.Module):
    """
    做一次 2D Fourier Transform + 截断 + 卷积 的层
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        in_channels:  输入通道数
        out_channels: 输出通道数
        modes1, modes2: 在频域截断的模式数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        # weights1、weights2 分别对应正向频率和负向频率上的可学习参数(复数用2个实数存储)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        """
        复数乘法:
        (batch, in_channel, x, y, 2) * (in_channel, out_channel, x, y, 2)
        -> (batch, out_channel, x, y, 2)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x: (batch, in_channels, H, W)

        # 1) 先做 2D FFT (rfft2)
        #    rfft2 => 只保留实频+虚频一半 => 结果是复数，但 shape: [batch, channel, H, W//2+1]
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])  # complex64
        # 转成 real+imag: shape => (batch, channel, H, W_half, 2)
        x_ft = torch.view_as_real(x_ft)

        batchsize = x_ft.shape[0]
        _, _, height, width_half, _ = x_ft.shape

        # 2) 构造输出在频域的张量
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width_half,
            2,               # real+imag
            device=x.device,
            dtype=x.dtype
        )

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
    """
    FNO (Fourier Neural Operator) 模型示例:
    - 输入: (batch, 2, 64, 64) (实部+虚部)
    - 输出: (batch, 1, 64, 64) (灰度图)
    """

    def __init__(self,
                 modes=16,
                 width=32,
                 num_layers=4,
                 lr=1e-3,
                 weight_decay=1e-5,
                 eta_min=1e-5,
                 # 如果你要使用别的调度器，可以保留这些参数:
                 step_size=20,
                 gamma=0.5
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        # 输入通道数2(实+虚)，输出1(灰度)
        self.input_channels = 2
        self.output_channels = 1

        # 首先用1×1卷积把 2->width
        self.fc0 = nn.Conv2d(self.input_channels, self.width, kernel_size=1)

        # 堆叠若干 Fourier_layer
        layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layers.append(Fourier_layer(self.width, self.modes, is_last=is_last))
        self.fno_layers = nn.Sequential(*layers)

        # 最后 1×1 卷积，把 width->1
        self.fc1 = nn.Conv2d(self.width, self.output_channels, kernel_size=1)

        # 损失函数
        self.loss_func = nn.MSELoss()

        # 优化器相关超参
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.step_size = step_size
        self.gamma = gamma

    def forward(self, x):
        """
        x: (batch, 2, 64, 64)
        """
        x = self.fc0(x)           # => (batch, width, 64, 64)
        x = self.fno_layers(x)    # => (batch, width, 64, 64)
        x = self.fc1(x)           # => (batch, 1, 64, 64)
        return x

    def training_step(self, batch, batch_idx):
        input_data, target_img = batch  # shapes: [B, 2, 64, 64], [B, 1, 64, 64]
        pred = self(input_data)        # => [B, 1, 64, 64]
        loss = self.loss_func(pred, target_img)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data, target_img = batch
        pred = self(input_data)
        loss = self.loss_func(pred, target_img)

        # 计算PSNR和SSIM (默认假设图像范围是[0,1])
        pred_np = pred.detach().cpu().numpy()  # shape: [B,1,64,64]
        tgt_np  = target_img.detach().cpu().numpy()  # shape: [B,1,64,64]
        psnr_values = []
        ssim_values = []

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
        """
        使用ReduceLROnPlateau以在后期Loss不下降时自动降低学习率。
        如果想改回CosineAnnealingLR或其他方式，请替换对应scheduler。
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # -- 1) ReduceLROnPlateau --------------------------------------
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",       # 监控val_loss最小化
                factor=0.5,       # 学习率衰减倍率
                patience=20,      # 连续多少epoch val_loss不提升就衰减
                min_lr=1e-6       # 学习率下限
            ),
            "monitor": "val_loss"  # 监控指标
        }

        # -- 2) 如果仍想用CosineAnnealingLR，可使用如下代码并注释上面的scheduler --
        # scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.step_size,  # 周期长度
        #     eta_min=self.eta_min
        # )

        return [optimizer], [scheduler]