import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class PreNorm(nn.Module):
    """
    对输入先进行LayerNorm，再喂给后续的功能模块
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GeGELU(nn.Module):
    """
    https://paperswithcode.com/method/geglu
    将x最后一维拆成前半部分和后半部分，前半进行GELU，和后半部分做element-wise乘法
    """
    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]
        x1, x2 = x[..., :c // 2], x[..., c // 2:]
        return self.fn(x1) * x2


class FeedForward(nn.Module):
    """
    MLP，全程使用GeGELU()作为激活函数
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    """
    采用ReLU作为激活函数的MLP
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RotaryEmbedding(nn.Module):
    """
    RoPE旋转位置编码
    """
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, dim//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, dim]


def rotate_half(x):
    # x [..., 2, d/2] => [-x2, x1]
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # t: [b, h, n, d],  freqs_x/freqs_y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d // 2], t[..., d // 2:]
    out_x = apply_rotary_pos_emb(t_x, freqs_x)
    out_y = apply_rotary_pos_emb(t_y, freqs_y)
    return torch.cat((out_x, out_y), dim=-1)