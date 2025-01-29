import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, orthogonal_
from einops import rearrange, repeat

from modules import (
    PreNorm, FeedForward, ReLUFeedForward,
    RotaryEmbedding,
    rotate_half, apply_rotary_pos_emb, apply_2d_rotary_pos_emb
)


class LinearAttention(nn.Module):
    """
    "Choose a Transformer: Fourier or Galerkin" 中提出的两种注意力:
      - Galerkin (对k,v做InstanceNorm)
      - Fourier (对q,k做InstanceNorm)
    """
    def __init__(self,
                 dim,
                 attn_type,   # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise ValueError(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim,
                                              min_freq=min_freq,
                                              scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise ValueError('Unknown initialization method')

        # 初始化 to_qkv
        for param in self.to_qkv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    if self.attn_type == 'fourier':
                        # init v
                        init_fn(param[(self.heads * 2 + h) * self.dim_head: (self.heads * 2 + h + 1) * self.dim_head, :],
                                gain=self.init_gain)
                        param.data[(self.heads * 2 + h) * self.dim_head: (self.heads * 2 + h + 1) * self.dim_head, :] += \
                            self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                    else:
                        # init q
                        init_fn(param[h * self.dim_head: (h + 1) * self.dim_head, :],
                                gain=self.init_gain)
                        param.data[h * self.dim_head: (h + 1) * self.dim_head, :] += \
                            self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        """
        对张量 x 在 (batch, head) 维度上拆开，然后沿 domain(n)做 InstanceNorm
        x: [b, h, n, d]
        """
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b
        )

    def forward(self, x, pos=None, not_assoc=False):
        """
        x: [b, n, c]   (batch, seq_len, channels)
        pos: [b, n, 2] or None
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 分成 q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if pos is None and self.relative_emb:
            raise ValueError('Must provide coordinates when using relative_emb=True')

        # 根据不同的attention类型选择归一化对象
        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        # 相对位置编码 (RoPE)
        if self.relative_emb:
            # 目前仅支持2D
            # freqs_x, freqs_y: [b, n, d//2] => 拼成 [b, n, d]
            freqs_x = self.emb_module(pos[..., 0], x.device)
            freqs_y = self.emb_module(pos[..., 1], x.device)
            freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=self.heads)
            freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=self.heads)

            q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
            k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)

        elif self.cat_pos:
            # 如果需要直接拼接坐标信息
            b, h, n, d = q.shape
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1).repeat(1, h, 1, 1)
            q, k, v = [torch.cat([p, xx], dim=-1) for p, xx in zip([pos, pos, pos], [q, k, v])]

        if not_assoc:
            # 可能的加速分支
            score = torch.matmul(q, k.transpose(-1, -2))  # [b,h,n,n]
            out = torch.matmul(score, v) * (1.0 / q.shape[2])
        else:
            # 缺省写法
            dots = torch.matmul(k.transpose(-1, -2), v)  # [b,h,d,d]
            out = torch.matmul(q, dots) * (1.0 / q.shape[2])

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossLinearAttention(nn.Module):
    """
    CrossAttention:
      - query来自 x (要预测的点)
      - key, value 来自 z (encoder编码后的特征)
    """
    def __init__(self,
                 dim,
                 attn_type,   # ['fourier','galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.attn_type = attn_type
        self.heads = heads
        self.dim_head = dim_head

        # 分别定义 q, k, v
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise ValueError(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim,
                                              min_freq=min_freq,
                                              scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise ValueError('Unknown initialization method')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # k
                    init_fn(param[h*self.dim_head:(h+1)*self.dim_head, :], gain=self.init_gain)
                    param.data[h*self.dim_head:(h+1)*self.dim_head, :] += \
                        self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                    # v
                    init_fn(param[(self.heads+h)*self.dim_head:(self.heads+h+1)*self.dim_head, :],
                            gain=self.init_gain)
                    param.data[(self.heads+h)*self.dim_head:(self.heads+h+1)*self.dim_head, :] += \
                        self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # q
                    init_fn(param[h*self.dim_head:(h+1)*self.dim_head, :], gain=self.init_gain)
                    param.data[h*self.dim_head:(h+1)*self.dim_head, :] += \
                        self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b
        )

    def forward(self, x, z, x_pos=None, z_pos=None):
        """
        x: [b, n1, d] => query
        z: [b, n2, d] => key/value
        """
        q = self.to_q(x)                   # [b, n1, heads*dim_head]
        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = kv
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise ValueError('Must pass x_pos, z_pos when relative_emb=True')

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        if self.relative_emb:
            # 2D
            x_freqs_x = self.emb_module(x_pos[..., 0], x.device)
            x_freqs_y = self.emb_module(x_pos[..., 1], x.device)
            x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
            x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

            z_freqs_x = self.emb_module(z_pos[..., 0], z.device)
            z_freqs_y = self.emb_module(z_pos[..., 1], z.device)
            z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
            z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

            q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
            k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

        elif self.cat_pos:
            # 如果需要拼接坐标信息
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1).repeat(1, self.heads, 1, 1)  # [b,h,n1, pos_dim]
            z_pos = z_pos.unsqueeze(1).repeat(1, self.heads, 1, 1)  # [b,h,n2, pos_dim]
            q = torch.cat([x_pos, q], dim=-1)
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        # 计算注意力
        dots = torch.matmul(k.transpose(-1, -2), v)   # [b,h,d,d]
        out = torch.matmul(q, dots) * (1.0 / z.shape[1])  # n2
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossFormer(nn.Module):
    """
    Decoder中的 Cross-Attn + FFN
    z' = z^0 + CrossAttn(z^0, f^L)
    z  = z' + FFN(z')
    """
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()
        self.cross_attn_module = CrossLinearAttention(
            dim, attn_type,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            relative_emb=relative_emb,
            scale=scale,
            relative_emb_dim=relative_emb_dim,
            min_freq=min_freq,
            cat_pos=cat_pos,
        )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        """
        x: [b, n1, c] => query
        z: [b, n2, c] => context
        """
        if self.use_ln:
            z = self.ln1(z)
            attn_out = self.cross_attn_module(x, z, x_pos, z_pos)
            if self.residual:
                x = self.ln2(attn_out) + x
            else:
                x = self.ln2(attn_out)
        else:
            attn_out = self.cross_attn_module(x, z, x_pos, z_pos)
            if self.residual:
                x = attn_out + x
            else:
                x = attn_out

        if self.use_ffn:
            x = self.ffn(x) + x

        return x