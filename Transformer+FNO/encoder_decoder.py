import torch
import torch.nn as nn
from einops import rearrange

from modules import (
    PreNorm, FeedForward, ReLUFeedForward
)
from attention import (
    LinearAttention, CrossFormer
)


# 若需要 standard attention，可自行添加 StandardAttention 实现
# 这里仅保留 'galerkin', 'fourier' 两种


class TransformerCatNoCls(nn.Module):
    """
    将多个 LinearAttention 层堆叠
    """
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['galerkin', 'fourier']
                 use_ln=False,
                 scale=16,
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads,
                                          dim_head=dim_head,
                                          dropout=dropout,
                                          relative_emb=True if not cat_pos else False,
                                          scale=scale[d],
                                          relative_emb_dim=relative_emb_dim,
                                          min_freq=min_freq,
                                          init_method=attention_init,
                                          init_gain=init_gain,
                                          cat_pos=cat_pos)

            if not use_ln:
                # [attn, ffn]
                ffn = (FeedForward(dim, mlp_dim, dropout) if not use_relu
                       else ReLUFeedForward(dim, mlp_dim, dropout))
                self.layers.append(nn.ModuleList([attn_module, ffn]))
            else:
                # [ln1, attn, ln2, ffn]
                ffn = (FeedForward(dim, mlp_dim, dropout) if not use_relu
                       else ReLUFeedForward(dim, mlp_dim, dropout))
                self.layers.append(nn.ModuleList([
                    nn.LayerNorm(dim),
                    attn_module,
                    nn.LayerNorm(dim),
                    ffn
                ]))

    def forward(self, x, pos_embedding):
        # x: [b, n, dim], pos_embedding: [b, n, 2]
        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                attn, ffn = attn_layer
                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                ln1, attn, ln2, ffn = attn_layer
                x_ = ln1(x)
                x = attn(x_, pos_embedding) + x
                x__ = ln2(x)
                x = ffn(x__) + x
        return x


class SpatialEncoder2D(nn.Module):
    """
    用于编码输入 (可视为 U-net 前半部分类似)，但这里是Transformer Encoder
    """
    def __init__(self,
                 input_channels,
                 in_emb_dim,
                 out_seq_emb_dim,
                 heads,
                 depth,
                 res,
                 use_ln=True,
                 emb_dropout=0.05):
        super().__init__()
        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerCatNoCls(
            dim=in_emb_dim,
            depth=depth,
            heads=heads,
            dim_head=in_emb_dim,
            mlp_dim=in_emb_dim,
            attn_type='galerkin',    # 也可换 'fourier'
            use_relu=False,
            use_ln=use_ln,
            scale=[res, res // 4] + [1] * (depth - 2) if depth >= 2 else [res],
            relative_emb_dim=2,
            min_freq=1 / res,
            dropout=0.03,
            attention_init='orthogonal'
        )

        self.to_out = nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False)

    def forward(self, x, input_pos):
        """
        x: [b, n, c], c=input_channels
        input_pos: [b, n, 2]
        """
        x = self.to_embedding(x)
        x = self.dropout(x)
        x = self.s_transformer(x, input_pos)
        x = self.to_out(x)
        return x


class GaussianFourierFeatureTransform(nn.Module):
    """
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    输入: [b, n, d_in]
    输出: [b, n, 2*mapping_size]
    """
    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        # 不参与训练
        B = torch.randn((num_input_channels, mapping_size)) * scale
        self.register_buffer('_B', B, persistent=False)

    def forward(self, x):
        """
        x: [b, n, num_input_channels]
        return: [b, n, mapping_size*2], 拼cos/sin
        """
        b, n, c = x.shape
        x = x.reshape(b*n, c)     # => [b*n, c]
        x = x @ self._B           # => [b*n, mapping_size]
        x = x.reshape(b, n, -1)
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class PointWiseDecoder2DSimple(nn.Module):
    """
    Decoder
    1) 对坐标做高频编码 (GaussianFourierFeatureTransform)
    2) Cross-Attn 融合 encoder 输出
    3) MLP 输出
    """
    def __init__(self,
                 latent_channels,
                 out_channels,
                 res=64,
                 scale=0.5):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.res = res

        # 坐标 -> Fourier特征 -> linear
        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(num_input_channels=2,
                                            mapping_size=latent_channels // 2,
                                            scale=scale),
            nn.Linear(latent_channels, latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels, latent_channels, bias=False),
        )

        # 一个 Cross-Attn (可根据需要添加多层)
        self.decoding_transformer = CrossFormer(
            dim=latent_channels,
            attn_type='galerkin',   # or 'fourier'
            heads=4,
            dim_head=latent_channels,
            mlp_dim=latent_channels,
            use_ln=False,
            residual=True,
            relative_emb=True,
            scale=16,
            relative_emb_dim=2,
            min_freq=1/res
        )

        self.to_out = nn.Sequential(
            nn.Linear(latent_channels + 2, latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels, latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels // 2, out_channels, bias=True)
        )

    def forward(self, z, query_pos, context_pos):
        """
        z: [b, n_ctx, latent_channels], 来自 encoder 的输出
        query_pos: [b, n_query, 2], 要预测的点坐标
        context_pos: [b, n_ctx, 2], encoder输入的坐标
        """
        # 对 query 坐标做 Fourier 映射
        x = self.coordinate_projection(query_pos)  # [b, n_query, latent_channels]
        # CrossAttention
        x = self.decoding_transformer(x, z, x_pos=query_pos, z_pos=context_pos)
        # 融合后与坐标 concat，再映射到最终 out
        x = self.to_out(torch.cat([x, query_pos], dim=-1))
        return x