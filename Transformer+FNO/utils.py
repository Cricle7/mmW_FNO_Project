# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_grid2D(shape, device):
    # 按需要保留，也可以直接在 model 中定义
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FC_nn(nn.Module):
    r"""Simple MLP to code lifting and projection"""
    def __init__(self, sizes = [2, 128, 128, 1], 
                        outermost_linear = True, 
                        outermost_norm = True,  
                        drop = 0.):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.net = nn.ModuleList([FCLayer(in_feature= m, out_feature= n, 
                                            activation='gelu', 
                                            is_normalized = False)   
                                for m, n in zip(sizes[:-2], sizes[1:-1])
                                ])
        if outermost_linear == True: 
            self.net.append(FCLayer(sizes[-2],sizes[-1], activation = None, 
                                    is_normalized = outermost_norm))
        else: 
            self.net.append(FCLayer(in_feature= sizes[-2], out_feature= sizes[-1], 
                                    activation='gelu',
                                    is_normalized = outermost_norm))

    def forward(self,x):
        for module in self.net:
            x = module(x)
            x = self.dropout(x)
        return x
    
# Fully Connected Layer
class FCLayer(nn.Module):
    """Fully connected layer """
    def __init__(self, in_feature, out_feature, 
                        activation = "gelu",
                        is_normalized = True): 
        super().__init__()
        if is_normalized:
            self.LinearBlock = nn.Sequential(
                            nn.Linear(in_feature,out_feature),
                            LayerNorm(out_feature),
                            )                               
        else:
            self.LinearBlock = nn.Linear(in_feature,out_feature)
        if activation:
            self.act = F.gelu
        else:
            self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.LinearBlock(x))
