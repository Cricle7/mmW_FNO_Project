import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FcnSingle(nn.Module):
    """
    简单的 MLP，用于 DeepONet 等分支/主干部分
    """
    def __init__(self, planes, activation="gelu", last_activation=False):
        super(FcnSingle, self).__init__()
        self.planes = planes

        if activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Identity

        layers = []
        for i in range(len(self.planes) - 2):
            layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            layers.append(act_fn())

        # 最后一层线性
        layers.append(nn.Linear(self.planes[-2], self.planes[-1]))
        if last_activation:
            layers.append(act_fn())

        self.layers = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.layers(x)

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

#####################################################################
# Fully Connected Neural Networks
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
        
#####################################################################
# LayerNorm Module
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
#####################################################################       
# Getting the 2D grid using the batch
def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    
    return torch.cat((gridx, gridy), dim=-1).to(device)