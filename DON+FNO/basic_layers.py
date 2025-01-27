import torch
import torch.nn as nn


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