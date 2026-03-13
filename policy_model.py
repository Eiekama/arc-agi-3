import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, num_layers=4):
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers

        self.dense = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(num_layers)
        ])
        self.normalize = nn.ModuleList([
            nn.LayerNorm(out_features, eps=1e-6)
            for _ in range(num_layers)
        ])

        # initialize weights
        sd = np.sqrt(1 / (3 * in_features)) # see https://github.com/wang-kevin3290/scaling-crl/blob/main/train.py#L108
        for d in self.dense:
            torch.nn.init.uniform_(d.weight, -sd, sd) # type: ignore
            torch.nn.init.zeros_(d.bias) # type: ignore
            
    def forward(self, x):
        for i in range(self.num_layers):
            output = self.dense[i](x)
            output = self.normalize[i](x)
            output = F.silu(output)
        return x + output
    
