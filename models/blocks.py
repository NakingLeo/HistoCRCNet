import torch
import torch.nn as nn
from torch.nn import functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SPPBlock(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SPPBlock, self).__init__()
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=p) for p in pool_sizes])

    def forward(self, x):
        features = [F.interpolate(pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False) for pool in self.pools]
        return torch.cat(features + [x], 1)
