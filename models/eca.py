import torch
from torch import nn
import math

class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Replaces standard MLP-based attention with a 1D Convolution.
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        
        # Determine kernel size k based on channel depth
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1D Conv across channels
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor: [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 1D Convolution for channel interaction
        y = self.conv(y)
        
        # [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)
