import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

def init_layer(layer) -> None:
    """
    Initialises each layer with Xavier Initialization
    Initialises BN layers with weight 1 and bias 0
    """

    if isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1.)
        layer.bias.data.fill_(0.)
    else:
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding=2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        #init_layer(self.conv1)
        # init_layer(self.conv2)
        # init_layer(self.bn1)
        # init_layer(self.bn2)

    def forward(self, x, pool_size=2) -> Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.bn1)
        init_layer(self.bn2)

    def forward(self, x, pool_size=(2,2), pool_type='avg') -> Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect pooling arg!')

        return x

class AttnBlock(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(AttnBlock, self).__init__()

        self.attn = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.classification = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.bn_attn = nn.BatchNorm1d(out_features)
        
        init_layer(self.attn)
        init_layer(self.classification)

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        # x -> (n_samples, n_in, n_time)
        att = torch.softmax(torch.tanh(self.attn(x)), dim=-1)
        classification = torch.sigmoid(self.classification(x))

        x = torch.sum(att * classification, dim=2)

        return x, classification