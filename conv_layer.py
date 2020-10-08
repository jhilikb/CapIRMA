
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Conventional Conv2d layer
    """

    def __init__(self, in_channel, out_channel, kernel_size):
        super(ConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out_conv0 = self.conv0(x)
        
        out_relu = self.relu(out_conv0)
        return out_relu
