import torch
import torch.nn as nn
from DoubleConvolutionBlock import DoubleConvolutionBlock as DoubleConv

class UpsampleBlock(nn.Module):
    """
    The upsampling block for the decoder.

    It upsamples the input feature map using transposed convolution,
    concatenates it with the corresponding skip connection from the encoder,
    and then applies a Double Convolution to blend the features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)