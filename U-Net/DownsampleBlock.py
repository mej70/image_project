import torch.nn as nn
from DoubleConvolutionBlock import DoubleConvolutionBlock as DoubleConv


class DownsampleBlock(nn.Module):
    """
    The downsampling block for the encoder.

    This block first applies a 3D Max Pooling operation to
    halve the spatial dimensions, followed by a Double Convolution to extract features and
    change the channel depth.
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=stride),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.downsample(x)

