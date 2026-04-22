import torch.nn as nn
from DoubleConvolutionBlock import DoubleConvolutionBlock as DoubleConv
from DownsampleBlock import DownsampleBlock as Downsample
from UpsampleBlock import UpsampleBlock as Upsample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=64):
        super().__init__()

        # Encoder
        self.entry = DoubleConv(in_channels, features)
        self.down1 = Downsample(features, features * 2)
        self.down2 = Downsample(features * 2, features * 4)
        self.down3 = Downsample(features * 4, features * 8)

        # Bottleneck
        self.bottleneck = Downsample(features * 8, features * 16)

        # Decoder
        self.up1 = Upsample(features * 16, features * 8)
        self.up2 = Upsample(features * 8, features * 4)
        self.up3 = Upsample(features * 4, features * 2)
        self.up4 = Upsample(features * 2, features)

        # Output
        self.out_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):

        skip1 = self.entry(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)


        x = self.bottleneck(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.out_conv(x)