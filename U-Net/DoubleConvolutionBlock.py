import torch.nn as nn


class DoubleConvolutionBlock(nn.Module):
    """
    A standard Double Convolution block for 3D CNNs.
    Performs two sequential operations of:
    3D Convolution -> 3D Batch Normalization -> ReLU activation.
    With default settings it maintains spatial dimensions (kernel=3, padding=1)
    while extracting features and increasing channel depth.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            # First Conv Block
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),

            # Second Conv Block
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
