import torch
from UnetCore import UNet

def test_unet3d():
    # 1. Define model parameters
    in_channels = 1  # e.g., 1 for grayscale/CT/MRI scans
    out_channels = 2  # e.g., 2 classes: background vs. tumor
    base_features = 16  # Lowered from 64 just to make the test run faster on CPU

    # Instantiate the model
    print("Instantiating UNet...")
    model = UNet(in_channels=in_channels, out_channels=out_channels, features=base_features)

    # 2. Create a dummy input tensor
    # Shape for 3D PyTorch tensors: (Batch, Channels, Depth, Height, Width)
    # IMPORTANT: Because we downsample 4 times (halving each time),
    # your spatial dimensions (D, H, W) MUST be divisible by 2^4 (16).
    batch_size = 1
    depth = 32
    height = 64
    width = 64

    print(f"Creating dummy input tensor of shape: ({batch_size}, {in_channels}, {depth}, {height}, {width})")
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # 3. Perform a forward pass
    print("Running forward pass...")
    # Using torch.no_grad() because we don't need to calculate gradients for a simple shape test
    with torch.no_grad():
        output = model(x)

    # 4. Verify the output
    print(f"Output tensor shape: {output.shape}")

    # The output spatial dimensions should perfectly match the input
    expected_shape = (batch_size, out_channels, depth, height, width)
    if output.shape == expected_shape:
        print("Success! The output shape matches the expected shape.")
    else:
        print(f"Error! Expected {expected_shape} but got {output.shape}")


if __name__ == "__main__":
    test_unet3d()