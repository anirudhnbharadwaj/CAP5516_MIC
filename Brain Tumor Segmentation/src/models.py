import torch
import torch.nn as nn
from monai.networks.nets import UNet

class UNet3D(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128, 256)):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=channels,
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

    def forward(self, x):
        return self.model(x)