"""The study's residual 3D U-Net, implemented entirely with torch.nn."""
import torch
from torch import nn


def block(inputs, outputs, stride=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, stride, 1, bias=True),
        nn.InstanceNorm3d(outputs, eps=1e-5, affine=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(outputs, outputs, 3, padding=1, bias=True),
        nn.InstanceNorm3d(outputs, eps=1e-5, affine=True),
        nn.LeakyReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 64, 128, 256, 320)
        strides = ((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
        self.encoder = nn.ModuleList()
        for i, (width, stride) in enumerate(zip(channels, strides)):
            self.encoder.append(block(channels[i-1] if i else 1, width, stride))
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(4, 0, -1):
            width = channels[i-1]
            self.up.append(nn.ConvTranspose3d(channels[i], width, strides[i], strides[i]))
            self.decoder.append(block(2 * width, width))
            # Stage-ordered head initialization preserves the seeded parameter
            # sequence; retain the full-resolution residual head.
            head = nn.Conv3d(width, 1, 1)
        self.output = head
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(n % s for n, s in zip(x.shape[2:], (8, 16, 16))):
            raise ValueError('Expected [N,1,D,H,W], with D divisible by 8 and H/W by 16')
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        for i, (up, stage) in enumerate(zip(self.up, self.decoder)):
            x = stage(torch.cat((up(x), skips[-i-2]), dim=1))
        return self.output(x)
