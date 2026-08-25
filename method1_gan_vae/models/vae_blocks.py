"""
Basic building blocks shared by the VAE encoder and decoder.

These are standard, well-known layer patterns (residual blocks, strided
conv downsampling, transposed-conv upsampling) -- not specific to any one
paper's architecture. The VAE class that assembles them into the actual
"Bringing Old Photos Back to Life"-style domain VAE is in vae.py, and that
assembly (encoder depth, bottleneck design, how mu/logvar are produced) is
the part you're implementing yourself from the paper's description.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A standard two-conv residual block with instance normalization.
    Used inside the encoder/decoder to add capacity without changing
    spatial resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    """Strided conv that halves spatial resolution and doubles channels
    (up to a cap), used to build the encoder's downsampling path."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Transposed conv that doubles spatial resolution, used to build the
    decoder's upsampling path (mirrors DownsampleBlock)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
