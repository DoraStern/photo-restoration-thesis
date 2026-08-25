"""
Building blocks for the Stage 1 restoration network.

Note the deliberate architectural difference from the VAE scaffold used for
the GAN+latent-translation method: Stage 1 here is a *deterministic*
restoration network with U-Net-style skip connections, not a VAE with a
compressive stochastic bottleneck. That's an intentional choice, not an
oversight -- Stage 1's job is "remove degradation while staying as
faithful as possible to the input," and skip connections are what let fine
detail (edges, texture) bypass the bottleneck entirely rather than being
forced through a compressed latent representation. The original DiffBIR
paper uses SwinIR (a transformer-based restorer) for this stage; a
convolutional U-Net plays the same functional role at a fraction of the
compute, which is a reasonable and defensible scope reduction for a thesis
rather than a full from-scratch SwinIR reproduction.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convs + norm + activation, resolution-preserving. The basic unit
    used at every U-Net stage (encoder, bottleneck, and decoder)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock followed by a strided-conv downsample. Returns both the
    pre-downsample features (kept as a skip connection) and the
    downsampled output (passed deeper into the encoder)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        down = self.downsample(skip)
        return down, skip


class UpBlock(nn.Module):
    """Upsamples, concatenates the matching encoder skip connection, then
    fuses with a ConvBlock. This is the actual mechanism that lets Stage 1
    stay faithful to fine input detail rather than smoothing everything
    through the bottleneck."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard against off-by-one size mismatches from odd input dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
