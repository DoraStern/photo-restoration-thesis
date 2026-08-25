"""
Stage 1 restoration network: a convolutional U-Net that maps a damaged
(scratches/smut-composited) photo to a restored one.

This is deliberately simpler than DiffBIR's own Stage 1 (which uses
SwinIR, a much heavier transformer-based restorer trained on a broad,
generic blur/noise/JPEG/downsampling degradation model). Two scope
decisions worth being explicit about in your thesis writeup:

1. Architecture: U-Net instead of SwinIR. Same functional role (faithful,
   detail-preserving degradation removal), far less compute. This is a
   standard, well-understood restoration architecture in its own right
   (used across denoising/inpainting literature), not a shortcut invented
   for this project.

2. Degradation model: trained specifically on YOUR FilmDamageSimulator
   scratches/smut compositing, not DiffBIR's generic blur/noise/JPEG/
   downsampling pipeline. This is actually a *better* fit for your thesis
   question (how well does this restoration paradigm handle these specific
   damage types) than reproducing their generic degradation model would be.
"""

import torch
import torch.nn as nn

from method2_diffbir.models.unet_blocks import ConvBlock, DownBlock, UpBlock


class RestorationUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, n_downsample: int = 4, max_channels: int = 512):
        super().__init__()

        # Build the channel sequence for each encoder stage, e.g. for
        # base_channels=64, n_downsample=4: [64, 128, 256, 512, 512]
        channels = [min(base_channels * (2 ** i), max_channels) for i in range(n_downsample + 1)]

        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channels[:-1]:
            self.down_blocks.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-2], channels[-1]),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.up_blocks = nn.ModuleList()
        up_in_ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.up_blocks.append(UpBlock(up_in_ch, skip_ch, skip_ch))
            up_in_ch = skip_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1], matching the dataset's normalization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.final_conv(x)


def restoration_loss(pred: torch.Tensor, target: torch.Tensor, l1_weight: float = 1.0):
    """
    L1 reconstruction loss, matching DiffBIR's own design reasoning for
    this stage: a regression loss here produces a faithful-but-slightly-
    smoothed output, and that's intentional -- Stage 2 (the diffusion
    prior, built separately) is what adds sharp detail back on top of
    this. Don't be tempted to add heavy perceptual/adversarial losses here
    to make Stage 1 alone look sharper; that would blur the separation of
    concerns the two-stage design is built around.
    """
    l1 = torch.nn.functional.l1_loss(pred, target)
    return l1_weight * l1, l1
