"""
The latent translation network -- this is where the actual "repair"
happens. Takes a latent map from either VAE1 (real damaged photo) or VAE2
(synthetic degraded photo), plus a damage mask resized to latent
resolution, and outputs a translated latent that VAE2's decoder can turn
into a clean-looking image.

Also defines the latent-space discriminator used for adversarial training
on real photos, where we have no ground-truth clean version to supervise
against directly (see train_translation_net.py for how the two training
signals -- supervised synthetic pairs + adversarial real photos -- combine).
"""

import torch
import torch.nn as nn

from method1_gan_vae.models.vae_blocks import ResidualBlock


class LatentTranslationNet(nn.Module):
    def __init__(self, latent_channels: int = 64, n_residual_blocks: int = 6):
        super().__init__()

        # Fuse the latent map with the (resized) damage mask -- this is the
        # actual mask-conditioning mechanism: the mask is just concatenated
        # as an extra input channel, giving every residual block access to
        # "is this spatial location damaged" throughout the network.
        self.input_conv = nn.Sequential(
            nn.Conv2d(latent_channels + 1, latent_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(latent_channels) for _ in range(n_residual_blocks)]
        )

        # Output projection back to latent space -- no activation, since
        # latent values (VAE means) aren't bounded to any fixed range
        self.output_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask arrives at image resolution; resize to match the latent
        # map's (much smaller) spatial size
        if mask.shape[-2:] != latent.shape[-2:]:
            mask = torch.nn.functional.interpolate(mask, size=latent.shape[-2:], mode="bilinear",
                                                     align_corners=False)
        x = torch.cat([latent, mask], dim=1)
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        return self.output_conv(x)


class LatentDiscriminator(nn.Module):
    """PatchGAN-style discriminator operating directly on latent maps
    (not images). Outputs a spatial grid of real/fake scores rather than
    one global score, which gives a stronger, more localized training
    signal than a single scalar would."""

    def __init__(self, latent_channels: int = 64, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # No sigmoid -- used with BCEWithLogitsLoss for numerical stability
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)
