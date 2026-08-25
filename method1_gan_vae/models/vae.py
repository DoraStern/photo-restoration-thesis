"""
A convolutional VAE with a *spatial* latent bottleneck (a small feature map,
not a single flattened vector) rather than the more familiar
flatten-to-a-vector VAE design.

Why spatial: "Bringing Old Photos Back to Life" needs the latent
representation to preserve rough spatial layout, so that later (in the
mapping network you'll build next) a damage mask can be used to tell the
model *where* in the latent space to focus repair. A flattened-vector
latent would throw that spatial correspondence away.

This same class is used for both VAE1 (domain A: real old photos) and VAE2
(domain B: clean photos) -- you'll instantiate two separate copies with
their own weights, one per domain, trained independently in
train_vae_domain_a.py / train_vae_domain_b.py.
"""

import torch
import torch.nn as nn

from method1_gan_vae.models.vae_blocks import ResidualBlock, DownsampleBlock, UpsampleBlock


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_downsample=3,
                 n_residual_blocks=4, latent_channels=64, max_channels=512):
        super().__init__()

        # Initial conv, no downsampling yet
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling path: halve spatial resolution each step, double
        # channels up to max_channels
        channels = base_channels
        for _ in range(n_downsample):
            next_channels = min(channels * 2, max_channels)
            layers.append(DownsampleBlock(channels, next_channels))
            channels = next_channels

        # Residual blocks at the bottleneck resolution, adding capacity
        # without further downsampling
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(channels))

        self.backbone = nn.Sequential(*layers)

        # Separate 1x1 convs producing the mean and log-variance maps of
        # the latent distribution -- same spatial size as the backbone
        # output, just a different channel count
        self.to_mu = nn.Conv2d(channels, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(channels, latent_channels, kernel_size=1)

        self.bottleneck_channels = channels

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        mu = self.to_mu(features)
        logvar = self.to_logvar(features)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, n_downsample=3,
                 n_residual_blocks=4, latent_channels=64, max_channels=512):
        super().__init__()

        # Figure out the bottleneck channel count the same way the
        # encoder did, so the shapes line up
        channels = base_channels
        for _ in range(n_downsample):
            channels = min(channels * 2, max_channels)

        layers = [nn.Conv2d(latent_channels, channels, kernel_size=1)]

        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(channels))

        # Upsampling path: mirror of the encoder's downsampling path
        channel_sequence = []
        c = base_channels
        for _ in range(n_downsample):
            channel_sequence.append(min(c * 2, max_channels))
            c = min(c * 2, max_channels)
        channel_sequence = [base_channels] + channel_sequence
        # channel_sequence e.g. [64, 128, 256, 512] for n_downsample=3;
        # we walk it backwards to go from bottleneck back to base_channels
        for i in range(n_downsample):
            in_ch = channel_sequence[n_downsample - i]
            out_ch = channel_sequence[n_downsample - i - 1]
            layers.append(UpsampleBlock(in_ch, out_ch))

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),  # output in [-1, 1], matches how we'll normalize images
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainVAE(nn.Module):
    """
    Full VAE: encode -> reparameterize -> decode.

    Instantiate one of these per domain (real old photos / clean photos).
    The `Encoder`/`Decoder` above are shared *class* definitions but each
    DomainVAE instance gets its own independently-trained weights.
    """

    def __init__(self, in_channels=3, base_channels=64, n_downsample=3,
                 n_residual_blocks=4, latent_channels=64, max_channels=512):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, n_downsample,
                                n_residual_blocks, latent_channels, max_channels)
        self.decoder = Decoder(in_channels, base_channels, n_downsample,
                                n_residual_blocks, latent_channels, max_channels)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """The standard VAE reparameterization trick: sample z = mu + eps*std
        where eps ~ N(0, 1), so gradients can flow through the sampling step."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, kl_weight: float = 1.0):
    """
    Standard VAE loss = reconstruction term + KL divergence term.

    Reconstruction uses L1 (tends to give sharper results than MSE for
    images -- this is a common choice in image-translation VAEs, not
    something unique to this paper).

    KL divergence pulls the latent distribution toward a standard normal,
    which is what makes the latent space smooth/well-structured enough for
    the mapping network to later translate between domains.
    """
    recon_loss = torch.nn.functional.l1_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def vae2_loss(recon_clean, recon_degraded, clean_target, mu_clean, logvar_clean,
              mu_degraded, logvar_degraded, kl_weight: float = 1.0, consistency_weight: float = 1.0):
    """
    VAE2's training objective is different from VAE1's plain reconstruction:
    it needs to learn a latent space where a clean photo AND a synthetically
    degraded version of it land close together, and BOTH decode back to the
    clean image. This is what lets the translation network later map a
    repaired latent through VAE2's decoder and get a clean-looking result.

    Four terms:
      1. Standard reconstruction of the clean branch (clean in -> clean out)
      2. Cross-reconstruction of the degraded branch (degraded in -> CLEAN
         out, not degraded out) -- this is the key difference from a plain
         autoencoder; it directly teaches "decode toward clean" regardless
         of which branch encoded the input
      3. Latent consistency: pulls the degraded branch's latent toward the
         clean branch's latent (clean side detached, so gradient flows
         into fixing the degraded encoder rather than both sides drifting
         together into a degenerate shortcut)
      4. KL divergence on both branches (standard VAE regularization)
    """
    recon_loss_clean = torch.nn.functional.l1_loss(recon_clean, clean_target)
    recon_loss_degraded = torch.nn.functional.l1_loss(recon_degraded, clean_target)

    consistency_loss = torch.nn.functional.l1_loss(mu_degraded, mu_clean.detach())

    kl_clean = -0.5 * torch.mean(1 + logvar_clean - mu_clean.pow(2) - logvar_clean.exp())
    kl_degraded = -0.5 * torch.mean(1 + logvar_degraded - mu_degraded.pow(2) - logvar_degraded.exp())
    kl_loss = kl_clean + kl_degraded

    total_loss = (recon_loss_clean + recon_loss_degraded
                  + consistency_weight * consistency_loss
                  + kl_weight * kl_loss)

    return total_loss, recon_loss_clean, recon_loss_degraded, consistency_loss, kl_loss
