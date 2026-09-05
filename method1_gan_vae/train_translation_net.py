"""
Train the latent translation network: the actual repair mechanism of this
method. Combines two training signals each step:

1. SUPERVISED (synthetic pairs, full ground truth available): encode a
   clean photo and its synthetically-degraded twin through frozen VAE2,
   translate the degraded latent with the mask, and directly supervise
   against the known clean latent with L1 loss. This teaches the network
   HOW to use the mask to repair damage.

2. ADVERSARIAL (real old photos, no ground truth): encode a real photo
   through frozen VAE1, translate it (no mask -- we don't know real
   damage locations), and train a discriminator to tell translated-real
   latents apart from genuine clean latents (from VAE2). The translation
   network is trained to fool it. This teaches the network to generalize
   the repair behavior learned from (1) to real-photo statistics, which
   is the actual domain-gap-closing step this whole method exists for.

Also includes an identity loss (translating an already-clean latent should
leave it roughly unchanged) which is a standard stabilizing trick borrowed
from CycleGAN-style unpaired translation training.

VAE1 and VAE2 are both loaded frozen from their own checkpoints and never
updated here -- only the translation network and discriminator train.

Usage:
    python train_translation_net.py \
        --vae1-checkpoint ./runs/vae_domain_a/checkpoints/vae_domain_a_epoch0050.pt \
        --vae2-checkpoint ./runs/vae_domain_b/checkpoints/vae_domain_b_epoch0050.pt \
        --real-photo-dir ./real_old_photos --clean-dir ./voc_data --masks-dir ./generated_masks \
        --epochs 50 --batch-size 8 --out-dir ./runs/translation_net
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils
from tqdm import tqdm

from method1_gan_vae.models.vae import DomainVAE
from method1_gan_vae.models.translation_net import LatentTranslationNet, LatentDiscriminator
from common.real_photo_dataset import RealOldPhotoDataset, denormalize as denorm_a
from method1_gan_vae.data.vae2_pair_dataset import VAE2PairDataset, denormalize as denorm_b


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class Logger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def load_frozen_vae(checkpoint_path, device):
    """Loads a VAE1 or VAE2 checkpoint, reconstructing the architecture
    from the checkpoint's own saved args (same pattern as evaluate.py) so
    there's no risk of mismatching --latent-channels etc. by hand.
    Freezes all parameters -- this VAE is used only for encoding/decoding,
    never updated during translation-network training."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt.get("args", {})
    model = DomainVAE(
        in_channels=3,
        n_downsample=train_args.get("n_downsample", 3),
        n_residual_blocks=train_args.get("n_residual_blocks", 4),
        latent_channels=train_args.get("latent_channels", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, train_args


def save_sample_grid(vae1, vae2, translator, real_batch, synth_batch, out_path, device, max_images=4):
    """Two rows of comparisons:
      - synthetic: degraded | translated->decoded | clean target (has ground truth)
      - real: real damaged photo | translated->decoded (no ground truth, this
        is the actual real-world use case the whole method targets)
    """
    translator.eval()
    with torch.no_grad():
        n = min(max_images, synth_batch["clean"].shape[0])
        clean = synth_batch["clean"][:n].to(device)
        degraded = synth_batch["degraded"][:n].to(device)
        mask = synth_batch["mask"][:n].to(device)

        mu_degraded, _ = vae2.encoder(degraded)
        translated_synth = translator(mu_degraded, mask)
        decoded_synth = vae2.decoder(translated_synth)

        n_real = min(max_images, real_batch["image"].shape[0])
        real = real_batch["image"][:n_real].to(device)
        mu_real, _ = vae1.encoder(real)
        zero_mask = torch.zeros((n_real, 1, real.shape[-2], real.shape[-1]), device=device)
        translated_real = translator(mu_real, zero_mask)
        decoded_real = vae2.decoder(translated_real)

        row1 = torch.cat([denorm_b(degraded), denorm_b(decoded_synth), denorm_b(clean)], dim=0)
        row2 = torch.cat([denorm_a(real), denorm_a(decoded_real)], dim=0)

        vutils.save_image(row1, out_path.replace(".png", "_synthetic.png"), nrow=n)
        vutils.save_image(row2, out_path.replace(".png", "_real.png"), nrow=n_real)
    translator.train()


def main():
    parser = argparse.ArgumentParser(description="Train the latent translation network.")
    parser.add_argument("--vae1-checkpoint", type=str, required=True)
    parser.add_argument("--vae2-checkpoint", type=str, required=True)
    parser.add_argument("--real-photo-dir", type=str, required=True)
    parser.add_argument("--clean-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True, nargs="+",
                         help="one or more mask folders; pass multiple to combine damage types kept "
                              "in separate folders, e.g. --masks-dir ./data/masks/scratches ./data/masks/smut")
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen")
    parser.add_argument("--out-dir", type=str, default="./runs/translation_net")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-residual-blocks", type=int, default=6)
    parser.add_argument("--sup-weight", type=float, default=10.0,
                         help="weight on the supervised synthetic-pair loss")
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=100,
                         help="both the real-photo and synthetic-pair dataloaders are capped to this many "
                              "batches per epoch, so they stay aligned regardless of underlying dataset size")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.out_dir, "checkpoints")
    samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    logger = Logger(os.path.join(args.out_dir, "train_log.txt"))
    log = logger.log

    device = torch.device(args.device)
    log(f"Using device: {device}")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(device)}")

    log(f"Loading frozen VAE1 from {args.vae1_checkpoint}")
    vae1, vae1_args = load_frozen_vae(args.vae1_checkpoint, device)
    log(f"Loading frozen VAE2 from {args.vae2_checkpoint}")
    vae2, vae2_args = load_frozen_vae(args.vae2_checkpoint, device)

    latent_channels = vae1_args.get("latent_channels", 64)
    assert latent_channels == vae2_args.get("latent_channels", 64), \
        "VAE1 and VAE2 must share the same --latent-channels -- they were trained with different values."

    log("Building datasets...")
    real_dataset = RealOldPhotoDataset(args.real_photo_dir, image_size=args.image_size, augment=True)
    synth_dataset = VAE2PairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size,
                                     augment=True, blend_mode=args.blend_mode)
    log(f"Real photos: {len(real_dataset)}. Synthetic pairs: {len(synth_dataset)} clean images, "
        f"{len(synth_dataset.mask_files)} masks.")

    real_sampler = RandomSampler(real_dataset, replacement=True,
                                  num_samples=args.steps_per_epoch * args.batch_size)
    synth_sampler = RandomSampler(synth_dataset, replacement=True,
                                   num_samples=args.steps_per_epoch * args.batch_size)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, sampler=real_sampler,
                              num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.num_workers > 0))
    synth_loader = DataLoader(synth_dataset, batch_size=args.batch_size, sampler=synth_sampler,
                               num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                               persistent_workers=(args.num_workers > 0))

    log("Fetching fixed sample batches for visualization...")
    fixed_real_batch = next(iter(real_loader))
    fixed_synth_batch = next(iter(synth_loader))
    log("Datasets ready.")

    translator = LatentTranslationNet(latent_channels=latent_channels,
                                       n_residual_blocks=args.n_residual_blocks).to(device)
    discriminator = LatentDiscriminator(latent_channels=latent_channels).to(device)

    opt_g = torch.optim.Adam(translator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    start_epoch = 1
    global_step = 0
    if args.resume:
        log(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        translator.load_state_dict(ckpt["translator_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        opt_g.load_state_dict(ckpt["opt_g_state_dict"])
        opt_d.load_state_dict(ckpt["opt_d_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)

    num_params = sum(p.numel() for p in translator.parameters())
    log(f"Translator has {num_params:,} parameters")
    log(f"Starting training: epochs {start_epoch}-{args.epochs}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running = {"sup": 0.0, "identity": 0.0, "adv_g": 0.0, "adv_d": 0.0}

        progress_bar = tqdm(zip(real_loader, synth_loader), total=args.steps_per_epoch,
                             desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        batch_end_time = time.time()

        for real_batch, synth_batch in progress_bar:
            data_time = time.time() - batch_end_time
            compute_start = time.time()

            real = real_batch["image"].to(device, non_blocking=True)
            clean = synth_batch["clean"].to(device, non_blocking=True)
            degraded = synth_batch["degraded"].to(device, non_blocking=True)
            mask = synth_batch["mask"].to(device, non_blocking=True)

            with torch.no_grad():
                mu_clean, _ = vae2.encoder(clean)
                mu_degraded, _ = vae2.encoder(degraded)
                mu_real, _ = vae1.encoder(real)

            zero_mask_real = torch.zeros((mu_real.shape[0], 1, real.shape[-2], real.shape[-1]), device=device)

            # ---- Train discriminator ----
            opt_d.zero_grad()
            translated_real_detached = translator(mu_real, zero_mask_real).detach()
            d_real_out = discriminator(mu_clean)
            d_fake_out = discriminator(translated_real_detached)
            loss_d_real = bce(d_real_out, torch.ones_like(d_real_out))
            loss_d_fake = bce(d_fake_out, torch.zeros_like(d_fake_out))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # ---- Train translator (generator) ----
            opt_g.zero_grad()

            translated_synth = translator(mu_degraded, mask)
            loss_sup = torch.nn.functional.l1_loss(translated_synth, mu_clean)

            zero_mask_clean = torch.zeros_like(mask)
            translated_identity = translator(mu_clean, zero_mask_clean)
            loss_identity = torch.nn.functional.l1_loss(translated_identity, mu_clean)

            translated_real = translator(mu_real, zero_mask_real)
            d_out_for_g = discriminator(translated_real)
            loss_adv_g = bce(d_out_for_g, torch.ones_like(d_out_for_g))

            loss_g = (args.sup_weight * loss_sup
                      + args.identity_weight * loss_identity
                      + args.adv_weight * loss_adv_g)
            loss_g.backward()
            opt_g.step()

            compute_time = time.time() - compute_start

            running["sup"] += loss_sup.item()
            running["identity"] += loss_identity.item()
            running["adv_g"] += loss_adv_g.item()
            running["adv_d"] += loss_d.item()
            global_step += 1

            progress_bar.set_postfix({"sup": f"{loss_sup.item():.4f}", "adv_d": f"{loss_d.item():.4f}"})

            if args.log_every and global_step % args.log_every == 0:
                log(f"  step {global_step}: data_time={data_time:.3f}s compute_time={compute_time:.3f}s")

            if global_step % args.sample_every == 0:
                sample_path = os.path.join(samples_dir, f"step_{global_step:07d}.png")
                save_sample_grid(vae1, vae2, translator, fixed_real_batch, fixed_synth_batch,
                                  sample_path, device)
                log(f"  Saved sample grids: {sample_path.replace('.png', '_synthetic.png')} / "
                    f"{sample_path.replace('.png', '_real.png')}")

            batch_end_time = time.time()

        elapsed = time.time() - start_time
        log(f"[Epoch {epoch}/{args.epochs}] "
            f"sup={running['sup'] / args.steps_per_epoch:.4f} "
            f"identity={running['identity'] / args.steps_per_epoch:.4f} "
            f"adv_g={running['adv_g'] / args.steps_per_epoch:.4f} "
            f"adv_d={running['adv_d'] / args.steps_per_epoch:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"total_elapsed={format_duration(elapsed)}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"translation_net_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "translator_state_dict": translator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            log(f"  Saved checkpoint: {ckpt_path}")

    log(f"Training complete. Total time: {format_duration(time.time() - start_time)}")
    logger.close()


if __name__ == "__main__":
    main()
