"""
Train VAE2 (domain B) on clean photos + their synthetically damaged
versions. Unlike VAE1, this trains with a dual-branch objective: both the
clean image AND its degraded version get encoded, and BOTH must decode
back to the CLEAN target, with a consistency term pulling their latents
together. See models/vae.py:vae2_loss for the full reasoning.

Usage:
    python train_vae_domain_b.py --clean-dir ./voc_data --masks-dir ./generated_masks \
        --epochs 50 --batch-size 8 --image-size 256 --out-dir ./runs/vae_domain_b
"""

import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils
from tqdm import tqdm

from method1_gan_vae.models.vae import DomainVAE, vae2_loss
from method1_gan_vae.data.vae2_pair_dataset import VAE2PairDataset, denormalize


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
        self.log_path = log_path
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


def save_comparison_grid(model, batch, out_path, device, max_images=6):
    """Saves [clean input | clean recon | degraded input | degraded->clean recon]
    so you can visually confirm both branches are converging toward the
    same clean output, which is the whole point of VAE2's training."""
    model.eval()
    with torch.no_grad():
        n = min(max_images, batch["clean"].shape[0])
        clean = batch["clean"][:n].to(device)
        degraded = batch["degraded"][:n].to(device)

        recon_clean, _, _ = model(clean)
        recon_degraded, _, _ = model(degraded)

        grid = torch.cat([
            denormalize(clean), denormalize(recon_clean),
            denormalize(degraded), denormalize(recon_degraded),
        ], dim=0)
        vutils.save_image(grid, out_path, nrow=n)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train VAE2 on clean photos + synthetic degradation (domain B).")
    parser.add_argument("--clean-dir", type=str, required=True, help="folder of clean photos (e.g. VOC2012)")
    parser.add_argument("--masks-dir", type=str, required=True, nargs="+",
                         help="one or more mask folders; pass multiple to combine damage types kept "
                              "in separate folders, e.g. --masks-dir ./data/masks/scratches ./data/masks/smut")
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen")
    parser.add_argument("--out-dir", type=str, default="./runs/vae_domain_b")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--consistency-weight", type=float, default=1.0,
                         help="weight pulling the degraded branch's latent toward the clean branch's latent")
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--n-downsample", type=int, default=3)
    parser.add_argument("--n-residual-blocks", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
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
    cpu_count = os.cpu_count()
    log(f"  CPUs available: {cpu_count}, --num-workers set to {args.num_workers}")

    log("Building dataset index...")
    dataset = VAE2PairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size,
                               augment=True, blend_mode=args.blend_mode)
    log(f"Loaded {len(dataset)} clean images, {len(dataset.mask_files)} masks "
        f"(blend_mode={args.blend_mode})")

    if args.steps_per_epoch:
        num_samples = args.steps_per_epoch * args.batch_size
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                                 persistent_workers=(args.num_workers > 0))
        log(f"Using --steps-per-epoch {args.steps_per_epoch}: {num_samples} images/epoch")
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                                 persistent_workers=(args.num_workers > 0))

    log("Fetching a fixed sample batch for visualization...")
    fixed_batch = next(iter(dataloader))
    log("Dataset ready.")

    model = DomainVAE(
        in_channels=3, n_downsample=args.n_downsample,
        n_residual_blocks=args.n_residual_blocks, latent_channels=args.latent_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    global_step = 0
    if args.resume:
        log(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)

    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model has {num_params:,} parameters")
    log(f"Starting training: epochs {start_epoch}-{args.epochs}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_rc, running_rd, running_cons, running_kl = 0.0, 0.0, 0.0, 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        batch_end_time = time.time()

        for batch in progress_bar:
            data_time = time.time() - batch_end_time
            compute_start = time.time()

            clean = batch["clean"].to(device, non_blocking=True)
            degraded = batch["degraded"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                recon_clean, mu_clean, logvar_clean = model(clean)
                recon_degraded, mu_degraded, logvar_degraded = model(degraded)
                loss, rc, rd, cons, kl = vae2_loss(
                    recon_clean, recon_degraded, clean, mu_clean, logvar_clean,
                    mu_degraded, logvar_degraded, kl_weight=args.kl_weight,
                    consistency_weight=args.consistency_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.time() - compute_start

            running_loss += loss.item()
            running_rc += rc.item()
            running_rd += rd.item()
            running_cons += cons.item()
            running_kl += kl.item()
            global_step += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.log_every and global_step % args.log_every == 0:
                log(f"  step {global_step}: data_time={data_time:.3f}s compute_time={compute_time:.3f}s")

            if global_step % args.sample_every == 0:
                sample_path = os.path.join(samples_dir, f"step_{global_step:07d}.png")
                save_comparison_grid(model, fixed_batch, sample_path, device)
                log(f"  Saved sample grid: {sample_path}")

            batch_end_time = time.time()

        n_batches = len(dataloader)
        elapsed = time.time() - start_time
        log(f"[Epoch {epoch}/{args.epochs}] loss={running_loss / n_batches:.4f} "
            f"recon_clean={running_rc / n_batches:.4f} recon_degraded={running_rd / n_batches:.4f} "
            f"consistency={running_cons / n_batches:.4f} kl={running_kl / n_batches:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"total_elapsed={format_duration(elapsed)}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"vae_domain_b_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            log(f"  Saved checkpoint: {ckpt_path}")

    log(f"Training complete. Total time: {format_duration(time.time() - start_time)}")
    logger.close()


if __name__ == "__main__":
    main()
