"""
Train VAE1 (domain A) on real old photos, unsupervised reconstruction.

This is the first, smallest independently-testable piece of the full
pipeline: encoder -> reparameterize -> decoder, trained to reconstruct
real old photos. Once this trains stably, VAE2 (domain B, on clean photos)
uses the exact same architecture/training loop against a different
dataset -- and both feed into the mapping network stage after that.

Usage:
    python train_vae_domain_a.py --data-root ./real_old_photos --epochs 50 \
        --batch-size 8 --image-size 256 --out-dir ./runs/vae_domain_a
"""

import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils
from tqdm import tqdm

from method1_gan_vae.models.vae import DomainVAE, vae_loss
from common.real_photo_dataset import RealOldPhotoDataset, denormalize


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
    """Prints with a wall-clock timestamp AND appends the same line to a log
    file, so you can check progress from outside the live session (e.g.
    tailing the file over SSH, or reopening a Kaggle notebook that's still
    running) rather than only trusting that the visible cell output is
    current. Opens in append mode, so resuming a run continues the same log
    rather than overwriting history."""

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


def save_reconstruction_grid(model, batch, out_path, device, max_images=8):
    """Saves a side-by-side grid of [original | reconstruction] so you can
    visually sanity-check training progress, not just watch the loss curve."""
    model.eval()
    with torch.no_grad():
        images = batch["image"][:max_images].to(device)
        recon, _, _ = model(images)
        comparison = torch.cat([denormalize(images), denormalize(recon)], dim=0)
        vutils.save_image(comparison, out_path, nrow=max_images)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train VAE1 on real old photos (domain A).")
    parser.add_argument("--data-root", type=str, required=True,
                         help="folder containing images/ and manifest.csv from the scraper scripts")
    parser.add_argument("--out-dir", type=str, default="./runs/vae_domain_a")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8,
                         help="the official repo uses 80-120 across 4 GPUs; start much smaller on one GPU/CPU")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01,
                         help="weight on the KL term; too high early on can collapse reconstructions to blur")
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--n-downsample", type=int, default=3)
    parser.add_argument("--n-residual-blocks", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5, help="save a checkpoint every N epochs")
    parser.add_argument("--sample-every", type=int, default=200,
                         help="save a reconstruction sample grid every N training steps")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true",
                         help="use automatic mixed precision (fp16) training -- meaningful speedup on modern "
                              "GPUs (T4 and newer) with minimal code cost; no effect on CPU")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                         help="if set, each 'epoch' samples this many random batches (with replacement) instead "
                              "of iterating the full dataset once -- use this to shorten epoch wall-clock time.")
    parser.add_argument("--log-every", type=int, default=20,
                         help="print a data-loading-vs-compute timing breakdown every N steps. Set to 0 to disable.")
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
    if args.num_workers > cpu_count:
        log(f"  Warning: --num-workers ({args.num_workers}) exceeds available CPUs ({cpu_count}); "
            f"this can hurt rather than help. Consider lowering it.")

    log("Building dataset index (scanning images/ and manifest.csv)...")
    dataset = RealOldPhotoDataset(args.data_root, image_size=args.image_size, augment=True)
    log(f"Loaded {len(dataset)} real old photos from {args.data_root}")

    if args.steps_per_epoch:
        num_samples = args.steps_per_epoch * args.batch_size
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                                 persistent_workers=(args.num_workers > 0))
        log(f"Using --steps-per-epoch {args.steps_per_epoch}: each epoch samples "
            f"{num_samples} images (with replacement) instead of the full {len(dataset)}-image dataset")
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                                 persistent_workers=(args.num_workers > 0))

    # A small fixed batch, held out purely for visualizing reconstruction
    # quality over time (not used for any gradient updates). This first
    # dataloader fetch is also what spins up worker processes -- can take a
    # few seconds even before any training happens, hence the explicit log.
    log("Fetching a fixed sample batch for visualization (this also starts the dataloader workers)...")
    fixed_batch = next(iter(dataloader))
    log("Dataset ready.")

    model = DomainVAE(
        in_channels=3,
        n_downsample=args.n_downsample,
        n_residual_blocks=args.n_residual_blocks,
        latent_channels=args.latent_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        log("Note: --amp has no effect on CPU, ignoring.")
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
        running_loss, running_recon, running_kl = 0.0, 0.0, 0.0
        running_data_time, running_compute_time = 0.0, 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)

        batch_end_time = time.time()
        for batch in progress_bar:
            data_time = time.time() - batch_end_time

            compute_start = time.time()
            images = batch["image"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_weight=args.kl_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.time() - compute_start

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            running_data_time += data_time
            running_compute_time += compute_time
            global_step += 1

            # Live, continuously-updating feedback -- this is what tells you
            # "still working" second by second, rather than waiting on a
            # periodic print that might be minutes away.
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "data_t": f"{data_time:.2f}s",
                "compute_t": f"{compute_time:.2f}s",
            })

            if args.log_every and global_step % args.log_every == 0:
                msg = (f"  step {global_step}: data_time={data_time:.3f}s compute_time={compute_time:.3f}s "
                       f"({'data-loading-bound' if data_time > compute_time else 'compute-bound'})")
                if device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    msg += f" | GPU mem: {mem_alloc:.2f}GB alloc / {mem_reserved:.2f}GB reserved"
                log(msg)

            if global_step % args.sample_every == 0:
                sample_path = os.path.join(samples_dir, f"step_{global_step:07d}.png")
                save_reconstruction_grid(model, fixed_batch, sample_path, device)
                log(f"  Saved sample grid: {sample_path}")

            batch_end_time = time.time()

        n_batches = len(dataloader)
        elapsed = time.time() - start_time
        log(f"[Epoch {epoch}/{args.epochs}] "
            f"loss={running_loss / n_batches:.4f} "
            f"recon={running_recon / n_batches:.4f} "
            f"kl={running_kl / n_batches:.4f} "
            f"avg_data_time={running_data_time / n_batches:.3f}s "
            f"avg_compute_time={running_compute_time / n_batches:.3f}s "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"total_elapsed={format_duration(elapsed)}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"vae_domain_a_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            log(f"  Saved checkpoint: {ckpt_path}")

    total_elapsed = time.time() - start_time
    log(f"Training complete. Total time: {format_duration(total_elapsed)}")
    logger.close()


if __name__ == "__main__":
    main()
