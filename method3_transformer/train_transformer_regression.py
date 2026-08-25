"""
Train the transformer regression restoration network (Method 3): pure
windowed self-attention, trained with a plain L1 loss, no adversarial
training, no diffusion sampling.

Reuses the exact same DegradedPairDataset from Method 2's Stage 1 --
clean photos + your FilmDamageSimulator masks, composited on the fly.
Nothing about the data pipeline changes between methods; only the model
architecture and loss do.

Usage:
    python train_transformer_regression.py \
        --clean-dir ./voc_data --masks-dir ./generated_masks \
        --epochs 50 --batch-size 8 --image-size 256 \
        --out-dir ./runs/transformer_regression
"""

import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils
from tqdm import tqdm

from method3_transformer.models.transformer_restoration import TransformerRestorationNet, transformer_regression_loss
from common.degraded_pair_dataset import DegradedPairDataset, denormalize


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


def save_comparison_grid(model, batch, out_path, device, max_images=6):
    """[damaged input | model output | clean target] -- same layout as
    Method 2's Stage 1 sample grids, for direct visual comparison between
    methods later."""
    model.eval()
    with torch.no_grad():
        n = min(max_images, batch["damaged"].shape[0])
        damaged = batch["damaged"][:n].to(device)
        clean = batch["clean"][:n].to(device)
        restored = model(damaged)
        comparison = torch.cat([denormalize(damaged), denormalize(restored), denormalize(clean)], dim=0)
        vutils.save_image(comparison, out_path, nrow=n)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train the transformer regression restoration network.")
    parser.add_argument("--clean-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen")
    parser.add_argument("--out-dir", type=str, default="./runs/transformer_regression")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--embed-dim", type=int, default=60,
                         help="channel width of the transformer features -- scaled down from full "
                              "SwinIR/MDTNet's 180 for thesis-scale compute")
    parser.add_argument("--depths", type=str, default="4,4,4,4",
                         help="comma-separated block count per RSTB group, e.g. '4,4,4,4' for 4 groups "
                              "of 4 blocks each -- scaled down from SwinIR's typical 6 groups of 6")
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--use-checkpoint", action="store_true",
                         help="gradient checkpointing -- trades extra compute time for substantially "
                              "lower memory use, recommended at image-size 256+ to avoid CUDA OOM")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    depths = tuple(int(d) for d in args.depths.split(","))

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
        log(f"  Warning: --num-workers ({args.num_workers}) exceeds available CPUs ({cpu_count}).")

    log("Building dataset index...")
    dataset = DegradedPairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size,
                                   augment=True, blend_mode=args.blend_mode)
    log(f"Loaded {len(dataset)} clean images, {len(dataset.mask_files)} damage masks "
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

    model = TransformerRestorationNet(
        in_channels=3, out_channels=3, embed_dim=args.embed_dim,
        depths=depths, num_heads=args.num_heads, window_size=args.window_size,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

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
    log(f"Model has {num_params:,} parameters (embed_dim={args.embed_dim}, depths={depths}, "
        f"num_heads={args.num_heads}, window_size={args.window_size})")
    log(f"Starting training: epochs {start_epoch}-{args.epochs}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_data_time, running_compute_time = 0.0, 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        batch_end_time = time.time()

        for batch in progress_bar:
            data_time = time.time() - batch_end_time
            compute_start = time.time()

            damaged = batch["damaged"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                restored = model(damaged)
                loss, l1 = transformer_regression_loss(restored, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.time() - compute_start

            running_loss += loss.item()
            running_data_time += data_time
            running_compute_time += compute_time
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
            f"avg_data_time={running_data_time / n_batches:.3f}s "
            f"avg_compute_time={running_compute_time / n_batches:.3f}s "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"total_elapsed={format_duration(elapsed)}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"transformer_regression_epoch{epoch:04d}.pt")
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
