"""
Train the Stage 1 restoration network on (damaged, clean) pairs, where
damage comes from your FilmDamageSimulator mask pool.

Usage:
    python train_stage1_restoration.py \
        --clean-dir ./voc_subset --masks-dir ./generated_masks \
        --epochs 50 --batch-size 8 --image-size 256 \
        --out-dir ./runs/stage1_restoration
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils

from method2_diffbir.models.restoration_net import RestorationUNet, restoration_loss
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


def save_comparison_grid(model, batch, out_path, device, max_images=6):
    """Saves [damaged input | model output | clean target] so you can
    visually confirm the network is actually removing damage, not just
    minimizing loss in some degenerate way (e.g. blurring everything)."""
    model.eval()
    with torch.no_grad():
        n = min(max_images, batch["damaged"].shape[0])
        damaged = batch["damaged"][:n].to(device)
        clean = batch["clean"][:n].to(device)
        restored = model(damaged)
        comparison = torch.cat([denormalize(damaged), denormalize(restored), denormalize(clean)], dim=0)
        # nrow=n (not a fixed max_images) guarantees each group of n images
        # forms exactly one row, regardless of how the actual batch size
        # compares to max_images.
        vutils.save_image(comparison, out_path, nrow=n)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train the Stage 1 restoration U-Net.")
    parser.add_argument("--clean-dir", type=str, required=True, help="folder of clean photos (e.g. VOC2012 subset)")
    parser.add_argument("--masks-dir", type=str, required=True, nargs="+",
                         help="one or more mask folders; pass multiple to combine damage types kept "
                              "in separate folders, e.g. --masks-dir ./data/masks/scratches ./data/masks/smut")
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen",
                         help="'screen' (default) = light/white damage marks (realistic for scratches/abrasion); "
                              "'multiply' = dark damage marks (realistic for soot/smut/heavy dirt)")
    parser.add_argument("--out-dir", type=str, default="./runs/stage1_restoration")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--n-downsample", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true",
                         help="use automatic mixed precision (fp16) training -- meaningful speedup on modern "
                              "GPUs (T4 and newer) with minimal code cost; no effect on CPU")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                         help="if set, each 'epoch' samples this many random batches (with replacement) instead "
                              "of iterating the full dataset once. Use this to shorten epoch wall-clock time "
                              "without changing the model or data -- e.g. --steps-per-epoch 200 with batch-size "
                              "16 processes 3200 images/epoch instead of the full ~17000 in VOC2012, roughly "
                              "proportionally cutting epoch time.")
    parser.add_argument("--log-every", type=int, default=20,
                         help="print a data-loading-vs-compute timing breakdown every N steps, so you can see "
                              "whether slowness is coming from data loading (CPU/disk-bound) or the actual "
                              "model forward/backward pass (GPU-bound). Set to 0 to disable.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.out_dir, "checkpoints")
    samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    cpu_count = os.cpu_count()
    print(f"  CPUs available: {cpu_count}, --num-workers set to {args.num_workers}")
    if args.num_workers > cpu_count:
        print(f"  Warning: --num-workers ({args.num_workers}) exceeds available CPUs ({cpu_count}); "
              f"this can hurt rather than help. Consider lowering it.")

    dataset = DegradedPairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size, augment=True,
                                   blend_mode=args.blend_mode)
    print(f"Loaded {len(dataset)} clean images, {len(dataset.mask_files)} damage masks "
          f"from {args.clean_dir} / {args.masks_dir} (blend_mode={args.blend_mode})")
    if args.steps_per_epoch:
        num_samples = args.steps_per_epoch * args.batch_size
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                                 persistent_workers=(args.num_workers > 0))
        print(f"Using --steps-per-epoch {args.steps_per_epoch}: each epoch samples "
              f"{num_samples} images (with replacement) instead of the full {len(dataset)}-image dataset")
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
                                 persistent_workers=(args.num_workers > 0))

    fixed_batch = next(iter(dataloader))

    model = RestorationUNet(
        in_channels=3, out_channels=3,
        base_channels=args.base_channels, n_downsample=args.n_downsample,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("Note: --amp has no effect on CPU, ignoring.")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_data_time = 0.0
        running_compute_time = 0.0

        batch_end_time = time.time()  # marks the end of the previous iteration
        for batch in dataloader:
            data_time = time.time() - batch_end_time  # time spent waiting for this batch

            compute_start = time.time()
            damaged = batch["damaged"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                restored = model(damaged)
                loss, l1 = restoration_loss(restored, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()  # so compute_time reflects actual GPU work, not just kernel launch
            compute_time = time.time() - compute_start

            running_loss += loss.item()
            running_data_time += data_time
            running_compute_time += compute_time
            global_step += 1

            if args.log_every and global_step % args.log_every == 0:
                msg = (f"  step {global_step}: data_time={data_time:.3f}s compute_time={compute_time:.3f}s "
                       f"({'data-loading-bound' if data_time > compute_time else 'compute-bound'})")
                if device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    msg += f" | GPU mem: {mem_alloc:.2f}GB alloc / {mem_reserved:.2f}GB reserved"
                print(msg)

            if global_step % args.sample_every == 0:
                sample_path = os.path.join(samples_dir, f"step_{global_step:07d}.png")
                save_comparison_grid(model, fixed_batch, sample_path, device)

            batch_end_time = time.time()

        n_batches = len(dataloader)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"loss={running_loss / n_batches:.4f} "
              f"avg_data_time={running_data_time / n_batches:.3f}s "
              f"avg_compute_time={running_compute_time / n_batches:.3f}s "
              f"epoch_time={format_duration(time.time() - epoch_start)} "
              f"total_elapsed={format_duration(elapsed)}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"stage1_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    total_elapsed = time.time() - start_time
    print(f"\nTraining complete. Total time: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
