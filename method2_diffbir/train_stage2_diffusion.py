"""
Train Method 2's Stage 2: the ControlNet-style adapter that steers a
frozen pretrained Stable Diffusion model using Stage 1's restored output
as a conditioning signal, adding back realistic detail that Stage 1's
regression loss necessarily smoothed away.

Per training step:
    1. Damaged image -> frozen Stage 1 model -> restored-but-smoothed
       output (this is the "hint")
    2. Clean target image -> frozen VAE -> clean latent
    3. Random noise added to the clean latent at a random timestep
       (standard diffusion training)
    4. ControlNet processes the noisy latent + the Stage 1 hint, producing
       residual signals injected into the frozen UNet's blocks
    5. Frozen UNet predicts the noise that was added; loss = MSE between
       predicted and actual noise
    6. Only the ControlNet's weights are updated

Requires unrestricted internet access to Hugging Face Hub (to download
the pretrained SD checkpoint on first run) -- run this on Kaggle or
similar, not in a network-sandboxed environment.

Usage:
    python train_stage2_diffusion.py \
        --stage1-checkpoint ./runs/stage1_restoration/checkpoints/stage1_epoch0050.pt \
        --clean-dir ./voc_data --masks-dir ./generated_masks \
        --epochs 20 --batch-size 1 --image-size 256 \
        --amp --gradient-checkpointing --out-dir ./runs/stage2_diffusion
"""

import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils
from tqdm import tqdm

from method2_diffbir.models.restoration_net import RestorationUNet
from method2_diffbir.models.diffusion_stage2 import (
    load_frozen_sd_components, build_controlnet, get_empty_prompt_embedding,
    encode_to_latent, decode_from_latent, DEFAULT_MODEL_ID,
)
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


def load_frozen_stage1(checkpoint_path, device):
    """Same architecture-from-checkpoint-args pattern as evaluate.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt.get("args", {})
    model = RestorationUNet(
        in_channels=3, out_channels=3,
        base_channels=train_args.get("base_channels", 64),
        n_downsample=train_args.get("n_downsample", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def sample_and_save(unet, controlnet, vae, hint_image, clean_image, empty_embedding,
                     noise_scheduler, num_inference_steps, device, out_path):
    """Runs a full multi-step diffusion sampling loop conditioned on a
    fixed hint, decodes the result, and saves a
    [Stage-1 hint | Stage-2 output | clean target] comparison grid."""
    controlnet.eval()
    batch_size = hint_image.shape[0]
    latent_h = hint_image.shape[-2] // 8
    latent_w = hint_image.shape[-1] // 8

    latents = torch.randn((batch_size, unet.config.in_channels, latent_h, latent_w), device=device)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * noise_scheduler.init_noise_sigma

    for t in noise_scheduler.timesteps:
        latent_model_input = noise_scheduler.scale_model_input(latents, t)

        down_res, mid_res = controlnet(
            latent_model_input, t, encoder_hidden_states=empty_embedding,
            controlnet_cond=hint_image, return_dict=False,
        )
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=empty_embedding,
            down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
        ).sample

        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    decoded = decode_from_latent(vae, latents)

    grid = torch.cat([denormalize(hint_image), denormalize(decoded.clamp(-1, 1)), denormalize(clean_image)], dim=0)
    vutils.save_image(grid, out_path, nrow=batch_size)
    controlnet.train()


def main():
    parser = argparse.ArgumentParser(description="Train Method 2's Stage 2 ControlNet-style adapter.")
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--clean-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True, nargs="+")
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                         help="Hugging Face Hub ID of the pretrained Stable Diffusion checkpoint")
    parser.add_argument("--out-dir", type=str, default="./runs/stage2_diffusion")
    parser.add_argument("--image-size", type=int, default=256,
                         help="must be divisible by 8 (the VAE's downsampling factor)")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Stage 2 is far heavier than Stage 1 -- start at 1 and only raise this "
                              "if you have headroom left after --gradient-checkpointing and --amp")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="ControlNet training typically uses a smaller LR than training from scratch")
    parser.add_argument("--num-inference-steps", type=int, default=20,
                         help="denoising steps used for sample-grid visualization during training "
                              "(fewer = faster preview, not used for final quality)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                         help="enables gradient checkpointing on the frozen UNet and the trainable "
                              "ControlNet -- strongly recommended, this stage is memory-heavy")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.image_size % 8 != 0:
        raise ValueError(f"--image-size must be divisible by 8 (VAE downsampling factor), got {args.image_size}")

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

    log(f"Loading frozen Stage 1 checkpoint from {args.stage1_checkpoint}")
    stage1_model = load_frozen_stage1(args.stage1_checkpoint, device)

    log(f"Loading frozen Stable Diffusion components from '{args.model_id}' "
        f"(requires internet access to Hugging Face Hub)...")
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_frozen_sd_components(args.model_id, device)
    log("Frozen SD components loaded and frozen (VAE, UNet, text encoder).")

    log("Building trainable ControlNet adapter from the frozen UNet...")
    controlnet = build_controlnet(unet).to(device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
        log("Gradient checkpointing enabled on UNet and ControlNet.")

    empty_embedding = get_empty_prompt_embedding(text_encoder, tokenizer, device)
    log(f"Empty-prompt text embedding ready, shape={tuple(empty_embedding.shape)}")

    log("Building dataset...")
    dataset = DegradedPairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size,
                                   augment=True, blend_mode=args.blend_mode)
    log(f"Loaded {len(dataset)} clean images, {len(dataset.mask_files)} masks")

    if args.steps_per_epoch:
        num_samples = args.steps_per_epoch * args.batch_size
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"))
        log(f"Using --steps-per-epoch {args.steps_per_epoch}: {num_samples} images/epoch")
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"))

    fixed_batch = next(iter(dataloader))

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    global_step = 0
    if args.resume:
        log(f"Resuming ControlNet weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        controlnet.load_state_dict(ckpt["controlnet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)

    num_trainable = sum(p.numel() for p in controlnet.parameters())
    log(f"ControlNet has {num_trainable:,} trainable parameters (UNet/VAE/text encoder frozen)")
    log(f"Starting training: epochs {start_epoch}-{args.epochs}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        batch_end_time = time.time()

        for batch in progress_bar:
            data_time = time.time() - batch_end_time
            compute_start = time.time()

            damaged = batch["damaged"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            with torch.no_grad():
                hint = stage1_model(damaged).clamp(-1, 1)

            clean_latent = encode_to_latent(vae, clean)
            noise = torch.randn_like(clean_latent)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                       (clean_latent.shape[0],), device=device).long()
            noisy_latent = noise_scheduler.add_noise(clean_latent, noise, timesteps)

            batch_embedding = empty_embedding.expand(clean_latent.shape[0], -1, -1)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                down_res, mid_res = controlnet(
                    noisy_latent, timesteps, encoder_hidden_states=batch_embedding,
                    controlnet_cond=hint, return_dict=False,
                )
                noise_pred = unet(
                    noisy_latent, timesteps, encoder_hidden_states=batch_embedding,
                    down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.time() - compute_start

            running_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.log_every and global_step % args.log_every == 0:
                log(f"  step {global_step}: loss={loss.item():.4f} "
                    f"data_time={data_time:.3f}s compute_time={compute_time:.3f}s")

            if global_step % args.sample_every == 0:
                sample_path = os.path.join(samples_dir, f"step_{global_step:07d}.png")
                fixed_damaged = fixed_batch["damaged"][:2].to(device)
                fixed_clean = fixed_batch["clean"][:2].to(device)
                with torch.no_grad():
                    fixed_hint = stage1_model(fixed_damaged).clamp(-1, 1)
                fixed_embedding = empty_embedding.expand(2, -1, -1)
                sample_and_save(unet, controlnet, vae, fixed_hint, fixed_clean, fixed_embedding,
                                 noise_scheduler, args.num_inference_steps, device, sample_path)
                log(f"  Saved sample grid: {sample_path}")

            batch_end_time = time.time()

        n_batches = len(dataloader)
        elapsed = time.time() - start_time
        log(f"[Epoch {epoch}/{args.epochs}] loss={running_loss / n_batches:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"total_elapsed={format_duration(elapsed)}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"stage2_controlnet_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "controlnet_state_dict": controlnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            log(f"  Saved checkpoint: {ckpt_path} (ControlNet weights only -- "
                f"frozen VAE/UNet/text encoder are not re-saved, re-download via --model-id instead)")

    log(f"Training complete. Total time: {format_duration(time.time() - start_time)}")
    logger.close()


if __name__ == "__main__":
    main()
