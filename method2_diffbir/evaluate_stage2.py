"""
Evaluate Method 2's full two-stage pipeline: frozen Stage 1 regression
network -> frozen Stable Diffusion + trained ControlNet-style adapter
(Stage 2), via the actual multi-step denoising sampling loop.

Unlike train_stage2_diffusion.py's periodic sample grids (qualitative,
saved during training), this computes real PSNR/SSIM/LPIPS numbers on a
held-out set -- the same evaluation pattern as your other three
evaluate.py scripts, for direct comparability in your results chapter.

Reports THREE levels, not just one, since that's genuinely informative
for your thesis rather than just convenient:
    1. Damaged vs. clean       -- the do-nothing baseline
    2. Stage 1 only vs. clean  -- what plain regression achieves alone
    3. Stage 1+2 vs. clean     -- what the diffusion refinement adds on top

Comparing (2) and (3) directly answers "how much does the diffusion prior
actually help over regression alone" -- the same ablation question raised
earlier when discussing why Method 3 (transformer regression) and Method
2's Stage 1 share the same underlying paradigm.

Requires unrestricted internet access to Hugging Face Hub on first run
(to download the pretrained SD checkpoint) -- run this on Kaggle or
similar, not a network-sandboxed environment.

Multi-step sampling is slow per image (many denoising steps, not one
forward pass) -- expect this to take meaningfully longer than your other
three evaluate.py scripts on the same --num-samples.

Usage:
    python evaluate_stage2.py \
        --stage1-checkpoint ./runs/stage1_restoration/checkpoints/<latest>.pt \
        --controlnet-checkpoint ./runs/stage2_diffusion/checkpoints/<latest>.pt \
        --clean-dir ./data/voc2012/... --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
        --num-samples 50 --num-inference-steps 20 --out-dir ./eval_results_stage2
"""

import argparse
import os

import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.utils as vutils

from models.diffusion_stage2 import (
    load_frozen_sd_components, build_controlnet, get_empty_prompt_embedding,
    decode_from_latent, DEFAULT_MODEL_ID,
)
from train_stage2_diffusion import load_frozen_stage1
from data.degraded_pair_dataset import DegradedPairDataset, denormalize


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return arr


@torch.no_grad()
def run_sampling(unet, controlnet, vae, hint_image, empty_embedding, noise_scheduler,
                  num_inference_steps, device):
    """Same multi-step denoising loop as train_stage2_diffusion.py's
    sample_and_save(), extracted here so it can be reused for a single
    sample at a time during evaluation."""
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

    return decode_from_latent(vae, latents).clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Method 2's full Stage 1 + Stage 2 pipeline.")
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                         help="Hugging Face Hub ID of the pretrained Stable Diffusion checkpoint")
    parser.add_argument("--clean-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True, nargs="+")
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen")
    parser.add_argument("--image-size", type=int, default=256,
                         help="must be divisible by 8 (the VAE's downsampling factor)")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="./eval_results_stage2")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lpips-net", type=str, choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.image_size % 8 != 0:
        raise ValueError(f"--image-size must be divisible by 8, got {args.image_size}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading frozen Stage 1 from {args.stage1_checkpoint}")
    stage1_model = load_frozen_stage1(args.stage1_checkpoint, device)

    print(f"Loading frozen Stable Diffusion components from '{args.model_id}' "
          f"(requires internet access to Hugging Face Hub)...")
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_frozen_sd_components(args.model_id, device)

    print(f"Building ControlNet and loading trained weights from {args.controlnet_checkpoint}")
    controlnet = build_controlnet(unet).to(device)
    ckpt = torch.load(args.controlnet_checkpoint, map_location=device)
    controlnet.load_state_dict(ckpt["controlnet_state_dict"])
    controlnet.eval()
    print(f"  Checkpoint was saved at epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')}")

    empty_embedding = get_empty_prompt_embedding(text_encoder, tokenizer, device)

    print(f"Loading LPIPS ({args.lpips_net}) perceptual similarity model...")
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_fn.eval()

    torch.manual_seed(args.seed)
    dataset = DegradedPairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size,
                                   augment=False, blend_mode=args.blend_mode)
    num_samples = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    print(f"Evaluating on {num_samples} samples (seed={args.seed}, {args.num_inference_steps} "
          f"denoising steps per sample -- this will take a while)")

    metrics = {level: {"psnr": [], "ssim": [], "lpips": []} for level in ("damaged", "stage1", "stage1_stage2")}
    comparison_rows = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            damaged = sample["damaged"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)

            hint = stage1_model(damaged).clamp(-1, 1)
            embedding = empty_embedding.expand(1, -1, -1)
            restored = run_sampling(unet, controlnet, vae, hint, embedding, noise_scheduler,
                                     args.num_inference_steps, device)

            clean_np = tensor_to_numpy_image(denormalize(clean[0]))
            for level, output in (("damaged", damaged), ("stage1", hint), ("stage1_stage2", restored)):
                output_np = tensor_to_numpy_image(denormalize(output[0]))
                metrics[level]["psnr"].append(peak_signal_noise_ratio(clean_np, output_np, data_range=255))
                metrics[level]["ssim"].append(structural_similarity(clean_np, output_np, channel_axis=2, data_range=255))
                metrics[level]["lpips"].append(lpips_fn(output, clean).item())

            print(f"  [{i+1}/{num_samples}] done")

            if i < 6:
                comparison_rows.append((denormalize(damaged[0]).cpu(), denormalize(hint[0]).cpu(),
                                         denormalize(restored[0]).cpu(), denormalize(clean[0]).cpu()))

    def summarize(name, values):
        arr = np.array(values)
        print(f"    {name}: mean={arr.mean():.3f}  std={arr.std():.3f}")

    for level, label in (("damaged", "Damaged (no restoration)"),
                          ("stage1", "Stage 1 only (regression)"),
                          ("stage1_stage2", "Stage 1 + Stage 2 (full pipeline)")):
        print(f"\n=== {label} vs. Clean ===")
        summarize("PSNR (dB, higher=better)", metrics[level]["psnr"])
        summarize("SSIM (0-1, higher=better)", metrics[level]["ssim"])
        summarize("LPIPS (0-1ish, lower=better)", metrics[level]["lpips"])

    stage2_psnr_delta = np.mean(metrics["stage1_stage2"]["psnr"]) - np.mean(metrics["stage1"]["psnr"])
    stage2_lpips_delta = np.mean(metrics["stage1"]["lpips"]) - np.mean(metrics["stage1_stage2"]["lpips"])
    print(f"\n=== What Stage 2 adds on top of Stage 1 alone ===")
    print(f"  PSNR change: {stage2_psnr_delta:+.3f} dB")
    print(f"  LPIPS change: {stage2_lpips_delta:+.3f} (positive = perceptually closer to clean)")
    print("  Note: it's common and expected for diffusion refinement to trade some PSNR for "
          "better LPIPS -- it adds plausible detail that may not exactly match ground truth pixels, "
          "but looks more realistic. Both directions are worth reporting, not just PSNR alone.")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("sample_index,damaged_psnr,damaged_ssim,damaged_lpips,"
                "stage1_psnr,stage1_ssim,stage1_lpips,"
                "stage1_stage2_psnr,stage1_stage2_ssim,stage1_stage2_lpips\n")
        for row_i, idx in enumerate(indices):
            f.write(f"{idx},"
                    f"{metrics['damaged']['psnr'][row_i]:.4f},{metrics['damaged']['ssim'][row_i]:.4f},{metrics['damaged']['lpips'][row_i]:.4f},"
                    f"{metrics['stage1']['psnr'][row_i]:.4f},{metrics['stage1']['ssim'][row_i]:.4f},{metrics['stage1']['lpips'][row_i]:.4f},"
                    f"{metrics['stage1_stage2']['psnr'][row_i]:.4f},{metrics['stage1_stage2']['ssim'][row_i]:.4f},{metrics['stage1_stage2']['lpips'][row_i]:.4f}\n")
    print(f"\nPer-sample metrics saved to {csv_path}")

    if comparison_rows:
        damaged_imgs = torch.stack([r[0] for r in comparison_rows])
        stage1_imgs = torch.stack([r[1] for r in comparison_rows])
        stage2_imgs = torch.stack([r[2] for r in comparison_rows])
        clean_imgs = torch.stack([r[3] for r in comparison_rows])
        grid = torch.cat([damaged_imgs, stage1_imgs, stage2_imgs, clean_imgs], dim=0)
        grid_path = os.path.join(args.out_dir, "comparison_grid.png")
        vutils.save_image(grid, grid_path, nrow=len(comparison_rows))
        print(f"Visual comparison grid saved to {grid_path} "
              f"(rows: damaged | stage1 only | stage1+stage2 | clean)")


if __name__ == "__main__":
    main()