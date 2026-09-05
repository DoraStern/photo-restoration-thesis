"""
Evaluate a trained Method 3 checkpoint (transformer regression restoration).

Computes PSNR, SSIM, and LPIPS for damaged-vs-clean (the do-nothing
baseline) and restored-vs-clean (the model), on a held-out set of
(clean, mask) pairs -- same evaluation pattern as Method 2's evaluate.py,
for direct side-by-side comparability in your results chapter.

LPIPS uses a pretrained network's internal features to judge perceptual
similarity, rather than comparing raw pixel values -- lower is better,
unlike PSNR/SSIM. Included alongside PSNR/SSIM specifically because this
method (pure pixel-wise regression, no adversarial or generative
component) tends to score well on PSNR/SSIM while potentially looking
blurrier than Methods 1/2 to LPIPS, which better reflects that
perceptual difference.

Reconstructs the model architecture automatically from the checkpoint's
own saved args (embed_dim, depths, num_heads, window_size, blend_mode) --
no need to re-specify them by hand and risk a mismatch.

Usage:
    python evaluate.py \
        --checkpoint ./runs/transformer_regression/checkpoints/transformer_regression_epoch0050.pt \
        --clean-dir ./data/voc2012/... --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
        --num-samples 100 --out-dir ./eval_results
"""

import argparse
import os

import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.utils as vutils

from models.transformer_restoration import TransformerRestorationNet
from data.degraded_pair_dataset import DegradedPairDataset, denormalize


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return arr


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Method 3 (transformer regression) checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clean-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True, nargs="+")
    parser.add_argument("--num-samples", type=int, default=100,
                         help="how many random (clean, mask) pairs to evaluate on")
    parser.add_argument("--image-size", type=int, default=None,
                         help="defaults to the image size the checkpoint was trained with")
    parser.add_argument("--out-dir", type=str, default="./eval_results")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lpips-net", type=str, choices=["alex", "vgg", "squeeze"], default="alex",
                         help="backbone network for LPIPS perceptual distance; 'alex' is the standard "
                              "default (fastest, most commonly reported in papers)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})

    image_size = args.image_size or train_args.get("image_size", 256)
    embed_dim = train_args.get("embed_dim", 60)
    depths_str = train_args.get("depths", "4,4,4,4")
    depths = tuple(int(d) for d in depths_str.split(",")) if isinstance(depths_str, str) else tuple(depths_str)
    num_heads = train_args.get("num_heads", 6)
    window_size = train_args.get("window_size", 8)
    blend_mode = train_args.get("blend_mode", "screen")

    print(f"  Reconstructed from checkpoint: image_size={image_size}, embed_dim={embed_dim}, "
          f"depths={depths}, num_heads={num_heads}, window_size={window_size}, blend_mode={blend_mode}")
    print(f"  Checkpoint was saved at epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')}")

    model = TransformerRestorationNet(
        in_channels=3, out_channels=3, embed_dim=embed_dim,
        depths=depths, num_heads=num_heads, window_size=window_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loading LPIPS ({args.lpips_net}) perceptual similarity model...")
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_fn.eval()

    torch.manual_seed(args.seed)
    dataset = DegradedPairDataset(args.clean_dir, args.masks_dir, image_size=image_size,
                                   augment=False, blend_mode=blend_mode)
    num_samples = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    print(f"Evaluating on {num_samples} samples (seed={args.seed})")

    damaged_psnrs, damaged_ssims, damaged_lpips = [], [], []
    restored_psnrs, restored_ssims, restored_lpips = [], [], []
    comparison_rows = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            damaged = sample["damaged"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)

            restored = model(damaged)

            damaged_np = tensor_to_numpy_image(denormalize(damaged[0]))
            clean_np = tensor_to_numpy_image(denormalize(clean[0]))
            restored_np = tensor_to_numpy_image(denormalize(restored[0]))

            d_psnr = peak_signal_noise_ratio(clean_np, damaged_np, data_range=255)
            d_ssim = structural_similarity(clean_np, damaged_np, channel_axis=2, data_range=255)
            r_psnr = peak_signal_noise_ratio(clean_np, restored_np, data_range=255)
            r_ssim = structural_similarity(clean_np, restored_np, channel_axis=2, data_range=255)

            # LPIPS expects [-1, 1]-range tensors -- damaged/clean/restored are
            # already in that range, so feed the raw tensors directly.
            d_lpips = lpips_fn(damaged, clean).item()
            r_lpips = lpips_fn(restored, clean).item()

            damaged_psnrs.append(d_psnr); damaged_ssims.append(d_ssim); damaged_lpips.append(d_lpips)
            restored_psnrs.append(r_psnr); restored_ssims.append(r_ssim); restored_lpips.append(r_lpips)

            if i < 6:
                comparison_rows.append((denormalize(damaged[0]).cpu(), denormalize(restored[0]).cpu(), denormalize(clean[0]).cpu()))

    def summarize(name, values):
        arr = np.array(values)
        print(f"  {name}: mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

    print("\n=== Damaged vs. Clean (baseline, no restoration) ===")
    summarize("PSNR (dB, higher=better)", damaged_psnrs)
    summarize("SSIM (0-1, higher=better)", damaged_ssims)
    summarize("LPIPS (0-1ish, lower=better)", damaged_lpips)

    print("\n=== Restored vs. Clean (Method 3 output) ===")
    summarize("PSNR (dB, higher=better)", restored_psnrs)
    summarize("SSIM (0-1, higher=better)", restored_ssims)
    summarize("LPIPS (0-1ish, lower=better)", restored_lpips)

    psnr_gain = np.mean(restored_psnrs) - np.mean(damaged_psnrs)
    ssim_gain = np.mean(restored_ssims) - np.mean(damaged_ssims)
    lpips_gain = np.mean(damaged_lpips) - np.mean(restored_lpips)
    print(f"\n=== Improvement from restoration ===")
    print(f"  PSNR gain: {psnr_gain:+.3f} dB")
    print(f"  SSIM gain: {ssim_gain:+.3f}")
    print(f"  LPIPS gain: {lpips_gain:+.3f} (positive = perceptually closer to clean after restoration)")
    if psnr_gain <= 0:
        print("  Note: non-positive PSNR gain means the model isn't yet improving over doing nothing -- "
              "expected for a short training run, but worth watching on longer runs.")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("sample_index,damaged_psnr,damaged_ssim,damaged_lpips,restored_psnr,restored_ssim,restored_lpips\n")
        for idx, dp, ds, dl, rp, rs, rl in zip(indices, damaged_psnrs, damaged_ssims, damaged_lpips,
                                                 restored_psnrs, restored_ssims, restored_lpips):
            f.write(f"{idx},{dp:.4f},{ds:.4f},{dl:.4f},{rp:.4f},{rs:.4f},{rl:.4f}\n")
    print(f"\nPer-sample metrics saved to {csv_path}")

    if comparison_rows:
        damaged_imgs = torch.stack([r[0] for r in comparison_rows])
        restored_imgs = torch.stack([r[1] for r in comparison_rows])
        clean_imgs = torch.stack([r[2] for r in comparison_rows])
        grid = torch.cat([damaged_imgs, restored_imgs, clean_imgs], dim=0)
        grid_path = os.path.join(args.out_dir, "comparison_grid.png")
        vutils.save_image(grid, grid_path, nrow=len(comparison_rows))
        print(f"Visual comparison grid saved to {grid_path}")


if __name__ == "__main__":
    main()