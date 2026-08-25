"""
Evaluate a trained Stage 1 restoration checkpoint: computes PSNR and SSIM
between (a) damaged vs. clean and (b) restored vs. clean, so you can see
the actual improvement the model provides, not just eyeball sample grids.

Reconstructs the model architecture from the checkpoint's saved args
(base_channels, n_downsample) automatically -- no need to re-specify them
and risk a mismatch.

Note: for a real baseline sanity check, evaluating on images the model saw
during training is fine (you're just confirming the pipeline/model works).
For your FINAL thesis comparison across all three restoration methods, use
a held-out clean-image set that was never used in ANY training stage --
see the pre-training checklist from earlier in this project.

Usage:
    python evaluate.py --checkpoint ./runs/baseline_check/checkpoints/stage1_epoch0008.pt \
        --clean-dir ./voc_subset_100/images --masks-dir ./generated_masks \
        --num-samples 20 --out-dir ./eval_results
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.utils as vutils

from method2_diffbir.models.restoration_net import RestorationUNet
from common.degraded_pair_dataset import DegradedPairDataset, denormalize


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor in [0, 1] -> (H, W, C) numpy array in [0, 255] uint8."""
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return arr


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 checkpoint with PSNR/SSIM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clean-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20,
                         help="how many random (clean, mask) pairs to evaluate on")
    parser.add_argument("--image-size", type=int, default=None,
                         help="defaults to the image size the checkpoint was trained with")
    parser.add_argument("--out-dir", type=str, default="./eval_results")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})

    image_size = args.image_size or train_args.get("image_size", 256)
    base_channels = train_args.get("base_channels", 64)
    n_downsample = train_args.get("n_downsample", 4)
    blend_mode = train_args.get("blend_mode", "screen")

    print(f"  Reconstructed from checkpoint: image_size={image_size}, base_channels={base_channels}, "
          f"n_downsample={n_downsample}, blend_mode={blend_mode}")
    print(f"  Checkpoint was saved at epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')}")

    model = RestorationUNet(in_channels=3, out_channels=3, base_channels=base_channels,
                             n_downsample=n_downsample).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.manual_seed(args.seed)
    dataset = DegradedPairDataset(args.clean_dir, args.masks_dir, image_size=image_size,
                                   augment=False, blend_mode=blend_mode)
    num_samples = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    print(f"Evaluating on {num_samples} samples (seed={args.seed})")

    damaged_psnrs, damaged_ssims = [], []
    restored_psnrs, restored_ssims = [], []

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

            damaged_psnrs.append(d_psnr)
            damaged_ssims.append(d_ssim)
            restored_psnrs.append(r_psnr)
            restored_ssims.append(r_ssim)

            if i < 6:  # keep a handful for a visual comparison grid
                comparison_rows.append((denormalize(damaged[0]).cpu(),
                                         denormalize(restored[0]).cpu(),
                                         denormalize(clean[0]).cpu()))

    def summarize(name, values):
        arr = np.array(values)
        print(f"  {name}: mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

    print("\n=== Damaged vs. Clean (baseline, no restoration) ===")
    summarize("PSNR (dB, higher=better)", damaged_psnrs)
    summarize("SSIM (0-1, higher=better)", damaged_ssims)

    print("\n=== Restored vs. Clean (model output) ===")
    summarize("PSNR (dB, higher=better)", restored_psnrs)
    summarize("SSIM (0-1, higher=better)", restored_ssims)

    psnr_gain = np.mean(restored_psnrs) - np.mean(damaged_psnrs)
    ssim_gain = np.mean(restored_ssims) - np.mean(damaged_ssims)
    print(f"\n=== Improvement from restoration ===")
    print(f"  PSNR gain: {psnr_gain:+.3f} dB")
    print(f"  SSIM gain: {ssim_gain:+.3f}")
    if psnr_gain <= 0:
        print("  Note: non-positive PSNR gain means the model isn't yet improving over doing nothing -- "
              "expected for a short baseline check, but worth watching on longer runs.")

    # Save per-sample results to CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("sample_index,damaged_psnr,damaged_ssim,restored_psnr,restored_ssim\n")
        for idx, dp, ds, rp, rs in zip(indices, damaged_psnrs, damaged_ssims, restored_psnrs, restored_ssims):
            f.write(f"{idx},{dp:.4f},{ds:.4f},{rp:.4f},{rs:.4f}\n")
    print(f"\nPer-sample metrics saved to {csv_path}")

    # Save a visual comparison grid
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
