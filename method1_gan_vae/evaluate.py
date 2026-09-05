"""
Evaluate a trained Method 1 pipeline (VAE1 + VAE2 + translation network).

Unlike Method 2/3's evaluate.py, this method has no ground truth for its
actual real-world use case (restoring real old photos -- there's no known
"correct" clean version of a genuinely old damaged photograph). So this
script supports two modes:

  --mode synthetic  Quantitative evaluation (PSNR/SSIM/LPIPS) on your
                     held-out synthetic test set, where ground truth IS
                     known. Pathway: degraded synthetic photo -> VAE2
                     encoder -> translation network (with its real mask)
                     -> VAE2 decoder -> compare against the known clean
                     target. This mirrors the network's supervised
                     training pathway and gives you numbers directly
                     comparable to Method 2 and Method 3's evaluate.py
                     output.

  --mode real        Qualitative-only evaluation on real old photos. No
                     metrics are computed here either (no ground truth
                     exists for LPIPS any more than for PSNR/SSIM) --
                     produces a visual comparison grid for manual
                     inspection.
                     Pathway: real photo -> VAE1 encoder -> translation
                     network (zero mask, matching training) -> VAE2
                     decoder.

Usage:
    python evaluate.py --mode synthetic \
        --vae1-checkpoint ./runs/vae_domain_a/checkpoints/<latest>.pt \
        --vae2-checkpoint ./runs/vae_domain_b/checkpoints/<latest>.pt \
        --translation-checkpoint ./runs/translation_net/checkpoints/<latest>.pt \
        --clean-dir ./data/voc2012/... --masks-dir ./data/generated_masks \
        --num-samples 100 --out-dir ./eval_results_synthetic

    python evaluate.py --mode real \
        --vae1-checkpoint ./runs/vae_domain_a/checkpoints/<latest>.pt \
        --vae2-checkpoint ./runs/vae_domain_b/checkpoints/<latest>.pt \
        --translation-checkpoint ./runs/translation_net/checkpoints/<latest>.pt \
        --real-photo-dir ./data/real_old_photos \
        --num-samples 20 --out-dir ./eval_results_real
"""

import argparse
import os

import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.utils as vutils

from models.vae import DomainVAE
from models.translation_net import LatentTranslationNet
from data.real_photo_dataset import RealOldPhotoDataset, denormalize as denorm_a
from data.vae2_pair_dataset import VAE2PairDataset, denormalize as denorm_b


def load_frozen_vae(checkpoint_path, device):
    """Reconstructs a VAE1 or VAE2 architecture from its checkpoint's own
    saved args, loads weights, freezes it. Same pattern used throughout
    this project's other evaluate.py / train_translation_net.py."""
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


def load_frozen_translator(checkpoint_path, latent_channels, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt.get("args", {})
    model = LatentTranslationNet(
        latent_channels=latent_channels,
        n_residual_blocks=train_args.get("n_residual_blocks", 6),
    ).to(device)
    model.load_state_dict(ckpt["translator_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return arr


def run_synthetic_eval(vae1, vae2, translator, lpips_fn, args, device):
    """Quantitative evaluation on synthetic pairs with known ground truth."""
    torch.manual_seed(args.seed)
    dataset = VAE2PairDataset(args.clean_dir, args.masks_dir, image_size=args.image_size,
                               augment=False, blend_mode=args.blend_mode)
    num_samples = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    print(f"Evaluating on {num_samples} synthetic samples (seed={args.seed})")

    damaged_psnrs, damaged_ssims, damaged_lpips = [], [], []
    restored_psnrs, restored_ssims, restored_lpips = [], [], []
    comparison_rows = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            degraded = sample["degraded"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)

            mu_degraded, _ = vae2.encoder(degraded)
            mu_clean, _ = vae2.encoder(clean)
            translated = translator(mu_degraded, mask)
            restored = vae2.decoder(translated)

            degraded_np = tensor_to_numpy_image(denorm_b(degraded[0]))
            clean_np = tensor_to_numpy_image(denorm_b(clean[0]))
            restored_np = tensor_to_numpy_image(denorm_b(restored[0]))

            d_psnr = peak_signal_noise_ratio(clean_np, degraded_np, data_range=255)
            d_ssim = structural_similarity(clean_np, degraded_np, channel_axis=2, data_range=255)
            r_psnr = peak_signal_noise_ratio(clean_np, restored_np, data_range=255)
            r_ssim = structural_similarity(clean_np, restored_np, channel_axis=2, data_range=255)

            # LPIPS expects [-1, 1]-range tensors -- degraded/clean/restored
            # here are already in that range.
            d_lpips = lpips_fn(degraded, clean).item()
            r_lpips = lpips_fn(restored, clean).item()

            damaged_psnrs.append(d_psnr); damaged_ssims.append(d_ssim); damaged_lpips.append(d_lpips)
            restored_psnrs.append(r_psnr); restored_ssims.append(r_ssim); restored_lpips.append(r_lpips)

            if i < 6:
                comparison_rows.append((denorm_b(degraded[0]).cpu(), denorm_b(restored[0]).cpu(), denorm_b(clean[0]).cpu()))

    def summarize(name, values):
        arr = np.array(values)
        print(f"  {name}: mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

    print("\n=== Damaged vs. Clean (baseline, no restoration) ===")
    summarize("PSNR (dB, higher=better)", damaged_psnrs)
    summarize("SSIM (0-1, higher=better)", damaged_ssims)
    summarize("LPIPS (0-1ish, lower=better)", damaged_lpips)
    print("\n=== Restored vs. Clean (Method 1 output) ===")
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

    os.makedirs(args.out_dir, exist_ok=True)
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


def run_real_eval(vae1, vae2, translator, args, device):
    """Qualitative-only evaluation on real old photos. No ground truth
    exists, so no metrics are computed here -- this is a visual check
    only, matching what actually happens at real-world inference time."""
    dataset = RealOldPhotoDataset(args.real_photo_dir, image_size=args.image_size, augment=False)
    num_samples = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    print(f"Running qualitative evaluation on {num_samples} real old photos")
    print("Note: no PSNR/SSIM is computed here -- there is no ground-truth clean version "
          "of a genuinely old damaged photograph to compare against. This is a visual check only.")

    comparison_rows = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            real = sample["image"].unsqueeze(0).to(device)

            mu_real, _ = vae1.encoder(real)
            zero_mask = torch.zeros((1, 1, real.shape[-2], real.shape[-1]), device=device)
            translated = translator(mu_real, zero_mask)
            restored = vae2.decoder(translated)

            if i < 8:
                comparison_rows.append((denorm_a(real[0]).cpu(), denorm_a(restored[0]).cpu()))

    os.makedirs(args.out_dir, exist_ok=True)
    if comparison_rows:
        real_imgs = torch.stack([r[0] for r in comparison_rows])
        restored_imgs = torch.stack([r[1] for r in comparison_rows])
        grid = torch.cat([real_imgs, restored_imgs], dim=0)
        grid_path = os.path.join(args.out_dir, "comparison_grid_real.png")
        vutils.save_image(grid, grid_path, nrow=len(comparison_rows))
        print(f"Visual comparison grid saved to {grid_path} (top row: real damaged photo, bottom row: restored)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Method 1 pipeline (VAE1 + VAE2 + translation net).")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], required=True)
    parser.add_argument("--vae1-checkpoint", type=str, required=True)
    parser.add_argument("--vae2-checkpoint", type=str, required=True)
    parser.add_argument("--translation-checkpoint", type=str, required=True)
    parser.add_argument("--clean-dir", type=str, help="required for --mode synthetic")
    parser.add_argument("--masks-dir", type=str, nargs="+", help="required for --mode synthetic")
    parser.add_argument("--blend-mode", type=str, choices=["screen", "multiply"], default="screen")
    parser.add_argument("--real-photo-dir", type=str, help="required for --mode real")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="./eval_results")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lpips-net", type=str, choices=["alex", "vgg", "squeeze"], default="alex",
                         help="backbone network for LPIPS perceptual distance (only used in --mode synthetic, "
                              "since --mode real has no ground truth to compare against)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "synthetic" and (not args.clean_dir or not args.masks_dir):
        parser.error("--mode synthetic requires --clean-dir and --masks-dir")
    if args.mode == "real" and not args.real_photo_dir:
        parser.error("--mode real requires --real-photo-dir")

    device = torch.device(args.device)

    print(f"Loading frozen VAE1 from {args.vae1_checkpoint}")
    vae1, vae1_args = load_frozen_vae(args.vae1_checkpoint, device)
    print(f"Loading frozen VAE2 from {args.vae2_checkpoint}")
    vae2, vae2_args = load_frozen_vae(args.vae2_checkpoint, device)

    latent_channels = vae1_args.get("latent_channels", 64)
    assert latent_channels == vae2_args.get("latent_channels", 64), \
        "VAE1 and VAE2 must share the same --latent-channels."

    print(f"Loading frozen translation network from {args.translation_checkpoint}")
    translator = load_frozen_translator(args.translation_checkpoint, latent_channels, device)

    if args.mode == "synthetic":
        print(f"Loading LPIPS ({args.lpips_net}) perceptual similarity model...")
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_fn.eval()
        run_synthetic_eval(vae1, vae2, translator, lpips_fn, args, device)
    else:
        run_real_eval(vae1, vae2, translator, args, device)


if __name__ == "__main__":
    main()