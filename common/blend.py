"""
Shared damage-compositing logic.

This used to be duplicated in three places: composite_damage.py (the
standalone CLI tool) and inline inside both DegradedPairDataset and
VAE2PairDataset. That duplication is exactly why the screen/multiply blend
fix earlier in the project had to be applied in multiple files instead of
one -- consolidating it here means any future fix (or new blend mode) only
needs to happen once.

Two blend modes:
  - "screen" (default): damage LIGHTENS toward white. Physically realistic
    for scratches/abrasion, where the print's emulsion is scraped away and
    the lighter paper base shows through.
  - "multiply": damage DARKENS toward black. More appropriate for damage
    that deposits dark material (soot/smut, heavy dirt, mold staining).
"""

import numpy as np
import torch


def composite_numpy(clean_img: np.ndarray, mask_img: np.ndarray, blend: str = "screen") -> np.ndarray:
    """For use with OpenCV-style uint8 arrays (e.g. the standalone CLI tool,
    or quick visual checks). clean_img: (H, W, 3), mask_img: (H, W), both
    uint8. Mask convention: 255 = clean, toward 0 = damaged."""
    import cv2 as cv

    if clean_img.shape[:2] != mask_img.shape[:2]:
        mask_img = cv.resize(mask_img, (clean_img.shape[1], clean_img.shape[0]), interpolation=cv.INTER_LINEAR)

    mask_norm = mask_img.astype(np.float32) / 255.0
    if clean_img.ndim == 3 and mask_norm.ndim == 2:
        mask_norm = mask_norm[:, :, None]

    clean_f = clean_img.astype(np.float32)

    if blend == "screen":
        damaged = 255.0 - (255.0 - clean_f) * mask_norm
    elif blend == "multiply":
        damaged = clean_f * mask_norm
    else:
        raise ValueError(f"Unknown blend mode '{blend}', expected 'screen' or 'multiply'")

    return np.clip(damaged, 0, 255).astype(np.uint8)


def composite_tensor(clean_tensor: torch.Tensor, mask_tensor: torch.Tensor, blend: str = "screen") -> torch.Tensor:
    """For use inside PyTorch Dataset classes, operating on [0, 1]-range
    tensors (before the [-1, 1] normalization step). clean_tensor: (C, H, W),
    mask_tensor: (1, H, W), same convention (255->1.0 = clean, 0 = damaged)."""
    if blend == "screen":
        return 1.0 - (1.0 - clean_tensor) * mask_tensor
    elif blend == "multiply":
        return clean_tensor * mask_tensor
    else:
        raise ValueError(f"Unknown blend mode '{blend}', expected 'screen' or 'multiply'")
