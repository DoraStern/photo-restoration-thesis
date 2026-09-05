"""
Dataset for Stage 1 training: pairs of (damaged, clean) images, where the
damage comes from YOUR existing FilmDamageSimulator mask pool (generated
via generate_synthetic_only.py) rather than DiffBIR's generic synthetic
blur/noise/JPEG degradation pipeline.

Masks are composited onto clean images on the fly (mask value 255 = clean,
toward 0 = damaged), so a given clean image can pair with a different
random mask each epoch -- more effective training variety than
pre-generating a fixed set of damaged/clean pairs once.

Blend mode matches composite_damage.py's two options:
  - "screen" (default): damage LIGHTENS toward white. Physically realistic
    for scratches/abrasion, where the print's emulsion is scraped away and
    the lighter paper base shows through.
  - "multiply": damage DARKENS toward black. More appropriate for damage
    that deposits dark material (soot/smut, heavy dirt, mold staining).
Since a mixed mask (e.g. scratches + smut generated together) doesn't track
which pixel came from which damage type, this is a dataset-wide setting --
if you want type-appropriate blending for a mixed mask pool, generate and
composite scratches and smut as separate mask batches with different
--blend-mode settings instead of one mixed pool.
"""

import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from common.blend import composite_tensor


class DegradedPairDataset(Dataset):
    """
    Expects:
        clean_dir/    -- folder of clean photos (e.g. a VOC2012 subset)
        masks_dir     -- one folder path, OR a list of folder paths, each
                          containing grayscale masks from
                          generate_synthetic_only.py (mask_*.png; NOT the
                          binarised_mask_*.png variants -- those are
                          thresholded and lose the soft edges that make
                          compositing look natural).

                          Passing multiple folders lets you keep damage
                          types in physically separate folders (e.g. one
                          for scratches, one for smut) and combine
                          whichever subset you want per run, rather than
                          always drawing from one mixed pool.
    """

    def __init__(self, clean_dir: str, masks_dir, image_size: int = 256, augment: bool = True,
                 blend_mode: str = "screen"):
        if blend_mode not in ("screen", "multiply"):
            raise ValueError(f"blend_mode must be 'screen' or 'multiply', got '{blend_mode}'")
        self.clean_dir = clean_dir
        self.masks_dirs = [masks_dir] if isinstance(masks_dir, str) else list(masks_dir)
        self.image_size = image_size
        self.augment = augment
        self.blend_mode = blend_mode

        valid_ext = (".jpg", ".jpeg", ".png")
        self.clean_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(valid_ext)]

        self.mask_files = []
        for d in self.masks_dirs:
            for f in os.listdir(d):
                if f.lower().endswith(".png") and not f.startswith("binarised_mask"):
                    self.mask_files.append(os.path.join(d, f))

        if len(self.clean_files) == 0:
            raise ValueError(f"No clean images found in {clean_dir}")
        if len(self.mask_files) == 0:
            raise ValueError(f"No usable masks found in {self.masks_dirs} "
                              f"(looking for mask_*.png, excluding binarised_mask_*.png)")

        load_size = int(image_size * 1.12)
        self.clean_resize = T.Resize(load_size)
        self.image_size_final = image_size

    def __len__(self):
        return len(self.clean_files)

    def _load_clean(self, idx):
        path = os.path.join(self.clean_dir, self.clean_files[idx])
        img = Image.open(path).convert("RGB")
        return self.clean_resize(img)

    def _load_random_mask(self):
        path = random.choice(self.mask_files)
        mask = Image.open(path).convert("L")  # single-channel grayscale
        return mask

    def _synchronized_crop_and_flip(self, clean_img, mask_img):
        """Applies the SAME random crop and flip to both the clean image
        and the mask, so the damage stays spatially aligned with the
        content it's composited onto."""
        # Resize mask to match the (already resized) clean image
        mask_img = mask_img.resize(clean_img.size, Image.BILINEAR)

        if self.augment:
            i, j, h, w = T.RandomCrop.get_params(clean_img, output_size=(self.image_size_final, self.image_size_final))
            clean_img = TF.crop(clean_img, i, j, h, w)
            mask_img = TF.crop(mask_img, i, j, h, w)

            if random.random() < 0.5:
                clean_img = TF.hflip(clean_img)
                mask_img = TF.hflip(mask_img)
            # Masks (unlike photo content) are safe to rotate freely --
            # scratches/smut don't have a "correct" orientation the way a
            # photo of a person or building does.
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                mask_img = TF.rotate(mask_img, angle)
        else:
            clean_img = TF.center_crop(clean_img, (self.image_size_final, self.image_size_final))
            mask_img = TF.center_crop(mask_img, (self.image_size_final, self.image_size_final))

        return clean_img, mask_img

    def __getitem__(self, idx):
        try:
            clean_img = self._load_clean(idx)
            mask_img = self._load_random_mask()
        except Exception:
            return self.__getitem__(random.randrange(len(self)))

        clean_img, mask_img = self._synchronized_crop_and_flip(clean_img, mask_img)

        clean_tensor = TF.to_tensor(clean_img)          # [0, 1], shape (3, H, W)
        mask_tensor = TF.to_tensor(mask_img)             # [0, 1], shape (1, H, W)

        # Composite damage onto the clean image using the shared blend logic
        # (common/blend.py) -- single source of truth, not duplicated inline.
        damaged_tensor = composite_tensor(clean_tensor, mask_tensor, blend=self.blend_mode)

        # Normalize both to [-1, 1] to match the restoration network's Tanh output
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        clean_tensor = normalize(clean_tensor)
        damaged_tensor = normalize(damaged_tensor)

        return {"damaged": damaged_tensor, "clean": clean_tensor}


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the Normalize(mean=0.5, std=0.5) above, for saving/viewing."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)
