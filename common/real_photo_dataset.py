"""
Dataset loader for VAE1 (domain A) training: real old photos, as collected
by loc_scraper.py / dpla_scraper.py. Unsupervised -- no labels needed, just
a folder of images.
"""

import os
import csv
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class RealOldPhotoDataset(Dataset):
    """
    Expects the folder layout produced by loc_scraper.py / dpla_scraper.py:

        <root>/images/*.jpg
        <root>/manifest.csv   (optional, only used to list valid filenames)

    If manifest.csv is present, filenames are read from it (keeps you in
    sync with whatever passed your download-time validation). Otherwise
    falls back to globbing every image file in <root>/images/.
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = True):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.image_size = image_size
        self.augment = augment

        self.filenames = self._load_filenames()
        if len(self.filenames) == 0:
            raise ValueError(f"No images found under {self.images_dir}")

        # Resize the short side up a bit past image_size so RandomCrop has
        # room to move -- this is a standard cheap augmentation for
        # unsupervised reconstruction training.
        load_size = int(image_size * 1.12)

        transform_list = [
            T.Resize(load_size),
        ]
        if augment:
            transform_list += [
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [T.CenterCrop(image_size)]

        transform_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
        ]
        self.transform = T.Compose(transform_list)

    def _load_filenames(self):
        manifest_path = os.path.join(self.root, "manifest.csv")
        if os.path.exists(manifest_path):
            filenames = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get("local_filename")
                    if fn:
                        filenames.append(fn)
            if filenames:
                return filenames

        # fallback: glob the images directory directly
        if not os.path.isdir(self.images_dir):
            return []
        valid_ext = (".jpg", ".jpeg", ".png")
        return [f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_ext)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.images_dir, filename)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # If something slipped past validation and is unreadable,
            # fall back to a random other item rather than crashing an
            # entire training run over one bad file.
            return self.__getitem__(random.randrange(len(self)))

        img_tensor = self.transform(img)
        return {"image": img_tensor, "filename": filename}


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the Normalize(mean=0.5, std=0.5) above, for saving/viewing
    reconstructed images. Maps [-1, 1] back to [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)
