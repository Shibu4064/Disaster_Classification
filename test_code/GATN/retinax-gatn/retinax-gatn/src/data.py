from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CLAHETransform:
    """Apply CLAHE on the L channel of LAB color space."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class MultiLabelDataset(Dataset):
    """CSV-driven dataset for multi-label image classification.

    Expects a CSV with an ``ID`` column (image stem, no extension) and one
    binary column per label. Images are resolved against ``img_dir`` by
    trying a list of common extensions.
    """

    def __init__(
        self,
        csv_path: Path | str,
        img_dir: Path | str,
        label_cols: Sequence[str],
        transform: Optional[transforms.Compose] = None,
        img_extensions: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.label_cols = list(label_cols)
        self.transform = transform
        self.img_exts = list(img_extensions) if img_extensions else [".png", ".jpg", ".jpeg", ".bmp"]

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, img_id: str) -> Path:
        for ext in self.img_exts:
            candidate = self.img_dir / f"{img_id}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image file found for ID '{img_id}' in {self.img_dir}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._find_image(str(row["ID"]))
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[self.label_cols].to_numpy(dtype=float), dtype=torch.float32)
        return img, labels


def build_transforms(
    resize: int,
    crop: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose([
        CLAHETransform(),
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val = transforms.Compose([
        CLAHETransform(),
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train, val
