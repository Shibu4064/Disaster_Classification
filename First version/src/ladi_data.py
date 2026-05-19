from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .ladi_config import IMG_EXTS, IMAGENET_MEAN, IMAGENET_STD


class LADIMultiLabelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str | Path,
        label_cols: Sequence[str],
        transform=None,
        img_exts: Optional[Sequence[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.label_cols = list(label_cols)
        self.transform = transform
        self.img_exts = list(img_exts) if img_exts else IMG_EXTS

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_id: str) -> Path:
        # ID can be a stem, a filename, or a relative path.
        p = Path(str(image_id))
        if p.suffix and (self.img_dir / p).exists():
            return self.img_dir / p
        if p.suffix and p.exists():
            return p
        stem = str(image_id)
        for ext in self.img_exts:
            cand = self.img_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Image for ID={image_id!r} not found in {self.img_dir}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row["ID"]))
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = torch.tensor(row[self.label_cols].to_numpy(dtype=float), dtype=torch.float32)
        return img, y


def detect_label_cols(df: pd.DataFrame, exclude: Optional[set[str]] = None) -> list[str]:
    exclude = set(exclude or {"ID", "image", "path", "split", "filename"})
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            vals = set(s.dropna().unique().tolist())
            if vals.issubset({0, 1, 0.0, 1.0, True, False}):
                cols.append(c)
    return cols


def build_transforms(img_size: int = 320):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def make_loaders(
    cache_dir: str | Path,
    label_cols: Optional[list[str]] = None,
    img_size: int = 320,
    batch_size: int = 16,
    num_workers: int = 2,
):
    cache_dir = Path(cache_dir)
    train_csv = cache_dir / "train_data.csv"
    val_csv = cache_dir / "val_data.csv"
    test_csv = cache_dir / "test_data.csv"
    img_dir = cache_dir / "images"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Expected train_data.csv and val_data.csv under {cache_dir}. Run notebook 00 first.")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv) if test_csv.exists() else val_df.copy()
    if label_cols is None:
        label_cols = detect_label_cols(train_df)
    train_tf, eval_tf = build_transforms(img_size=img_size)
    train_ds = LADIMultiLabelDataset(train_df, img_dir, label_cols, train_tf)
    val_ds = LADIMultiLabelDataset(val_df, img_dir, label_cols, eval_tf)
    test_ds = LADIMultiLabelDataset(test_df, img_dir, label_cols, eval_tf)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, label_cols, train_df, val_df, test_df
