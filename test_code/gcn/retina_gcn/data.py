from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]


class RetinaMultiLabelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        label_cols: List[str],
        transform=None,
        img_exts: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.label_cols = label_cols
        self.transform = transform
        self.img_exts = img_exts or list(DEFAULT_IMG_EXTS)

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, img_id: str) -> Path:
        for ext in self.img_exts:
            candidate = self.img_dir / f"{img_id}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No image file found for ID '{img_id}' in {self.img_dir}"
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = str(row["ID"])
        img_path = self._find_image(img_id)

        with Image.open(img_path) as im:
            img = im.convert("RGB")

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(
            row[self.label_cols].to_numpy(dtype=float), dtype=torch.float32
        )
        return img, labels


def detect_label_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"ID"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna().unique()
            if len(vals) <= 3 and set(vals).issubset({0, 1, 0.0, 1.0}):
                cols.append(c)
    return cols
