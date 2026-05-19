from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class LabelEmbeddings:
    """Compute or load [CLS] embeddings for label names using a HF model."""

    @classmethod
    @torch.no_grad()
    def load(
        cls,
        out_path: Path | str,
        label_order: Sequence[str],
        model_name: str = "bert-base-uncased",
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        out_path = Path(out_path)
        device = torch.device(device)

        if out_path.exists():
            arr = np.load(out_path, mmap_mode="r")
            return torch.from_numpy(arr).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        tokens = tokenizer(list(label_order), padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(**tokens)
        embs = outputs.last_hidden_state[:, 0, :].cpu()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embs.numpy())
        return embs.to(device)
