from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch


def normalize_adj_np(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    row_sum = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(row_sum + eps)
    return (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]


def topk_sparsify(A: np.ndarray, k: Optional[int] = 6, keep_diag: bool = True) -> np.ndarray:
    if k is None or k <= 0 or k >= A.shape[0]:
        return A.copy()
    C = A.shape[0]
    B = np.zeros_like(A, dtype=np.float64)
    diag = np.diag(A).copy()
    for i in range(C):
        row = A[i].copy()
        if keep_diag:
            row[i] = -1e12
        idx = np.argsort(-row)[:k]
        B[i, idx] = A[i, idx]
    B = np.maximum(B, B.T)
    if keep_diag:
        np.fill_diagonal(B, diag)
    return B


def build_adj_from_labels(
    Y: np.ndarray,
    mode: str = "pmi",
    topk: Optional[int] = 6,
    add_self_loops: bool = True,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """Build a static label graph from train-only multi-label annotations."""
    Y = np.asarray(Y, dtype=np.float64)
    assert Y.ndim == 2, "Y must be an N x C binary label matrix"
    N, C = Y.shape
    cooc = Y.T @ Y
    freq = np.diag(cooc)

    if mode.lower() == "condprob":
        # P(i|j) = cooc(i,j) / count(j); symmetrised because GCN uses an undirected graph.
        P = cooc / (freq[None, :] + eps)
        A = 0.5 * (P + P.T)
        mx = A.max()
        if mx > 0:
            A = A / (mx + eps)
    elif mode.lower() == "pmi":
        Pij = cooc / max(N, 1)
        Pi = freq / max(N, 1)
        pmi = np.log((Pij + eps) / (Pi[:, None] * Pi[None, :] + eps))
        A = np.maximum(pmi, 0.0)
        mx = A.max()
        if mx > 0:
            A = A / (mx + eps)
    elif mode.lower() == "cooc":
        A = cooc.copy()
        mx = A.max()
        if mx > 0:
            A = A / (mx + eps)
    elif mode.lower() == "identity":
        A = np.eye(C, dtype=np.float64)
    else:
        raise ValueError("mode must be one of: pmi, condprob, cooc, identity")

    if add_self_loops and mode.lower() != "identity":
        A = A + np.eye(C, dtype=np.float64)
    if topk is not None and topk > 0 and mode.lower() != "identity":
        A = topk_sparsify(A, k=topk, keep_diag=True)
    if normalize:
        A = normalize_adj_np(A, eps=eps)
    return torch.tensor(A, dtype=torch.float32)


def permute_adj(A: torch.Tensor, seed: int = 123) -> torch.Tensor:
    C = A.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(C)
    return A[perm][:, perm]


def graph_stats(A: torch.Tensor, labels: list[str]) -> pd.DataFrame:
    A_np = A.detach().cpu().numpy()
    rows = []
    for i, lab in enumerate(labels):
        row = A_np[i].copy()
        top = np.argsort(-row)[: min(6, len(labels))]
        rows.append(
            {
                "label": lab,
                "degree_weighted": float(row.sum()),
                "max_neighbor": labels[int(top[1])] if len(top) > 1 and int(top[0]) == i else labels[int(top[0])],
                "max_weight": float(row[top[1]] if len(top) > 1 and int(top[0]) == i else row[top[0]]),
                "density_nonzero": float((row > 1e-9).mean()),
            }
        )
    return pd.DataFrame(rows)


def save_graphs_from_csv(
    train_csv: str | Path,
    label_cols: list[str],
    out_dir: str | Path,
    topk: int = 6,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(train_csv)
    Y = df[label_cols].values.astype(np.float32)
    paths = {}
    for mode in ["pmi", "condprob", "cooc", "identity"]:
        A = build_adj_from_labels(Y, mode=mode, topk=topk)
        p = out_dir / f"A_{mode}.pt"
        torch.save(A, p)
        stats = graph_stats(A, label_cols)
        stats.to_csv(out_dir / f"A_{mode}_stats.csv", index=False)
        paths[mode] = p
    return paths
