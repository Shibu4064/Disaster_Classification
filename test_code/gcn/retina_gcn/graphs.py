from typing import Optional

import numpy as np
import torch


def normalize_adj(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    D = A.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + eps)
    return (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]


def topk_sparsify(A: np.ndarray, k: int = 6, keep_diag: bool = True) -> np.ndarray:
    C = A.shape[0]
    B = np.zeros_like(A)
    diag = np.diag(A).copy()

    for i in range(C):
        row = A[i].copy()
        if keep_diag:
            row[i] = -1e9
        idx = np.argsort(-row)[:k]
        B[i, idx] = A[i, idx]

    if keep_diag:
        np.fill_diagonal(B, diag)

    return B


def build_adj_from_train_labels(
    Y: np.ndarray,
    mode: str = "pmi",
    add_self_loops: bool = True,
    eps: float = 1e-8,
    topk: Optional[int] = 6,
) -> torch.Tensor:
    assert Y.ndim == 2
    N, C = Y.shape
    cooc = (Y.T @ Y).astype(np.float64)
    freq = np.diag(cooc).astype(np.float64)

    if mode == "condprob":
        denom = freq[None, :] + eps
        P = cooc / denom
        A = 0.5 * (P + P.T)
        mx = A.max()
        if mx > 0:
            A = A / (mx + eps)

    elif mode == "pmi":
        Pij = cooc / max(N, 1)
        Pi = freq / max(N, 1)
        denom = Pi[:, None] * Pi[None, :] + eps
        pmi = np.log((Pij + eps) / denom)
        A = np.maximum(pmi, 0.0)
        mx = A.max()
        if mx > 0:
            A = A / (mx + eps)
    else:
        raise ValueError("mode must be 'pmi' or 'condprob'")

    if add_self_loops:
        A = A + np.eye(C, dtype=np.float64)

    if topk is not None and topk > 0:
        A = topk_sparsify(A, k=int(topk), keep_diag=True)

    return torch.tensor(normalize_adj(A, eps=eps), dtype=torch.float32)


def build_identity_adj(C: int) -> torch.Tensor:
    A = np.eye(C, dtype=np.float64)
    return torch.tensor(normalize_adj(A), dtype=torch.float32)


def permute_adj(A: torch.Tensor, seed: int = 123) -> torch.Tensor:
    C = A.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(C)
    return A[perm][:, perm]
