from typing import Dict, Optional

import numpy as np


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    n_pos = int(y_true.sum())
    n = y_true.shape[0]
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _binary_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = 0
    ap_sum = 0.0
    for i, yi in enumerate(y_sorted, start=1):
        if yi == 1:
            tp += 1
            ap_sum += tp / i
    return float(ap_sum / n_pos)


def f1_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-8) -> float:
    return float((2 * tp) / (2 * tp + fp + fn + eps))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    train_freq: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict:
    assert y_true.shape == y_prob.shape
    N, C = y_true.shape
    y_pred = (y_prob >= threshold).astype(np.int32)

    per_auc = np.full(C, np.nan, dtype=np.float64)
    per_ap = np.full(C, np.nan, dtype=np.float64)
    per_f1 = np.full(C, np.nan, dtype=np.float64)

    for c in range(C):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        per_auc[c] = _binary_auc(yt, yp)
        per_ap[c] = _binary_ap(yt, yp)

        tp = float(((y_pred[:, c] == 1) & (yt == 1)).sum())
        fp = float(((y_pred[:, c] == 1) & (yt == 0)).sum())
        fn = float(((y_pred[:, c] == 0) & (yt == 1)).sum())
        if (yt.sum() > 0) or (y_pred[:, c].sum() > 0):
            per_f1[c] = f1_from_counts(tp, fp, fn)

    macro_auc = float(np.nanmean(per_auc))
    macro_ap = float(np.nanmean(per_ap))
    macro_f1 = float(np.nanmean(per_f1))

    yt_flat = y_true.reshape(-1).astype(np.int32)
    yp_flat = y_prob.reshape(-1).astype(np.float64)
    ypred_flat = y_pred.reshape(-1).astype(np.int32)

    micro_auc = _binary_auc(yt_flat, yp_flat)
    micro_ap = _binary_ap(yt_flat, yp_flat)

    tp = float(((ypred_flat == 1) & (yt_flat == 1)).sum())
    fp = float(((ypred_flat == 1) & (yt_flat == 0)).sum())
    fn = float(((ypred_flat == 0) & (yt_flat == 1)).sum())
    micro_f1 = f1_from_counts(tp, fp, fn)

    out = {
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "micro_auc": float(micro_auc),
        "micro_ap": float(micro_ap),
        "micro_f1": micro_f1,
        "per_label_auc": per_auc.tolist(),
        "per_label_ap": per_ap.tolist(),
        "per_label_f1": per_f1.tolist(),
    }

    if train_freq is not None:
        order = np.argsort(train_freq)
        k1 = C // 3
        k2 = 2 * C // 3
        rare = order[:k1]
        med = order[k1:k2]
        freq = order[k2:]

        def bucket(idxs: np.ndarray) -> Dict[str, float]:
            if len(idxs) == 0:
                return {
                    "auc": float("nan"),
                    "ap": float("nan"),
                    "f1": float("nan"),
                    "n_labels": 0,
                }
            return {
                "auc": float(np.nanmean(per_auc[idxs])),
                "ap": float(np.nanmean(per_ap[idxs])),
                "f1": float(np.nanmean(per_f1[idxs])),
                "n_labels": int(len(idxs)),
            }

        out["bucket_rare"] = bucket(rare)
        out["bucket_medium"] = bucket(med)
        out["bucket_frequent"] = bucket(freq)
        out["bucket_indices"] = {
            "rare": rare.tolist(),
            "medium": med.tolist(),
            "frequent": freq.tolist(),
        }

    return out
