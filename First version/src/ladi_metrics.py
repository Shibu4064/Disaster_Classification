from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
except Exception:  # pragma: no cover
    average_precision_score = roc_auc_score = f1_score = precision_score = recall_score = None


def _safe_metric(fn, y_true, y_score_or_pred, default=np.nan, **kwargs):
    try:
        return float(fn(y_true, y_score_or_pred, **kwargs))
    except Exception:
        return float(default)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
    train_freq: Optional[np.ndarray] = None,
) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    assert y_true.shape == y_prob.shape
    N, C = y_true.shape

    if thresholds is None:
        thresholds = np.full(C, 0.5, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    per_ap, per_auc, per_f1, per_p, per_r = [], [], [], [], []
    for c in range(C):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        yh = y_pred[:, c]
        if yt.sum() == 0:
            per_ap.append(np.nan)
            per_auc.append(np.nan)
        else:
            per_ap.append(_safe_metric(average_precision_score, yt, yp))
            per_auc.append(_safe_metric(roc_auc_score, yt, yp))
        per_f1.append(_safe_metric(f1_score, yt, yh, zero_division=0))
        per_p.append(_safe_metric(precision_score, yt, yh, zero_division=0))
        per_r.append(_safe_metric(recall_score, yt, yh, zero_division=0))

    out = {
        "macro_ap": float(np.nanmean(per_ap)),
        "macro_auc": float(np.nanmean(per_auc)),
        "macro_f1": float(np.nanmean(per_f1)),
        "macro_precision": float(np.nanmean(per_p)),
        "macro_recall": float(np.nanmean(per_r)),
        "micro_ap": _safe_metric(average_precision_score, y_true.reshape(-1), y_prob.reshape(-1)),
        "micro_auc": _safe_metric(roc_auc_score, y_true.reshape(-1), y_prob.reshape(-1)),
        "micro_f1": _safe_metric(f1_score, y_true.reshape(-1), y_pred.reshape(-1), zero_division=0),
        "per_label_ap": [float(x) if not np.isnan(x) else np.nan for x in per_ap],
        "per_label_auc": [float(x) if not np.isnan(x) else np.nan for x in per_auc],
        "per_label_f1": [float(x) if not np.isnan(x) else np.nan for x in per_f1],
        "thresholds": thresholds.tolist(),
    }
    if train_freq is not None:
        train_freq = np.asarray(train_freq)
        order = np.argsort(train_freq)
        k1, k2 = C // 3, 2 * C // 3
        buckets = {"rare": order[:k1], "medium": order[k1:k2], "frequent": order[k2:]}
        for name, idx in buckets.items():
            out[f"{name}_macro_ap"] = float(np.nanmean(np.asarray(per_ap)[idx])) if len(idx) else np.nan
            out[f"{name}_macro_f1"] = float(np.nanmean(np.asarray(per_f1)[idx])) if len(idx) else np.nan
    return out


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid=None) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    C = y_true.shape[1]
    thresholds = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            yp = (y_prob[:, c] >= t).astype(int)
            score = _safe_metric(f1_score, y_true[:, c], yp, zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, t
        thresholds[c] = best_t
    return thresholds


def per_label_table(y_true: np.ndarray, y_prob: np.ndarray, label_cols: list[str], thresholds=None) -> "pd.DataFrame":
    import pandas as pd

    metrics = compute_metrics(y_true, y_prob, thresholds=thresholds)
    rows = []
    th = metrics["thresholds"]
    for i, lab in enumerate(label_cols):
        rows.append({
            "label": lab,
            "support": int(y_true[:, i].sum()),
            "prevalence": float(y_true[:, i].mean()),
            "AP": metrics["per_label_ap"][i],
            "AUC": metrics["per_label_auc"][i],
            "F1": metrics["per_label_f1"][i],
            "threshold": th[i],
        })
    return pd.DataFrame(rows)
