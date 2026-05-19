from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multi_label_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Mean AP plus per-class (C*) and overall micro (O*) precision/recall/F1."""
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)

    return {
        "mAP": average_precision_score(y_true, y_score, average="macro"),
        "CP": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "CR": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "CF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "OP": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "OR": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "OF1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }
