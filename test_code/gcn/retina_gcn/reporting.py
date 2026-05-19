import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .train import ExperimentResult


def save_per_label_table(
    out_path: str, label_cols: List[str], train_freq: np.ndarray, metrics: Dict
) -> None:
    df = pd.DataFrame(
        {
            "label": label_cols,
            "train_prevalence": train_freq,
            "auc": np.array(metrics["per_label_auc"], dtype=np.float64),
            "ap": np.array(metrics["per_label_ap"], dtype=np.float64),
            "f1@0.5": np.array(metrics["per_label_f1"], dtype=np.float64),
        }
    )
    df.to_csv(out_path, index=False)


def to_summary_row(res: ExperimentResult) -> Dict:
    t = res.test
    return {
        "exp": res.name,
        "mode": res.mode,
        "best_epoch": res.best_epoch,
        "test_macro_ap": t["macro_ap"],
        "test_macro_auc": t["macro_auc"],
        "test_macro_f1": t["macro_f1"],
        "test_micro_f1": t["micro_f1"],
        "rare_ap": t.get("bucket_rare", {}).get("ap", float("nan")),
        "medium_ap": t.get("bucket_medium", {}).get("ap", float("nan")),
        "frequent_ap": t.get("bucket_frequent", {}).get("ap", float("nan")),
        "ckpt_path": res.ckpt_path,
    }


def mean_std_aggregate(rows: List[Dict], keys: List[str]) -> Dict:
    out = {}
    for k in keys:
        vals = [
            r[k]
            for r in rows
            if r.get(k) is not None
            and not (isinstance(r[k], float) and math.isnan(r[k]))
        ]
        if len(vals) == 0:
            out[k + "_mean"] = float("nan")
            out[k + "_std"] = float("nan")
        else:
            arr = np.array(vals, dtype=np.float64)
            out[k + "_mean"] = float(np.nanmean(arr))
            out[k + "_std"] = float(np.nanstd(arr))
    return out
