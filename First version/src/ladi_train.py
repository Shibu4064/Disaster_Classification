from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .ladi_metrics import compute_metrics, tune_thresholds, per_label_table


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def predict_all(model: nn.Module, loader, device: torch.device):
    model.eval()
    ys, probs = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        ys.append(y.numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(probs, axis=0)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp: bool = True):
    model.train()
    use_amp = amp and device.type == "cuda"
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def save_json(obj: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    out_dir: str | Path,
    run_name: str,
    label_cols: list[str],
    train_freq: Optional[np.ndarray] = None,
    epochs: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    pos_weight: Optional[torch.Tensor] = None,
    amp: bool = True,
    freeze_backbone_epochs: int = 0,
    selection_metric: str = "macro_ap",
):
    out_dir = Path(out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_epoch = -1e9, -1
    best_path = out_dir / "best_model.pt"
    history = []

    # Optional initial backbone freeze for graph heads.
    def set_backbone_grad(flag: bool):
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = flag

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0:
            set_backbone_grad(epoch > freeze_backbone_epochs)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, amp=amp)
        yv, pv = predict_all(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, train_freq=train_freq)
        score = float(val_metrics.get(selection_metric, -1e9))
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if not k.startswith("per_label") and k != "thresholds"}}
        history.append(row)
        print(f"[{run_name}] epoch {epoch:02d}/{epochs} loss={train_loss:.4f} val_macroAP={val_metrics['macro_ap']:.4f} val_macroF1={val_metrics['macro_f1']:.4f} val_microF1={val_metrics['micro_f1']:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), best_path)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    yv, pv = predict_all(model, val_loader, device)
    yt, pt = predict_all(model, test_loader, device)
    thresholds = tune_thresholds(yv, pv)
    val_05 = compute_metrics(yv, pv, train_freq=train_freq)
    test_05 = compute_metrics(yt, pt, train_freq=train_freq)
    val_tuned = compute_metrics(yv, pv, thresholds=thresholds, train_freq=train_freq)
    test_tuned = compute_metrics(yt, pt, thresholds=thresholds, train_freq=train_freq)
    np.savez_compressed(out_dir / "val_predictions.npz", y_true=yv, y_prob=pv, thresholds=thresholds)
    np.savez_compressed(out_dir / "test_predictions.npz", y_true=yt, y_prob=pt, thresholds=thresholds)
    per_label_table(yt, pt, label_cols, thresholds=thresholds).to_csv(out_dir / "per_label_test_tuned.csv", index=False)
    metrics = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "val_at_0_5": val_05,
        "test_at_0_5": test_05,
        "val_tuned": val_tuned,
        "test_tuned": test_tuned,
        "label_cols": label_cols,
    }
    save_json(metrics, out_dir / "metrics.json")
    return metrics, str(best_path)


def collect_run_metrics(root: str | Path):
    root = Path(root)
    rows = []
    for p in root.glob("*/metrics.json"):
        with open(p) as f:
            m = json.load(f)
        rows.append({
            "run_name": m.get("run_name", p.parent.name),
            "best_epoch": m.get("best_epoch"),
            "val_macro_ap": m.get("val_at_0_5", {}).get("macro_ap"),
            "test_macro_ap": m.get("test_at_0_5", {}).get("macro_ap"),
            "test_macro_f1@0.5": m.get("test_at_0_5", {}).get("macro_f1"),
            "test_micro_f1@0.5": m.get("test_at_0_5", {}).get("micro_f1"),
            "test_macro_f1_tuned": m.get("test_tuned", {}).get("macro_f1"),
            "test_micro_f1_tuned": m.get("test_tuned", {}).get("micro_f1"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("test_macro_ap", ascending=False)
    return df
