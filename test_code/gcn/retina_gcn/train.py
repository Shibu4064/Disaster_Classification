import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .models import (
    CNNWithLabelGCN,
    init_bias_from_prior,
    load_backbone_and_head_from_baseline,
    set_requires_grad,
)


@dataclass
class ExperimentResult:
    name: str
    mode: str  # "baseline" | "full" | "refonly"
    best_epoch: int
    val: Dict
    test: Dict
    ckpt_path: str
    run_dir: str


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = True,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


def run_baseline(
    exp_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_freq: np.ndarray,
    amp: bool = True,
    selection_metric: str = "macro_ap",
) -> ExperimentResult:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{exp_name}.pt")

    model = model.to(device)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_score = -1e9
    best_epoch = -1
    best_val_metrics = None

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, amp=amp)

        yv, pv = predict_all(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, train_freq=train_freq)
        score = float(val_metrics.get(selection_metric, -1e9))

        print(
            f"[{exp_name}] epoch {epoch:02d}/{epochs} "
            f"train_loss={tr_loss:.4f} "
            f"val_macroAP={val_metrics['macro_ap']:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f} "
            f"val_microF1={val_metrics['micro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    yt, pt = predict_all(model, test_loader, device)
    test_metrics = compute_metrics(yt, pt, train_freq=train_freq)

    with open(os.path.join(out_dir, f"{exp_name}_metrics.json"), "w") as f:
        json.dump(
            {"best_epoch": best_epoch, "val": best_val_metrics, "test": test_metrics},
            f,
            indent=2,
        )

    return ExperimentResult(
        exp_name, "baseline", best_epoch, best_val_metrics, test_metrics, ckpt_path, out_dir
    )


def run_gcn_full(
    exp_name: str,
    gcn_model: CNNWithLabelGCN,
    baseline_ckpt: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epochs_total: int,
    lr_main: float,
    weight_decay: float,
    train_freq: np.ndarray,
    amp: bool,
    warmup_epochs: int = 3,
    lr_refiner: float = 1e-3,
) -> ExperimentResult:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{exp_name}.pt")

    gcn_model = gcn_model.to(device)
    load_backbone_and_head_from_baseline(gcn_model, baseline_ckpt, device)

    set_requires_grad(gcn_model.backbone, False)
    set_requires_grad(gcn_model.head, False)
    set_requires_grad(gcn_model.refiner, True)

    opt_warm = torch.optim.AdamW(
        gcn_model.refiner.parameters(), lr=lr_refiner, weight_decay=weight_decay
    )
    print(f"[{exp_name}] Warmup refiner only for {warmup_epochs} epochs...")
    for w in range(1, warmup_epochs + 1):
        tr_loss = train_one_epoch(gcn_model, train_loader, opt_warm, device, amp=amp)
        yv, pv = predict_all(gcn_model, val_loader, device)
        vm = compute_metrics(yv, pv, train_freq=train_freq)
        print(
            f"[{exp_name}] warmup {w:02d}/{warmup_epochs} "
            f"train_loss={tr_loss:.4f} val_macroAP={vm['macro_ap']:.4f}"
        )

    set_requires_grad(gcn_model.backbone, True)
    set_requires_grad(gcn_model.head, True)
    set_requires_grad(gcn_model.refiner, True)

    opt = torch.optim.AdamW(
        gcn_model.parameters(), lr=lr_main, weight_decay=weight_decay
    )

    finetune_epochs = max(1, epochs_total - warmup_epochs)
    print(f"[{exp_name}] Finetune all for {finetune_epochs} epochs...")

    best_score = -1e9
    best_epoch = -1
    best_val_metrics = None

    for epoch in range(1, finetune_epochs + 1):
        tr_loss = train_one_epoch(gcn_model, train_loader, opt, device, amp=amp)

        yv, pv = predict_all(gcn_model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, train_freq=train_freq)

        score = float(val_metrics["macro_ap"])
        print(
            f"[{exp_name}] finetune {epoch:02d}/{finetune_epochs} "
            f"train_loss={tr_loss:.4f} "
            f"val_macroAP={val_metrics['macro_ap']:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f} "
            f"val_microF1={val_metrics['micro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(gcn_model.state_dict(), ckpt_path)

    gcn_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    yt, pt = predict_all(gcn_model, test_loader, device)
    test_metrics = compute_metrics(yt, pt, train_freq=train_freq)

    with open(os.path.join(out_dir, f"{exp_name}_metrics.json"), "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "val": best_val_metrics,
                "test": test_metrics,
                "warmup_epochs": warmup_epochs,
                "finetune_epochs": finetune_epochs,
            },
            f,
            indent=2,
        )

    return ExperimentResult(
        exp_name, "full", best_epoch, best_val_metrics, test_metrics, ckpt_path, out_dir
    )


def run_gcn_refiner_only(
    exp_name: str,
    gcn_model: CNNWithLabelGCN,
    baseline_ckpt: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epochs: int,
    lr_refiner: float,
    weight_decay: float,
    train_freq: np.ndarray,
    amp: bool,
) -> ExperimentResult:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{exp_name}.pt")

    gcn_model = gcn_model.to(device)
    load_backbone_and_head_from_baseline(gcn_model, baseline_ckpt, device)

    set_requires_grad(gcn_model.backbone, False)
    set_requires_grad(gcn_model.head, False)
    set_requires_grad(gcn_model.refiner, True)

    opt = torch.optim.AdamW(
        gcn_model.refiner.parameters(), lr=lr_refiner, weight_decay=weight_decay
    )

    best_score = -1e9
    best_epoch = -1
    best_val_metrics = None

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(gcn_model, train_loader, opt, device, amp=amp)

        yv, pv = predict_all(gcn_model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, train_freq=train_freq)
        score = float(val_metrics["macro_ap"])

        print(
            f"[{exp_name}] epoch {epoch:02d}/{epochs} "
            f"train_loss={tr_loss:.4f} "
            f"val_macroAP={val_metrics['macro_ap']:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f} "
            f"val_microF1={val_metrics['micro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(gcn_model.state_dict(), ckpt_path)

    gcn_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    yt, pt = predict_all(gcn_model, test_loader, device)
    test_metrics = compute_metrics(yt, pt, train_freq=train_freq)

    with open(os.path.join(out_dir, f"{exp_name}_metrics.json"), "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "val": best_val_metrics,
                "test": test_metrics,
                "refiner_only": True,
            },
            f,
            indent=2,
        )

    return ExperimentResult(
        exp_name,
        "refonly",
        best_epoch,
        best_val_metrics,
        test_metrics,
        ckpt_path,
        out_dir,
    )


__all__ = [
    "ExperimentResult",
    "predict_all",
    "train_one_epoch",
    "run_baseline",
    "run_gcn_full",
    "run_gcn_refiner_only",
]
