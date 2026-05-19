from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from src.data import MultiLabelDataset, build_transforms
from src.embeddings import LabelEmbeddings
from src.engine import collect_predictions, eval_epoch, train_epoch
from src.metrics import compute_multi_label_metrics
from src.model import GATNResnet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_cols = pd.read_csv(cfg.TRAIN_CSV).columns.drop("ID").tolist()
    print(f"Found {len(label_cols)} labels: {label_cols}")

    label_embs = LabelEmbeddings.load(
        out_path=cfg.EMB_CACHE,
        label_order=label_cols,
        model_name=cfg.BERT_MODEL,
        device="cpu",
    ).to(device)

    train_tf, val_tf = build_transforms(cfg.IMG_RESIZE, cfg.IMG_CROP, cfg.RETINA_MEAN, cfg.RETINA_STD)

    train_ds = MultiLabelDataset(cfg.TRAIN_CSV, cfg.IMG_DIR, label_cols, transform=train_tf)
    val_ds = MultiLabelDataset(cfg.VAL_CSV, cfg.IMG_DIR, label_cols, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = GATNResnet(
        label_embs=label_embs,
        train_backbone=True,
        gat_num_heads=cfg.GAT_HEADS,
        gcn_hidden_dim=cfg.GCN_HIDDEN,
        alpha=cfg.ALPHA,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.BASE_LR,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )

    cfg.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        y_true, y_score = collect_predictions(model, val_loader, device)
        m = compute_multi_label_metrics(y_true, None, y_score)

        print(
            f"Epoch {epoch}/{cfg.NUM_EPOCHS}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"mAP={m['mAP']:.4f}, CP={m['CP']:.4f}, CR={m['CR']:.4f}, CF1={m['CF1']:.4f}, "
            f"OP={m['OP']:.4f}, OR={m['OR']:.4f}, OF1={m['OF1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.CHECKPOINT_PATH)
            print(f"  -> Saved checkpoint to {cfg.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
