from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, sparse_loss = model(imgs)
        loss = criterion(logits, targets) + sparse_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Validation", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        logits, sparse_loss = model(imgs)
        loss = criterion(logits, targets) + sparse_loss
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    targets_all, scores_all = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        scores_all.append(torch.sigmoid(logits).cpu().numpy())
        targets_all.append(targets.numpy())
    return np.vstack(targets_all), np.vstack(scores_all)
