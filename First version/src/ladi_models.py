from __future__ import annotations

from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_backbone(name: str = "resnet18", pretrained: bool = True):
    """Return torchvision feature extractor and feature dimension. Falls back to random weights if downloads fail."""
    name = name.lower()
    try:
        if name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feat_dim
        if name == "resnet34":
            weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet34(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feat_dim
        if name == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = torchvision.models.efficientnet_b0(weights=weights)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, feat_dim
        if name == "efficientnet_b2":
            weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            net = torchvision.models.efficientnet_b2(weights=weights)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, feat_dim
    except Exception as e:
        print(f"[warning] Could not load pretrained weights for {name}: {e}. Falling back to random weights.")
        return get_backbone(name=name, pretrained=False)
    raise ValueError("Unsupported backbone. Use resnet18, resnet34, efficientnet_b0, or efficientnet_b2.")


class BaselineCNN(nn.Module):
    def __init__(self, num_labels: int, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim = get_backbone(backbone, pretrained=pretrained)
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_labels)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(self.dropout(feats))


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # H: C x F, A_norm: C x C
        out = A_norm @ (H @ self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class BatchGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # H: B x C x F, A_norm: B x C x C
        return self.lin(torch.bmm(A_norm, H))


class StaticGCNClassifier(nn.Module):
    """ML-GCN style classifier: static train-label graph produces one classifier vector per label."""

    def __init__(
        self,
        num_labels: int,
        A_norm: torch.Tensor,
        backbone: str = "resnet18",
        pretrained: bool = True,
        label_dim: int = 256,
        gcn_hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim = get_backbone(backbone, pretrained=pretrained)
        self.feat_dim = feat_dim
        self.label_embeddings = nn.Parameter(torch.empty(num_labels, label_dim))
        nn.init.xavier_uniform_(self.label_embeddings)
        self.gc1 = GraphConvolution(label_dim, gcn_hidden)
        self.gc2 = GraphConvolution(gcn_hidden, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.register_buffer("A_norm", A_norm.float())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def label_weights(self) -> torch.Tensor:
        H = F.relu(self.gc1(self.label_embeddings, self.A_norm))
        H = self.dropout(H)
        W = self.gc2(H, self.A_norm)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        W = self.label_weights()
        logits = feats @ W.t() + self.bias
        return logits


def batch_normalize_adj(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # A: B x C x C, non-negative. Add self-loops then symmetric normalize.
    B, C, _ = A.shape
    I = torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0)
    A = F.relu(A) + I
    deg = A.sum(dim=-1).clamp_min(eps)
    d_inv_sqrt = deg.pow(-0.5)
    return A * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)


class DynamicGCNClassifier(nn.Module):
    """Image-conditioned dynamic GCN.

    Static A gives the disaster-label prior from training co-occurrence; the model also creates
    a sample-specific label adjacency from the current image feature vector. The learned gate
    mixes static and dynamic adjacency before batch-wise GCN propagation.
    """

    def __init__(
        self,
        num_labels: int,
        A_static: torch.Tensor,
        backbone: str = "resnet18",
        pretrained: bool = True,
        label_dim: int = 256,
        gcn_hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim = get_backbone(backbone, pretrained=pretrained)
        self.feat_dim = feat_dim
        self.label_embeddings = nn.Parameter(torch.empty(num_labels, label_dim))
        nn.init.xavier_uniform_(self.label_embeddings)
        self.img_to_context = nn.Sequential(
            nn.Linear(feat_dim, label_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(label_dim, label_dim)
        )
        self.q_proj = nn.Linear(label_dim, label_dim, bias=False)
        self.k_proj = nn.Linear(label_dim, label_dim, bias=False)
        self.gc1 = BatchGraphConvolution(label_dim, gcn_hidden)
        self.gc2 = BatchGraphConvolution(gcn_hidden, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.graph_gate_raw = nn.Parameter(torch.tensor(-2.0))  # starts close to static graph
        self.register_buffer("A_static", A_static.float())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def dynamic_adj(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.size(0)
        C = self.A_static.size(0)
        context = self.img_to_context(feats).unsqueeze(1)  # B x 1 x D
        H0 = self.label_embeddings.unsqueeze(0).expand(B, -1, -1) + context
        Q = self.q_proj(H0)
        K = self.k_proj(H0)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
        A_dyn = torch.softmax(scores, dim=-1)
        A_dyn = 0.5 * (A_dyn + A_dyn.transpose(1, 2))
        gate = torch.sigmoid(self.graph_gate_raw)
        A = (1.0 - gate) * self.A_static.unsqueeze(0) + gate * A_dyn
        return batch_normalize_adj(A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        B = feats.size(0)
        A = self.dynamic_adj(feats)
        context = self.img_to_context(feats).unsqueeze(1)
        H = self.label_embeddings.unsqueeze(0).expand(B, -1, -1) + context
        H = F.relu(self.gc1(H, A))
        H = self.dropout(H)
        W = self.gc2(H, A)  # B x C x feat_dim
        logits = torch.einsum("bd,bcd->bc", feats, W) + self.bias
        return logits


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def init_classifier_bias(module: nn.Module, train_freq: np.ndarray, eps: float = 1e-4):
    p = np.clip(np.asarray(train_freq), eps, 1 - eps)
    bias = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    # The models expose either .head.bias or .bias.
    with torch.no_grad():
        if hasattr(module, "head") and hasattr(module.head, "bias") and module.head.bias is not None:
            module.head.bias.copy_(bias.to(module.head.bias.device))
        elif hasattr(module, "bias"):
            module.bias.copy_(bias.to(module.bias.device))


def load_backbone_from_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location=device)
    target = model.state_dict()
    copied = 0
    for k, v in sd.items():
        if k.startswith("backbone.") and k in target and target[k].shape == v.shape:
            target[k] = v
            copied += 1
    model.load_state_dict(target, strict=False)
    print(f"Loaded {copied} backbone tensors from {ckpt_path}.")


def pos_weight_from_df(train_df, label_cols, max_weight: float = 10.0) -> torch.Tensor:
    import numpy as np
    y = train_df[label_cols].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pw = neg / np.clip(pos, 1, None)
    pw = np.clip(pw, 1.0, max_weight)
    return torch.tensor(pw, dtype=torch.float32)
