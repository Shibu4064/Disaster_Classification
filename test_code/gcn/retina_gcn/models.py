import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_backbone(name: str):
    name = name.lower()
    if name == "resnet18":
        net = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    if name == "densenet121":
        net = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.DEFAULT
        )
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        return net, feat_dim
    raise ValueError(f"Unsupported backbone: {name} (use resnet18 or densenet121)")


class BaselineCNN(nn.Module):
    def __init__(self, num_labels: int, backbone: str):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone)
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        AX = torch.einsum("ij,bjf->bif", A_norm, X)
        return self.lin(AX)


class LabelGCNRefiner(nn.Module):
    """Refines per-label logits via a 2-layer GCN with a bounded residual gate."""

    def __init__(
        self,
        num_labels: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        alpha_max: float = 0.5,
    ):
        super().__init__()
        self.g1 = GCNLayer(1, hidden_dim)
        self.g2 = GCNLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.alpha_raw = nn.Parameter(torch.tensor(-5.0))
        self.alpha_max = float(alpha_max)

    def forward(self, z: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        X = z.unsqueeze(-1)
        X = F.relu(self.g1(X, A_norm))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = F.relu(self.g2(X, A_norm))
        X = F.dropout(X, p=self.dropout, training=self.training)
        delta = self.out(X).squeeze(-1)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_raw)
        return z + alpha * delta


class CNNWithLabelGCN(nn.Module):
    def __init__(
        self,
        num_labels: int,
        backbone: str,
        A_norm: torch.Tensor,
        gcn_hidden: int = 64,
        gcn_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone)
        self.head = nn.Linear(feat_dim, num_labels)
        self.refiner = LabelGCNRefiner(
            num_labels, hidden_dim=gcn_hidden, dropout=gcn_dropout, alpha_max=0.5
        )
        self.register_buffer("A_norm", A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        z = self.head(feats)
        return self.refiner(z, self.A_norm)


def init_bias_from_prior(linear: nn.Linear, train_freq: np.ndarray, eps: float = 1e-4):
    p = np.clip(train_freq, eps, 1.0 - eps)
    b = np.log(p / (1.0 - p))
    with torch.no_grad():
        linear.bias.copy_(torch.tensor(b, dtype=linear.bias.dtype))


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def load_backbone_and_head_from_baseline(
    gcn_model: nn.Module, baseline_ckpt_path: str, device: torch.device
):
    sd = torch.load(baseline_ckpt_path, map_location=device)
    gsd = gcn_model.state_dict()
    copied = 0
    for k, v in sd.items():
        if k.startswith("backbone.") or k.startswith("head."):
            if k in gsd and gsd[k].shape == v.shape:
                gsd[k] = v
                copied += 1
    gcn_model.load_state_dict(gsd)
    print(f"Loaded {copied} tensors from baseline into GCN model.")
