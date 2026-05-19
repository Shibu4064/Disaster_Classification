from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

from .graph import GraphAttentionTransformerLayer, GraphConvolution, normalize_adj


class GATNResnet(nn.Module):
    """Graph Attention Transformer Network with an EfficientNet-B3 backbone.

    The model builds a label correlation graph from BERT embeddings, refines
    it through a multi-head attention layer, and runs two GCN layers to
    produce per-label classifier weights. Image features from the backbone
    are dot-producted with these weights to get logits.
    """

    IMG_FEAT_DIM = 1536  # EfficientNet-B3 output

    def __init__(
        self,
        label_embs: torch.Tensor,
        train_backbone: bool = False,
        gat_num_heads: int = 4,
        gat_hidden_dim: Optional[int] = None,
        gcn_hidden_dim: int = 1024,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer("E", label_embs)
        self.num_classes = label_embs.size(0)
        emb_dim = label_embs.size(1)

        backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone

        self.gat_layer = GraphAttentionTransformerLayer(
            num_classes=self.num_classes,
            num_heads=gat_num_heads,
            hidden_dim=gat_hidden_dim,
        )
        self.gc1 = GraphConvolution(emb_dim, gcn_hidden_dim)
        self.gc2 = GraphConvolution(gcn_hidden_dim, self.IMG_FEAT_DIM)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)

        E_norm = F.normalize(self.E, dim=1)
        A0 = E_norm @ E_norm.t()
        A_trans = self.gat_layer(A0)
        A_hat = normalize_adj(A_trans)

        H = self.act(self.gc1(self.E, A_hat))
        W = self.gc2(H, A_hat)

        logits = feats @ W.t()

        I = torch.eye(self.num_classes, device=A_hat.device)
        sparse_loss = (A_hat - I).abs().mean() * self.alpha
        return logits, sparse_loss

    def get_config_optim(self, lr: float, lrp: float, lrt: float) -> List[dict]:
        return [
            {"params": self.backbone.parameters(), "lr": lr * lrp},
            {"params": self.gat_layer.parameters(), "lr": lr * lrt},
            {"params": self.gc1.parameters(), "lr": lr},
            {"params": self.gc2.parameters(), "lr": lr},
        ]
