from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def normalize_adj(A_raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetrically normalise an affinity matrix with self-loops."""
    C = A_raw.size(0)
    A_pos = torch.relu(A_raw)
    I = torch.eye(C, device=A_raw.device, dtype=A_raw.dtype)
    A_tilde = A_pos + I
    degree = A_tilde.sum(dim=1).clamp_min(eps)
    inv_sqrt = degree.pow(-0.5)
    D_inv_sqrt = torch.diag(inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


class GraphConvolution(nn.Module):
    """Standard graph convolution: H' = A_hat H W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        out = A_hat @ (H @ self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class GraphAttentionTransformerLayer(nn.Module):
    """Multi-head self-attention over an initial label correlation matrix.

    Treats the C-by-C affinity matrix as a sequence of C tokens of dimension
    C, applies multi-head scaled dot-product attention, and projects back
    to C dimensions. The output is forced non-negative and symmetric.
    """

    def __init__(
        self,
        num_classes: int,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim or num_classes

        self.q_proj = nn.Linear(num_classes, self.hidden_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(num_classes, self.hidden_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(num_classes, self.hidden_dim * num_heads, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim * num_heads, num_classes, bias=False)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        C = A.size(0)
        H, d = self.num_heads, self.hidden_dim

        A_in = A.unsqueeze(0)
        Q = self.q_proj(A_in).view(1, C, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(A_in).view(1, C, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(A_in).view(1, C, H, d).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.permute(0, 2, 1, 3).reshape(1, C, H * d)
        out = self.out_proj(out).squeeze(0)
        out = F.relu(out)
        return 0.5 * (out + out.t())
