import torch
import torch.nn as nn
from einops import rearrange


class GatedProjectionFusion(nn.Module):
    def __init__(self, nmods=2, dim=1024):
        super().__init__()
        self.nmods = nmods
        self.proj_src = nn.Linear(dim * (nmods - 1), dim)
        self.proj2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, src, target):
        """
        x1, x2: [B, N, D] token streams from two modalities
        returns: [B, N, D] fused embedding
        """

        p1 = self.proj_src(src)
        p2 = self.proj2(target)

        cat = torch.cat([p1, p2], dim=-1)
        g = torch.sigmoid(self.gate(cat))

        fused = g * p1 + (1 - g) * p2
        return fused


class TokenFusionTransformer(nn.Module):
    def __init__(self, dim=1024, depth=2, heads=8, mlp_dim=1024, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(depth)
            ]
        )

    def forward(self, src, target):
        """
        x1,x2: [B, N, D] tokens from two modalities
        returns: [B, N1+N2, D] fused tokens
        """
        b, n, d = target.shape

        x = torch.cat([src, target], dim=1)
        x = rearrange(x, "b n d -> n b d")

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, "n b d -> b n d")
        return x[:, -target.shape[1] :]
