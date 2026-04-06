import torch
import torch.nn as nn


class GatedProjectionFusion(nn.Module):
    def __init__(self, nmods=2, dim=1024):
        super().__init__()
        self.nmods = nmods
        self.proj_src = nn.Linear(dim, dim)
        self.proj_tgt = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.gate.weight.data.zero_()
        self.gate.bias.data.zero_()
        self.proj_tgt.weight = torch.nn.Parameter(torch.eye(dim))
        self.proj_tgt.bias.data.zero_()

    def forward(self, src, target):
        """
        src=event tokens
        target=rgb tokens
        """

        p1 = self.proj_tgt(target)
        p2 = self.proj_src(src)

        cat = torch.cat([p1, p2], dim=-1)
        g = torch.tanh(self.gate(cat))

        fused = p1 + g * p2
        return fused


class TokenFusionTransformer(nn.Module):
    def __init__(
        self, dim=1024, depth=2, heads=8, mlp_dim=1024, dropout=0.0, attn_type="sa"
    ):
        super().__init__()
        self.attn_type = attn_type
        self.t_cls = (
            nn.TransformerEncoderLayer
            if attn_type == "sa"
            else nn.TransformerDecoderLayer
        )
        self.layers = nn.ModuleList(
            [
                self.t_cls(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
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

        if self.attn_type == "sa":
            x = torch.cat([src, target], dim=1)
            for layer in self.layers:
                x = layer(x)
            x = x[:, -target.shape[1] :]
        else:
            x = target
            for layer in self.layers:
                x = layer(tgt=x, memory=src)
        return x
