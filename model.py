import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from AmbiDOA import log_print
from datetime import datetime


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic non-learnable sinusoidal positional encoding for Transformers.

    Why this is needed:
      - A Transformer does not inherently know the order of time samples.
      - Positional encoding injects sample-index information into the sequence.

    How it works:
      - For each position (t) and feature dimension (d), sin/cos values are
        generated at different frequencies.
      - This creates a [L, d_model] table that is added to input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        # Build positional matrix with shape [max_len, d_model].
        pe = torch.zeros(max_len, d_model)  # [L, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]

        # Original Transformer formula (Vaswani et al.) with frequency scaling
        # for even/odd channels.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        # Even channels -> sin, odd channels -> cos.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, d_model]

        # register_buffer: not trainable parameters, but moved with the model
        # between CPU/GPU devices.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        Returns x plus positional slice [1, L, d_model].
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


class AmbiTransformerDOA(nn.Module):
    """
    Transformer model for multi-source DOA in Ambisonics:
      - HEAD 1 (count): predicted number of sources (scalar regression)
      - HEAD 2 (positions): [max_sources, 3] unit vectors (x, y, z)

    High-level architecture:
      1) Channel projection (16) -> d_model.
      2) Optional temporal positional encoding.
      3) Transformer encoder stack.
      4) Temporal pooling (mean across time).
      5) Two output MLP heads: count and positions.

    Terminology:
      - d_model: internal feature size in Transformer (not signal length).
      - nhead: number of attention heads (d_model must be divisible by nhead).
      - num_layers: number of encoder layers.
      - dim_feedforward: MLP size inside each encoder layer.
    """

    def __init__(
        self,
        n_channels: int = 16,  # number of input channels
        input_len: int = 2756,  # window length in samples (used for pos-enc max_len)
        d_model: int = 256,  # projected model dimension
        nhead: int = 8,  # number of attention heads
        num_layers: int = 8,  # encoder depth
        dim_feedforward: int = 1024,  # encoder MLP size
        dropout: float = 0.3,  # regularization
        max_sources: int = 12,  # maximum number of predicted sources
        use_posenc: bool = True,  # enable positional encoding
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.max_sources = max_sources

        # (1) Project channel features 16 -> d_model.
        self.proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # (2) Add temporal positional encoding.
        self.pos_enc = (
            SinusoidalPositionalEncoding(d_model, max_len=max(4096, input_len))
            if use_posenc
            else nn.Identity()
        )

        # (3) Transformer encoder (multi-head attention + feedforward blocks).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input shape [B, T, d_model]
            norm_first=True,  # Pre-LN
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pre_head_norm = nn.LayerNorm(d_model)  # normalization before output heads

        # (4) HEAD 1: source-count regression.
        self.n_sources_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # (5) HEAD 2: source positions (max_sources * 3).
        self.positions_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_sources * 3),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [batch, n_channels, n_samples] -> [B, 16, T]
        Returns:
          n_sources: [B, 1]
          positions: [B, max_sources, 3] unit vectors
        """
        # Convert to Transformer layout [B, T, C].
        x = x.permute(0, 2, 1)  # [B, T, C]

        # Project channels to d_model.
        x = self.proj(x)  # [B, T, d_model]

        # Add positional encoding.
        x = self.pos_enc(x)  # [B, T, d_model]

        # Transformer encoding.
        feats = self.transformer(x)  # [B, T, d_model]

        # Temporal mean pooling.
        feats = feats.mean(dim=1)  # [B, d_model]

        # Normalize before heads.
        feats = self.pre_head_norm(feats)

        # HEAD 1: source count.
        n_sources = self.n_sources_head(feats)  # [B, 1]

        # HEAD 2: source positions.
        positions = self.positions_head(feats)  # [B, S * 3]
        positions = positions.view(-1, self.max_sources, 3)  # [B, S, 3]

        # Normalize to unit vectors on the sphere.
        positions = positions / (positions.norm(dim=2, keepdim=True) + 1e-8)

        return n_sources, positions
