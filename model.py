import copy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _drop_path(x_residual: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth applied to a layer's residual contribution.

    Inverted scaling: multiply the residual by a per-sample Bernoulli mask and
    divide by keep_prob, so expected magnitude is preserved without any rescale
    at eval time. No-op when drop_prob == 0 or in eval mode.
    """
    if drop_prob <= 0.0 or not training:
        return x_residual
    keep_prob = 1.0 - drop_prob
    mask_shape = (x_residual.shape[0],) + (1,) * (x_residual.ndim - 1)
    mask = torch.empty(mask_shape, device=x_residual.device, dtype=x_residual.dtype).bernoulli_(keep_prob)
    return x_residual.div(keep_prob).mul(mask)


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


class _IndependentEncoder(nn.Module):
    """Stack of independent TransformerEncoderLayers with optional per-layer DropPath
    (stochastic depth) and optional gradient checkpointing.

    Uses the same `layers` ModuleList as nn.TransformerEncoder, so state dicts are
    key-compatible across the two. Stochastic depth is applied to each layer's
    residual contribution via the residual-extraction trick (out - x). When
    use_checkpoint is True each layer is wrapped in torch.utils.checkpoint so the
    O(T^2) attention activations are recomputed in backward instead of stored.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int,
                 drop_rates=None, use_checkpoint: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.drop_rates = list(drop_rates) if drop_rates is not None else [0.0] * num_layers
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dp in zip(self.layers, self.drop_rates):
            if self.use_checkpoint and self.training:
                out = checkpoint(layer, x, use_reentrant=False)
            else:
                out = layer(x)
            if dp > 0.0:
                x = x + _drop_path(out - x, dp, self.training)
            else:
                x = out
        return x


class AmbiTransformerDOA(nn.Module):
    """
    Transformer model for multi-source DOA in Ambisonics (query-decoder, presence).

    Architecture:
      1) Channel projection (n_channels) -> d_model.
      2) Optional temporal positional encoding.
      3) Transformer encoder stack (independent layers, optional stochastic depth
         and gradient checkpointing).
      4) Query decoder (DETR-style set prediction): `max_sources` learned slot
         queries cross-attend the encoder output and self-attend (de-duplicating),
         each predicting its own unit-vector direction and a presence logit. Count
         is derived downstream as sum_j 1{sigmoid(presence_j) > threshold}; targets
         are assigned by permutation-invariant (PI) matching.

    Terminology:
      - d_model: internal feature size in the Transformer (not signal length).
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
        gradient_checkpointing: bool = False,  # recompute activations in backward to save memory
        stochastic_depth: float = 0.0,  # max DropPath rate at the final layer (0 disables)
        decoder_layers: int = 3,  # query-decoder depth
        decoder_dropout: float = 0.1,  # dropout inside the decoder + slot heads (independent of encoder `dropout`)
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.max_sources = max_sources

        # Linear decay schedule: layer 0 keeps everything, layer L-1 drops at the
        # configured max rate. p_ℓ = stochastic_depth * ℓ / (L-1). When num_layers
        # == 1 the schedule collapses to a single rate. drop_rates=None means
        # disabled — preserves bit-identical baseline against pre-SD checkpoints.
        if stochastic_depth > 0.0:
            if num_layers == 1:
                drop_rates = [stochastic_depth]
            else:
                drop_rates = [stochastic_depth * i / (num_layers - 1) for i in range(num_layers)]
        else:
            drop_rates = None

        # (1) Project channel features (n_channels) -> d_model.
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
        # Independent-layer encoder when stochastic depth or gradient checkpointing
        # is needed (both produce `transformer.layers.*` keys, matching the stock
        # nn.TransformerEncoder used otherwise).
        if drop_rates is not None or gradient_checkpointing:
            self.transformer = _IndependentEncoder(
                enc_layer, num_layers, drop_rates=drop_rates, use_checkpoint=gradient_checkpointing
            )
        else:
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pre_head_norm = nn.LayerNorm(d_model)  # normalization before output heads

        # (4) Query-decoder readout (presence-only set prediction).
        # One learned query embedding per source slot; these carry slot identity
        # (DETR-style), no sinusoidal PE on the query side since PI matching makes
        # the slots order-free. The decoder has its own dropout so it can be tuned
        # independently of the (typically heavier) encoder dropout.
        self.source_queries = nn.Parameter(torch.randn(max_sources, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
            batch_first=True,  # [B, S, d_model]
            norm_first=True,  # Pre-LN, matching the encoder
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        # Per-slot heads: each query token predicts its own unit direction and a
        # single presence logit.
        self.slot_pos_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(d_model, 3),
        )
        self.slot_presence_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, n_channels, n_samples] -> [B, 16, T]
        Returns:
          n_sources: [B, max_sources] per-slot presence logits.
          positions: [B, max_sources, 3] unit vectors.
        """
        # Convert to Transformer layout [B, T, C].
        x = x.permute(0, 2, 1)  # [B, T, C]

        # Project channels to d_model, add positional encoding, encode.
        x = self.proj(x)  # [B, T, d_model]
        x = self.pos_enc(x)  # [B, T, d_model]
        memory = self.transformer(x)  # [B, T, d_model] (full sequence, no pooling)

        # Query decoder: learned slots cross-attend the sequence and self-attend.
        B = memory.size(0)
        q = self.source_queries.unsqueeze(0).expand(B, -1, -1)  # [B, S, d_model]
        slots = self.decoder(tgt=q, memory=memory)  # [B, S, d_model]
        slots = self.pre_head_norm(slots)  # LayerNorm over feature dim, per slot

        positions = self.slot_pos_head(slots)  # [B, S, 3]
        positions = positions / (positions.norm(dim=2, keepdim=True) + 1e-8)

        n_sources = self.slot_presence_head(slots).squeeze(-1)  # [B, S] presence logits
        return n_sources, positions
