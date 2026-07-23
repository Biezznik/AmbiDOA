import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from AmbiDOA import log_print
from datetime import datetime


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


class _SharedEncoder(nn.Module):
    """Applies a single TransformerEncoderLayer num_layers times (ALBERT-style weight sharing).

    All passes through the stack share identical parameters, reducing memorizable
    capacity while preserving the depth of iterative refinement.

    use_checkpoint: when True, each pass is wrapped in torch.utils.checkpoint so
    intermediate activations (including the O(T²) attention matrix) are not stored
    for backward — they are recomputed on demand. This trades compute for memory
    and is essential for long sequences or large batch sizes.

    drop_rates: optional list of length num_layers with per-pass DropPath rates.
    When None (or all zeros), behavior is bit-identical to the unregularized baseline.
    Drop is applied to the residual contribution (out - x) so layer internals
    stay untouched.
    """

    def __init__(self, layer: nn.TransformerEncoderLayer, num_layers: int, use_checkpoint: bool = False, drop_rates=None):
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.drop_rates = list(drop_rates) if drop_rates is not None else [0.0] * num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            if self.use_checkpoint and self.training:
                out = checkpoint(self.layer, x, use_reentrant=False)
            else:
                out = self.layer(x)
            dp = self.drop_rates[i]
            if dp > 0.0:
                x = x + _drop_path(out - x, dp, self.training)
            else:
                x = out
        return x


class _StochasticDepthEncoder(nn.Module):
    """Stack of independent TransformerEncoderLayers with per-layer DropPath.

    Mirrors nn.TransformerEncoder but applies stochastic depth on each layer's
    residual contribution via the residual-extraction trick (out - x). Treats the
    full block (attn + FFN) as one residual unit — coarser than per-sublayer
    drop but does not require cracking the layer's internals.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int, drop_rates):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.drop_rates = list(drop_rates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dp in zip(self.layers, self.drop_rates):
            out = layer(x)
            if dp > 0.0:
                x = x + _drop_path(out - x, dp, self.training)
            else:
                x = out
        return x


class AmbiTransformerDOA(nn.Module):
    """
    Transformer model for multi-source DOA in Ambisonics:
      - HEAD 1: source count or per-slot presence. Three modes:
          * "regression" — scalar SmoothL1 target (one logit).
          * "ordinal"    — max_sources independent BCE heads predicting P(k>=j).
          * "presence"   — max_sources per-slot presence logits (BCE against the
                           PI-matched assignment). Count is derived from the
                           position head's slot usage, not predicted separately.
      - HEAD 2 (positions): [max_sources, 3] unit vectors (x, y, z)

    High-level architecture:
      1) Channel projection (16) -> d_model.
      2) Optional temporal positional encoding.
      3) Transformer encoder stack.
      4) Readout — one of two builds (A/B), selected by use_query_decoder:
           A (default): temporal mean-pool -> pooled count/presence + positions heads.
           B (presence-only): query decoder — max_sources learned slots cross-attend
             the encoder output and predict per-slot direction + presence.

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
        share_encoder_weights: bool = False,  # ALBERT-style cross-layer weight sharing
        gradient_checkpointing: bool = False,  # recompute activations in backward to save memory
        stochastic_depth: float = 0.0,  # max DropPath rate at the final layer (0 disables)
        count_loss_type: str = "regression",  # "regression" -> 1 logit; "ordinal" -> max_sources BCE logits; "presence" -> per-slot presence logits
        count_head_hidden: bool = True,  # False -> single Linear count head (~3K params vs ~70K). Ignored in "presence" mode.
        use_query_decoder: bool = False,  # A/B readout: False -> mean-pool + pooled heads (baseline); True -> query decoder (presence-only)
        decoder_layers: int = 3,  # query-decoder depth (only used when use_query_decoder=True)
        decoder_dropout: float = 0.1,  # dropout inside the decoder + slot heads (independent of encoder `dropout`)
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        if count_loss_type not in ("regression", "ordinal", "presence"):
            raise ValueError(
                f"count_loss_type must be 'regression', 'ordinal', or 'presence', got {count_loss_type!r}."
            )
        if use_query_decoder and count_loss_type != "presence":
            raise ValueError(
                "use_query_decoder=True currently supports count_loss_type='presence' only."
            )
        self.max_sources = max_sources
        self.count_loss_type = count_loss_type
        self.use_query_decoder = use_query_decoder

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
        if share_encoder_weights:
            # One set of weights applied num_layers times — reduces memorizable
            # parameters while preserving iterative refinement depth.
            self.transformer = _SharedEncoder(
                enc_layer,
                num_layers,
                use_checkpoint=gradient_checkpointing,
                drop_rates=drop_rates,
            )
        elif drop_rates is not None:
            self.transformer = _StochasticDepthEncoder(enc_layer, num_layers, drop_rates)
        else:
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pre_head_norm = nn.LayerNorm(d_model)  # normalization before output heads

        # (4)+(5) Readout heads. Two mutually exclusive builds (A/B), selected by
        # use_query_decoder so a run can be reverted by flipping one config flag:
        #   A (default): temporal mean-pool feature -> pooled count/presence head
        #                + pooled positions MLP (all slots from one vector).
        #   B (presence-only): query decoder -> per-slot direction + presence
        #                heads (each slot predicts itself).
        if use_query_decoder:
            # --- Build B: query-decoder readout (presence-only, validated above). ---
            # Pooled-path heads are not built in this mode.
            self.n_sources_head = None
            self.presence_head = None
            self.positions_head = None
            # One learned query embedding per source slot. These carry the slot
            # identity (DETR-style); no sinusoidal PE is added on the query side
            # because PI matching makes the slots order-free.
            self.source_queries = nn.Parameter(torch.randn(max_sources, d_model) * 0.02)
            # Decoder has its own dropout so it can be tuned independently of the
            # (typically heavier) encoder dropout.
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
            # Per-slot heads: each query token predicts its own unit direction and
            # a single presence logit.
            self.slot_pos_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(d_model, 3),
            )
            self.slot_presence_head = nn.Linear(d_model, 1)
        else:
            # --- Build A: pooled-feature readout (baseline; all count modes). ---
            self.source_queries = None
            self.decoder = None
            self.slot_pos_head = None
            self.slot_presence_head = None

            # (4) HEAD 1: source count OR per-slot presence.
            #   - regression: single scalar logit (SmoothL1 target).
            #   - ordinal: max_sources logits; head j predicts P(k >= j+1) via BCE.
            #   - presence: max_sources per-slot presence logits (BCE against PI
            #               assignment). Count is derived downstream as
            #               sum_j 1{sigmoid(presence_j) > 0.5}.
            # count_head_hidden=False collapses the 2-layer MLP to a single Linear
            # (d_model -> count_out_dim), a remedy against count-head memorization
            # of per-sample k from the pooled feature. Ignored in presence mode.
            if count_loss_type == "presence":
                self.n_sources_head = None
                self.presence_head = nn.Linear(d_model, max_sources)
            else:
                count_out_dim = max_sources if count_loss_type == "ordinal" else 1
                if count_head_hidden:
                    self.n_sources_head = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model, count_out_dim),
                    )
                else:
                    self.n_sources_head = nn.Linear(d_model, count_out_dim)
                self.presence_head = None

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
          head1:     [B, 1] in "regression" mode, [B, max_sources] logits in "ordinal"
                     mode, [B, max_sources] per-slot presence logits in "presence" mode.
          positions: [B, max_sources, 3] unit vectors
        """
        # Convert to Transformer layout [B, T, C].
        x = x.permute(0, 2, 1)  # [B, T, C]

        # Project channels to d_model.
        x = self.proj(x)  # [B, T, d_model]

        # Add positional encoding.
        x = self.pos_enc(x)  # [B, T, d_model]

        # Transformer encoding (kept as a full sequence — no pooling yet).
        memory = self.transformer(x)  # [B, T, d_model]

        # --- Build B: query-decoder readout (presence-only). ---
        if self.use_query_decoder:
            B = memory.size(0)
            # Broadcast the learned slot queries across the batch.
            q = self.source_queries.unsqueeze(0).expand(B, -1, -1)  # [B, S, d_model]
            # Each slot cross-attends the full temporal sequence (and the slots
            # self-attend, letting them de-duplicate), replacing the mean-pool
            # single-vector bottleneck.
            slots = self.decoder(tgt=q, memory=memory)  # [B, S, d_model]
            slots = self.pre_head_norm(slots)  # LayerNorm over feature dim, per slot

            positions = self.slot_pos_head(slots)  # [B, S, 3]
            positions = positions / (positions.norm(dim=2, keepdim=True) + 1e-8)

            # Per-slot presence logits: [B, S].
            n_sources = self.slot_presence_head(slots).squeeze(-1)
            return n_sources, positions

        # --- Build A: temporal mean-pool + pooled heads (baseline). ---
        feats = memory.mean(dim=1)  # [B, d_model]
        feats = self.pre_head_norm(feats)

        # HEAD 1: source count OR per-slot presence. Output shape depends on
        # count_loss_type:
        #   "regression" -> [B, 1] scalar logit,
        #   "ordinal"    -> [B, max_sources] cumulative BCE logits,
        #   "presence"   -> [B, max_sources] per-slot presence logits.
        if self.count_loss_type == "presence":
            n_sources = self.presence_head(feats)
        else:
            n_sources = self.n_sources_head(feats)

        # HEAD 2: source positions.
        positions = self.positions_head(feats)  # [B, S * 3]
        positions = positions.view(-1, self.max_sources, 3)  # [B, S, 3]

        # Normalize to unit vectors on the sphere.
        positions = positions / (positions.norm(dim=2, keepdim=True) + 1e-8)

        return n_sources, positions
