"""Encoder: token sequence → fixed M_max bottleneck slots."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from vqvae import cs_attention


class SinusoidalPositionalEmbedding(nn.Module):
    # max_len only needs to cover the training seq_len (<=64); kept small so the
    # one-hot select matmul below stays cheap.
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)  # (max_len, d), constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Select rows 0..T-1 of the pe table via a one-hot matmul. Alternatives
        # both fail on the WSE: pe[:, :T] is a buffer slice (slice_filter kernel
        # asserts), and F.embedding(arange, pe) streams the arange as an i64 index
        # (act_host_to_wse) that has no valid layout. onehot @ pe == pe[:T], using
        # arange only in a compare (the RVQ one-hot pattern that lowers) + a matmul.
        T = x.size(1)
        rows = torch.arange(T, device=x.device).unsqueeze(1)              # (T,1)
        cols = torch.arange(self.pe.size(0), device=x.device).unsqueeze(0)  # (1,max_len)
        onehot = (rows == cols).to(x.dtype)                              # (T,max_len)
        return x + (onehot @ self.pe).unsqueeze(0)                       # (1,T,d) bcast


class Encoder(nn.Module):
    """Token IDs → (B, M_max, D_code) pre-quantization vectors.

    Architecture:
      - input embedding lookup (frozen, shared with decoder output)
      - sinusoidal positional embedding
      - bidirectional Transformer encoder (n_enc_layers, d_model)
      - Perceiver-style readout: M_max learned queries cross-attend to encoded tokens
      - down-projection d_model → d_code
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        d_code: int = 256,
        n_enc_layers: int = 4,
        n_heads: int = 6,
        d_ff: int = 1024,
        m_max: int = 64,
        dropout: float = 0.1,
        pad_token_id: int = 1,
        embedding_table: torch.Tensor | None = None,
        d_emb: int | None = None,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.m_max = m_max
        self.d_model = d_model
        # Phase 8 B: the token embedding lives in d_emb dims (pinned to 384 by the
        # pretrained XLM-R table). When d_model != d_emb, project up after lookup
        # so the transformer can be wider than the embedding. d_emb == d_model
        # (default) keeps the original path exactly (no projection params).
        d_emb = d_emb or d_model
        self.d_emb = d_emb

        self.token_emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_token_id)
        if embedding_table is not None:
            assert embedding_table.shape == (vocab_size, d_emb), \
                f"embedding_table shape {embedding_table.shape} != ({vocab_size}, {d_emb})"
            self.token_emb.weight.data.copy_(embedding_table)
            self.token_emb.weight.requires_grad = False  # frozen
        self.emb_proj = nn.Linear(d_emb, d_model) if d_emb != d_model else None
        self.pos = SinusoidalPositionalEmbedding(d_model)

        # cs_attention: explicit-ops, cstorch-compatible, weight-compatible with
        # nn.TransformerEncoder(nn.TransformerEncoderLayer(norm_first, gelu)).
        # torch's fused attention does not lower on the CS-3 (see cs_attention.py).
        self.encoder = cs_attention.TransformerEncoder(
            lambda: cs_attention.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout),
            num_layers=n_enc_layers)

        # Perceiver-style readout: M_max learned queries cross-attend to encoded tokens
        self.readout_queries = nn.Parameter(torch.randn(m_max, d_model) * 0.02)
        self.readout_attn = cs_attention.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        # cs_attention.LayerNorm: explicit-ops LayerNorm — cstorch's fused kernel
        # fails on the readout's M (learned-query) dimension (see cs_attention.py).
        self.readout_norm_q = cs_attention.LayerNorm(d_model)
        self.readout_norm_kv = cs_attention.LayerNorm(d_model)
        self.readout_ff = nn.Sequential(
            cs_attention.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.proj = nn.Linear(d_model, d_code)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T_in) int64. Returns (B, M_max, D_code)."""
        B, T = token_ids.shape
        pad_mask = token_ids == self.pad_token_id  # (B, T) — True where padding

        x = self.token_emb(token_ids)             # (B, T, d_emb)
        if self.emb_proj is not None:
            x = self.emb_proj(x)                  # (B, T, d_model)
        x = self.pos(x)
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        # Perceiver readout — M learned queries cross-attend to encoded tokens.
        # The queries must live on the activation path: the WSE streams samples
        # through its tiles, and cstorch has no wafer LayerNorm kernel for a
        # purely weight-derived tensor (the learned readout_queries alone, with
        # no batch/data dependence). Seed them with the per-batch encoder summary
        # so q is a genuine activation; readout_norm_q renormalizes each query, so
        # this conditions (rather than corrupts) the queries.
        # NOTE: this diverges from the pre-port readout — reproduce the frozen
        # GB10 model with pre-port code (see PORT_CS3 §8).
        q = self.readout_queries.unsqueeze(0) + h.mean(dim=1, keepdim=True)  # (B, M, d_model)
        q_norm = self.readout_norm_q(q)
        h_norm = self.readout_norm_kv(h)
        attn_out = self.readout_attn(
            q_norm, h_norm, h_norm, key_padding_mask=pad_mask)
        z = q + attn_out
        z = z + self.readout_ff(z)

        return self.proj(z)  # (B, M_max, d_code)
