"""Encoder: token sequence → fixed M_max bottleneck slots."""

import math

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # Perceiver-style readout: M_max learned queries cross-attend to encoded tokens
        self.readout_queries = nn.Parameter(torch.randn(m_max, d_model) * 0.02)
        self.readout_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.readout_norm_q = nn.LayerNorm(d_model)
        self.readout_norm_kv = nn.LayerNorm(d_model)
        self.readout_ff = nn.Sequential(
            nn.LayerNorm(d_model),
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

        # Perceiver readout — M queries attend to T tokens
        q = self.readout_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_model)
        q_norm = self.readout_norm_q(q)
        h_norm = self.readout_norm_kv(h)
        attn_out, _ = self.readout_attn(
            q_norm, h_norm, h_norm, key_padding_mask=pad_mask, need_weights=False)
        z = q + attn_out
        z = z + self.readout_ff(z)

        return self.proj(z)  # (B, M_max, d_code)
