"""Decoder: bottleneck + target-language tag → autoregressive token logits."""

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


class Decoder(nn.Module):
    """Autoregressive Transformer decoder, cross-attends to bottleneck.

    The bottleneck (B, M, D_code) is up-projected to d_model, then a learned
    target-language tag is prepended. Decoder input ids are causally
    self-attended and cross-attend to this memory.

    Output projection is tied to the input embedding table (frozen) so that
    decoded tokens live in the same vocabulary space.
    """

    def __init__(
        self,
        vocab_size: int,
        n_langs: int,
        d_model: int = 384,
        d_code: int = 256,
        n_dec_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 1,
        embedding_table: torch.Tensor | None = None,
        d_emb: int | None = None,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Phase 8 B: embedding is d_emb-dim (384, pinned by XLM-R). When d_model
        # differs, project up on input and back down before the tied output
        # matmul, so tying to the d_emb table survives a wider transformer.
        # d_emb == d_model (default) keeps the original path (no extra params).
        d_emb = d_emb or d_model
        self.d_emb = d_emb

        self.token_emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_token_id)
        if embedding_table is not None:
            assert embedding_table.shape == (vocab_size, d_emb), \
                f"embedding_table shape {embedding_table.shape} != ({vocab_size}, {d_emb})"
            self.token_emb.weight.data.copy_(embedding_table)
            self.token_emb.weight.requires_grad = False
        self.emb_proj = nn.Linear(d_emb, d_model) if d_emb != d_model else None
        self.out_proj = nn.Linear(d_model, d_emb) if d_emb != d_model else None
        self.pos = SinusoidalPositionalEmbedding(d_model)

        self.lang_emb = nn.Embedding(n_langs, d_model)
        self.bottleneck_proj = nn.Linear(d_code, d_model)

        # cs_attention: explicit-ops, cstorch-compatible, weight-compatible with
        # nn.TransformerDecoder(nn.TransformerDecoderLayer(norm_first, gelu)).
        # torch's fused attention does not lower on the CS-3 (see cs_attention.py).
        self.decoder = cs_attention.TransformerDecoder(
            lambda: cs_attention.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout),
            num_layers=n_dec_layers)
        self.out_norm = cs_attention.LayerNorm(d_model)  # explicit-ops LayerNorm

        # Output projection ties to embedding table (parameter sharing).
        # We keep a separate bias so the frozen emb table stays untouched.
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(
        self,
        bottleneck: torch.Tensor,         # (B, M, D_code)
        bottleneck_mask: torch.Tensor,    # (B, M) bool — True where valid (before <stop>)
        target_ids: torch.Tensor,         # (B, T) int64 (teacher-forced input)
        lang_id: torch.Tensor,            # (B,) int64 (target language index)
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        B, T = target_ids.shape
        device = target_ids.device

        # Project bottleneck to d_model and ADD the learned lang tag to every
        # memory slot (broadcast) instead of prepending it as a separate slot.
        # The old torch.cat([lang_token, mem]) was fine forward, but its BACKWARD
        # slices the (B, M+1, d) grad back into (B,1,d)+(B,M,d) — ws_km.slice on a
        # float/lane tensor, which the slice_filter kernel asserts on during size
        # estimation (crashing the compile). Broadcast-add carries the same lang
        # conditioning with no concat/slice. Memory length is now M (no lang slot).
        mem = self.bottleneck_proj(bottleneck)              # (B, M, d_model)
        lang = self.lang_emb(lang_id).unsqueeze(1)          # (B, 1, d_model)
        mem = mem + lang                                    # (B, M, d_model)
        mem_pad_mask = bottleneck_mask.to(torch.int32) == 0  # (B, M) True = pad

        # Decoder input: token embeddings + positional
        tgt_emb = self.token_emb(target_ids)                    # (B, T, d_emb)
        if self.emb_proj is not None:
            tgt_emb = self.emb_proj(tgt_emb)                    # (B, T, d_model)
        tgt_emb = self.pos(tgt_emb)
        tgt_pad_mask = target_ids == self.pad_token_id

        # Causal mask (True = disallowed, strictly above the diagonal). Built via
        # an arange compare rather than torch.triu: cstorch's triu decomposition
        # does arithmetic on the input and rejects a bool tensor. col > row is the
        # identical upper-triangular (excl. diagonal) mask in pure static ops.
        _rows = torch.arange(T, device=device).view(-1, 1)
        _cols = torch.arange(T, device=device).view(1, -1)
        causal_mask = _cols > _rows

        h = self.decoder(
            tgt=tgt_emb,
            memory=mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
        )
        h = self.out_norm(h)                                    # (B, T, d_model)
        if self.out_proj is not None:
            h = self.out_proj(h)                                # (B, T, d_emb)

        # Tied output projection. F.linear(h, W, b) == h @ W.t() + b, but avoids
        # an explicit .t() on the embedding weight — cstorch cannot transfer a
        # transposed/sliced weight parameter to the WSE (same class as the QKV
        # weight-split issue). Numerically identical. With d_emb != d_model, h has
        # been projected back to d_emb by out_proj above, so this matches W.
        logits = F.linear(h, self.token_emb.weight, self.out_bias)  # (B, T, V)
        return logits
