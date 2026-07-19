"""Decoder: bottleneck + target-language tag → autoregressive token logits."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from vqvae import cs_attention


class SinusoidalPositionalEmbedding(nn.Module):
    """Adds sinusoidal positions to (B, T, d).

    See the twin in vqvae/encoder.py for why the table must be sized EXACTLY to
    T on the CS-3: every in-graph row select (slice / embedding-gather / one-hot
    matmul) either asserts in slice_filter, has no valid layout, or lands on
    GlobalHost with no matmul kernel. max_len == T reduces it to a broadcast-add.
    """

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
        T = x.size(1)
        assert T <= self.pe.size(0), \
            f"seq_len {T} exceeds positional table {self.pe.size(0)}"
        if T == self.pe.size(0):
            return x + self.pe.unsqueeze(0)      # (1,T,d) bcast — the WSE path
        return x + self.pe[:T].unsqueeze(0)      # off-wafer only (in-graph slice)


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
        seq_len: int | None = None,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        if embedding_table is not None:
            self.token_emb.weight.data.copy_(embedding_table)
            self.token_emb.weight.requires_grad = False
        # seq_len is the DECODER INPUT length (tgt_len-1 under pre_shifted), not
        # the raw target length. See SinusoidalPositionalEmbedding.
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=seq_len or 512)

        # Causal mask as a constant WEIGHT rather than an in-graph compare.
        # Both operands of `cols > rows` are aranges, so the mask is fully
        # data-independent; on the CS-3 the compiler then tries to stream it from
        # the host as an i64 activation, and i64 activations have no valid layout
        # ("SelectLanesAndLayout failed: StreamFromActHost has no valid layouts").
        # Materializing it here puts it in weight memory instead. Stored as float
        # because cs_attention consumes it as `mask * _MASK_NEG` and cast it to
        # scores.dtype anyway — so this is numerically identical to the bool mask.
        if seq_len is not None:
            _r = torch.arange(seq_len).view(-1, 1)
            _c = torch.arange(seq_len).view(1, -1)
            self.register_buffer("causal_mask", (_c > _r).float(), persistent=False)
        else:
            self.causal_mask = None  # off-wafer: built per-forward from T

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
        tgt_emb = self.token_emb(target_ids)
        tgt_emb = self.pos(tgt_emb)
        tgt_pad_mask = target_ids == self.pad_token_id

        # Causal mask (disallowed = strictly above the diagonal). Precomputed as a
        # float weight buffer when seq_len is known (required on the CS-3 — see
        # __init__). Off-wafer it is built here from an arange compare rather than
        # torch.triu, whose cstorch decomposition does arithmetic on the input and
        # rejects a bool tensor.
        if self.causal_mask is not None:
            assert T == self.causal_mask.size(0), \
                f"decoder input T={T} != seq_len {self.causal_mask.size(0)}"
            causal_mask = self.causal_mask
        else:
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
        h = self.out_norm(h)

        # Tied output projection. F.linear(h, W, b) == h @ W.t() + b, but avoids
        # an explicit .t() on the embedding weight — cstorch cannot transfer a
        # transposed/sliced weight parameter to the WSE (same class as the QKV
        # weight-split issue). Numerically identical.
        logits = F.linear(h, self.token_emb.weight, self.out_bias)  # (B, T, V)
        return logits
