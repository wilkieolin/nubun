"""Decoder: bottleneck + target-language tag → autoregressive token logits."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from vqvae import cs_attention


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
        # Gather the first T rows with F.embedding rather than slicing the pe
        # buffer: slicing a GlobalHost buffer emits ws_waf.slice, which cstorch
        # cannot transfer to the WSE (same class as weight slicing). F.embedding
        # is the lowering-friendly gather (same op as token_emb). Values are
        # identical, and the (1, max_len, d) buffer shape is unchanged.
        pos = torch.arange(x.size(1), device=x.device)
        return x + F.embedding(pos, self.pe.squeeze(0)).unsqueeze(0)


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
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        if embedding_table is not None:
            self.token_emb.weight.data.copy_(embedding_table)
            self.token_emb.weight.requires_grad = False
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

        # Project bottleneck to d_model; prepend learned lang tag at memory position 0
        mem = self.bottleneck_proj(bottleneck)              # (B, M, d_model)
        lang_token = self.lang_emb(lang_id).unsqueeze(1)    # (B, 1, d_model)
        mem = torch.cat([lang_token, mem], dim=1)           # (B, M+1, d_model)
        # Lang token is always valid; bottleneck_mask appended after. cstorch's
        # concat rejects bool (i1) tensors ("integer dtype must be i32 or i16"),
        # so build validity in int32 and derive the pad mask by comparison.
        lang_valid = torch.ones(B, 1, dtype=torch.int32, device=device)
        mem_valid = torch.cat([lang_valid, bottleneck_mask.to(torch.int32)], dim=1)  # (B, M+1)
        mem_pad_mask = mem_valid == 0                                # True = pad

        # Decoder input: token embeddings + positional
        tgt_emb = self.token_emb(target_ids)
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
        h = self.out_norm(h)

        # Tied output projection. F.linear(h, W, b) == h @ W.t() + b, but avoids
        # an explicit .t() on the embedding weight — cstorch cannot transfer a
        # transposed/sliced weight parameter to the WSE (same class as the QKV
        # weight-split issue). Numerically identical.
        logits = F.linear(h, self.token_emb.weight, self.out_bias)  # (B, T, V)
        return logits
