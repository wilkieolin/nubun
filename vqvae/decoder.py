"""Decoder: bottleneck + target-language tag → autoregressive token logits."""

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

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.out_norm = nn.LayerNorm(d_model)

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
        # Lang token is always valid; bottleneck_mask appended after
        lang_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        mem_valid = torch.cat([lang_mask, bottleneck_mask], dim=1)  # (B, M+1)
        mem_pad_mask = ~mem_valid                                    # nn expects "True = pad"

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

        # tgt_is_causal=True tells nn.TransformerDecoder our tgt_mask is causal,
        # so it skips _detect_is_causal_mask — whose internal torch.triu(bool)
        # does not lower on cstorch. The mask is genuinely causal, so this is a
        # correct hint and leaves GB10 behavior unchanged.
        h = self.decoder(
            tgt=tgt_emb,
            memory=mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
            tgt_is_causal=True,
        )
        h = self.out_norm(h)

        # Tied output projection
        logits = h @ self.token_emb.weight.t() + self.out_bias  # (B, T, V)
        return logits
