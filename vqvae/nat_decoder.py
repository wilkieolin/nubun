"""Non-autoregressive decoder (Phase 6).

The AR decoder can route around the bottleneck: given the target-language tag it
is a competent LM and minimizes token-CE by fluent unconditional generation, so
the codes become optional (measured posterior collapse). This NAT decoder removes
that escape hatch — there is NO teacher-forced target input and NO causal mask, so
every output token must be explained by the codes.

Design (length-conditioned parallel prediction, the simplest NAT first cut):
  - query positions = sinusoidal position embeddings, length = target length
    (gold at train, length_head prediction at inference)
  - a learned target-language tag is added to every query position
  - full (non-causal) self-attention across positions + cross-attention to the
    bottleneck memory (lang tag prepended, same as the AR decoder)
  - position-wise CE against gold tokens under a monotonic-alignment assumption

Upgrade path if monotonic alignment hurts: CTC / latent-alignment loss (emit a
longer position grid and marginalize alignments). Documented, not implemented here.

Interface mirrors Decoder so model.py can route to either:
    logits = nat_decoder(bottleneck, bottleneck_mask, out_len, lang_id)
"""

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

    def forward(self, length: int, device) -> torch.Tensor:
        return self.pe[:, :length].to(device)          # (1, length, d_model)


class NATDecoder(nn.Module):
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
        max_len: int = 128,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.max_len = max_len

        # Output projection tied to the (frozen) embedding table, matching the AR
        # decoder so both arms share the same vocabulary geometry.
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        if embedding_table is not None:
            self.token_emb.weight.data.copy_(embedding_table)
            self.token_emb.weight.requires_grad = False
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=max(max_len, 4096))

        self.lang_emb = nn.Embedding(n_langs, d_model)
        self.bottleneck_proj = nn.Linear(d_code, d_model)

        # No causal mask is ever applied -> this stack is a cross-attention decoder
        # over a fixed position grid; self-attention is bidirectional.
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(
        self,
        bottleneck: torch.Tensor,        # (B, M, D_code)
        bottleneck_mask: torch.Tensor,   # (B, M) bool, True = valid
        out_len: torch.Tensor,           # (B,) int64 target lengths
        lang_id: torch.Tensor,           # (B,) int64
    ) -> torch.Tensor:
        """Returns logits (B, T, V) where T = out_len.max(). Positions >= out_len[i]
        are still produced but should be masked by the caller in the loss."""
        B = bottleneck.size(0)
        device = bottleneck.device
        T = int(out_len.max().clamp(min=1, max=self.max_len))

        # Memory = lang tag + projected bottleneck (same layout as the AR decoder)
        mem = self.bottleneck_proj(bottleneck)
        lang_token = self.lang_emb(lang_id).unsqueeze(1)
        mem = torch.cat([lang_token, mem], dim=1)
        # int32 concat (cstorch's concat rejects bool/i1), then derive pad mask.
        lang_valid = torch.ones(B, 1, dtype=torch.int32, device=device)
        mem_valid = torch.cat([lang_valid, bottleneck_mask.to(torch.int32)], dim=1)
        mem_pad_mask = mem_valid == 0

        # Query grid: position embeddings + broadcast lang tag; no token content.
        q = self.pos(T, device).expand(B, -1, -1).clone()      # (B, T, d_model)
        q = q + self.lang_emb(lang_id).unsqueeze(1)

        # Mask query positions beyond each example's target length.
        ar = torch.arange(T, device=device).unsqueeze(0)       # (1, T)
        q_pad_mask = ar >= out_len.unsqueeze(1)                 # (B, T) True = pad

        h = self.decoder(
            tgt=q,
            memory=mem,
            tgt_mask=None,                                      # NON-causal
            tgt_key_padding_mask=q_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
        )
        h = self.out_norm(h)
        logits = h @ self.token_emb.weight.t() + self.out_bias
        return logits
