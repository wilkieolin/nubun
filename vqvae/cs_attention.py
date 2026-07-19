"""cstorch-compatible attention — explicit ops, weight-compatible with torch's.

PyTorch's nn.MultiheadAttention / nn.TransformerEncoder / nn.TransformerDecoder
route through the fused scaled_dot_product_attention, which does NOT lower on the
Cerebras CS-3 (the compile fails with an internal assertion in ws_waf.aamatmul,
or the whole graph lowers to an empty CIRH module). These modules recompute the
identical math with explicit ops (Q/K/V linears, softmax(QKᵀ/√d)·V, out proj),
which lowers cleanly (verified on the wafer via cerebras/minimal_probe.py
--model attn2).

They are drop-in and **state-dict compatible** with the torch modules they
replace — same submodule/parameter names and shapes:
  MultiheadAttention: in_proj_weight (3E,E), in_proj_bias (3E,), out_proj.{weight,bias}
  TransformerEncoderLayer: self_attn.*, linear1.*, linear2.*, norm1.*, norm2.*
  TransformerDecoderLayer: self_attn.*, multihead_attn.*, linear1/2.*, norm1/2/3.*
  Transformer{Encoder,Decoder}: layers.{i}.*
so existing GB10 checkpoints load unchanged and the GB10 (CUDA) path stays
numerically equivalent (masking uses a large finite negative instead of -inf;
identical up to floating-point noise, and eval runs with dropout off).

Only batch_first=True, norm_first=True, GELU — the exact config Nubun uses.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

# Large finite negative for additive masking. Pure arithmetic (no masked_fill /
# no -inf), so it lowers on cstorch and never produces NaNs on fully-valid rows.
# exp(-1e4) underflows to 0 in fp16 and fp32 alike — same effect as -inf here.
_MASK_NEG = -1e4


class LayerNorm(nn.Module):
    """Explicit-ops LayerNorm, weight-compatible with nn.LayerNorm (weight, bias).

    cstorch's fused LayerNorm kernel fails to lower ("Failed to convert op to
    WAF") when the tensor's non-batch dim is weight-shaped rather than a streamed
    activation dim (e.g. the Perceiver readout's M learned-query slots). Computing
    mean/var/normalize with primitive ops sidesteps the specialized kernel and is
    numerically identical (biased variance, matching nn.LayerNorm)."""

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        centered = x - mu
        var = centered.pow(2).mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(var + self.eps) * self.weight + self.bias


class MultiheadAttention(nn.Module):
    """Explicit-ops replacement for nn.MultiheadAttention (batch_first, packed
    in_proj). Returns just the attention output (Nubun never uses the weights)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        # Names/shapes match nn.MultiheadAttention exactly (packed QKV).
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # -> out_proj.{weight,bias}
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """query (B,Lq,E), key/value (B,Lk,E).
        key_padding_mask (B,Lk) bool: True = ignore (pad).
        attn_mask (Lq,Lk) bool: True = disallowed (e.g. causal upper triangle)."""
        E, H, dh = self.embed_dim, self.num_heads, self.head_dim
        # Project with the FULL packed in_proj, then slice the ACTIVATION outputs.
        # Splitting the (3E,E) weight parameter emits aten::split_copy on a
        # WGT_HOST tensor, which cstorch cannot transfer to the WSE; slicing the
        # projected activation (on the WSE) is fine. F.linear(x, W)[..., :E] ==
        # F.linear(x, W[:E]), so this is numerically identical to a QKV split.
        if query is key and key is value:                 # self-attention
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv[..., :E], qkv[..., E:2 * E], qkv[..., 2 * E:]
        else:                                             # cross-attention
            q = F.linear(query, self.in_proj_weight, self.in_proj_bias)[..., :E]
            kv = F.linear(key, self.in_proj_weight, self.in_proj_bias)
            k, v = kv[..., E:2 * E], kv[..., 2 * E:]

        B, Lq, _ = q.shape
        Lk = k.shape[1]
        # (B, H, L, dh)
        q = q.view(B, Lq, H, dh).transpose(1, 2)
        k = k.view(B, Lk, H, dh).transpose(1, 2)
        v = v.view(B, Lk, H, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)  # (B,H,Lq,Lk)
        if attn_mask is not None:
            scores = scores + attn_mask.view(1, 1, Lq, Lk).to(scores.dtype) * _MASK_NEG
        if key_padding_mask is not None:
            scores = scores + key_padding_mask.view(B, 1, 1, Lk).to(scores.dtype) * _MASK_NEG

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)                       # (B,H,Lq,dh)
        out = out.transpose(1, 2).reshape(B, Lq, E)       # (B,Lq,E)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """norm_first, GELU — matches nn.TransformerEncoderLayer's params."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        a = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                           key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        x = x + F.dropout(a, p=self.dropout, training=self.training)
        h = self.norm2(x)
        h = self.linear2(F.dropout(F.gelu(self.linear1(h)), p=self.dropout,
                                   training=self.training))
        x = x + F.dropout(h, p=self.dropout, training=self.training)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_factory, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_factory() for _ in range(num_layers)])

    def forward(self, src, src_key_padding_mask=None, src_mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """norm_first, GELU — matches nn.TransformerDecoderLayer's params."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        x = tgt
        s = self.norm1(x)
        a = self.self_attn(s, s, s, key_padding_mask=tgt_key_padding_mask,
                           attn_mask=tgt_mask)
        x = x + F.dropout(a, p=self.dropout, training=self.training)
        c = self.norm2(x)
        ca = self.multihead_attn(c, memory, memory,
                                 key_padding_mask=memory_key_padding_mask)
        x = x + F.dropout(ca, p=self.dropout, training=self.training)
        h = self.norm3(x)
        h = self.linear2(F.dropout(F.gelu(self.linear1(h)), p=self.dropout,
                                   training=self.training))
        x = x + F.dropout(h, p=self.dropout, training=self.training)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer_factory, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer_factory() for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return x
