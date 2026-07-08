"""Loss helpers for VQ-VAE training."""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    logits: torch.Tensor,        # (B, T, V)
    targets: torch.Tensor,        # (B, T)  shifted target ids
    pad_token_id: int,
    token_weight: torch.Tensor | None = None,  # (V,) per-token loss weight (P5b)
) -> torch.Tensor:
    """Cross-entropy over target tokens.

    If token_weight is given (Phase 5b), each target token's loss is scaled by
    token_weight[target], then averaged as a weighted mean over non-pad
    positions. Downweighting frequent punctuation/function tokens stops them
    from dominating the gradient, so the bottleneck must encode content.
    """
    logits2d = logits.reshape(-1, logits.size(-1))
    targets1d = targets.reshape(-1)
    if token_weight is None:
        return F.cross_entropy(logits2d, targets1d, ignore_index=pad_token_id)

    ce = F.cross_entropy(logits2d, targets1d, ignore_index=pad_token_id,
                         reduction="none")               # (N,), 0 at pad
    w = token_weight[targets1d]                          # (N,)
    w = w * (targets1d != pad_token_id).to(w.dtype)      # exclude pad from denom
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def length_penalty(
    indices: torch.Tensor,       # (B, M)
    stop_index: int = 0,
    m_max: int | None = None,
) -> torch.Tensor:
    """Mean fractional length used: 0 = stop at slot 0, 1 = no stop emitted.

    Used to push the encoder toward shorter encodings. Scaled to [0, 1].
    """
    B, M = indices.shape
    if m_max is None:
        m_max = M
    is_stop = indices == stop_index                         # (B, M)
    # First-stop position per row (or M if no stop)
    has_any_stop = is_stop.any(dim=1)                       # (B,)
    # argmax on bool returns the first True index, or 0 if all-False — we need M for "no stop"
    first_stop = torch.where(
        has_any_stop, is_stop.float().argmax(dim=1), torch.full_like(has_any_stop, M, dtype=torch.long))
    # Cast to float for division
    return (first_stop.float() / m_max).mean()


def semantic_loss(
    sem_pred: torch.Tensor,      # (B, D_sem) predicted from pooled bottleneck
    target_emb: torch.Tensor,    # (B, D_sem) frozen sentence embedding (detached)
) -> torch.Tensor:
    """Cosine-distance loss pushing the pooled bottleneck toward the frozen
    multilingual sentence embedding of the source. 0 = identical direction,
    2 = opposite. Both sides are L2-normalized by cosine_similarity internally.

    Because the target encoder (MiniLM) aligns translations across languages,
    a sentence and its translation share ~the same target — so this pressures
    the codebook toward language-independent *meaning*, not token boilerplate.
    """
    cos = F.cosine_similarity(sem_pred.float(), target_emb.float(), dim=-1)
    return (1.0 - cos).mean()


def usage_entropy(
    usage: torch.Tensor,         # (K,) per-step counts (or running average)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Empirical entropy of code usage. Higher = more balanced. Negate for a loss."""
    p = usage / (usage.sum() + eps)
    return -(p * (p + eps).log()).sum()
