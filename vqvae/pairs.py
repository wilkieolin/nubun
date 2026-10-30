"""Minimal-pair Hamming penalty (PHASE9 §8 branch 2).

A batch carries minimal pairs in consecutive rows (2i, 2i+1): the same sentence
with one factor changed (negation or tense; augment/minpairs.py). The two
members' code stacks should differ in at most `budget` slots, so the changed
factor is pushed into one radical (the discrete form of Locatello 2020's
shared-factor aggregation).

Depth: the quantiser returns levels 0..pair_levels-1 (default here: level 0
only). In the converged seed-2 model, pair members already differ in a median
of 2 level-0 slots for tense (random sentences: 62), but in 63 of 64 slots
once all 8 levels count, as unrelated sentences do: the deep levels carry
surface detail, so a small budget on the full stack would fight reconstruction.

Value: the exact number of slots whose stack (over those levels) differs, from the hard
codes, minus the budget, floored at 0, averaged over the pairs in the batch.
Gradient: from a soft surrogate, the per-level disagreement of the softmax
assignments summed over slots (straight-through: forward = hard count). Written
with reshapes and sums only (no slicing, argmin, prod or int64), which the CS-3
compiler accepts in the quantiser already.
"""
import torch
from torch.nn import functional as F


def _agreement(assign):
    """(L, B, M, k) -> (L, B/2, M): sum_k a_k b_k for rows 2i and 2i+1.
    a.b = ((a + b)^2 - a^2 - b^2) / 2, summed over k: only sums over the pair axis."""
    L, B, M, k = assign.shape
    x = assign.reshape(L, B // 2, 2, M, k)
    return ((x.sum(dim=2) ** 2) - (x ** 2).sum(dim=2)).sum(dim=-1) * 0.5


def pair_hamming_loss(hard, soft, pair_mask, budget):
    """hard, soft: (L, B, M, k) from ResidualVectorQuantizer with pair_tau set;
    pair_mask: (B,) float per row, 1 on both rows 2i, 2i+1 of a minimal pair
    (batch-major, as CS runs require; reduced to per pair by reshape + mean).
    Returns (penalty scalar, mean slot Hamming over the masked pairs)."""
    L, B, M, _ = hard.shape
    same_level = _agreement(hard.float())                  # (L, P, M), 1 = same code
    level_diffs = (1.0 - same_level).sum(dim=0)            # (P, M)
    ham = (level_diffs > 0.5).float().sum(dim=-1)          # (P,) slots with any level differing
    surrogate = (1.0 - _agreement(soft.float())).mean(dim=0).sum(dim=-1)   # (P,)
    ham_st = ham + surrogate - surrogate.detach()
    mask = pair_mask.float().reshape(B // 2, 2).mean(dim=-1)   # (P,), no slicing
    n = mask.sum().clamp(min=1.0)
    penalty = (F.relu(ham_st - budget) * mask).sum() / n
    return penalty, (ham * mask).sum() / n
