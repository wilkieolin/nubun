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

Two forms of the pairing: pair_hamming_loss reshapes the batch axis to
(B/2, 2), for GPU batches where every row pair is a minimal pair.
pair_hamming_loss_partner compares each row with its partner encoded in a second
pass, for CS-3 runs, where no op may mix rows across the batch: splitting the
batch axis, a weighted batch sum and a matmul over the batch all fail
(HANDOFF_MINPAIRS.md).
"""
import torch
from torch.nn import functional as F


def _agreement(assign):
    """(L, B, M, k) -> (L, B/2, M): sum_k a_k b_k for rows 2i and 2i+1.
    a.b = ((a + b)^2 - a^2 - b^2) / 2, summed over k: only sums over the pair axis."""
    L, B, M, k = assign.shape
    x = assign.reshape(L, B // 2, 2, M, k)
    return ((x.sum(dim=2) ** 2) - (x ** 2).sum(dim=2)).sum(dim=-1) * 0.5


def _penalty(same_hard, same_soft, mask, budget, grad_scale=1.0):
    """same_*: (L, N, M) code agreement for N pairs (or rows); mask: (N,) float."""
    level_diffs = (1.0 - same_hard).sum(dim=0)             # (N, M)
    ham = (level_diffs > 0.5).float().sum(dim=-1)          # (N,) slots with any level differing
    surrogate = (1.0 - same_soft).mean(dim=0).sum(dim=-1)  # (N,)
    ham_st = ham + grad_scale * (surrogate - surrogate.detach())
    n = mask.sum().clamp(min=1.0)
    penalty = (F.relu(ham_st - budget) * mask).sum() / n
    return penalty, (ham * mask).sum() / n


def pair_hamming_loss_partner(hard, soft, hard_p, soft_p, pair_mask, budget):
    """hard, soft: (L, B, M, k) for the batch; hard_p, soft_p: the same for each
    row's partner, from a detached second pass over partner_src_ids; pair_mask:
    (B,) float, 1 on paired rows. Per row, so each pair counts twice: the same
    mean as pair_hamming_loss. With the partner detached, each row's surrogate
    gives half the pair gradient (d(a.b')/da = b over 2n rows, against b over n
    pairs), so grad_scale=2 makes it exact."""
    same_hard = (hard.float() * hard_p.float().detach()).sum(dim=-1)   # (L, B, M)
    same_soft = (soft.float() * soft_p.float().detach()).sum(dim=-1)
    return _penalty(same_hard, same_soft, pair_mask.float(), budget, grad_scale=2.0)


def pair_hamming_loss(hard, soft, pair_mask, budget):
    """hard, soft: (L, B, M, k) from ResidualVectorQuantizer with pair_tau set;
    pair_mask: (B,) float per row, 1 on both rows 2i, 2i+1 of a minimal pair
    (reduced to per pair by reshape + mean; GPU only, see the module docstring).
    Returns (penalty scalar, mean slot Hamming over the masked pairs)."""
    B = hard.shape[1]
    mask = pair_mask.float().reshape(B // 2, 2).mean(dim=-1)   # (P,), no slicing
    return _penalty(_agreement(hard.float()), _agreement(soft.float()), mask, budget)
