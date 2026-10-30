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
pair_hamming_loss_sel takes a (B, P) one-hot row-to-pair map and sums over the
batch with it, keeping the result per row: the CS-3 compiler rejects splitting
the batch axis ("failed to propagate named dimension" on the (L, B, M, k) ->
(L, B/2, 2, M, k) reshape).
"""
import torch
from torch.nn import functional as F


def _agreement(assign):
    """(L, B, M, k) -> (L, B/2, M): sum_k a_k b_k for rows 2i and 2i+1.
    a.b = ((a + b)^2 - a^2 - b^2) / 2, summed over k: only sums over the pair axis."""
    L, B, M, k = assign.shape
    x = assign.reshape(L, B // 2, 2, M, k)
    return ((x.sum(dim=2) ** 2) - (x ** 2).sum(dim=2)).sum(dim=-1) * 0.5


def _agreement_sel(assign, sel):
    """(L, B, M, k), sel (B, P) -> (L, B, M): each row's agreement with its pair
    partner, batch-major. The pair sum (L, P, M, k) is a batch reduction weighted
    by sel, scattered back to the rows by sel; partner = pair sum - own row.
    Rows outside a pair get garbage, masked by the caller. Everything after the
    pair sum stays batch-major: compares on non-batch tensors land on GlobalHost
    ("Unable to insert transfer from WGT_HOST to WSE" on the relu)."""
    w = sel[None, :, :, None, None]                        # (1, B, P, 1, 1)
    pair_sum = (w * assign.unsqueeze(2)).sum(dim=1)        # (L, P, M, k)
    row_sum = (w * pair_sum.unsqueeze(1)).sum(dim=2)       # (L, B, M, k)
    return (assign * (row_sum - assign)).sum(dim=-1)


def _penalty(same_hard, same_soft, mask, budget):
    """same_*: (L, N, M) code agreement for N pairs (or rows); mask: (N,) float."""
    level_diffs = (1.0 - same_hard).sum(dim=0)             # (N, M)
    ham = (level_diffs > 0.5).float().sum(dim=-1)          # (N,) slots with any level differing
    surrogate = (1.0 - same_soft).mean(dim=0).sum(dim=-1)  # (N,)
    ham_st = ham + surrogate - surrogate.detach()
    n = mask.sum().clamp(min=1.0)
    penalty = (F.relu(ham_st - budget) * mask).sum() / n
    return penalty, (ham * mask).sum() / n


def pair_hamming_loss_sel(hard, soft, pair_sel, budget):
    """hard, soft: (L, B, M, k) as below; pair_sel: (B, P) float, 1 at
    (2i, i) and (2i+1, i) for pair i, rows outside a pair all 0 (CS runs: batch-major).
    Per row against its partner, so each pair counts twice: same mean, same gradient."""
    sel = pair_sel.float()
    mask = sel.sum(dim=1)                                  # (B,), 1 on paired rows
    return _penalty(_agreement_sel(hard.float(), sel),
                    _agreement_sel(soft.float(), sel), mask, budget)


def pair_hamming_loss(hard, soft, pair_mask, budget):
    """hard, soft: (L, B, M, k) from ResidualVectorQuantizer with pair_tau set;
    pair_mask: (B,) float per row, 1 on both rows 2i, 2i+1 of a minimal pair
    (reduced to per pair by reshape + mean; GPU only, see the module docstring).
    Returns (penalty scalar, mean slot Hamming over the masked pairs)."""
    B = hard.shape[1]
    mask = pair_mask.float().reshape(B // 2, 2).mean(dim=-1)   # (P,), no slicing
    return _penalty(_agreement(hard.float()), _agreement(soft.float()), mask, budget)
