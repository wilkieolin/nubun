# Handoff: minimal-pair wafer runs (Phase 9 branch 2)

For a Claude Code session started on the ALCF Cerebras user node, in
`~/code/nubun`. It closes the loop that was being run by hand: compile, launch,
monitor, read out, and record. Background is PHASE9.md §8 ("Branch 2 status").
Last updated 2026-10-30, at commit `dbe05ee` or later.

## Where things stand

- **Code (all pushed, untested on the wafer):**
  - `cerebras/opus_input.py`: `--minpairs-dir`, `--pairs-per-batch P`. Rows
    (2i, 2i+1) for i < P are a minimal pair, with the same language and
    direction. `pair_sel` (B, P) maps each row to its pair, one-hot; unpaired
    rows are all 0.
  - `vqvae/pairs.py`: the level-0 slot-Hamming penalty
    `relu(ham - budget)`. The value is exact; the gradient comes from softmax
    assignments. It uses only sums and reshapes.
  - `cerebras/train_cstorch.py` flags: `--lambda-pair --pair-budget
    --pair-tau --pair-levels`. With them it logs `pair_hamming`.
  - `cerebras/runs/minpairs_pair.sh`: the runs. `compile` mode, or `mix` and
    `penalty` runs (B=128, lr 6e-4, 50k steps, seed 1, 8 pairs per batch,
    lambda 1.0, budget 2, level 0).
- **Data**, in the repo's own `data/`; `submit_alcf.sh` defaults
  `DATA_DIR=<repo>/data`. `~/code/nubun-copy` was deleted for quota.
  - `embedding_table.pt`, `token_weights.pt`, `parallel_corpus.npz`
  - `opus100_npy/`, unpacked with `<lang>_en.sem/` semantic targets
  - `minpairs_npy/`: the medium corpus, 32.5k pairs, 9 languages, 64 tokens
    max.
  - If `data/opus100/` (packed) is still present after unpacking, delete it
    to save quota.
- **Fixed on the way here:**
  - the minpairs scan now reads unpacked directories;
  - `pair_mask` is now (B,), because CS runs reject tensors whose leading
    dimension is not B; then replaced by `pair_sel` (B, P), below.
  - The penalty graph compiles as of 2026-10-30 (`logs/m2-20261030T220352Z.log`,
    about 40 min). Three compiler failures on the way, all in `vqvae/pairs.py`
    or the soft-assignment path:
    1. the reshape `(L, B, M, k) -> (L, B/2, 2, M, k)`: "failed to propagate
       named dimension"; the batch axis cannot be split. Pairs are now
       batch reductions weighted by `pair_sel` (`pair_hamming_loss_sel`);
    2. `F.relu` and compares on per-pair (P,) tensors: "Unable to insert transfer
       from WGT_HOST to WSE"; non-batch tensors land on GlobalHost. The pair
       sum is scattered back to rows, so everything after it is per row;
    3. backward of `amin` (via `gap = dist - dmin` in `quantizer.py`): "ReduceOp
       supports only f16/f32..." on a bool tie mask. `dmin` is detached, which
       is exact (a per-row shift under softmax).
    The pair term disables gradient accumulation ("will NOT be applied"): the
    penalty run takes B=128 in one pass, so its step time may differ from the
    baseline's. The `mix` graph is unchanged.

## Loop

1. `git pull`, then `bash cerebras/runs/minpairs_pair.sh compile`. It runs in
   the foreground for about 20 minutes; use tmux.
   - **Passes:** go to 2.
   - **Fails in the client** (a Python traceback before "Starting cstorch
     run" or in the data path): fix it, test the input function on the user
     node (see "Testing without the wafer"), commit, push, and retry.
   - **Fails in the compiler:** read `logs/m2-<stamp>.log` and
     `cerebras_logs/m2-<stamp>/`, and export logs with
     `bash cerebras/export_logs.sh <wsjob>`. The penalty is the only new
     graph. Suspects:
     - the `> 0.5` compare and the `.float()` cast in `pair_hamming_loss`;
     - the straight-through `ham + s - s.detach()`;
     - `F.softmax` over the 128 codes per level in `quantizer.py`, which
       runs when `pair_tau` is set;
     - the reshape `(L, B/2, 2, M, k)`.

     Keep to the wafer rules: static shapes, no int64, no argmin, no
     in-graph slicing or weight gather. CS3_BRINGUP.md lists what has failed
     before. To bisect, temporarily return `surrogate` alone in place of
     `ham_st`.
2. `bash cerebras/runs/minpairs_pair.sh` launches `mix` and `penalty`, one
   wafer each. They run detached, and the logs go to
   `logs/minpairs-{mix,penalty}-<stamp>.out`. Each takes about 5 h.
   `RUNS=mix` or `RUNS=penalty` launches just one. If only one wafer is
   free, run `penalty` first, since `mix` is the cheaper control.
3. Monitor with `csctl get jobs | grep nubun-mp` and the tail of the logs.
   Every 1000 steps a run logs loss, recon and `pair_hamming` (penalty run
   only). Things to check:
   - recon should track the seed-1 baseline: 4.21 final at 50k
     (`seed_pair.sh`, PHASE9 §7 "Cross-seed runs");
   - `pair_hamming` should settle near or below 2;
   - a NaN or a recon blow-up means stop (`csctl cancel job <id>`) and
     report.
4. After both runs finish, export each final checkpoint. Each run keeps
   only its newest `.mdl` (1.6 GB).

   ```bash
   python cerebras/export_ckpt.py model_dir/minpairs-<run>-seed1-<stamp>/nubun_cs_step50000.mdl \
       --embedding-table data/embedding_table.pt
   ```

   Readouts, each against the seed-1 baseline
   (`model_dir/seed1-b128-20261013T221048Z/nubun_cs_step50000.eval.pt`, if
   it is still on ALCF; otherwise seed 2 at 50k, below):
   - `python scripts/pair_hamming.py --checkpoint <eval.pt> --minpairs-dir data/minpairs_npy`.
     It reads packed or unpacked shards.
   - `python roundtrip_eval.py` (recon and round-trip quality, as for the earlier runs)
   - `python code_usage_audit.py`: distinct level-0 codes per sentence;
     seed 1 at 50k is 2.3 (PHASE9 §7).
   - `python codebook_stability.py --a <mix eval.pt> --b <penalty eval.pt>`
5. Record the results in PHASE9.md §8 (Branch 2 status, "Training"), then
   commit and push.

## How to read the result

Baseline, `seed2-b128` at 50k, no pairs in training, X side, medium pairs,
level 0:

| | mean | median | identical (0) | within 2 slots |
|---|---|---|---|---|
| negation pairs | 9.1 | 1 | 44% | 69% |
| tense pairs | 9.6 | 1 | 43% | 68% |
| random pairs | 62.9 | 64 | 0.2% | 1% |

At 50k, level 0 is still coarse (2.3 distinct codes per sentence), so most
pairs already sit within the budget. The penalty acts on about a third of
them.

The converged 500k model is very different:
- negation median 44 (25% within 2 slots);
- tense median 2;
- level 0 has diversified to 8.5 distinct codes per sentence.

So a 50k run mainly tests three things:
1. whether the penalty compiles and trains without hurting recon;
2. whether the tail of high-Hamming pairs collapses;
3. whether the penalty holds back level-0 diversification (distinct codes
   per sentence well below 2.3 would be a warning sign).

A level-0 Hamming of 0 is not the goal either: it means level 0 ignores the
factor. Report the share of pairs at exactly 0 alongside the share within 2.
Whether the penalty helps in the regime that matters needs a longer run, or
a fine-tune from the long run. That decision is the user's; ask before
launching anything over 50k.

## Testing without the wafer

The input function runs on the user node without a wafer. This is the check
that caught both earlier bugs:

```python
import sys, numpy as np; sys.path.insert(0, "cerebras")
from opus_input import make_opus_input_fn
fn = make_opus_input_fn("data/opus100_npy", 64, 128, 2, seed=1, with_semantic=True, pad=1,
     sem_dir="data/opus100_npy", parallel_corpus="data/parallel_corpus.npz",
     minpairs_dir="data/minpairs_npy", pairs_per_batch=8, with_pair_sel=True)
b = next(iter(fn()))
print({k: tuple(v.shape) for k, v in b.items()})   # every leading dim must be 128
```

`train_cstorch.py --synthetic-data` batches also include `pair_sel` when
`--lambda-pair > 0`. `bash cerebras/submit_alcf.sh smoke-m2 --lambda-pair 1.0
--pairs-per-batch 8 --compile-only` therefore compiles the penalty graph with
no real data, which helps separate a graph problem from a data problem.

## Ground rules

- Do not use `/eagle`.
- Never print full tokens or credentials; prefixes only.
- Keep within the home quota:
  - 1.6 GB per `.mdl`; prune old `model_dir/` runs only with the user's
    say-so;
  - `--keep-ckpts 1` is already set for these runs.
- Ask before launching runs beyond the two above, or before changing
  lambda, budget, steps or pairs per batch. Report what you would change and
  why.
- Commit with the attribution line the session gives you, then push.
