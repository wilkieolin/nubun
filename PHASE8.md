# Phase 8 — Scale-Up: Closing the Quantization Gap

Goal: raise round-trip reconstruction past the Phase 7 mark and produce more
robust, interpretable radicals — then freeze the architecture and scope a port
to alternate hardware (Cerebras CS-3).

All round-trip numbers below are **free-generation cross-lingual meaning** on
one **consistent seeded 200-sentence split** (`roundtrip_eval.py --combine-splits
--langs en,zh,es,fr,ar,ru --max-gen-len 48`), so they are directly comparable.
Gold (translation) ceiling on this split = **0.861**.

## Results

| model | real cross | notes |
|-------|-----------|-------|
| gold (translation ceiling) | 0.861 | upper bound |
| **E3 @ 100k — FROZEN WINNER** | **0.435** | RVQ 8×128, tied emb, deep decoder |
| shallow continuous ceiling (no-VQ) | 0.430 | Phase 6 unfrozen+cosine |
| E2 — RVQ 8×128, tied, deep | 0.417 | cooler-LR 100k schedule |
| E1 — RVQ 4×128, tied, deep | 0.315 | decoder capacity only |
| baseline — RVQ 4×128 (Phase 7) | 0.305 | two separate emb tables |
| frozen no-VQ | 0.184 | frozen-embedding lower bound |

**Sequence outcome: 0.305 → 0.435, +43% over baseline**, reaching >97% of the
architecture's continuous ceiling. E3@100k even edges past the *shallow*
continuous ceiling (0.430), because the deep decoder raised the continuous
ceiling itself.

## What moved the needle (and what didn't)

- **E1 (decoder capacity): marginal, +0.010.** Tying encoder/decoder embeddings
  (211.5M → 135.2M params) plus a 2× deeper/wider decoder barely helped free
  generation, even though every *teacher-forced* metric improved. Refuted the
  "decoder is the bottleneck" hypothesis: a bigger decoder can't recover
  information the codes discard.
- **Recalibration on a consistent split** corrected two errors: (a) the "0.328"
  Phase 7 number was a different eval config — the true same-split baseline is
  0.305; (b) the free-gen ceiling is *not* exposure-bias-capped (the shallow
  *continuous* model already free-generates at 0.430). The real lever is the
  **quantization gap** (continuous 0.430 → RVQ 0.315 = −0.115).
- **E2 (RVQ 4→8 levels): the win, +0.102.** Closed all but 0.013 of the
  quantization gap, and — unlike E1 — it *transferred* to free generation,
  because code capacity is the true information bottleneck. 8 levels is the sweet
  spot; 16 would chase 0.013 and cost interpretability.
- **E3 (200k full-horizon cosine): collapsed after 100k, salvaged to the win.**
  Stretching cosine over 200k held LR too hot too long (~1.66e-4 at step 100k vs
  E2's ~3e-5), collapsing the RVQ codebook to 1/128. But the *first* 100k, with
  its hotter LR, produced the best checkpoint (0.435 > E2 0.417). Lesson: for
  runs >100k, decay LR faster — don't stretch cosine to the full step count.

## Frozen architecture (E3 @ 100k)

Checkpoint: `data/phase8_e3_rvq8_200k_step100000.pt`

- Perceiver encoder (4 layers) → **RVQ 8×128** (8 residual codebooks, k=128) →
  autoregressive decoder (**10 layers, d_ff 2048**), output projection tied to
  input embedding.
- `d_model=384` (pinned by the pretrained XLM-R embedding table), `d_code=256`,
  `m_max=64`.
- **Tied** encoder/decoder token embeddings, **unfrozen** (trained).
- opus100 (~8.5M pairs, 10 langs), 100k steps, batch 32, cosine LR + semantic
  loss (λ=5) + token-weighted CE.

## Radicals stay interpretable (recon/interpretability tension resolved)

Going to 8 RVQ levels did **not** dilute the level-0 radicals — level 0 carries
the primary concept, levels 1–7 handle refinement. Level-0 discovery
(`results/e3_100k_codebook_L0.txt`) yields a rich cross-lingual inventory:

- **Pronouns:** we (16), I (21), she (31), you (39)
- **Number:** two (45)
- **Negation:** nothing (9), not (53)
- **Content:** file (1), medicine/health (7), law (11), know (12), father (24),
  yes (26), work (27), war/military (28), day (33), water (36), woman (44),
  name (49), committee/council (50), money (54), say (59), love (0), leave (5),
  republic/nation (4), market/economy (47), Israel/world (52)

Each fires on the same meaning across 10 languages and 5 scripts.

## Interlingua holds

`crosslingual_consistency.py` on the frozen winner (all 10 langs, 45 pairs):

- mean real Jaccard = **0.420**, chance = 0.181, **lift = +0.239**
- all 45 language pairs positive (+0.18 … +0.27)

Parallel translations share ~42% of their primary radicals vs ~18% at chance —
the interlingua property holds at the bag-of-codes level.

## Next lever: the capacity gap

Quantization is closed. The dominant remaining gap is now **capacity**:
continuous 0.430 → gold 0.861 (−0.431). Pushing past ~0.43 requires raising the
*continuous* ceiling — more training (safely scheduled), direct non-English-
centric data, or more model capacity (decoupling `d_model` from the 384-pinned
embedding). This is compute-heavy, which motivates the **CS-3 port** (see
`reference_accelerator_options` memory): our small, bandwidth-bound model is
SRAM-resident on the wafer, and the custom RVQ traces cleanly under `cstorch`
(argmin/gather/detach in a static unrolled loop) — the port lift is the
training-loop idiom, not the ops.

## Update — B (d_model decouple) result: width raises the ceiling

After freezing E3@100k (0.435), two follow-ups on the `gb10-scaleup` branch:

- **A (WSD schedule): failed, informative.** Holding LR flat at peak 3e-4 for
  50k steps diverged at ~55–60k (decoder/recon blowup; codebook stayed intact).
  Flat-at-peak is more destabilizing than E2/E3's always-decaying cosine at the
  same peak. Conclusion: the schedule lever is tapped — E3's gently-decaying
  cosine is the optimum.
- **B (d_model 384→512): the win.** Round-trip real cross = **0.469** — beats
  E3@100k (0.435) by +0.034 and, decisively, **exceeds the d_model-384
  continuous ceiling (0.430)**. A quantized wider model beating the narrower
  *continuous* ceiling proves widening d_model raised the continuous ceiling
  itself; unlike decoder depth (E1, +0.01, didn't transfer), width transfers to
  free generation. shuffle-gap +0.421 (strongest of any run). 54.5% of gold.

Leaderboard (same seeded 200-sentence split, real cross):

| model | real cross |
|-------|-----------|
| gold (translation ceiling) | 0.861 |
| **B — d_model 512, RVQ8, tied, deep** | **0.469** |
| E3 @ 100k — d_model 384 | 0.435 |
| continuous ceiling — d_model 384 | 0.430 |
| E2 — d_model 384 | 0.417 |

Frozen-winner candidate: `data/phase8_B_dmodel512_100k_step100000.pt` (155.4M).
Next width/schedule levers: B used E2's baseline cosine-100k (not E3's gentler
schedule), so width+gentle-schedule may add more; and d_model 768 tests whether
width keeps paying.
