# Phase 6 — Force code reliance, and measure meaning (not tokens)

## Why

Phase 5c's diagnostic verdict: **the characters are nearly inert.** Shuffling the
bottleneck costs only ~2–3% content-acc, i.e. the decoder reconstructs almost as
well from the *wrong* codes as the right ones (severe posterior collapse). The
central thesis — a discrete, language-independent, composable character set that
*carries* meaning — is therefore **untested, and currently being falsified** by our
own numbers.

Two root causes, both structural (not hyperparameters):

1. **The decoder can succeed without the codes.** A strong autoregressive decoder
   given the target-language tag is a competent LM; token-CE is minimized by fluent
   unconditional generation. Word dropout (5c) is the weak remedy and is plateauing.
2. **We never measure the thesis.** Our accuracy is **teacher-forced** (`evaluate_pair`
   argmaxes teacher-forced logits; `diagnose_ablation` feeds `tgt[:, :-1]`). The gold
   prefix leaks the answer, so the shuffle-gap is measured in the one regime where the
   decoder is *allowed* to ignore codes. And token-exact CE punishes valid paraphrases,
   so content-acc is a poor proxy for meaning either way.

Encoder note: the bottleneck is **already** decoupled from source structure — the
encoder uses a Perceiver readout (`m_max` learned queries cross-attending the source),
not per-source-token codes. So the fix is on the **decoder + metric + interlingua-test**
side, not the encoder.

## Hypothesis

If we (a) remove the decoder's ability to route around the codes and (b) measure
free-generation *meaning* preservation, then either the shuffle-gap opens up
(thesis supported) or it stays flat under a decoder that *cannot* ignore the codes
(thesis in serious trouble). Both outcomes are clean — unlike today's ambiguity.

## The falsifiable metric (headline)

**Round-trip cross-lingual meaning preservation, free-generation, through the codes.**

    src ──encode→ quantize→ codes ──free-generate(tgt lang)→ ŷ ──MiniLM→ e(ŷ)
    cos( e(ŷ), e(src) )                      = round-trip meaning
    cos_real − cos_shuffle                    = MEANING shuffle-gap  ← the number that matters

- Free generation (greedy AR / parallel NAT), **not** teacher forced.
- MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) is XLM-R-based and shares our
  vocab, so generated ids re-embed directly. It aligns translations cross-lingually,
  so a faithful reconstruction in *any* target language scores high vs the source.
- Report same-lang and cross-lang separately. The **meaning shuffle-gap** replaces
  the teacher-forced content-acc shuffle-gap as our primary KPI.

## Baselines — bracket every number

Numbers are meaningless without an upper and lower bound. Every Phase 6 run reports
the headline metric against these four points:

| id | arm            | bottleneck to decoder            | tells us                              |
|----|----------------|----------------------------------|---------------------------------------|
| B0 | no-code (LM)   | zeroed / detached                | **lower bound**: what the decoder does with 0 bits |
| B1 | no-VQ          | continuous `z_e` (skip quantize) | **upper bound**: reconstruction ceiling; VQ's cost |
| B2 | AR + VQ (5c)   | quantized codes                  | reference (today's model)             |
| B3 | NAT + VQ       | quantized codes                  | **experimental arm**                  |

Reading: B3 is only interesting *relative to* B0 and B1. If `B3_real ≈ B0`, codes are
inert even under NAT (thesis fails). If `B0 ≪ B3_real ≤ B1` and the meaning
shuffle-gap is large, the thesis holds and the characters carry content.

The eval-time versions of B0/B2 (zero/real/shuffle on an existing checkpoint) are
computable **now** via `roundtrip_eval.py` — no training needed. B1/B3 are trained arms.

## Architectural change — NAT decoder

`vqvae/nat_decoder.py` (new, written): a non-autoregressive decoder that **cannot**
model target-token dependencies, so all content must come from the codes.

- No teacher-forced target tokens, no causal mask.
- Query positions = sinusoidal position embeddings (+ lang tag), length = target
  length (gold at train, `length_head` prediction at inference).
- Full self-attention across positions + cross-attention to the bottleneck.
- Position-wise CE against gold tokens (monotonic-alignment first cut). CTC/latent
  alignment is the documented upgrade if the monotonic assumption hurts.
- Reuses the existing `length_head` (already trained in 5c) for inference length.

This is the strongest available test of code reliance: a NAT decoder with inert codes
produces garbage, so the shuffle-gap *must* open if the codes carry anything.

## New scripts (all safe — no edits to sweep files)

- `roundtrip_eval.py` — free-generation round-trip meaning metric + real/shuffle/zero
  baselines (B0/B2 eval-time), same & cross lang. Runnable on existing checkpoints.
- `crosslingual_consistency.py` — the interlingua test we've never run: encode the
  *same meaning* in each language, measure code-sequence agreement. Directly probes
  the "language-independent characters" claim.
- `vqvae/nat_decoder.py` — the NAT decoder module.

## Deferred edits (BLOCKED until the wd50 sweep finishes)

These touch files the running sweep imports; do **not** edit until the sweep is done:

- `vqvae/model.py`: `decoder_type={"ar","nat"}`; `no_vq` (pass `z_e` to decoder);
  `no_code` (zero the bottleneck) flags; route to `nat_decoder` when selected.
- `train_vqvae.py`: `--decoder-type`, `--no-vq`, `--no-code` flags; NAT loss branch
  (position-wise CE w/ length target); wire round-trip metric into `evaluate()`.
- `run_phase6.sh`: trains B0, B1, B2(reuse 5c best), B3; then `roundtrip_eval.py` +
  `crosslingual_consistency.py` + `diagnose_ablation.py` on each.

## Success / falsification criteria

- **Thesis supported** if the NAT arm (B3) reaches `round-trip cos` well above the
  no-code floor (B0) with a **meaning shuffle-gap ≥ ~0.10**, and cross-lingual code
  agreement is materially above chance.
- **Thesis in trouble** if even the NAT decoder shows `B3_real ≈ B0` / a flat meaning
  shuffle-gap — the codes cannot be made to carry content under the current
  quantizer/capacity, and the pivot is the bottleneck itself (RVQ/product-VQ/FSQ,
  larger codebook, sparse-composition) per the sparse-dictionary framing.

## Sequencing

1. Finish wd50 (nearly free; settles the word-dropout dose-response). Then stop
   iterating on word dropout.
2. Run `roundtrip_eval.py` on the Phase 5b / 5c_wd15/wd30/wd50 checkpoints — get the
   *meaning* shuffle-gap and B0/B2 eval baselines with zero new training.
3. If that confirms the teacher-forced gap was hiding collapse (expected), do the
   deferred edits and run the B0/B1/B3 training matrix.
