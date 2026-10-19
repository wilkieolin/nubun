# Phase 9 — From interlingua to logography: review and plan

Written 2026-10-12, after the CS-3 500k run (round-trip 0.582) became the best
model. This document (1) restates the goal and scores the project against it,
(2) reports new measurements taken for this review, (3) draws the lessons from
Phases 1-8, and (4) proposes the next phase as gated workstreams. The
literature behind the proposals is in `LITERATURE.md`.

## 1. The goal, and where we stand

The goal is a **composable series of radicals forming a sparse character set
that combines efficiently to convey semantic meaning**: Hangul's composability
with the higher-level semantics of Chinese radicals and characters. It must be
**human-readable** and at the same time a **natively tokenized semantic space**
in which AI models can reason in compressed form.

That goal decomposes into six properties. Phases 3-8 built and validated the
first two with a multilingual VQ-VAE. The other four are the work of Phase 9.

| # | property | state | evidence |
|---|---|---|---|
| 1 | codes carry meaning, independent of language | **strong** | round-trip cross-lingual meaning 0.583 (seed 42) and **0.651** (seed 2, same recipe; gold 0.861, shuffle 0.044), so the seed error bar on round-trip is ~0.07 (A2); interlingua Jaccard lift +0.278 on all 45 pairs; same slot holds the same level-0 code across translations 53% vs 10% chance (A6); code cost varies 15% across languages where text varies 42% (A5) |
| 2 | codes are interpretable radicals | **good, seed-specific, with a small stable core** | level-0 inventories (pronoun paradigm, negation, modals, dozens of content concepts) in `RADICALS.md`; only 19% of a converged run's level-0 radicals recur in a second seed (§7 A4), and of those the content radicals that recur decode to the same five concepts in both models: negation, today/date, child, health, European Union (~7% of the inventory); the rest is a property of one run |
| 3 | the representation is sparse | **no** | 64 fixed slots = 3584 raw bits/sentence, but only ~8.5 distinct level-0 codes and ~17 distinct stacks per sentence are used (§2); under a learned prior the full code costs 280-484 bits, levels 0-1 49-83 (A5); no length or sparsity pressure exists in the loss |
| 4 | radicals compose | **weak, not established** | posdis excess +0.027 and 3x bosdis at 500k (the code is positional, keep RVQ); but additive composition explains only 10-14% of the code vector (TRE 0.86-0.90) and topographic similarity fell from 0.20 to 0.13 as round-trip rose (A3); every A3 number is reproduced by the second seed within noise, so this is a property of the recipe, not of one run; depth >= 2 is language-specific (A6); no held-out-combination or substitution test yet (C) |
| 5 | human-readable glyph layer | **not started** | nothing renders a code to a glyph; no reading, no legibility or learnability test |
| 6 | AI reasoning substrate | **measured, failing** | code prior trained (A5): the full 8-level code costs 1.3-1.7x (seed 42) to 2.1-2.2x (seed 2) the bits of the text it encodes and is lossy; levels 0-1 cost 0.22-0.34x at 40% of gold fidelity, levels 0-3 0.53-0.85x at 55-58%; the better-reconstructing seed has the costlier code; no LLM has seen Nubun codes |

The honest summary (updated 2026-10-19, after workstream A and its cross-seed
checks): **we have a working, cross-lingual discrete bottleneck whose level 0
is interpretable; we do not yet have a character set.** The bottleneck is
dense, its compositionality is weak and falling with training, its inventory is
90% seed-specific, and its full code costs more bits than text. What is
seed-robust is everything about the meaning as a whole (round-trip, interlingua,
the A3 metrics) plus a core of about half a dozen level-0 radicals for frequent,
sharply bounded concepts. §8 is the consolidated status and the plan: confirm
with a third seed, then add the inductive bias (minimal pairs) that the
identifiability argument of `LITERATURE.md` §3 turns out not to supply.

## 2. New measurements for this review (2026-10-12)

All on the seeded FLORES validation split (200 sentences x 10 languages, mean
36.5 source tokens), encoder-only, CPU. Scripts: `crosslingual_consistency.py`
(all 10 languages) and the new `code_usage_audit.py`; codebook geometry from a
one-off script on the two checkpoints.

| metric | E3 @100k (GB10, B=32) | **CS-3 @500k (B=128)** |
|---|---|---|
| round-trip cross (from PHASE8) | 0.435 | **0.582** |
| interlingua Jaccard: real / chance / lift | 0.420 / 0.181 / +0.239 | 0.600 / 0.322 / **+0.278** |
| level-0 codes active (of 128) | 127 | 102 |
| level-0 marginal entropy (bits) | 5.88 | 5.69 |
| level-0 per-slot entropy: mean (min) | 4.35 (2.81) | 4.96 (**0.02**) |
| distinct level-0 codes per sentence (of 64 slots) | 14.9 | **8.5** |
| distinct full 8-level stacks per sentence (of 64) | 63.5 | **16.7** |
| level-0 codes shared by all 10 languages of a sentence (union) | 2.6 (38.7) | 3.5 (16.9) |
| level-0 codes covering 50% / 90% of slot mass | 15 / 57 | 14 / 46 |
| level-0 rank-frequency log-log slope | not run | -1.88 |
| mean code norm, level 0 to level 7 | 4.58 to 1.19 | 10.93 to 3.50 |
| adjacent-level 16-d subspace overlap (1 = same, 0 = orthogonal) | 0.49 to 0.19 | 0.37 to 0.16 |
| raw bits per sentence (64 x 8 x 7) | 3584 | 3584 |

What this says:

- **The long run made the code radically more redundant while reconstructing
  better.** E3 filled all 64 slots with distinct stacks; the 500k model
  collapses them to ~17 distinct stacks and ~8.5 distinct primary radicals per
  sentence, with the other ~47 slots repeating one of them. Some slot
  positions are now near-constant (minimum per-slot entropy 0.02 bits): they
  are fillers. The decoder is reading a sparse code; the architecture just
  does not let the encoder emit one. A 36-token sentence is being carried by
  roughly 17 characters built from 8-9 radicals, which is already in the range
  a logography would want. **Property 3 is closer than the raw numbers
  suggest, and a variable-length bottleneck is the natural next step.**
- **The interlingua lift is slightly higher, but the chance baseline nearly
  doubled** (0.181 to 0.322) because sentences now share a small, heavy-headed
  set of radicals. Jaccard-vs-chance remains a valid test, but the next metric
  should be stack-level or order-aware (see workstream A).
- **Coarse-to-fine is real.** Code norms fall monotonically across levels and
  level codebooks occupy partially distinct subspaces (adjacent overlap 0.16 to
  0.51, never near 1). The vector-symbolic condition for reading levels
  independently (near-orthogonal bundles) is only partly met; levels 1-3
  overlap most with each other.
- **Usage is heavy-headed.** 14 radicals cover half the slot mass and the
  rank-frequency slope is steeper than Zipf. Combined with the near-constant
  slots, a part of level-0 is acting as structural filler rather than meaning.
  Which codes are fillers is a decoding question for `analyze_codebook.py`.

Not measured, and needed: the 500k model's radical inventory (needs decoding
on the GB10), the round-trip length curve (100k/250k/480k `.eval.pt` files are
written), and every compositionality metric in workstream A.

## 3. Lessons from Phases 1-8

- **Direct sparse decomposition of embeddings (Phases 1-2) could not have
  worked**, and the literature now says why: dictionary recovery is
  non-identifiable for coherent atoms, embedding geometry is seed-unstable, and
  unsupervised disentanglement needs an inductive bias. Parallel sentences in
  10 languages are conditionally independent views of one meaning, which is
  exactly the condition under which a discrete latent is identifiable. The
  pivot to end-to-end multilingual training was the right call for a
  principled reason, not just an empirical one (`LITERATURE.md` §0.2, §3.1).
- **Measure the thesis, not a proxy.** Teacher-forced accuracy hid posterior
  collapse for two phases (Phase 5c/6). Free-generation round-trip meaning
  with shuffle and zero controls has been the right KPI since. The same
  discipline must apply to compositionality (property 4): it is currently an
  assertion.
- **The code, not the decoder, is the bottleneck.** Decoder capacity (E1)
  barely moved free generation; RVQ depth (E2), width (B), schedule (C1) and
  training length (CS-3) did. Information that the codes discard cannot be
  recovered downstream.
- **Hot learning-rate tails collapse codebooks; data mixing did not help.**
  E3 at 200k and A (flat peak) diverged; D1 (30% non-English pivot pairs) hurt
  generation and left the interlingua lift unchanged: the interlingua is not
  data-limited.
- **The wafer works and is the training platform.** 500k steps at B=128 in 46
  hours; the compile rules are recorded in `cerebras/CS3_BRINGUP.md`. Decoding
  and MiniLM scoring still happen on the GB10.

## 4. Proposed next phase: six workstreams, gated

Order of operations: **A first** (cheap, no training, de-risks everything
else), **B and C** are the training work and run on the CS-3, **D and E** are
new code that can start in parallel with B/C on the existing 500k model,
**F** is the compute plan. Each workstream names a decision it feeds.

### A. Measure compositionality and information content (now, no training)

Decision fed: are the levels an ordered hierarchy or an unordered bag, and
how many bits does a sentence really need?

1. **Radical inventory of the 500k model** (`analyze_codebook.py --rvq-level 0`
   on the GB10). Also decode levels 1-2. Identify which heavy-use level-0 codes
   are fillers. Update `RADICALS.md`.
2. **Round-trip length curve**: 100k / 250k / 480k / 500k `.eval.pt` on the same
   seeded split. Tells us whether 0.582 is saturating.
3. **Compositionality suite** (new script `compositionality_metrics.py`), on
   the 500k, C1 and E3 checkpoints, per level and cumulative:
   topographic similarity (code Hamming or edit distance vs MiniLM cosine);
   posdis and bosdis (Chaabouni et al. 2020) against cheap attributes
   (negation present, pronoun person, number, tense, lexical field from
   BabelNet or a keyword list); tree reconstruction error with additive
   composition over code embeddings (Andreas 2019); MIG per level;
   normalised PMI between each code and attributes, compared across the 10
   languages for stability (Boldt & Mortensen). **Gate:** if posdis is high,
   keep RVQ (ordered levels); if bosdis dominates, move to additive
   quantisation with joint encoding (workstream B4).
4. **Cross-seed stability**: two short runs on the wafer with different seeds;
   Hungarian-match level-0 codebooks and report the fraction of radicals that
   recur. Locatello's warning applies to us until this is done.
   *Done 2026-10-16 (§7 A4):* two converged 500k runs (seeds 42 and 2) share
   19% of their level-0 radicals (chance 8%), 32% of slot mass; mostly the
   structural codes plus a few frequent topical ones. Levels >= 1: none.
5. **Bits per sentence under a code prior**: train a small autoregressive
   transformer over the 500k model's code sequences (on the GB10) and report
   bits/sentence, versus a byte-level or token-level LM at equal round-trip
   fidelity. This is the first number that speaks to property 6.
6. Replace or complement bag-of-codes Jaccard with a stack-level match
   (distinct stacks shared across languages, chance-corrected) now that the
   chance baseline has drifted.

### B. Make the code sparse (training, CS-3)

Decision fed: what is the natural number of characters per sentence?

1. **Variable-length bottleneck.** The 500k model already uses ~17 distinct
   stacks for a 36-token sentence. Reintroduce a length mechanism compatible
   with RVQ and the wafer: either a host-side target length (compression
   ratio on source tokens, as in Phases 3-5, which needs the stop-mask helpers
   rewritten without int64) or a learned slot gate with a sparsity penalty
   and a straight-through mask. Target: 12-24 slots used, round-trip within
   0.02 of the 64-slot model.
2. **Variable depth per slot** (quantizer dropout, SoundStream): train with
   random truncation of the level stack so levels 0-2 must carry the meaning
   alone, and read "deep stack = rare, specific concept" off the result. This
   also gives the glyph layer a principled cut between radical (levels 0-1),
   modifiers (2-3) and optional diacritics (4-7).
3. **Explicit level-0 pressure.** A reconstruction-only objective minimises the
   information in the codes (Kharitonov et al.); add a cross-lingual agreement
   loss directly on level-0 assignments of parallel pairs (soft assignment
   agreement, or the deterministic-IB form H(code0) - beta I(code0; meaning)),
   so radical semantics is trained for, not accidental.
4. **Quantiser alternatives, each as a control at matched bits** (56
   bits/slot): FSQ (collapse-free; if it shows the same cross-lingual sharing,
   radicals are a property of the data); additive quantisation with beam
   encoding and a joint least-squares codebook refit (removes greedy RVQ
   suboptimality and tests whether level-0 radicals survive non-greedy
   encoding); SimVQ or CVQ-VAE as drop-in collapse remedies if the learned
   codebook stays. Choose by two-part MDL (codebook description + corpus code
   length under the prior from A5), not reconstruction alone.
5. **Width on the wafer.** C1's d_model 512 (merged as the `d_emb` decouple) at
   the 500k/B=128 recipe is the open PHASE8 item and the cheapest likely
   round-trip gain; run it with the sparse bottleneck rather than before it.

### C. Make the radicals compose (training and evaluation)

Decision fed: is "composable" true, and does it generalise?

1. **COGS-style split over code combinations**: hold out sentences whose
   attribute combinations (a given level-0 radical plus negation; a pronoun
   plus a tense) never co-occur in training; test whether the encoder emits
   the predicted combination and the decoder reconstructs it. This is the
   only test of "composable" as opposed to "interpretable".
2. **Code substitution tests**: swap one level-0 code in an encoded sentence
   (we to they, not inserted or removed) and measure the targeted change in
   the regenerated sentence across languages. Cheap, uses existing decoder.
3. **Optional structural pressure**: iterated learning (periodically
   re-initialise the decoder so only easily-learnable code structure survives;
   Ren et al. 2020). Try only if C1 shows poor combinatorial generalisation.

### D. The human-readable layer (design and tooling, no training)

Decision fed: can a person read a Nubun sentence, and how fast can they learn?

1. **Glyph grammar** (`render_nubun.py`, specimen page from FLORES sentences):
   level-0 code = base radical glyph; levels 1-2 = modifiers in fixed
   positions around the base (a Hangul-style block, at most 4 components per
   block: Cowan 2001, Taft & Zhu 1997); levels 3+ = optional diacritics, omitted
   in casual writing. Use the Unicode Ideographic Description Sequence
   operators as the formal description of each block so the grammar is
   machine-checkable.
2. **Stroke design under the perceptual constraints**: ~3 strokes per radical
   with ~50% redundancy (Changizi & Shimojo 2005), built from the frequent
   natural-scene junction types with cardinal orientations (Changizi et al.
   2006; Morin 2018); score every glyph by perimetric complexity (Pelli et al.
   2006) and keep the inventory in the 100-150 range for the starter set,
   growing toward the 200-600 that real logographies use.
3. **A reading for every glyph.** Phonology is recruited even in semantic
   scripts (Perfetti et al. 2005) and the ideographic-myth critique
   (DeFrancis, Unger) is answered only if glyphs can be said: map code ids to
   a small featural syllabary (consonant = level-0 class, vowel = level-1), so
   every character is pronounceable by rule.
4. **Seed and validate against attested primitive sets**: NSM's ~65 primes,
   Swadesh 100/200, Ogden's 850, Jackendoff's ontological categories, Dixon's
   verb types. Report which primes have a level-0 radical and which do not;
   the gaps say what the bottleneck is not representing.
5. **Learnability test**: Pelli's ~3,000-trial budget. A small in-browser
   study (glyph to meaning, sentence to gloss) with the specimen page gives the
   first human-readability number. Blissymbolics' transparency deficit says
   abstract compositional glyphs will not be guessable; teach the radicals
   explicitly (Shen & Ke 2007) and measure learning speed, not guessing.

### E. The AI reasoning substrate (new code, GB10)

Decision fed: do models compute more cheaply or more faithfully over codes?

1. **Code language model** (shares A5): an autoregressive transformer over
   code sequences, next-sentence prediction in code space (the Large Concept
   Model setting, discrete). Report bits/sentence and next-sentence meaning
   accuracy versus text.
2. **Token Assorted recipe**: fine-tune a small open LLM on text where spans
   are replaced by their Nubun code strings (randomised replacement), then
   test reasoning with mixed code/text chains. The first test of "reason in
   a compressed format".
3. **Semantic-hub probe**: linear probes from a multilingual LLM's middle
   layers to Nubun level-0 codes and back (Wendler et al. 2024; Wu et al.
   2025). If codes are linearly decodable from the hub, they are a readout of
   an existing structure rather than an autoencoder artefact.

### F. Compute plan

- CS-3 (two wafers usable in parallel): B1-B5, A4, C1 training. One wafer for
  the sparse-bottleneck line, one for controls (FSQ, AQ). Keep the B=128 /
  lr 6e-4 / cosine-to-length recipe; compaction of the vocabulary to the
  117k used ids remains an untested throughput lever.
- GB10: all decoding (radical inventories, round-trip, substitution tests),
  the code LM, LLM fine-tunes, probes.
- Any CPU box: `code_usage_audit.py`, `crosslingual_consistency.py`, the
  compositionality suite without decoding, the glyph renderer.

## 5. Success and falsification criteria for Phase 9

- **Sparse**: a model using at most 24 slots (or an adaptive mean of ~16) and
  a truncated level stack whose round-trip is within 0.02 of the best model's
  (0.651, seed 2; the seed error bar is itself ~0.07, so compare at matched
  seed or over several). Fail: the
  round-trip drops by more than 0.05 when slots are reduced below 32, meaning
  the duplication seen in §2 was load-bearing after all.
- **Composable**: posdis or bosdis clearly above shuffled-attribute chance;
  TRE with additive composition well below a permuted-code control; held-out
  attribute combinations reconstructed at least as well as seen ones within
  0.05 round-trip. Fail: combinatorial generalisation collapses to the
  no-code floor. **Interim (2026-10-13, §7 A3): not established.** posdis is
  above shuffle (+0.027) but small; TRE 0.86-0.90 against a 1.00 control;
  topsim 0.13 and falling. The held-out-combination test (C1) has not run.
- **Stable**: at least two thirds of level-0 radicals recur across seeds after
  matching. Fail: inventories are seed-specific. **Outcome (2026-10-16,
  §7 A4): FAIL.** 19% of level-0 radicals recur between two converged 500k
  runs (chance 8%); the shared part is the structural codes and a few
  frequent topical radicals. **Decoded in both models (2026-10-19), the
  recurring content radicals mean the same thing**: negation, today/date,
  child, health, European Union, plausibly numbers; about 7% of the
  inventory is a stable, meaningful core and the rest is seed-specific.
  Phase 9 must add a mechanism that makes the inventory converge (§8, branch
  2) or redefine the character set as per-model.
- **Readable**: a learner reaches above-chance sentence glossing of specimen
  sentences within the 3,000-exposure budget. Fail: no learning curve.
- **Substrate**: the code prior is materially more compressible than a text LM
  at equal round-trip fidelity, and the mixed code/text fine-tune does not
  lose accuracy relative to text-only. Fail: codes cost more bits than text.
  **Interim (2026-10-13, §7 A5): FAIL for the full code** (1.3-1.7x text;
  2.1-2.2x for the seed-2 model, which reconstructs better and pays for it);
  the shallow code (levels 0-1, 0-3) costs 0.22-0.85x text but at 40-58% of
  gold fidelity, and no equal-fidelity text baseline exists yet. The
  fine-tune (E2) has not run.

## 6. Risks and the standing objections

- **The ideographic myth** (DeFrancis 1984, Unger 2004): no pure-meaning script
  has ever worked at scale, and real logographies recruited phonetics for
  ~85% of their characters. Our answer must be D3 (a reading for every glyph)
  plus the fact that our radicals are learned from language rather than
  posited above it.
- **Transparency**: Blissymbolics is the closest existing system and is the
  least guessable AAC symbol set. We should promise learnability, not
  iconicity.
- **Compositionality is not free**: effective codes need not be compositional
  (Kottur 2017), and a reconstruction objective minimises code information
  (Kharitonov 2020). Workstreams A and C exist because the thesis could still
  fail here.
- **Hierarchy may not pay for reconstruction** (Reyhanian & Wiskott 2026); its
  justification must come from MDL and compositionality, which is why B4
  chooses by description length.
- **English pivot**: opus100 is X-to-English; the data lever (D1) showed the
  interlingua is not data-limited, but the glyph inventory may still be
  English-shaped. The NSM/Swadesh comparison in D4 and the per-language nPMI
  stability in A3 are the checks.

## 7. Workstream A results (2026-10-13)

Infrastructure first: pre-port checkpoints (E3, C1) give different codes under
main's `vqvae/` (the port seeds the readout queries with the encoder mean), so
`vqvae_select.py` runs them on a pre-port worktree (`../nubun_preport`,
`origin/gb10-scaleup`); it reproduces §2's E3 and 500k audit rows exactly.
Decoding on the GB10 must run under `scripts/mem_watchdog.sh`: greedy decode at
chunk 1024 grew to 66 GB through allocator fragmentation (chunk 512 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` peaks at 10.5 GB).

### A3 + A6: compositionality (`compositionality_metrics.py`)

All 2009 FLORES sentences x 10 languages (held out from opus100). Attributes
labelled on the English side (spaCy + keyword lists) and copied across the
parallel languages: negation, tense, subject person, subject number, lexical
field, and five named-entity presence flags. Controls: attributes shuffled at
sentence level; meaning or code rows permuted. Level-0 numbers (excess = real
minus shuffle):

| level 0 | E3 @100k | C1 @130k | CS-3 @500k (seed 42) | CS-3 @500k seed 2 (2026-10-19) |
|---|---|---|---|---|
| posdis excess (slot specialises to one attribute) | +0.008 | +0.017 | **+0.027** | +0.032 |
| bosdis excess (code count specialises) | **+0.020** | +0.018 | +0.009 | +0.009 |
| MIG, bag / positional | 0.049 / 0.022 | 0.033 / 0.014 | 0.021 / 0.002 | 0.014 / 0.003 |
| topsim, bag Jaccard vs MiniLM (perm ~0) | **0.20** | 0.19 | 0.13 | 0.10 |
| TRE, additive, held-out 1-R^2 (perm 1.00) | **0.86** | 0.88 | 0.90 | 0.89 |
| nPMI cross-language stability r (shuffled) | 0.57 (0.50) | 0.60 (0.50) | **0.67 (0.59)** | 0.66 (0.57) |
| A6 level-0 bag Jaccard real / chance / kappa | 0.41 / 0.19 / 0.28 | 0.42 / 0.18 / 0.30 | 0.59 / 0.33 / **0.38** | n/a |
| A6 same code in same slot, real / chance / kappa | 0.37 / 0.10 / 0.30 | 0.44 / 0.09 / 0.38 | 0.53 / 0.10 / **0.48** | n/a |
| A6 depth-2 stack prefix bag kappa | 0.10 | 0.11 | 0.12 | n/a |
| decode-probe codes strong in >=3 languages (A1) | **55** | 19 | 17 | n/a |

The seed-2 column (`results/a4/seed2_500000/compositionality_cs3_seed2_500000.txt`)
reproduces every seed-42 number within noise, with round-trip 0.07 higher:
the weak-compositionality result is a property of the recipe, not of one run,
and the extra fidelity of seed 2 did not buy any compositional structure.

What this says:

- **The gate (A3) points to keeping ordered RVQ, on a weak signal.** With more
  training the code moves from bag-like (E3: bosdis > posdis) to positional
  (500k: posdis 3x bosdis), and the same slot carries the same level-0 code
  across translations 53% of the time (chance 10%, kappa 0.48). Slot position
  is part of the interlingua, which bag-of-codes Jaccard could not see. The
  absolute disentanglement values are small: FLORES sentences are long (36
  tokens) and the attributes are coarse sentence-level properties.
- **Compositionality in the topsim/TRE sense is weak and falling.** Additive
  composition from the attributes explains 10-14% of the pooled code vector,
  and topographic similarity falls from 0.20 (E3) to 0.13 (500k) while
  round-trip rises from 0.435 to 0.582. Better reconstruction is being bought
  with a less geometric, more positional code. Property 4 is not established;
  workstream C (held-out combinations, substitution) is the real test.
- **Depth >= 2 is not interlingual, but it is not surface detail either.**
  Stack-prefix agreement across languages falls from kappa 0.38 at depth 1 to
  0.12 at depth 2 and 0.04 at depth 3, yet dropping levels 2-7 costs 40% of
  round-trip (A2 below: 0.583 -> 0.346), and cross-lingual output loses as
  much as same-language output. The deep levels carry meaning, encoded
  differently depending on the source language. They are where the
  interlingua is still incomplete, not something the glyph layer can drop;
  B3 (cross-lingual agreement pressure) should target them, not only level 0.
- MI estimates on stack prefixes of depth >= 3 saturate (nearly every slot is a
  unique symbol; real ~ shuffle) and should be ignored.

### A1: radical inventory of the 500k model

`results/cs3_500k_codebook_L{0,1,2}.txt` (64 samples x 10 languages per code).
Clean cross-lingual radicals at level 0: SHE (3: 她 / her / Elle / Ela / Она),
WOMEN (50: женщин / Frauen / femmes / 女性), NEGATION "No" (58: No / Не / Non),
NEVER (66: Nunca / jamais / never), 1sg (71: me / tenho / могу / Je), 2nd person
(105: você / tu), HONORIFIC (93: Herr / Sr / M), DATE (55, 79: 年 / 月 / 日 /
numerals), URL (76: http). Level 1 has one clean code, THREE (40: 3 / três /
tres / three / 三), a number modifier; levels 1-2 are otherwise uninterpretable
in isolation, as expected for residual levels.

- **The probe undercounts the 500k model.** It decodes random 3-16-slot code
  sequences; the 500k decoder only ever saw 64 heavily duplicated slots. 17
  codes are strong in >= 3 languages (C1 19, E3 55). In-distribution nPMI (A3)
  agrees with the probe on function words (negation 66/107 "any", 2nd person
  105, plural 20 they/their) but finds topical radicals the probe cannot see
  (health 96/73, politics 113/16, disaster 42, numbers 30): content radicals
  are readable only in context. A probe that substitutes codes into real
  encodings (C2) is the right inventory tool for this model.
- **Fillers identified.** Codes 43 (present in 100% of encodings, always the
  same slot), 29 (85%), 118 (71%), 75 (63%), 99 (40%) and 49 (27%) sit in one
  fixed slot each and decode to nothing consistent. These are §2's
  near-constant slots: structure, not meaning, and the first thing a
  variable-length bottleneck (B1) should remove.
- The heaviest-used content codes (91, 14, 45, 127, 101; 4-6% of slot mass
  each) also decode to noise under the probe; 101 is the top nPMI code for
  past tense and for the person, organisation and time entity flags at once, i.e. a generic
  "news event" radical.

### A2: round-trip length curve, and round-trip by RVQ depth

PHASE8 protocol (`roundtrip_eval.py --combine-splits --langs en,zh,es,fr,ar,ru
--max-gen-len 48`, seeded 200-sentence split, gold cross 0.861); the 500k
checkpoint reproduces 0.5824 (0.5828 here, batch 64 instead of 16). New
`--rvq-depth d` decodes from only the first d levels.

| CS-3 run checkpoint | 100k | 250k | 480k | 500k |
|---|---|---|---|---|
| round-trip cross | 0.450 | 0.511 | 0.580 | **0.583** |
| shuffle-gap cross | +0.405 | +0.465 | +0.532 | +0.536 |

| 500k, decoded from levels 0..d-1 | d=1 | d=2 | d=4 | d=8 |
|---|---|---|---|---|
| seed 42: round-trip cross | 0.235 | 0.346 | 0.478 | 0.583 |
| seed 42: round-trip same-language | 0.237 | 0.351 | 0.490 | 0.612 |
| seed 2 (2026-10-19): round-trip cross | n/a | 0.348 | 0.496 | **0.651** |
| seed 2: round-trip same-language | n/a | 0.347 | 0.507 | 0.691 |

The seed-2 model (same recipe, `--seed 2`, logged recon 2.77 against 2.89)
scores 0.651, +0.068 over seed 42 on the identical protocol
(`results/a4/seed2_500000/roundtrip_seed2_500000*.txt`). The shuffle and zero
controls are unchanged (0.042 / 0.034). Seed 2 is therefore the best Nubun
model, and the difference between two seeds of one recipe is as large as
several of the single-run lever comparisons in PHASE8 and above (E3 vs C1,
100k vs 250k, width). Those comparisons carry an unmeasured ~0.07 error bar
until a third seed gives a variance estimate (§8, branch 1).

- **Round-trip rose steadily and flattened only in the last 20k steps**
  (+0.003 from 480k to 500k), where the cosine LR reaches its floor; the
  flattening is the schedule ending, not the model saturating. Together with
  A4 (the inventory is fixed only in that same window), a longer or
  higher-floor schedule is still a live lever.
- **Every level pays.** Each doubling of depth adds about 0.11-0.13
  round-trip (seed 2: 0.15 per doubling). Level 0 alone carries 40% of the
  full model's round-trip, levels 0-1 59% (seed 2: 53%).
- Protocol warning: on FLORES devtest (article-ordered, the script's default
  when `--combine-splits` is omitted) the `shuffle` control rolls each batch by
  one sentence, so "shuffled" codes come from the neighbouring sentence of the
  same article: shuffle reads 0.227 there against 0.047 on the seeded split.
  Always use `--combine-splits`.

### A4: cross-seed stability (`codebook_stability.py`) — done 2026-10-16

CPU, §2's seeded split (200 sentences x 10 languages). Two checkpoints
encode the same 2000 sentences; per RVQ level, codes are compared in a
**slot view** (k x k contingency table, Hungarian matching, fraction of slots
that agree under the best one-to-one relabelling) and a slot-agnostic
**sentence view** (each code's footprint is the set of sentences it occurs
in; Hungarian on pairwise Jaccard; a radical "recurs" when its matched partner
shares >= 0.5 of its sentences). Chance for every number is the same statistic
with B's sentences rolled by one. Before running it across seeds, we ran it
along the 500k run itself, every checkpoint against the final one:

| level-0, vs the 500k checkpoint | 480k | 250k | 100k | 50k | chance |
|---|---|---|---|---|---|
| slot agreement under best relabelling | **0.891** | 0.612 | 0.452 | 0.359 | 0.14-0.19 |
| NMI | 0.865 | 0.622 | 0.515 | 0.458 | 0.27-0.35 |
| mean matched sentence-Jaccard (active codes) | 0.740 | 0.418 | 0.272 | 0.207 | 0.08-0.13 |
| fraction of active radicals that recur (J >= 0.5) | **0.973** | 0.329 | 0.082 | 0.068 | 0.01-0.04 |
| slot mass carried by recurring radicals | 0.99 | 0.41 | 0.11 | 0.08 | - |

(Active = taking >= 0.1% of slots: 73 codes at 500k. Level 1 recurrence is
0.92 at 480k and 0.03 at 250k; levels 2-7 fall off faster.)

What this says:

- **The radical inventory keeps turning over until the learning rate is near
  its floor.** Half-way through the run, two thirds of the level-0 radicals
  the final model uses did not yet exist as the same sentence-footprints; at
  100k steps, more than 90% did not. The 100k checkpoint has a similar
  number of active codes but they are not the 500k symbols (the cross-seed
  analysis below shows why: they are coarser, sentence-level ancestors).
  Only in the last 20k steps (LR within 10% of its floor) is the inventory
  essentially fixed (97% recurrence, Hungarian returns the identity).
- **Consequence for A4.** Two independent short runs cannot be expected to
  agree with each other more than a run agrees with its own past, so a
  cross-seed number at 50k steps is a lower bound on what converged runs
  would share, and a converged cross-seed test needs full-length runs. The
  within-run drift is itself the first stability result: whatever the
  radicals are, they are a property of the end-point of the schedule, not of
  the data alone. Any RADICALS.md inventory must name the checkpoint.
- **Consequence for the hierarchy.** Level 0 stabilises first and levels 1-7
  later and less, consistent with coarse-to-fine, but at 250k even level 1 is
  almost entirely re-drawn. The levels are not independently stable
  components. This matches A6 above: depth >= 2 is neither interlingual nor
  stable in time, so the radical is levels 0-1 at most. (It still carries
  meaning: decoding without levels 2-7 loses 40% of round-trip, A2. The
  deep levels are unstable, language-dependent detail that the glyph layer
  must still represent somehow.)
- **The fillers are the most stable codes.** Of A1's six fixed-slot fillers,
  43 (the always-present one) is already in place at 50k (J 0.71) and 43, 75,
  29, 99 and 49 all recur at 250k (J 0.64-0.83) when only 28% of the content
  radicals do; a slot that always holds the same code recurs trivially.
  Excluding fillers, level-0 recurrence is 0.970 / 0.284 / 0.060 / 0.060 at
  480k / 250k / 100k / 50k. Filler 118 does not track its own past (J 0.56
  even at 480k), so it is not a fixed slot in the same sense.

**Cross-seed runs** (2026-10-13/14): two 50k-step runs, seeds 1 and 2,
otherwise the B=128 recipe with a 50k cosine (`cerebras/runs/seed_pair.sh`,
`model_dir/seed{1,2}-b128-20261013T221048Z`, ~5 h each on one wafer; final
training recon 4.21 / 4.26, the long run at its own 50k: 4.15). Level 0,
active codes 65 (seed 1) / 58 (seed 2) / 66 (long @50k):

| level 0 | **seed 1 vs seed 2** | seed 1, 50k vs 40k | seed 2, 50k vs 40k | seed 1 vs long @50k | seed 2 vs long @50k | seed 1 / 2 vs long @500k |
|---|---|---|---|---|---|---|
| slot agreement, best relabelling (chance) | **0.422** (0.145) | 0.826 | 0.827 | 0.442 | 0.393 | 0.370 / 0.337 |
| NMI (chance) | **0.516** (0.260) | 0.797 | 0.786 | 0.531 | 0.494 | 0.430 / 0.407 |
| mean matched sentence-Jaccard (chance) | **0.232** (0.070) | 0.628 | 0.586 | 0.273 | 0.250 | 0.226 / 0.212 |
| active radicals that recur, J >= 0.5 (chance) | **0.092** (0.000) | 0.785 | 0.707 | 0.138 | 0.103 | 0.062 / 0.017 |

Level 1 across seeds: slot agreement 0.244 (chance 0.116), recurrence 0.018.
Levels >= 2: nothing recurs. Development of level 0 along the long run
(`code_usage_audit.py`), distinct level-0 codes per 64-slot sentence:

| checkpoint | seed 1 @50k | seed 2 @50k | long @50k | long @100k | long @250k | long @500k |
|---|---|---|---|---|---|---|
| distinct level-0 codes per sentence | 2.3 | 2.3 | 2.5 | 4.0 | 5.9 | 8.5 |
| distinct full stacks per sentence | 11.2 | 11.1 | 14.1 | 14.7 | 15.1 | 16.7 |

What this says:

- **At 50k steps, two seeds share about 9% of their level-0 inventory, where
  the same run 10k steps apart shares 71-79%.** Taken at face value this is
  Locatello's result: which codes exist is a property of the seed. The
  agreement is not zero (slot agreement 0.42 against 0.15 chance, NMI 0.52
  against 0.26), so the two inventories are correlated, but they are not the
  same set of symbols. A 50k run with a different seed resembles the long run
  at 50k (0.14 / 0.10) and at 500k (0.06 / 0.02) no better than it resembles
  the other seed.
- **But level 0 at 50k is not yet a radical inventory; it is a sentence
  clustering.** A 50k model writes 2.3 distinct level-0 codes per sentence,
  typically one code in ~50 of the 64 slots: the level-0 code is a label for
  the whole sentence. The codes that do recur across seeds are exactly the
  ones that isolate the same small set of sentences in all 10 languages (e.g.
  seed-1 code 103 and seed-2 code 123: 7 sentence ids, 39 of 42 occurrences
  shared, 3-6 per language; code 41 and 126: 84 sentence ids, 278 of 380
  shared). The interlingua is present at level 0 before any radicals are: the
  two seeds agree on *which sentences are alike*, not on a symbol set.
- **The within-run drift (§ above) is differentiation, not relabelling.** The
  distinct-codes-per-sentence count grows monotonically, 2.5 to 4.0 to 5.9 to
  8.5 over the 500k run: sentence-cluster codes split into slot-level radicals
  as the learning rate falls. The 100k inventory recurs at 8% in the 500k
  model because the 500k radicals are mostly its descendants, not because the
  model keeps redrawing its symbols at random. The radical regime (fillers in
  fixed slots plus ~8 content radicals per sentence) exists only late in the
  schedule.
- **Consequence.** The cross-seed question that matters for RADICALS.md, "do
  two converged runs learn the same radicals?", cannot be answered with short
  runs: they have no radicals to compare. It needs a second 500k run with a
  different seed (one wafer, ~45 h, the existing `long_b128.sh` with
  `--seed`). Until then, every inventory claim names its checkpoint and is
  understood to be seed-specific. The result also sharpens B1: a
  variable-length bottleneck should be introduced late or with the code
  already in the radical regime, since a model in the one-code-per-sentence
  regime would simply shorten to one slot.
**Interim, seed 2 at 250k (2026-10-15).** The seed-2 500k run reached its
250k checkpoint (mid-schedule; the seed-42 run has a 250k `.eval.pt` at the
same learning-rate state), and it is already in the radical regime: 8.5
distinct level-0 codes per sentence against 5.9 for seed 42 at 250k, a
fixed-slot filler (code 38, slot 34, 100% of sentences) and a near-constant
slot (entropy 0.00). Level 0, active codes 75 / 71:

| level 0 | seed 2 @250k vs seed 42 @250k | seed 2 @250k vs seed 42 @500k | (seed 42, 250k vs 500k) |
|---|---|---|---|
| slot agreement, best relabelling (chance) | 0.346 (0.128) | 0.318 (0.121) | 0.612 (0.177) |
| NMI (chance) | 0.397 (0.232) | 0.367 (0.218) | 0.622 (0.330) |
| mean matched sentence-Jaccard (chance) | 0.301 (0.118) | 0.309 (0.139) | 0.418 (0.109) |
| active radicals that recur, J >= 0.5 (chance) | **0.133** (0.053) | 0.133 (0.067) | 0.329 (0.041) |
| slot mass carried by recurring radicals | 0.26 | 0.27 | 0.41 |

- Of the ten seed-2 radicals that recur in seed 42 at 250k, four are the
  structural codes present in most sentences (seed 2's 88 / 86 / 38 / 89, in
  98 / 86 / 100 / 87% of sentences, matched to seed 42's 62 / 29 / 27 / 75),
  and six are content codes used in 1-8% of sentences, several matching
  radicals A1/A3 named: 41 -> 42 (disaster), 42 -> 73 (health), 64 -> 91 and
  26 -> 101 (the heavy "news event" / past-tense codes, J 0.46-0.47, just
  under the threshold).
- At 250k the two seeds' fixed-slot fillers both sat in slot 34 (seed 2 code
  38, seed 42 code 27). At 500k this turns out to be a coincidence of the
  matching: seed 2 keeps its only fixed slot at 34, seed 42's always-present
  filler is at slot 52 with secondary fillers at 34 and 57. Filler *slots* are
  seed-specific; that there are one to five of them is not.
- Read against the 50k result (9%, clustering regime) and the within-run
  number (33% for seed 42 between 250k and 500k), 13% with a quarter of the
  slot mass is a weak but real shared core: structure plus a handful of
  frequent topical radicals. The converged comparison at 500k is still the
  one that counts.

**Converged, seed 2 at 500k (2026-10-16).** The seed-2 run finished at 13:33
UTC on 2026-10-16 (final logged recon 2.77; seed 42: 2.89). Seed 2 at 500k:
79 active level-0 codes (seed 42: 73), 9.3 distinct level-0 codes per sentence
(8.5), 24.6 distinct stacks (16.7), one fixed slot. Level 0:

| level 0 | **seed 2 @500k vs seed 42 @500k** | seed 2, 500k vs 480k | seed 42, 500k vs 480k |
|---|---|---|---|
| slot agreement, best relabelling (chance) | **0.344** (0.121) | 0.899 (0.240) | 0.891 (0.191) |
| NMI (chance) | **0.387** (0.229) | 0.879 | 0.865 |
| mean matched sentence-Jaccard (chance) | **0.307** (0.136) | 0.750 | 0.740 |
| active radicals that recur, J >= 0.5 (chance) | **0.190** (0.076) | 0.949 | 0.973 |
| slot mass carried by recurring radicals | **0.32** | 0.98 | 0.99 |

Level 1 across seeds: slot agreement 0.189 (chance 0.098), NMI 0.261 (0.189),
no radical recurs. Levels 2-7: at chance.

What this says:

- **Two converged runs share about a fifth of their level-0 inventory and a
  third of their slot mass.** Each inventory is internally stable (95-97%
  recurrence over its own last 20k steps), so this is not noise in the
  measurement: the runs settled on different symbol sets. The number is
  above chance (8%) and above the 50k result (9%), and it did not move between
  250k and 500k (13% then, 19% now, with the same mass), so more training is
  not closing the gap.
- **What is shared.** 15 of seed 2's 79 active radicals recur: six structural
  codes present in 53-100% of sentences (seed 2's 38 / 86 / 89 / 53 / 120 /
  98, matched to seed 42's 43 / 29 / 62 / 118 / 75 / 27, including the two
  always-present fillers at J 1.00), the "news event" code pair 26 -> 101
  (26% / 20% of sentences), and eight content radicals used in 1-8% of
  sentences. **Decoded in both models (2026-10-19, `analyze_codebook.py
  --rvq-level 0` on seed 2, `scripts/stable_pairs_report.py`,
  `results/a4/seed2_500000/stable_pairs_seed2_seed42_500k.txt`), five of the
  eight content pairs name the same concept on both sides**: EUROPEAN UNION
  (114 -> 35: Europäische Union, Parliament, Consejo / Europäische Union,
  European), CHILD (33 -> 104: 子, niño, Kinder, filhos / детей, enfants,
  filhos), NEGATION-NOTHING (5 -> 60: No, Не, Нет, 不, नहीं, شيء, じゃない /
  No, nada, ничего), TODAY-DATE (102 -> 79: 今天, आज, hoje, heute, اليوم /
  seed 42's DATE radical from A1), HEALTH (42 -> 73: disease, 疫苗, 癌, 薬,
  saúde, здоровья / seed 42's health code from A3); 49 -> 30 is plausibly
  NUMBERS on both sides; two pairs are unreadable. The six structural pairs
  decode to function-word noise on both sides and recur because frequent
  codes match frequent codes. So the data forces a core of about half a dozen
  level-0 radicals into existence regardless of seed, and they are frequent,
  categorical, basic-vocabulary concepts (three are Swadesh-list items, one is
  the FLORES news domain's dominant entity). Most of the dozens of content
  radicals in `RADICALS.md`-style inventories are seed-specific partitions of
  meaning space.
- **Levels >= 1 share nothing across seeds**, which closes the question A6
  raised: the residual levels are both language-specific and seed-specific.
  Only level 0 is a candidate for a character inventory, and only part of it.
  This is not evidence that the deep levels are meaningless (A2: they carry
  0.35 -> 0.65 of round-trip for seed 2). A residual code is defined relative
  to the level-0 partition, so when level 0 differs the level-1 codebooks
  cannot match whatever they mean; the joint 0..1 stack has ~3,300 distinct
  values and a real-chance NMI gap of 0.014, so the matching has no power
  above level 0. Levels 1-7 should be compared as a coordinate system
  conditional on level 0 (e.g. 0..1 stacks within the stable radicals), not
  as inventories.
- **Consequence for the goal.** Property 2 is downgraded in §1 to "good,
  seed-specific". A human-readable character set cannot be read off one run;
  either the inventory must be made to converge (a prior over the codebook, a
  fixed or data-derived level-0 initialisation such as k-means over
  sentence embeddings, or a consistency loss between two encoders as in
  B3's agreement pressure applied across seeds) or the character set must be
  defined by the meaning-space partition it induces and re-fitted per model,
  the way a tokenizer is. The second is compatible with property 6 (an AI
  substrate needs a fixed vocabulary per model, not across models) but not
  with property 5 as stated. This is a Phase 9 decision, recorded in §5.
- **Why this is Locatello's result, and what the identifiability argument
  missed.** `LITERATURE.md` §3 argued that parallel sentences are 3+
  conditionally independent views of one discrete latent and so identify it
  (Kruskal). They do: everything about the whole-sentence meaning is
  seed-robust (round-trip, A6, the A3 metrics, nPMI stability). But Kruskal
  identifies the latent, not its factorisation into 64 x 8 codes; any bijective
  recoding of the stacks satisfies the same conditional independence, which is
  Locatello's symmetry one level up. Translations share every meaning factor
  and differ only in language, so they identify the meaning/language split and
  nothing inside meaning. Pairs that share most factors and differ in a known
  few (Locatello et al. 2020, weakly-supervised disentanglement) identify the
  factorisation; minimal pairs (same sentence with negation, tense, number or
  one entity changed) are that kind of data, and we already label those
  attributes. §8, branch 2.
- Cross-seed checks of A3 and A5 on the seed-2 model: done 2026-10-19, in
  their own sections (A3 column, A5 rows).

- `codebook_stability.py` is the tool for any later pair, including E3 vs C1
  on the pre-port worktree (`vqvae_select`); its sentence view is the right
  one for models in the clustering regime, its slot view for the radical
  regime.

### A5: bits per sentence under a code prior (`dump_codes.py`, `code_prior.py`)

2.1M opus100 sentences (200k pairs per shard, X side plus a de-duplicated 25%
of English sides) encoded with the 500k encoder; train/test split by
translation pair (a translation in train would leak the test meaning, since
the codes are language-independent); FLORES (2009 x 10) as out-of-domain test.
The same causal transformer (6 layers, d 384, 30k steps x 64 sentences, no
language tag) is trained on each stream: code stacks truncated to depth d,
slot-major, or the sentences' own XLM-R tokens (vocabulary compacted to the
101k types seen in training; unseen test tokens charged log2 p(UNK) plus a
uniform choice among the 148k unseen types). Round-trip from A2.

| stream | bits/sent, opus test | bits/sent, FLORES | x text (FLORES) | round-trip | % of gold |
|---|---|---|---|---|---|
| text (XLM-R tokens) | 164.7 | 378.8 | 1.00 | lossless | - |
| code, levels 0-1 (raw 896 bits) | **48.7** | **82.5** | 0.22 | 0.346 | 40% |
| code, levels 0-3 (raw 1792) | 117.7 | 199.3 | 0.53 | 0.478 | 55% |
| code, levels 0-7 (raw 3584) | 280.2 | 484.3 | 1.28 | 0.583 | 68% |
| seed 2, levels 0-1 (2026-10-19) | 55.1 | 127.2 | 0.34 | 0.348 | 40% |
| seed 2, levels 0-3 | 139.2 | 320.8 | 0.85 | 0.496 | 58% |
| seed 2, levels 0-7 | 363.3 | 791.8 | 2.09 | 0.651 | 76% |

FLORES per language, levels 0-3 code: 182 (zh) to 210 (hi), spread 15%;
text: 300 (en) to 426 (ru), spread 42%.

What this says:

- **The full code fails §5's substrate criterion.** Under a learned prior the
  complete 8-level stack costs 1.3x (FLORES) to 1.7x (opus) the bits of the
  text it encodes, and it is lossy. As it stands the code is not a compressed
  format; it is an expansion with a discrete interface.
- **The shallow code is a genuine compression, at a fidelity cost.** Levels
  0-1 cost a fifth of the text's bits and keep 40% of the gold ceiling; levels
  0-3 cost half and keep 55%. The marginal price rises steeply: depth 2 -> 4
  costs 117 bits for +0.13 round-trip, depth 4 -> 8 costs 285 bits for +0.10.
  The deep levels are the expensive, language-dependent part (A6), and they
  are where most of the bits go.
- **The prior sees the redundancy §2 measured.** 64 x 2 slots at 7 bits are
  896 raw bits; the prior needs 49-83. Most slots are predictable from the
  others (fillers and duplicated stacks), so the dense layout is not where
  the cost is; a variable-length bottleneck (B1) would mainly change the
  representation, not the bit count.
- **Code cost is nearly language-independent; text cost is not.** The same
  meaning costs within 15% across ten languages as code and within 42% as
  text. That is the interlingua seen as a bit count.
- **What would make the substrate case.** A code whose rate-distortion curve
  passes below text at useful fidelity: either the deep levels become
  interlingual (B3 pressure on levels 1-7) so they compress across languages,
  or fidelity at depth <= 4 rises (B2 quantizer dropout trains the shallow
  levels to carry more). Both are measured by re-running this script, which
  now gives every B experiment a rate-distortion point.
- **The better seed has the costlier code (cross-seed check, 2026-10-19,
  `results/a4/seed2_500000/prior_codes_d{2,4,8}.json`).** Seed 2 reconstructs
  better (0.651) and its code costs 1.1-1.3x seed 42's on opus and 1.5-1.6x on
  FLORES at every depth; the full stack is 2.1x text on FLORES and 2.2x on
  opus. The opus-to-FLORES ratio is 2.2-2.3x for seed 2 against 1.7x for seed
  42: seed 2's code generalises worse out of domain under the prior, i.e. it
  is more sentence-specific. More meaning in the code and more bits; the
  substrate criterion fails harder for the better model. Code cost per
  language still spreads ~17% (292 zh to 343 hi at depth 4).
- Caveats: the priors were still improving slowly at 30k steps (d8 281 ->
  280 over the last 5k), equally for all streams; the text stream is scored
  losslessly while the code is scored at its own fidelity, so "equal
  fidelity" is not reached by any row; a lossy text baseline (for example a
  summariser at matched round-trip) is the missing comparison.

### Still open in A

Nothing. The seed-2 cross-seed checks of A2, A3, A4 (decoded core) and A5
ran on the GB10 on 2026-10-19 (`scripts/seed2_gb10.sh`, outputs in
`results/a4/seed2_500000/`; a first run on the 50k seed-pair checkpoint by
mistake is kept in `results/a4/seed2_50k/` as an early-training data point:
round-trip 0.443, levels 0-1 prior 40.6 bits on FLORES). The convergence
decision is taken in §8.

## 8. Status after workstream A (2026-10-19) and the plan

### What is in

Every A item has a result, and A2, A3, A5 have a second seed. A1 inventory
of the 500k model (probe undercounts; nPMI finds the content radicals), A2
length curve and depth curve for both seeds, A3 compositionality suite on E3 /
C1 / 500k / 500k seed 2, A4 converged cross-seed test with the shared radicals
decoded in both models, A5 code prior against a text LM for both seeds, A6
stack-level interlingua. Two 500k wafer models exist
(`model_dir/long-b128-20261010T042150Z`, seed 42, round-trip 0.583;
`model_dir/long-b128-seed2-20261014T160630Z`, seed 2, **0.651, the best
model**; `.eval.pt` at 50k/100k/250k/480k/500k and 250k/480k/500k
respectively). A third seed (3) was started on 2026-10-19 (branch 1 below).

### What the results say, together

1. **Level 0 is the script; everything deeper is the model's private
   detail.** Level 0 is interlingual (A6 kappa 0.48 for same slot, same
   code), cheap (A5: levels 0-1 at 0.22x text), coarse-to-fine first (§2),
   and the only level with any cross-seed recurrence (A4). Levels >= 2 are
   language-specific (A6), seed-specific (A4), and carry most of the bits
   (A5) for the last +0.10 of round-trip (A2).
2. **Training buys fidelity with structure.** From E3 to 500k, round-trip
   rose 0.435 to 0.583 while topsim fell 0.20 to 0.13 and the inventory went
   from 15 distinct radicals per sentence to 8.5 plus fillers (§2, A3). The
   reconstruction objective is doing what Kharitonov predicted: minimising
   code information, not organising it.
3. **The inventory is a property of the run, except for a small core the
   data forces.** 19% of level-0 radicals recur across converged seeds. Six
   of those are structural codes matched by frequency; five content radicals
   recur and decode to the same concept in both models: negation, today/date,
   child, health, European Union (A4). The core is ~7% of the inventory and
   consists of frequent, categorical, basic-vocabulary concepts. A fixed
   human-readable inventory cannot be read off a single model as the plan
   assumed, but a small one can be read off the intersection of models.
   Levels 1-7 cannot be compared across seeds at all (residual codes are
   defined relative to level 0), so "seed-specific" there means "unmeasurable",
   not "meaningless".
4. **Sparsity is latent, not realised.** The decoder reads ~17 distinct
   stacks per sentence and the prior needs 49-83 bits for levels 0-1, but the
   model emits 64 slots x 8 levels every time (§2, A5).
5. **The interlingua is the strongest and earliest result**: it is present at
   50k steps as a whole-sentence clustering before any radicals exist (A4),
   and it shows up as a bit count (A5, 15% spread across languages).
6. **Seed variance is large and was never measured.** Two seeds of one
   recipe differ by 0.07 round-trip (0.583 vs 0.651) and by 1.5x in code cost
   under the prior, while every compositionality metric agrees. Single-run
   lever comparisons smaller than ~0.07 (several in PHASE8) are not
   established; the whole-meaning properties are robust and the inventory and
   bit cost are not.
7. **Why: Kruskal identifies the meaning, not its factorisation** (A4, last
   bullets). Parallel data pins down what is encoded, not the coordinates;
   only concepts that act as factors in the data (frequent, sharp,
   independent) get a seed-independent code. Identifying the factorisation
   needs pairs that differ in a known few factors (Locatello 2020), which
   translations are not and minimal pairs are.

### The decision the results force

Two readings of "character set" are now distinct:

- **Shared script**: one inventory that any model trained on the data
  recovers. Requires making level 0 converge. Not achieved (19%).
- **Per-model script**: the inventory is fitted per model like a tokenizer;
  glyphs are assigned by the meaning-space partition a model induces. Works
  for property 6 (one model, one vocabulary) and for a reading of property 5
  where each release of the script is one model's, but not for a script
  people learn once.

Decision (2026-10-19): pursue the shared script by changing the data, not
only the model (branch 2), because finding 7 says the current data cannot
identify a factorisation however the model is regularised; keep D per-model-
capable so the glyph layer does not wait. Revisit after the first minimal-pair
run.

### The plan: two branches (agreed 2026-10-19)

**Branch 1, confirm.** A third 500k seed (`SEED=3 cerebras/runs/long_b128.sh`,
started 2026-10-19, wafer otherwise idle, ~2 days) answers what two seeds
cannot: whether the stable core is a fixed set (the same five concepts recur
in all three pairwise matchings and the three-way intersection) or a pairwise
coincidence (the intersection shrinks), and what the seed error bar on
round-trip is (three runs give a variance). Then `scripts/seed2_gb10.sh
<seed3 ckpt>` on the GB10 (round-trip, A3, A5, level-0 decode) and
`codebook_stability.py` for the two new pairs. This branch cannot change the
conclusion, only its confidence; it never blocks branch 2.

**Branch 2, supply the inductive bias.** Minimal pairs, mixed into the opus100
corpus (tens of thousands, not a replacement): the same sentence with one
labelled factor changed (negation, tense, subject number, one entity swap;
the attributes `compositionality_metrics.py` already labels), generated by
rule-based edits on English opus100 sentences and machine-translated into the
other nine languages so each pair is itself parallel. Training routine:
the two members of a pair co-located in a batch and a penalty on the Hamming
distance between their code stacks beyond a budget of one or two slots, the
discrete form of Locatello 2020's shared-factor aggregation, so the changed
factor is pushed into one radical. New compile on the wafer. Success is read
by the substitution test (below) and by the cross-seed recurring fraction
rising above the current 7% core on two seeds of the new recipe. A cheaper
probe needing no new data, a smaller level-0 codebook (k0 = 16 or 32) on two
seeds, tests whether forcing coarseness alone raises recurrence and can run
between seed 3 and the first minimal-pair run.

**Readout for both: the substitution test (C2).** Swap a stable radical in or
out of real encodings and decode: code 5 (negation) and 102 (today) in seed
2, 60 and 79 in seed 42. On branch 1 it tests whether today's stable core is
composable; on branch 2 it is the success criterion. One script against the
decoder, written first, on the GB10, because both branches wait on it.

Order: seed 3 on the wafer now; the substitution script and then the
minimal-pair corpus on the GB10 while it trains; the branch-2 training run
follows seed 3 on the wafer so the CS-3 queue stays full.

### Previous ranking (2026-10-16), kept for reference

Ranked by what the A results said would move the scorecard, cheapest first
within each tier. Item 3 is done; items 1, 2, 4-6 stand but now sit behind
the two branches above.

1. **B2 quantiser dropout, then B1 variable length, on one wafer (CS-3).**
   The shallow levels must carry more meaning (A5) and the fillers must go
   (§2, A1). Train with random stack truncation so levels 0-1 reach well above
   today's 0.346 round-trip; then introduce the length mechanism late in the
   schedule or on the B2 model, never from scratch (A4: a model in the
   one-code-per-sentence regime would shrink to one slot). Scored by A2
   (round-trip by depth) and A5 (bits by depth): the goal is a
   rate-distortion point below text at >= 55% of gold.
2. **Inventory convergence on the second wafer.** Three cheap controls, each
   a 500k run, each followed by `codebook_stability.py` against seed 42:
   (a) data-derived level-0 initialisation (k-means over encoder means from a
   pretrained run, or over MiniLM sentence embeddings projected to d_code),
   (b) B3's level-0 agreement loss on parallel pairs, (c) FSQ at matched bits
   (B4), where the "codebook" is fixed by construction: if FSQ cells recur
   across seeds the radicals are a data property and the learned codebook is
   the problem. Success is the §5 Stable threshold (two thirds). A6 says
   the same pressure applied to levels 1-7 is also needed for the shallow
   code to be interlingual.
3. **GB10 follow-ups that need no training.** *Done 2026-10-19 except C2*
   (`scripts/seed2_gb10.sh`): seed-2 round-trip 0.651, A3 and A5 cross-seed
   checks, the 15 cross-seed-stable radicals decoded in both models (five
   content concepts, three of them Swadesh items). C2 substitution tests are
   now the shared readout of the two branches above.
4. **D1 glyph grammar on level 0 only, per model.** Render levels 0-1 as base
   plus one modifier, nothing deeper (A6), from a code-to-glyph table that is
   explicitly a per-checkpoint artefact. Specimen page from FLORES. This can
   start now on any CPU and is unblocked by A4 as long as it is per-model.
5. **C1 held-out combinations** on the B2 model, not on today's models: the
   positional code (A3) is the right substrate for it, but only once the
   shallow levels carry the meaning.
6. **E deferred** until a B model passes the A5 rate-distortion test; E1 is
   already served by `code_prior.py`.

### Standing caveats

All compositionality and stability numbers are on the FLORES held-out split
(long sentences, coarse attributes); the A1 probe undercounts the 500k
model; the A5 text baseline is lossless while the code is lossy; the
cross-seed core rests on two seeds until branch 1 reports; the decoded
concepts of the core are read from the A1 probe's aggregate tokens, which
undercounts, so "unreadable" pairs may still be meaningful.

