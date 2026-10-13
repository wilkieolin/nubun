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
| 1 | codes carry meaning, independent of language | **strong** | round-trip cross-lingual meaning 0.582 (gold 0.861, shuffle 0.044); interlingua Jaccard lift +0.278, positive on all 45 pairs |
| 2 | codes are interpretable radicals | **good** | level-0 inventories (pronoun paradigm, negation, modals, dozens of content concepts) in `RADICALS.md`; the 500k model's inventory is not decoded yet |
| 3 | the representation is sparse | **no** | 64 fixed slots = 3584 raw bits/sentence, but only ~8.5 distinct level-0 codes and ~17 distinct stacks per sentence are used (§2); no length or sparsity pressure exists in the loss |
| 4 | radicals compose | **unmeasured** | coarse-to-fine RVQ structure seen qualitatively (code norms fall 10.9 to 3.5 across levels); no posdis/bosdis, TRE, nPMI or held-out-combination test has been run |
| 5 | human-readable glyph layer | **not started** | nothing renders a code to a glyph; no reading, no legibility or learnability test |
| 6 | AI reasoning substrate | **not started** | no language model over code sequences; no bits-per-sentence of a code prior; no LLM has seen Nubun codes |

The honest summary: **we have a working, interpretable, cross-lingual discrete
bottleneck; we do not yet have a character set.** The bottleneck is dense (every
sentence is 64 x 8 codes), its compositionality is asserted rather than
measured, and the two user-facing halves of the goal (glyphs for humans, tokens
for models) have no code behind them.

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
  a truncated level stack whose round-trip is within 0.02 of 0.582. Fail: the
  round-trip drops by more than 0.05 when slots are reduced below 32, meaning
  the duplication seen in §2 was load-bearing after all.
- **Composable**: posdis or bosdis clearly above shuffled-attribute chance;
  TRE with additive composition well below a permuted-code control; held-out
  attribute combinations reconstructed at least as well as seen ones within
  0.05 round-trip. Fail: combinatorial generalisation collapses to the
  no-code floor.
- **Stable**: at least two thirds of level-0 radicals recur across seeds after
  matching. Fail: inventories are seed-specific.
- **Readable**: a learner reaches above-chance sentence glossing of specimen
  sentences within the 3,000-exposure budget. Fail: no learning curve.
- **Substrate**: the code prior is materially more compressible than a text LM
  at equal round-trip fidelity, and the mixed code/text fine-tune does not
  lose accuracy relative to text-only. Fail: codes cost more bits than text.

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

| level 0 | E3 @100k | C1 @130k | CS-3 @500k |
|---|---|---|---|
| posdis excess (slot specialises to one attribute) | +0.008 | +0.017 | **+0.027** |
| bosdis excess (code count specialises) | **+0.020** | +0.018 | +0.009 |
| MIG, bag / positional | 0.049 / 0.022 | 0.033 / 0.014 | 0.021 / 0.002 |
| topsim, bag Jaccard vs MiniLM (perm ~0) | **0.20** | 0.19 | 0.13 |
| TRE, additive, held-out 1-R^2 (perm 1.00) | **0.86** | 0.88 | 0.90 |
| nPMI cross-language stability r (shuffled) | 0.57 (0.50) | 0.60 (0.50) | **0.67 (0.59)** |
| A6 level-0 bag Jaccard real / chance / kappa | 0.41 / 0.19 / 0.28 | 0.42 / 0.18 / 0.30 | 0.59 / 0.33 / **0.38** |
| A6 same code in same slot, real / chance / kappa | 0.37 / 0.10 / 0.30 | 0.44 / 0.09 / 0.38 | 0.53 / 0.10 / **0.48** |
| A6 depth-2 stack prefix bag kappa | 0.10 | 0.11 | 0.12 |
| decode-probe codes strong in >=3 languages (A1) | **55** | 19 | 17 |

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
- **Depth > 2 is not interlingual.** Stack-prefix agreement across languages
  falls from kappa 0.38 at depth 1 to 0.12 at depth 2 and 0.04 at depth 3:
  levels 2-7 carry language-specific surface detail. This supports B2's cut
  (radical = levels 0-1, everything deeper optional) and suggests the glyph
  layer should not render levels >= 2 at all.
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

### A2: round-trip length curve

Running (100k / 250k / 480k, plus 500k as a protocol check against 0.582).

### Not started in A

A4 (cross-seed stability) needs two wafer runs; A5 (code prior) needs a code
dump of the opus100 training corpus with the 500k encoder.
