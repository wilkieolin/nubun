# Nubun — a learned, composable synthetic logography

Nubun is an experiment in deriving a **sparse, composable set of semantic
"radicals"**: a character set that combines like Korean Hangul (a small inventory
of components assembled into blocks by fixed rules) but carries meaning like
Chinese radicals and characters (each component a concept, compounds built from
concepts). The end goal is a script that is **human-readable** and at the same
time a **natively tokenized semantic space** in which AI models could reason in a
compressed, language-neutral form.

The current instrument is a multilingual sentence **VQ-VAE**: a Perceiver encoder
maps a sentence in any of 10 languages to 64 slots, each slot is quantized by an
8-level residual vector quantizer (8 codebooks x 128 codes), and an autoregressive
decoder regenerates the sentence in any target language from the codes alone.
The codes behave as a cross-lingual interlingua: one level-0 code fires for
"water", another for "we", another for "not", in every language and script.

## Status (2026-10-28, after Phase 9 workstream A on three seeds)

| property of the goal | state | evidence |
|---|---|---|
| codes carry meaning across languages | **strong** | free-generation round-trip meaning 0.583 / 0.651 / 0.647 for three seeds of one recipe (sd 0.038) against a 0.861 gold ceiling; interlingua Jaccard lift +0.278 on all 45 language pairs; same slot, same level-0 code across translations 53% vs 10% chance; code cost varies 15% across languages, text 42% |
| codes are interpretable radicals | **good, seed-specific, small stable core** | level-0 inventories (`RADICALS.md`; the 500k model's in `PHASE9.md` §7 A1); runs differing only in seed share 14-27% of their level-0 radicals, a fixed set across three seeds whose content members decode to the same concepts (European Union, health, an international-body radical, weakly child; `PHASE9.md` §7 A4); the other ~95% is seed-specific |
| representation is sparse | **not yet** | 64 fixed slots, 3584 raw bits/sentence; ~8.5 distinct level-0 codes and ~17 distinct stacks per sentence are used; a learned prior needs 49-83 bits for levels 0-1 |
| radicals compose | **weak, not established** | the code is positional (keep RVQ), but additive composition explains 10-14% of it and topographic similarity falls as round-trip rises; identical on three seeds; no held-out-combination or substitution test yet |
| human-readable glyph layer | **not started** | no rendering, no reading, no legibility test; must be per-model until the inventory converges |
| AI reasoning substrate | **measured, failing** | full 8-level code costs 1.3x to 2.5x the bits of the text it encodes depending on seed; levels 0-1 cost 0.22-0.39x at 40% of gold fidelity |

Models: `model_dir/long-b128-seed2-20261014T160630Z/nubun_cs_step500000.eval.pt`
(CS-3, seed 2, d_model 384, RVQ 8x128, 500k steps at B=128, **round-trip
0.651, best model**) and `model_dir/long-b128-20261010T042150Z/nubun_cs_step500000.eval.pt`
(same recipe, seed 42, 0.583) and `model_dir/long-b128-seed3-20261019T210243Z/nubun_cs_step500000.eval.pt`
(seed 3, 0.647). Best GB10 model:
`data/phase8_C1_d512_cos130k_step130000.pt` (0.5045).

Where to go next: `PHASE9.md` §8 (consolidated status and the two-branch plan:
confirm with a third seed, done 2026-10-28; add minimal pairs to supply the
inductive bias). Branch 2: the minimal-pair corpus pipeline and its LLM-judge
filter are built and calibrated against a human check (`augment/README.md`
§9-10). The trainers are being extended to mix the pairs in.

## Documents

| file | what it is |
|---|---|
| `PHASE9.md` | **start here**: goal scorecard (§1), workstream plan (§4), success criteria with outcomes (§5), workstream A results (§7), consolidated status and next steps (§8) |
| `LITERATURE.md` | literature review across ML, linguistics/writing systems, and information theory, with implications for Nubun |
| `RADICALS.md` | curated level-0 radical inventories with cross-lingual evidence |
| `PHASE8.md` | scale-up results: RVQ depth, width, schedule, data lever, CS-3 500k run |
| `PHASE6.md` | the posterior-collapse diagnosis and the round-trip meaning metric |
| `PORT_CS3.md` | cold-start guide to the Cerebras CS-3 port |
| `cerebras/CS3_BRINGUP.md` | wafer bring-up log: cstorch op rules, run hygiene, run records |

## History in one paragraph

Phases 1-2 tried to derive radicals directly by sparse decomposition (NMF,
k-means + OMP) of multilingual word embeddings; 500 atoms covered 93% of 5k
concepts at 4 atoms each, but the atoms were not stable or usable. Phases 3-5
moved to an end-to-end sentence VQ-VAE and added a semantic-target loss and
frequency-weighted CE to push punctuation out of the codes. Phase 5c/6 found the
decoder was ignoring the codes (posterior collapse) and replaced teacher-forced
accuracy with the free-generation round-trip meaning metric that is used ever
since. Phase 7 introduced residual VQ (a stack of codes per slot) and the first
radical report. Phase 8 scaled: 8 RVQ levels, tied embeddings, deeper decoder,
width 512, gentler schedule (0.305 to 0.5045), then a 500k-step run on the
Cerebras CS-3 (0.582). `LITERATURE.md` §3 explains, after the fact, why the
Phase 1-2 approach could not have worked and why the multilingual VQ-VAE could.

## Reproducing the headline numbers

```bash
# round-trip meaning (needs a GPU box with MiniLM; the seeded 200-sentence split)
python roundtrip_eval.py --checkpoint <ckpt> --langs en,zh,es,fr,ar,ru --combine-splits --max-gen-len 48

# interlingua: bag-of-level-0-codes Jaccard vs chance, all 45 language pairs (CPU is fine)
python crosslingual_consistency.py --checkpoint <ckpt> --langs en,zh,es,fr,ar,ru,hi,ja,pt,de --combine-splits

# code usage / effective bits per sentence (CPU is fine)
python code_usage_audit.py --checkpoint <ckpt>

# radical discovery (decodes thousands of code sequences; GPU)
python analyze_codebook.py --checkpoint <ckpt> --rvq-level 0
```

Checkpoint and code versions must match. GB10 checkpoints up to C1 load with the
pre-port `vqvae/` (commit `df5c6ae` or branch `gb10-scaleup`); CS-3 `.eval.pt`
checkpoints need the current `vqvae/` (the wafer port changed the attention
packing, memory layout and output bias). Use a `git worktree` for the old code.
`data/` is not in git; see `PORT_CS3.md` §4 for the files and `CS3_BRINGUP.md`
§0 for where they are staged on the ALCF node. If `parallel_corpus.npz` was
written with numpy 2 and you load it under numpy 1, alias `numpy._core` to
`numpy.core` before loading.
