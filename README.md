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

## Status (2026-10-12)

| property of the goal | state | evidence |
|---|---|---|
| codes carry meaning across languages | **strong** | free-generation round-trip meaning 0.582 of a 0.861 gold ceiling; shuffle-gap +0.538; interlingua Jaccard lift +0.278 on all 45 language pairs |
| codes are interpretable radicals | **good** | level-0 inventories for E3 and C1 (`RADICALS.md`); the 500k wafer model's inventory is not yet decoded |
| representation is sparse | **not yet** | 64 fixed slots, 3584 raw bits/sentence; only ~8.5 distinct level-0 codes and ~17 distinct code stacks per sentence are actually used (`PHASE9.md` §2) |
| radicals compose | **unmeasured** | coarse-to-fine RVQ structure observed qualitatively; no compositionality metric has been run |
| human-readable glyph layer | **not started** | no rendering, no reading, no legibility test |
| AI reasoning substrate | **not started** | no language model over codes, no bits-per-sentence of a code prior |

Best model: `model_dir/long-b128-20261010T042150Z/nubun_cs_step500000.eval.pt`
(trained on the Cerebras CS-3, d_model 384, RVQ 8x128, 500k steps at B=128).
Best GB10 model: `data/phase8_C1_d512_cos130k_step130000.pt` (0.5045).

## Documents

| file | what it is |
|---|---|
| `PHASE9.md` | **start here**: review of progress against the goal, new measurements, and the proposed next phase |
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
