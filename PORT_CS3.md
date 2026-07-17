# Porting Nubun to the Cerebras CS-3 — cold-start guide

Read this if you're picking up the Nubun VQ-VAE on an x86 box with the
`cerebras-pytorch` wheel and no prior context. Companion: `PHASE8.md` (how the
frozen model was built and why), `cerebras/` (the port scaffold).

## 1. What Nubun is (30 seconds)

Nubun learns a **synthetic composable logography**: a multilingual sentence is
encoded by a Perceiver encoder into a discrete bottleneck of "characters"
(a residual vector-quantized codebook), and an autoregressive decoder
reconstructs the sentence *in any target language* from those codes. The codes
behave as a cross-lingual **interlingua** — the same "radical" fires on the same
meaning across 10 languages / 5 scripts (e.g. one code = "water", another =
"woman", another = the pronoun "we"). Success is measured by **free-generation
round-trip meaning**: generate, re-embed with MiniLM, cosine to the source.

The frozen best model reaches round-trip cross-lingual meaning **0.435**
(gold/translation ceiling 0.861), with interpretable radicals and a measured
interlingua lift of +0.239. See `PHASE8.md`.

## 2. Why the CS-3

The dominant remaining quality gap is **capacity** (continuous ceiling 0.430 →
gold 0.861), which needs bigger/longer training runs. The model is small
(~135M params) and **memory-bandwidth-bound** — on the GB10's ~273 GB/s LPDDR5X
that bandwidth is the ceiling. On the CS-3 the whole model is resident in ~44 GB
of on-wafer SRAM at ~21 PB/s, which is the ideal case for a small bandwidth-bound
job. The gate check (below) confirmed the custom RVQ traces cleanly under
`cstorch`; the port's real work is re-expressing the training loop in the cstorch
idiom, not the ops.

## 3. The frozen model

Checkpoint: `data/phase8_e3_rvq8_200k_step100000.pt` (PyTorch, ~1.6 GB).
Architecture (all reconstructable from `ckpt["args"]`):

- Perceiver encoder (4 layers) → **RVQ 8×128** (8 residual codebooks, k=128,
  `d_code=256`) → AR decoder (**10 layers, d_ff 2048**), output projection tied
  to the input embedding.
- `d_model=384` — **pinned** by the pretrained XLM-R embedding table
  (`data/embedding_table.pt`, shape `(250002, 384)`). You cannot change d_model
  without adding input/output projections (see "capacity levers" below).
- Encoder & decoder token embeddings are **tied** (one table) and **unfrozen**.
- Trained on opus100 (~8.5M pairs, 10 langs), 100k steps, cosine LR + semantic
  loss (λ=5) + token-weighted CE.

Model code: `vqvae/model.py`, `vqvae/encoder.py`, `vqvae/decoder.py`,
`vqvae/quantizer.py` (the `ResidualVectorQuantizer`). Loss helpers:
`vqvae/losses.py`. The GB10 training loop (reference): `train_vqvae.py`.

## 4. Files transferred separately (NOT in git — `data/` is gitignored)

`rsync`/`scp` these from the GB10 box into `data/`:

| file | ~size | needed for |
|------|------|-----------|
| `data/phase8_e3_rvq8_200k_step100000.pt` | 1.6 G | port/eval the frozen model |
| `data/embedding_table.pt` | 367 M | build the model (required always) |
| `data/parallel_corpus.npz` | 3.5 M | eval / semantic-target demo |
| `data/token_weights.pt` | 1 M | token-weighted CE (M2) |
| `data/opus100/*.npz` | ~1.3 G | only if you will *train* on CS |

## 5. Environment

x86 only. Make a **separate** env from the GB10 `nubun` conda env (different
Torch build):

    python -m venv ~/venvs/nubun-cs && source ~/venvs/nubun-cs/bin/activate
    pip install -r cerebras/requirements-cerebras.txt   # pin cerebras_pytorch to the cluster release

Client and appliance `cerebras_pytorch` versions **must match** the CS-3
cluster's software stack — check with the cluster admin / `csctl` before pinning.

## 6. Step 1 — the gate check (no wafer time)

    python cerebras/compile_check.py \
        --checkpoint data/phase8_e3_rvq8_200k_step100000.pt \
        --embedding-table data/embedding_table.pt

This builds the frozen model, wraps it with `cstorch.compile`, and traces ONE
train step on synthetic data in **`compile_only`** mode (no wafer). Success =
`COMPILE CHECK PASSED`. Any failure names the op/pattern that won't lower —
that's your work-list. The known culprit (`bincount`) is already fixed in
`vqvae/quantizer.py` (replaced by an equivalent static-shape `one_hot().sum()`).

## 7. Step 2 — train

    # Milestone 1: recon + RVQ only (prove the model trains on the WSE)
    python cerebras/train_cstorch.py --no-semantic --steps 2000 --compile-only   # dry compile
    python cerebras/train_cstorch.py --no-semantic --steps 2000                  # real run

    # Milestone 2: full recipe (precompute semantic targets first)
    python cerebras/precompute_semantic_targets.py \
        --token-ids data/parallel_corpus.npz --out data/sem_targets_parallel.npy
    python cerebras/train_cstorch.py --steps 100000

`train_cstorch.py` mirrors `train_vqvae.py` in cstorch form. **You must replace
its synthetic `make_input_fn` with the real opus pipeline** — adapt
`vqvae/data.py::Opus100Dataset` into a cstorch `DataLoader` input function that
yields `{src_ids, tgt_ids, tgt_lang_id[, sem_target]}`. That data-pipeline
adaptation is the largest remaining task and is version-specific.

## 8. cstorch constraints & how our code already handles them

Everything inside a `@cstorch.trace` step becomes a static graph. Rules and our
status:

- **No tensor-value reads in the traced step** (`.item()`, `.max()`-in-a-python-
  `if`, `print(tensor)`, `.to("cpu")`) — they raise `aten::_local_scalar_dense`.
  → All logging/`.item()` lives in `@cstorch.step_closure` in the scaffold.
- **No data-dependent Python control flow** — the RVQ's `for level in
  range(n_levels)` is a *fixed-count* loop (static unroll, fine) and `if
  level == 0` is a Python-int test (trace-time, fine). ✓
- **Straight-through estimator** (`x + (q - x).detach()`) is pure tensor ops. ✓
- **`bincount`** (data-dependent output size) → already swapped for
  `F.one_hot(idx, k).sum(0)` in `vqvae/quantizer.py`. ✓
- **Frozen MiniLM must not be in the trace** → precompute its targets offline
  (`cerebras/precompute_semantic_targets.py`), stream them as `sem_target`. 
- **Optimizer / LR scheduler** must be `cstorch.optim.*` (plain torch ones may
  not lower). The scaffold uses `cstorch.optim.AdamW` + a cosine `LambdaLR`.
- **LR schedule length**: decay cosine over the *actual* run length. Do NOT
  stretch cosine past the step count — a hot-LR tail collapsed the RVQ codebook
  on the GB10 (a 200k run died at 110k). See `PHASE8.md`.

Every cstorch call in the scaffold is marked `# VERIFY` — API names/kwargs can
shift between releases; check them against your installed version.

## 9. Getting a checkpoint back to the GB10 for eval

The round-trip metric lives on the GB10 (`roundtrip_eval.py`, needs the MiniLM
re-embedder). To score a cstorch-trained model, convert its saved `model`
state_dict into a `train_vqvae`-style checkpoint dict:

    ckpt = {"step": STEP, "args": vars_of_the_frozen_config,
            "model_state": cs_state["model"]}
    torch.save(ckpt, "data/phase8_csX_stepSTEP.pt")

then on the GB10:

    CUDA_MEM_FRACTION=0.40 python roundtrip_eval.py \
        --checkpoint data/phase8_csX_stepSTEP.pt \
        --langs en,zh,es,fr,ar,ru --combine-splits --max-gen-len 48

Compare against the frozen winner's **0.435** on the same seeded split. (State
dict keys match because the model class is identical — tying saves both
`encoder.token_emb.weight` and `decoder.token_emb.weight` with equal data.)

## 10. Capacity levers to try once training runs on CS

The point of the port is to afford bigger continuous-ceiling runs:

- **More / longer training** with a correctly-scoped cosine (don't repeat the
  hot-LR-tail collapse).
- **Decouple `d_model`** from the 384-pinned embedding: add an emb→d_model input
  projection and a d_model→emb output projection (keep output-tying via the
  projection), then widen d_model. This is the main width lever.
- **Direct, non-English-centric pairs** (opus100 is all X↔en) to pressure the
  codes to be language-neutral and strengthen the interlingua.
