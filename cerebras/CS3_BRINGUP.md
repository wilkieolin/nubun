# CS-3 bring-up log & handoff — getting the Nubun VQ-VAE to compile on the wafer

Written for an agent/engineer picking this up **on the ALCF Cerebras user node**
(`cer-usn-01`/`cer-usn-02`), where you can run `cszoo`/`csctl` and read compile
logs directly — no GCE↔ALCF file sync. Companion docs: `PORT_CS3.md` (cold-start),
`PHASE8.md` (the frozen model). This file is the *debugging* record: what breaks,
why, and the loop to keep going.

## 0. TL;DR status (2026-07-19)

**`minprobe-vqvae` COMPILES AND EXECUTES ON THE WAFER.** As of `c404248`
(branch `phase8-cs3-wafer-compile`) the full VQVAE — RVQ on, embeddings tied —
compiles and runs 5 steps on the CS-3 with the loss descending:

```
minimal_probe: model=vqvae rvq=True tie=True
step 0: loss=97.0147   step 3: loss=45.7841
minimal_probe: DONE — VQVAE compiled and executed.
```

That closes §7 item 1. Next action: §7 item 2 (propagate int32 ids + host-side
label one-hot to `train_cstorch.py` / `opus_input.py` / `compile_check.py`),
then `smoke`.

Known gaps left deliberately: `VectorQuantizer` (non-RVQ) still uses the
argmin+i64 pattern, so `--no-rvq` bisection hits the i64 abort; `nat_decoder.py`
has an untouched buffer slice (:42) and arange (:113); no `GradScaler` yet.

**The model now diverges from the frozen GB10 model** (see §5) — CS-3 trains from
scratch; reproduce the frozen 0.435 model with *pre-port* code (git before the
Phase-8 CS-3 commits).

## 1. THE key fact: offline traces catch nothing

Our local `compile_check.py` / `minimal_probe.py` runs stop at
`ClusterConfigError: mgmt_address has no default value`. That is NOT success — it
is the client giving up right before dispatch. The pipeline has 4 stages:

1. build lazy graph (client) — a *recorder*; every `aten` op "traces clean".
2. lower lazy → CIRH (client).
3. dispatch to cluster ← **offline dies here (ClusterConfigError)**.
4. **compile CIRH → WSE image (cluster coordinator)** ← every real error is here.

So all op-lowering / kernel-match / layout / placement errors only appear on the
**cluster**. `--compile_only` still compiles on the cluster. There is no offline
WSE compiler. `cstorch.backend("CPU")` runs eagerly and catches nothing either.
Consequence: you discover the compiler's supported-op surface one wafer compile
at a time. On-node you can at least iterate fast (keep a warm `tmux` session).

## 2. THE diagnostic loop (do this every iteration)

Client only ever sees a terse `502` / `Cannot compile empty CIRH module`. The
real reason is server-side. Loop:

```bash
# 1. run the canary (from the repo root, in the R_2.10.0 venv + proxy):
bash cerebras/submit_alcf.sh minprobe-vqvae      # or smoke / m1 / minprobe-attn2

# 2. get the failing compile wsjob id:
csctl get jobs -a | grep name=minprobe-vqvae | tail

# 3. export its coordinator logs (run from repo root so it lands here):
csctl log-export <wsjob-id>

# 4. read the REAL error (not the client 502):
cd log-export-<wsjob-id>-*/coordinator-0
grep -niE "FATAL|error:|Failed to convert|no corresponding kernel|Unable to insert|SelectLanesAndLayout|no valid layouts|slice_filter|Exception|Precompiling|\.cc:[0-9]" dbg_crd_0.out | grep -viE "0x0|libMLIR|libLLVM" | head

# 5. pin the op to a source line via the MLIR loc:
cd ../cs_*_user*/
grep -nE "ws_km\.<offending_op>" ws_km.mlir | sed -E 's/#cirh[^>]*>//g' | head   # shapes
grep -nE "loc\(" ws_km.mlir | grep -iE "vqvae/|arange|<opname>" | head            # source lines
```

`dbg_crd_0.out` = compile stdout (progress + FATAL). The `loc(...)` strings in
`cirh.mlir` / `ws_km.mlir` carry the full Python callsite (`vqvae/....py:line`)
— that's how you know which line produced the bad op.

**`dbg_crd_0.err` is NOT always ignorable.** It is usually just the
graceful-shutdown message, *but when the compiler dies by signal it is the only
place the reason appears*. If `dbg_crd_0.out` ends mid-sentence with no FATAL —
e.g. trailing off at `Exploring floorplans... 100%` — the coordinator was
aborted, `.out` was truncated, and the assertion is in `.err`:

```bash
grep -niE "Assertion|CEREBRAS_THROW|killed by signal" .../coordinator-0/dbg_crd_0.err | grep -v '^\s*#'
# e.g. target_arch.cc:1293: get_mov_simd_width: Assertion `num_bits == 16 || num_bits == 32' failed.
```

This cost a full misdiagnosis once: the client showed only a bare 502, `.out`
had no error, and the real cause (an i64 tensor on the wafer) was in `.err`.

Then: fix → verify numerically + trace locally (see §4) → commit → re-run.

## 3. Discoveries: the failure classes and the rules

### THE governing rule — ask about the tensor, not the op

> **Nothing reaches the wafer except 16/32-bit floats and int32, arriving either
> as a streamed activation (with a batch/streaming dim) or as a weight.**

Two corollaries cause most failures, and neither is visible if you reason about
which *ops* are supported:

- **i64 cannot live on the wafer.** There is no 64-bit datapath;
  `get_mov_simd_width` asserts `num_bits == 16 || 32` and the compiler *aborts*
  (SIGABRT, no diagnostic — see §2 on `.err`). Casting on-wafer does not help:
  reading the i64 to cast it is itself a 64-bit move. **Integers must be cast on
  the host, or the computation reworked in float.**
- **A constant-only subgraph has nowhere to live.** With every operand
  data-independent there is no streaming dim, so the compiler places the whole
  thing on `GlobalHost` — where there is no kernel ("Failed to convert op to
  WAF") — or tries to stream the constant in ("no valid layouts"). **If a tensor
  is knowable at construction time, build it as a buffer in `__init__`; do not
  compute it in the traced step.**

Why this framing and not an op table: `5e47df6` replaced a failing positional
embedding with `onehot @ pe` — a matmul, an unambiguously supported op — and it
still failed, because *both operands were constant*. Likewise `cols > rows` is a
compare, and compares lower fine, but a compare of two aranges is constant-only.
The op was never the problem. Check the **dtype** and the **provenance** of every
tensor you introduce.

One trap: **some i64 is silently tolerated, which hides the rest.** Embedding
lookups lower to `ws_km.onehot`, which absorbs i64 indices itself. So i64
`src_ids` feeding an embedding is fine — until the same tensor also feeds a
pad-mask compare, which is not. Expect to remove i64 in several rounds, each
revealing the next.

### The two older op-shaped classes (still valid, now secondary)

**When adding any op, apply these rules; do not reach for stock-PyTorch idioms.**

### Class A — fused / specialized kernels don't lower → rewrite in primitives
| Don't use | Symptom | Use instead |
|---|---|---|
| `nn.MultiheadAttention` / `nn.Transformer*` (fused SDPA) | empty CIRH module; `aamatmul` assert | explicit `softmax(QKᵀ/√d)·V`, separate q/k/v Linears (`cs_attention.py`) |
| `F.cross_entropy` (fused `cross_entropy_loss`) | `wgth.cast` placement error over vocab | `log_softmax` + `gather` + mask |
| `nn.LayerNorm` (when the normalized-adjacent dim is weight-shaped, e.g. the readout's M query slots) | `Failed to convert op to WAF` | explicit mean/var/rsqrt (`cs_attention.LayerNorm`) |

### Class B — touching a weight/buffer directly → route through F.linear/F.embedding/onehot/broadcast
| Don't do | Symptom | Use instead |
|---|---|---|
| `weight.split(...)` / `weight[:E]` (slice a param) | `aten::split_copy` "Unable to insert transfer WGT_HOST→WSE" | project with full weight; or separate Linears |
| `x @ weight.t()` (explicit weight transpose) | transfer/placement error | `F.linear(x, weight, bias)` |
| `codebook[idx]` (advanced-index a weight) | weight-gather to WSE | `onehot @ codebook` (onehot via `arange` compare) |
| `self.pe[:, :T]` (slice a buffer) | `ws_km.slice` → **slice_filter kernel asserts** | one-hot matmul `onehot @ pe` |
| `F.embedding(arange(T), pe)` (arange as gather index) | `act_host_to_wse i64` "no valid layouts" | one-hot matmul (arange in a *compare*, not an index) |

### Class C — an i64 or constant-only tensor on the wafer
| Don't do | Symptom | Use instead |
|---|---|---|
| int64 ids from the dataloader | `act_host_to_wse i64` "no valid layouts", or SIGABRT in `get_mov_simd_width` | yield `int32` (F.embedding accepts int32 indices) |
| `logits.gather(1, labels)` (i64 gather index) | i64 streamed to wafer | pass labels **already one-hot** as a float activation from the host |
| `(labels[:,None] == arange(V))` on-wafer | (N,V) integer tensor + big backward fan → "did not fit in memory" | same — build the one-hot host-side |
| `dist.argmin(1)` then `idx == arange` | argmin yields i64 → (N,k) i64 | `onehot = (dist == dist.amin(1,keepdim=True))`, normalize row sums for ties |
| `x.min(dim=1).values` | also emits an i64 **indices** output that survives | `x.amin(dim=1)` |
| `pe[:T]` / `onehot @ pe` / `arange` mask | constant-only → GlobalHost or streamed constant | size the buffer to exactly T in `__init__`; broadcast-add |

### Other hard rules discovered
- **NO in-graph slicing at all.** `slice_filter` asserts on *every* `ws_km.slice`
  we produced — float channel-slices AND int64 sequence-slices. Anything that
  would slice a tensor in the traced step must be moved to the host / dataloader
  or expressed as a matmul/compare. This is why the AR teacher-forcing shift
  (`tgt[:, :-1]`/`tgt[:, 1:]`) is now done on the host (`pre_shifted=True`).
- **`concat` rejects bool (i1).** Build masks in `int32`, derive with `== 0`.
- **`bincount` / `F.one_hot` don't lower** (call `.item()` internally). Use
  `(idx[:,None] == arange(k))`.
- **`torch.triu(bool)` doesn't lower.** Causal mask = `arange<arange` (a compare).
- **`arange` is fine in a COMPARE, not as a gather INDEX or a streamed constant.**
- **You must fetch a graph output every step** (unconditional `loss.item()` in a
  `step_closure`) or the whole step is DCE'd to an empty module.
- **Weight-only / data-independent subgraphs are dangerous.** A `LayerNorm` whose
  input is purely parameter-derived has no WSE kernel (the wafer streams samples;
  a constant has no streaming dim). Fixed by seeding readout queries with the
  encoder summary. (Elementwise/matmul on constants seem OK; norms are not.)

### Non-issues (ruled out — don't chase these)
- **`client 1.14.0 vs server 1.20.2` version warning is BENIGN.** A fresh modelzoo
  gpt3 compile succeeds through it. Never the cause.
- **Bare `502` with no coordinator FATAL** = transient infra — **but only if it
  does not reproduce at the same stage.** Transient infra is real on this cluster
  (image-build jobs fail and auto-retry), so the temptation to wave a 502 away is
  strong. The test is reproducibility: retry once and compare *where* the
  coordinator log stops. Two runs dying at the identical point (e.g. both at
  `micro batch size 2` → `Exploring floorplans... 100%`) is a real graph bug, not
  infra — go read `dbg_crd_0.err` for the assertion (§2). Confirm cluster health
  with `bash cerebras/submit_alcf.sh minprobe-attn2` (known-good), but note it
  runs in EXECUTE mode, so it can fail on wafer contention (`ClockSync
  DEADLINE_EXCEEDED`) while the compile path is perfectly healthy.
- Compile-only vs execute, raw-generator vs torch DataLoader, AdamW+warmup-from-0:
  all tested and fine.

## 4. Verify locally before every wafer run (cheap, no cluster)

Two checks, in the `cerebras` conda env on any x86 box (or on-node):

```bash
# (a) numeric equivalence of the rewrite vs the original op (must be ~1e-6):
python - <<'PY'  # e.g. F.linear vs @W.t(); onehot@cb vs cb[idx]; explicit LN vs nn.LayerNorm
...
PY
# (b) it still traces (expect ClusterConfigError locally = the graph built fine):
python cerebras/minimal_probe.py --model vqvae 2>&1 | grep -iE "ClusterConfigError|Error|Traceback"
```

Every fix so far was verified numerically identical (diffs 1e-7..0.0) so GB10
numerics are preserved *where the op wasn't structurally changed*. Structural
changes (separate q/k/v, mem broadcast-add, readout seeding) are documented as
divergences (§5).

## 5. Divergences from the frozen GB10 model (important)

These change the architecture/state_dict, so the frozen `phase8_e3` checkpoint no
longer loads into the current code, and forward numerics differ. **This is
accepted** — CS-3 does fresh capacity runs; frozen eval uses pre-port code.
- Attention: `in_proj_weight` (packed) → separate `q_proj/k_proj/v_proj`.
- Readout queries: seeded with `h.mean` (input-conditioned) instead of pure learned.
- Decoder memory: lang tag broadcast-added to every slot instead of a prepended
  slot (memory length M, not M+1).
- Positional embedding `pe` buffer: `(1,4096,d)` persistent → `(seq_len,d)`
  non-persistent, sized exactly to T via the new `src_len`/`tgt_len` args, and
  added by broadcast (no selection op at all).
- Causal mask: computed per-forward → precomputed `(T,T)` float buffer.
- AR shift moved to the host (`pre_shifted=True`; dataloader yields `dec_in`/`labels`).
- RVQ code selection: `argmin` → float `amin` compare, tie-normalized. Identical
  when the row min is unique; on an exact tie (reachable under codebook collapse,
  where dead/duplicate codes give equal distances) it splits the assignment
  across tied codes instead of argmin's lowest-index-wins.
- `quantizer` level-0 `idx0` is now **float**, not int64. Cast on the host where
  integer codes are needed (`idx0.round().to(torch.int64)`). The stop-mask
  helpers (`get_stop_mask`/`force_stop_at`/`first_stop_position`) still assume
  integer codes and need updating when `use_stop_mask` is enabled (M6).
- Dataloader dtypes: token ids are `int32`, and the decoder labels cross over
  already one-hot as f16 plus a separate float pad mask.

CS-3-trained checkpoints WILL round-trip to GB10 **if GB10 runs this same code**
(§9 of PORT_CS3). To eval them, run the current `vqvae/` on the GB10, not pre-port.

## 6. The canary: `cerebras/minimal_probe.py`

Standalone raw-cstorch loop with bisection modes (via `submit_alcf.sh <mode>`):
- `minprobe` — canonical MLP (baseline sanity / cluster health).
- `minprobe-attn2` — hand-rolled attention; **known-good**, use as cluster health check.
- `minprobe-vqvae` — the real `VQVAE` (small synthetic config) = the actual canary.
  Flags: `--no-rvq`, `--no-tie` for further bisection.
- `minprobe`/`-raw`/`-adamw`/`-attn` — historical bisection probes (input, optimizer,
  torch attention); keep for regression.

`smoke` = the real M1 graph (train_cstorch) in execute mode, synthetic data.
`compile`/`m1`/`m2` = the actual training milestones.

## 7. What's left to do (in order)

1. ~~**Finish the compile of `minprobe-vqvae`**~~ — **DONE** (`c404248`). Compiles
   and executes; see §0.
2. **Propagate to the real path** (held until the canary compiled — now unblocked).
   Everything the canary needed applies here too:
   - `cerebras/train_cstorch.py` `train_step`: call `model(..., pre_shifted=True)`.
     `weighted_ce` must take a host-side **one-hot** label activation and a float
     pad mask — its current `.gather()`/`tgt[:,1:]` will hit the i64 abort.
   - `cerebras/opus_input.py`: yield `dec_in`/`labels_onehot`/`label_mask`
     pre-shifted on the host, token ids as **int32**, not int64.
   - `cerebras/compile_check.py`: same pre-shift and dtypes.
   - Cast `idx0` back to int on the host wherever codes are consumed.
3. **Apply the Class-C fix to `VectorQuantizer`** (non-RVQ, `quantizer.py:73-78`)
   — it still uses argmin + i64 arange, so `--no-rvq` bisection currently aborts.
4. **`nat_decoder.py`**: buffer slice at :42 and arange at :113 are untouched
   Class-C/slice landmines; fix before any NAT compile.
5. **Add `cstorch.amp.GradScaler`** for the real fp16 runs (the persistent
   "half dtype float16 but no grad scaler" warning → gradient underflow in M1/M2).
   Held deliberately so it didn't muddy the compile bisection.
6. **`smoke` → `m1`** (needs data staged on ALCF: embedding_table.pt,
   token_weights.pt, opus100 shards; M2 also needs `precompute_opus_semantic.py`).
7. Update `PORT_CS3.md §8/§10` with the final op-rewrite list and the divergences.

## 8. Commit trail (Phase 8 CS-3 bring-up), newest first

```
c404248 no i64 / constant-only subgraphs on the wafer — VQVAE COMPILES + RUNS
        (pe sized to T + broadcast-add; causal mask as float buffer; int32 ids;
         host-side label one-hot; RVQ select via float amin) [supersedes 5e47df6]
5e47df6 positional embedding via one-hot matmul (arange index won't stream)
1375913 host-side AR shift (pre_shifted) — remove the last in-graph slices
6e40af8 add lang tag to every memory slot (drop cat → no float slice)
868e3a0 separate q/k/v projections (packed-split slice crashes the compiler)
6b8ec7a explicit-ops LayerNorm (fused kernel fails on weight-shaped dims)
18ae4d2 positional embedding via F.embedding [superseded by 5e47df6]
aafe808 remove weight-transpose / weight-index ops (RVQ dist/select, tied out)
dae18d4 attention full packed projection [superseded by 868e3a0]
4188749 seed readout queries with encoder summary
78b3775 materialize expanded readout queries [superseded]
12fc7d1 explicit log_softmax+gather CE (fused cross_entropy won't lower)
4a20865 memory validity mask in int32 (concat rejects bool)
6b786bf cstorch-compatible attention — fixes empty CIRH module
2915d97 fetch a graph output every step (avoid empty-module DCE)
```
Earlier `vqvae/` op fixes (pre-committed): `torch.triu(bool)` → arange causal
mask; `F.one_hot`/`bincount` → arange compare for usage counts.

## 9. Mental model for the next op that fails

Read the coordinator FATAL, get the op + `loc` from the MLIR (§2). **First ask
the two §3 questions about every tensor involved: what dtype is it, and is any
operand data-dependent?** That catches more than the op-shaped table below. Then
classify:
- **coordinator log truncates with NO error** (ends mid-progress-line, client
  shows a bare 502) → the compiler aborted. Read `dbg_crd_0.err` for the
  assertion. `get_mov_simd_width` = an i64 tensor on the wafer (Class C).
- **`did not fit in memory` at micro-batch 1** on a small model → not really
  size; look for a big integer intermediate or a wide backward fan (an on-wafer
  one-hot compare does this). Move the construction host-side.
- **`Failed to convert op to WAF`** → fused/specialized kernel missing → rewrite in
  primitives (Class A).
- **`Unable to insert transfer WGT_HOST→WSE` / `split_copy`** → touching a weight →
  F.linear/F.embedding/onehot (Class B).
- **`slice_filter` assert / `ws_km.slice`** → an in-graph slice → move to host or
  express as matmul/compare.
- **`SelectLanesAndLayout` / `no valid layouts` / `act_host_to_wse`** → check the
  dtype in the op text. `actbuf<i64>` → Class C: cast on the host, or rework in
  float. Otherwise a streamed constant → build it as a buffer in `__init__`, or
  pass it from the dataloader as a batched float activation. Note the shape is
  reported at the **micro-batch** size the compiler is trying, so a `(B,T)`
  input shows up as `[T]` — divide by B before trying to identify it.
- **empty CIRH module** → the step has no fetched output, OR a fused kernel nuked
  the whole graph → check the `step_closure` fetch and Class A ops.

Keep every rewrite numerically identical and verify locally (§4) before spending
wafer time.
