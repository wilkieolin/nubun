# CS-3 bring-up log & handoff — getting the Nubun VQ-VAE to compile on the wafer

Written for an agent/engineer picking this up **on the ALCF Cerebras user node**
(`cer-usn-01`/`cer-usn-02`), where you can run `cszoo`/`csctl` and read compile
logs directly — no GCE↔ALCF file sync. Companion docs: `PORT_CS3.md` (cold-start),
`PHASE8.md` (the frozen model). This file is the *debugging* record: what breaks,
why, and the loop to keep going.

## 0. TL;DR status (2026-10-09)

**Corrected history (the 2026-07-20 version of this section was wrong).**
Reconstructed from `cerebras_logs/executors/000001/run_meta.json` (24 compile /
4 execute jobs) and the 14 exported coordinator logs:

- The real graph compiled cleanly three times in July (Jul 19 21:54, Jul 20
  00:33, Jul 21 15:51; ~17 min each; floorplan `batch=32, lanes=2`) — but with
  **three different compile-cache hashes**, i.e. three slightly different graphs.
- **The real graph never executed a training step.** Its single execute job
  (`wsjob-gwhzz9cp2dxmqwucyyzu6h`, Jul 20 00:50:43Z, placed on
  `cer-anl-wse001-cs-sy02`) died after 146.6 s. Its logs were **never exported**
  and the cluster purged the record (~1 week retention). No surviving log
  contains `DEADLINE_EXCEEDED`; the "ClockSync = infra" diagnosis below was
  written from the client's terse message. The client's `clock_sync()` deadline
  is 300 s (`appliance_client.py:328`), so a 147 s death was not that deadline.
- The known-good canary executed fine the next day (Jul 21 17:07), so the
  execute path was healthy; the failure was specific to our run. Leading
  suspects, unproven: `--mount-dirs` included `$REPO/data`, which does not
  exist on the user node (`disable_mount_check` defaults to True, so nothing
  flagged it; compile jobs have no worker pods, so only execute saw it), and the
  always-failing custom worker-image build (no internet from the build pod).
- The two "hangs" were client-side: the server finished the compile but the
  client never collected it. `python … |& tee` block-buffers stdout, so the log
  looked dead and the run was abandoned.

**Today (2026-10-09), same cluster release R_2.10.0, cluster healthy:**
- `minprobe-attn2` canary: compile + execute, 5 steps — OK
  (`wsjob-37madtw8whfsgulkrcypdx` / `wsjob-anhstf3gyetgt4yp5dwtpp`). The
  `client 1.14.0 vs server 1.20.2` warning is unchanged from July.
- **Real M1 graph EXECUTED on the wafer for the first time** (`smoke`,
  synthetic data, bf16, filtered AdamW, clip 1.0, `token_weight_t` streamed,
  schedule constants pinned to m1's): run `smoke-20261009T013455Z`, compile
  `wsjob-bxn53ckbfbttzj3edcnqht` (fresh, ~18 min after ~2 h 10 m queue),
  execute `wsjob-exybxrbrubrixuu6hspevm` on `cer-anl-wse001-cs-sy02` —
  10 steps, `nubun_cs_step10.mdl` (867 MB, all 360 model tensors finite),
  python rc 0. `track.json`: execute_prep 230 s, execute_till_recv_loss 281 s
  — so the July job that died at 147 s died *during execute prep*, consistent
  with the nonexistent `$REPO/data` mount (still unproven, but no longer
  reproducible). (Losses not printed: `--log-every` defaulted to 100; smoke
  modes now pass `--log-every 1`.)
- **`smoke-m2` (M2 graph) and `m1` (real data, 2000 steps) both succeeded**
  the same night — see §7 items 4–5. `m1`: 208 samples/s (~6.5 steps/s at
  B=32, vs ~3.0 on the GB10), recon 10.15 @100 → ~8.8–9.1 from step 1200 on,
  9.14 @2000; checkpoints round-trip to the eval format (`export_ckpt.py`).
- Data is staged at `~/code/nubun-copy/data` (embedding table, token weights,
  parallel corpus, 9 opus100 shards **and** their `.sem.npz`), unpacked to
  `~/code/nubun-copy/data/opus100_npy` for memory-mapped worker access.

**Run hygiene is now enforced by `cerebras/submit_alcf.sh`** (see §2b): per-run
log/artifact/model dirs, proxy-free unbuffered client, mount dirs validated
before submit, wsjob ids printed and logs auto-exported on failure.

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

## 2b. Run hygiene (what the July runs got wrong — now in the wrapper)

- **Export the EXECUTE job's logs within the hour of any failure**:
  `bash cerebras/export_logs.sh <wsjob>` (lands in `logx/<wsjob>/`, prints the
  §2 grep over `dbg_crd_0.out` AND `.err` plus worker/chief/weight errors). Job
  records expire from the cluster in about a week; the July execute failure is
  unrecoverable because this was skipped. The wrapper prints the ids on exit
  and exports automatically when python exits non-zero.
- **Unbuffered client** (`python -u`) and one log per run
  (`logs/<mode>-<UTC stamp>.log`); one artifact dir per run
  (`cerebras_logs/<mode>-<stamp>/`: `run_meta.json` with every wsjob id and
  timing, `cirh.mlir`, `track.json`, `performance.json`) — the default
  `cerebras_logs/executors/000001` is overwritten by every run.
- **No proxy on the appliance client.** gRPC C-core honours `http(s)_proxy`
  and `~/.bashrc` exports the ALCF proxy; the wrapper sets `no_proxy` for
  `10.125.8.2,.cerebras.internal` and runs python with the proxy variables
  unset. Keep the proxy only for pip/git.
- **Every mount dir must exist** (`cs_common.make_cluster_config` checks, and
  passes `disable_mount_check=False` so the cluster validates too).
- **The custom worker-image build always fails here** (`pip-index-connectivity-
  check`: no internet from the build pod; no `allow-venv` cluster volume
  either) and the run continues on the stock image after ~1 min. That is what
  every successful run used. `--skip-image-build` exists but also skips
  automount configuration — only use it after the canary proves it.
- **Graph-affecting constants are part of the compile hash**: `--steps`,
  `--warmup-steps`, `--lr`, `--lr-min-ratio` feed the TRACED LR schedulers.
  `smoke`/`compile`/`m1` pin identical values (`M1_STEPS`, `WARMUP`) and
  `smoke` executes only `--max-steps 10` of that schedule, so one ~17-minute
  compile serves all three. Do not toggle stall detection / POL /
  micro-batch between runs you want cache-shared.
- **The torch seed is part of the compile hash too**: cstorch bakes
  `torch.initial_seed()` into every dropout `RandomUniform` as `global_seed`.
  `train_cstorch.py` now calls `torch.manual_seed(args.seed)` first thing;
  keep `--seed` fixed across runs that should share a compile. To find what
  broke a cache hit, diff the two runs' `cirh.mlir` with `loc(...)` stripped.
- Compile jobs queue behind other users' P2 compile jobs (37 min of queue for
  the Oct 9 canary). Budget for it; avoid queueing several of ours at once (Oct 9: `smoke-m2` + `m1` queued
  together waited ~1.5 h, then ran back to back without trouble).

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
| `logits.gather(1, labels)` (i64 gather index) | i64 streamed to wafer | frozen-embedding lookup CE (Class D + below) |
| `dist.argmin(1)` then `idx == arange` | argmin yields i64 → (N,k) i64 | `onehot = (dist == dist.amin(1,keepdim=True))`, normalize row sums for ties |
| `x.min(dim=1).values` | also emits an i64 **indices** output that survives | `x.amin(dim=1)` |
| `pe[:T]` / `onehot @ pe` / `arange` mask | constant-only → GlobalHost or streamed constant | size the buffer to exactly T in `__init__`; broadcast-add |

### Class D — embeddings/gathers: the index must be RAW (the table may train)
Every `F.embedding` lowers to a `cirh.Gather`. Two independent conditions decide
whether it survives; both were discovered the hard way (a full session of wrong
theories about width, rank, and dtype — see the git log for `13abd18`… onward).
| Don't do | Symptom | Use instead |
|---|---|---|
| index the table with a **derived** tensor (`emb(labels.reshape(N))`, a slice, a cast) | `acth.gather` on the weight → "Unable to insert transfer WGT_HOST→ACT_HOST" | look up with the **raw load_input** (`emb(labels)`), reshape the RESULT. The gather→`ws_km.onehot` rewrite is a peephole that only matches `Gather(weight, load_input)`. |
| look up a trainable **width-1** per-vocab table (an output bias, `250037x1`) | its backward `cirh.ScatterNd` → `Failed to convert op to WAF` (`logx-ce4`) | drop the bias (the tied, biasless projection needs none). |

**CORRECTED 2026-10-09: trainable embedding tables DO lower.** The July rule
"the table must be FROZEN" over-generalized from the width-1 bias failure; the
`logx-ce2` probe that also had a `250037x384` ScatterNd failed earlier on a
derived-index forward gather, so the full table was never actually judged.
Evidence it works: the attn2 canary (512x128 ScatterNd) and the real graph's
10x384 lang embedding always trained; `smoke-emb` (`--train-embeddings`,
`wsjob-4iqmfba8mcmctvxycpsrwk` / `wsjob-jtkf3vft7yfpg2ppadyk4k`) compiled and
ran 10 steps with three `250037x384` ScatterNds (encoder lookup, decoder
lookup, CE label lookup); all rows updated, tied copies equal, finite.
This matters: on the GB10 the frozen arm trails the unfrozen arm by ~1 nat of
recon at every horizon (Phase 6: 9.20 vs 8.12 @2k, 6.43 vs 5.12 @50k) and
round-trip 0.184 vs 0.430 — and wafer `m1` (frozen) reproduced the frozen GB10
curve (9.14 @2k).

Width/rank/frozen-vs-trainable do NOT affect the *forward* — only these two do.
Neither is visible offline; both need a wafer compile to observe.

**Full-vocab CE (the real reconstruction loss), `cerebras/train_cstorch.weighted_ce`.**
Because the output projection is tied and biasless, `logit[i,label] ==
h_i·emb[label]`, so cross-entropy needs no gather and no (N,V) one-hot:
```
lse     = detached-max logsumexp(logits)            # NOT torch.logsumexp — see below
emb_t   = F.embedding(labels_raw, frozen_emb_weight)  # raw index, frozen table
logit_t = (dec_h * emb_t).sum(-1)                   # == logits[i,label]
nll     = lse - logit_t
```
`labels` must reach this un-sliced (hence host-side `pre_shifted` AR shift), and
`dec_h` (the decoder's post-out_norm hidden state) is returned alongside
`logits`. The pad mask and any per-token weights come from the host as (B,T)
float activations. This is the ONLY full-vocab CE that lowers: a host-built
one-hot is ~1 GB/step of bandwidth, an on-wafer one-hot needs a second (N,V)
tensor beside `logits` that does not co-fit, and a gather streams an i64 index.

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
- **`torch.logsumexp` doesn't lower.** Its backward runs through `amax`, whose
  gradient is `(x==max)/mask.sum(dim)` — an ADD-reduce over an i1 tensor, and
  `ReduceOp` only supports f16/f32/bf16/cb16 (`ReduceOp … IN data type is 1`).
  Use a **detached max**: `mx=x.amax(-1,keepdim=True).detach(); mx+log(exp(x-mx).sum(-1))`.
  Value and gradient are identical (the max cancels analytically); `amax` never
  enters the backward graph.
- **`ReduceOp` is float-only.** Any reduction (`sum`/`mean`/`logsumexp`-tail) over
  a bool/int tensor fails. Cast to float first, or restructure so the reduce is
  over floats. (Bool/int are second-class everywhere except as embedding indices.)
- **No trainable per-vocab output bias.** Dropped (Class D). A `(V,)`/`(V,1)`
  trainable table scatters on backward. A tied softmax without an output bias is
  standard; `decoder.out_bias` is removed (§5).
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
- **RETRACTED (2026-10-09): "`ClockSync … DEADLINE_EXCEEDED` in execute mode is
  infra — do NOT edit the graph."** No log ever recorded that message; the
  only real-graph execute job died at 147 s, under the 300 s clock-sync
  deadline, and its logs were never exported. Treat any execute-mode death as
  unexplained until `cerebras/export_logs.sh <execute wsjob>` has been read.
  Transient infra does exist (the test is reproducibility at the same stage),
  but the rule as previously written stopped the investigation.

### A meta-lesson worth its own line
- **A failed compile stops at its FIRST error and tells you nothing downstream.**
  Reading structure off a *failing* graph produced three wrong theories this
  session (Class-D width/rank/frozen). The reliable move is to DIFF the failing
  IR against a *known-good* graph (a prior SUCCEEDED job's `ws_km.mlir`), not to
  infer a mechanism from the failure alone. And when the coordinator dies by
  signal, `dbg_crd_0.out` truncates mid-line — the assertion is in `.err` (§2).

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
- **Output bias removed.** `decoder.out_bias` (a `(V,)` Parameter) is gone — a
  trainable per-vocab table scatters on backward (Class D). Tied softmax, no bias.
- **Decoder returns `(logits, h)`;** the model dict gains `dec_h`. The full-vocab
  CE (`weighted_ce`) is the frozen-embedding lookup form (Class D), not gather.
- Dataloader dtypes / shapes (real path): token ids are `int32`; the host does the
  AR shift and yields `dec_in`=tgt[:-1] and un-sliced `labels`=tgt[1:] (both
  length `seq_len-1`), plus a float `label_mask` and, when weighted, a
  host-looked-up `token_weight_t`; positional tables sized `src_len=seq_len`,
  `tgt_len=seq_len-1`.

- **Precision: bfloat16 activations, fp32 master weights, no loss scaling**
  (`--half-dtype bfloat16`; GB10 used bf16 autocast). The CSX default is
  float16 + a required GradScaler. A static-scale `GradScaler(loss_scale=1.0)`
  must NOT be used with an LR scheduler (it registers on the optimizer but
  never enters its conditional-update context → `lr_scheduler.step()` raises
  "Failed to mark tensor"); modelzoo's `MixedPrecision` builds no scaler for
  bf16 either. `cbfloat16`/`float16` get `GradScaler(loss_scale="dynamic")`.
- **Optimizer recipe restored to the E3 winner**: AdamW over TRAINABLE params
  only (the frozen 96M tied table was previously given to AdamW, which
  preallocates ~770 MB of dead `exp_avg/exp_avg_sq`), `weight_decay=0.01`,
  `eps=1e-8`, global grad-norm clip 1.0, warmup from `lr/warmup`. Remaining
  known difference: cstorch AdamW adds `eps` to the *uncorrected* `sqrt(v)`
  (`AdamBase.py`), so eps acts ~`1/sqrt(1-beta2^t)` larger on early steps;
  CPU parity check (scratch `verify_c1.py`): gradients identical (4e-9), losses
  identical to 5 digits over 3 steps, params within 3e-4 from that eps placement.
- **Embeddings frozen vs E3's unfrozen tied embeddings.** Forced by Class D
  (trainable embedding lookups scatter on backward). A real capacity
  difference; the E3 checkpoint's embedding was trained.
- **Semantic head (M2)**: pooled bottleneck is `z_q.mean(dim=1)` when
  `use_stop_mask` is off (identical value; the old masked mean reduced a
  constant-only `ones_like` mask), and the cosine is explicit primitives
  (`train_cstorch.cosine_sim`, verified == `F.cosine_similarity` fwd+grad).
- Host activations `label_mask`/`token_weight_t`/`sem_target` are float32 (the
  graph casts); synthetic `smoke` streams a random `token_weight_t` so it is the
  same graph as `m1`.
- Real-data workers memory-map UNPACKED `.npy` shards (`opus100_npy/`, built by
  `cerebras/unpack_opus.sh`): numpy ignores `mmap_mode` for `.npz`, so the old
  loader pulled ~1.5 GB per semantic shard into each worker's RAM (first batch
  took 134 s); unpacked it is 0.1 s to construct and ~100 ms/batch over cold
  NFS. Token weights (250002 entries) are padded with 1.0 to the 250037 vocab.
  Each streaming worker offsets the seed by `get_streaming_rank()`.

CS-3-trained checkpoints WILL round-trip to GB10 **if GB10 runs this same code**
(§9 of PORT_CS3). To eval them, run the current `vqvae/` on the GB10, not pre-port.

## 6. The canary: `cerebras/minimal_probe.py`

Standalone raw-cstorch loop with bisection modes (via `submit_alcf.sh <mode>`):
- `minprobe` — canonical MLP (baseline sanity / cluster health).
- `minprobe-attn2` — hand-rolled attention; **known-good**, use as cluster health check.
- `minprobe-vqvae` — the real `VQVAE` (small synthetic config) = the actual canary.
  Flags: `--no-rvq`, `--no-tie`; also `--vocab/--batch-size/--seq-len` (capacity
  probing at real dims), `--loss {ce,logsumexp}`, `--compile-only`.
- `minprobe-cehead` — ONLY the output head + full-vocab CE (`h` from the loader).
  Isolates the CE primitives from the VQVAE; how the Class-D lookup CE was proven
  at vocab 250037. Writes its OWN log (probe logs must not share a file).
- `minprobe`/`-raw`/`-adamw`/`-attn` — historical bisection probes (input, optimizer,
  torch attention); keep for regression.

`smoke` = the real M1 graph (train_cstorch) in execute mode, synthetic data,
`--max-steps 10` of the pinned m1 schedule — no data, mounts only the repo.
`smoke-m2` = the same with the semantic head + loss on (the M2 graph).
`compile` = the M1 graph with real data, compile-only. `m1`/`m2` = training
milestones on real data (`DATA_DIR`, default `~/code/nubun-copy/data`; the
wrapper unpacks `opus100_npy/` once). Every mode: `logs/<mode>-<stamp>.log`,
`cerebras_logs/<mode>-<stamp>/`, `model_dir/<mode>-<stamp>/`.

## 7. What's left to do (in order)

1. ~~Compile `minprobe-vqvae`~~ — DONE (`c404248`).
2. ~~Compile the real graph~~ — DONE (three times in July; never executed).
3. ~~Execute the real M1 graph~~ — DONE 2026-10-09
   (`wsjob-exybxrbrubrixuu6hspevm`, 10 steps, checkpoint finite; §0).
4. ~~`smoke-m2`~~ — DONE 2026-10-09: run `smoke-m2-20261009T041116Z`, compile
   `wsjob-nmhw8wjzcexauzfaxu9pgk` (fresh), execute `wsjob-eh6uu5synsddksnkquaq4y`,
   10 finite steps (loss 209.9 → 202.1), checkpoint at step 10. The ~200 loss
   scale is the synthetic *unscaled* `randn` tied embedding table (row norm
   ~√384 → huge logits), not a bug: the CPU parity script at D=32 shows the
   same effect (≈32 vs ln 512 ≈ 6.2). Real-data m1 should start near ~12.
5. ~~`m1`~~ — DONE 2026-10-09: run `m1-20261009T041121Z`, compile
   `wsjob-faws8nrhwjypkm66fqz2j5` (fresh, 18 min), execute
   `wsjob-epx7tgvcjv4bvdde4mj5xu` (6 m 28 s wall, 307 s of steps, 208
   samples/s, flops_utilization 0.077, no input starvation). Loss (recon):
   @100 10.15, @500 9.46, @1000 9.43, @1200 8.77, @1500 8.76, @2000 9.14 —
   descends, then flattens near 9 as the 2000-step cosine decays. GB10
   reference: Phase 7 (unfrozen embeddings, semantic on) had recon 8.15 @2000.
   Leading hypothesis for the gap: the tied token table is FROZEN on the wafer
   (§5), so the output projection cannot learn — a capacity difference, not a
   bug. Checkpoints @500/1000/1500/2000, all finite.
   **Expected compile-cache hit did NOT happen** — the smoke and m1 `cirh.mlir`
   differed only in the dropout `RandomUniform {global_seed}`: cstorch bakes
   `torch.initial_seed()` into the graph (`ltc_backend.py` `set_rng_state`) and
   `train_cstorch.py` never seeded torch. FIXED: `torch.manual_seed(args.seed)`
   right after arg parsing; synthetic and real-data traces now produce
   identical `cirh.mlir` (verified with validate-only). The next run recompiles
   once (seed 42 is a new graph); later runs with the same constants should hit.
5b. **`m1 --train-embeddings` — DONE 2026-10-09, and it is the answer to the
   frozen-embedding question.** Run `m1-20261009T164423Z`, compile
   `wsjob-qyeuzwnpgtktmpwaocl3ns` (42 s — cache hit on the `smoke-emb` graph:
   the seed fix works and synthetic smoke ≡ real-data m1), execute
   `wsjob-bq8pkctybbirpiksayxaah`, 166 samples/s (vs 208 frozen, −20%),
   flops_utilization 0.080. Recon, trainable vs frozen at the same step:
   @100 9.77/10.15, @500 8.02/9.46, @1000 7.88/9.43, @1500 7.22/8.76,
   @2000 7.73/9.14; mean of the last six logged points (1500–2000)
   **7.48 vs 8.96, ≈1.5 nats better** — better even than the GB10 unfrozen
   arm (8.12 @2k, which also carried the semantic loss). Frozen embeddings DO
   drag training; train them. Same-batch CPU scoring of both step-2000
   checkpoints on the SAME 16 real batches (`logs/score_m1_frozen_vs_emb.out`):
   **9.04 frozen vs 7.48 trainable.**
   `submit_alcf.sh` now passes `--train-embeddings` to every training mode
   (`FROZEN_EMB=1` opts out). m2 was then launched as a detached chain
   (`logs/chain_smoke_m2_then_m2.sh` → `.out`): `smoke-m2` with the trainable
   table, gate (rc 0, 10 finite steps), then `m2` (100k steps, cosine over
   100k — the E2-style schedule; E3's 200k-stretched cosine was the PHASE8
   lesson NOT to repeat). At ~5 steps/s m2 fits in one 23 h job (~6 h). Untested idea if throughput matters: the corpus uses
   only 117,323 of 250,037 token ids (100k cover 99.988%) — a compacted vocab
   would halve the table, the logits matmul and the ScatterNds.
5c. **m2 — DONE 2026-10-09** (first full-length wafer run): run
   `m2-20261009T171835Z`, execute `wsjob-jlmxznppsjc49mkdiulace`, 100k steps,
   B=32, trainable tied table, semantic on, cosine over 100k, ~5.5 steps/s
   (~5 h wall). Smoothed logged recon @10k 6.05 · @25k ~5.5 · @50k 5.09 ·
   @70k 4.76 · @90k 4.63 · last 5k 4.78; no late collapse (one single-batch
   spike at 30k). CPU recon on the 16 fixed real batches: **4.58** (vs 7.48 for
   m1-trainable @2k). GB10 Phase 7 (closest surviving log) ended at 4.75.
   Eval-format checkpoint: `model_dir/m2-20261009T171835Z/nubun_cs_step100000.eval.pt`
   → run `roundtrip_eval.py` on the GB10 and compare with E3's 0.435.

5d. **Throughput sweep (synthetic, M2 graph, trainable emb), 2026-10-09** — wafer 2
   (`cer-anl-wse002`) is usable in parallel with wafer 1. samples/s:
   B=32 ~178 (m2) · B=64 217 · B=128 **298** · B=256 253 · B=512 325; the
   compiler chose `microbatch=64` for every B>=64 (micro-batch 128 was built and
   rejected on estimated perf; forcing `--micro-batch-size 128` at B=128 gave
   234 — worse). Per-sample cost dominates: 70.5% of every 64-slot row is
   padding (median pair length 13; 82.3% of pairs fit T=32), so the main lever
   is a short-T phase (`--seq-len 32 --fit-seq-len`). T=32/B=128 with auto
   micro-batch ABORTS in the compiler ("Memory allocation was unsuccessful",
   microbatch=32) — use `--min-micro-batch 64`. Equal-samples A/B vs m2
   (3.2M samples, B=128, lr 6e-4): `logs/ab_chain.sh` arms `t64` and `t32`
   (20k steps T=32 → 5k steps T=64 via --resume).
5e. **Equal-samples A/B vs m2 — DONE 2026-10-10** (3.2M samples each; CPU recon on
   the same 16 fixed real batches at T=64; wall = wafer execute time):
   | run | recipe | wall | recon |
   |---|---|---|---|
   | m2 `m2-20261009T171835Z` | B=32, lr 3e-4, 100k steps | ~5 h | 4.58 |
   | **t64** `m2-20261009T205948Z` | **B=128, lr 6e-4, 25k steps** | **~2 h 15 m** (~400 samples/s) | **4.42** |
   | t32 `m2-20261009T221230Z` → `m2-20261010T002355Z` | B=128: 20k steps T=32 (`--fit-seq-len`, 614 samples/s) → 5k steps T=64 (`--resume`) | ~1 h 37 m | 4.77 |
   | t64 @17.5k (same wall as t32) | | ~1 h 37 m | 4.63 |
   **Verdict: use B=128, T=64, lr 6e-4 (sqrt-scaled).** Better than m2 at 2.2x the
   speed. The short-T phase is NOT worth it as configured: it is 1.5x faster per
   sample but loses 0.14 recon even at equal wall time (long pairs are absent for
   80% of training; recon jumps to 5.19 at the T=64 switch and only partly
   recovers in 5k steps). `--resume` across a seq-len change works (positional
   tables are not in the state_dict). Eval-format checkpoints for the GB10:
   `model_dir/m2-20261009T205948Z/nubun_cs_step25000.eval.pt` (t64),
   `model_dir/m2-20261009T171835Z/nubun_cs_step100000.eval.pt` (m2).
   Untested levers left: vocab compaction (117k used ids), both wafers
   (`--num-csx 2`) for a final long run.
5f. **Weekend run — DONE 2026-10-12**: B=128, lr 6e-4, T=64, 500k steps (64M
   samples, cosine scoped to 500k), `cerebras/runs/long_b128.sh` →
   `model_dir/long-b128-20261010T042150Z/`. Three ~23 h segments chained by
   `--resume` (0→240k, 240k→480k, 480k→500k), all rc=0, no retries, compile
   cache hit on every resume; ~2.9 steps/s (~375 samples/s), 46 h wall total.
   CPU recon on the same 16 fixed batches:
   | step | 100k | 250k | 480k | **500k** |
   |---|---|---|---|---|
   | recon | 3.77 | 3.27 | 2.81 | **2.79** |
   (m2 B=32 @100k: 4.58.) Wafer-logged 10-point recon at the end ≈ 3.0–3.1; each
   segment restarts the data stream with a new seed, so small bumps at the
   joins are expected. Milestones kept every 50k; `.eval.pt` written for
   100k/250k/480k/500k.
   **Round-trip (GB10, 2026-10-12)** — `roundtrip_eval.py --combine-splits --langs
   en,zh,es,fr,ar,ru --max-gen-len 48`, same seeded 200-sentence split (gold cross
   0.8614), run from a worktree of this branch (the wafer port changed the math —
   lang tag added to every memory slot, readout-query seeding — so the `.eval.pt`
   needs this branch's `vqvae/`, not the pre-port code):
   | condition | same | cross |
   |---|---|---|
   | **real** | 0.6114 | **0.5824** |
   | shuffle | 0.0438 | 0.0443 |
   | zero | 0.0644 | 0.0643 |
   Shuffle-gap cross **+0.538**. **0.582 vs C1's 0.5045 (the GB10 frozen winner,
   d_model 512) and E3's 0.435**: the CS-3 500k B=128 run is the new best model,
   at d_model 384. Codes carry the meaning (shuffle/zero ≈ 0.05).
6. ~~Round-trip~~ — DONE for `m1` step 500: `cerebras/export_ckpt.py`
   loads the `.mdl`, rebuilds VQVAE exactly as `roundtrip_eval.py` does, loads
   strict (0 missing / 0 unexpected), writes `<mdl>.eval.pt`; CPU recon on 4
   real batches 9.70 vs wafer-logged 9.54. Still to do: run `roundtrip_eval.py`
   on the GB10 (needs MiniLM). `--resume`
   (`cstorch.load` of model/optimizer/lr_scheduler/step) is implemented;
   checkpoints written before it existed lack `lr_scheduler`.
7. **`m2`** — `--steps` = the real run length (cosine scoped to it), `.sem.npz`
   already staged and unpacked.
8. Deferred (off the M1/M2 path): non-RVQ `VectorQuantizer` argmin/i64
   (`quantizer.py:72-91`), all of `nat_decoder.py`, word dropout
   (`model.py` — rewrites `dec_in`, breaking the raw-index lookup rule),
   `target_len`/stop-mask helpers (i64), `--skip-image-build` evaluation.

## 8. Commit trail (Phase 8 CS-3 bring-up), newest first

```
(uncommitted, 2026-10-09) run hygiene + recipe parity + mmap data path
        cerebras/cs_common.py (shared ClusterConfig/backend/debug args),
        submit_alcf.sh rewrite, export_logs.sh, unpack_opus.sh;
        train_cstorch: bf16, trainable-only AdamW wd 0.01 eps 1e-8 clip 1.0,
        explicit cosine, --max-steps, --artifact-dir; model.py mean pooling;
        opus_input: mmap .npy shards, abspath, per-worker seed, fp32 floats.
377aab7 real path: lookup CE + no output bias — smoke COMPILES at real dims
        (decoder returns (logits,h), out_bias dropped; weighted_ce = frozen-emb
         lookup + detached-max logsumexp; opus/compile_check int32 + host shift;
         Class D: raw gather index, no trainable per-vocab table). Execute gated
         on ClockSync infra.
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
