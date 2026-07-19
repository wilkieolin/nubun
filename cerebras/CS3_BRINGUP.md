# CS-3 bring-up log & handoff — getting the Nubun VQ-VAE to compile on the wafer

Written for an agent/engineer picking this up **on the ALCF Cerebras user node**
(`cer-usn-01`/`cer-usn-02`), where you can run `cszoo`/`csctl` and read compile
logs directly — no GCE↔ALCF file sync. Companion docs: `PORT_CS3.md` (cold-start),
`PHASE8.md` (the frozen model). This file is the *debugging* record: what breaks,
why, and the loop to keep going.

## 0. TL;DR status (2026-07-19)

We are bringing up `cerebras/minimal_probe.py --model vqvae` (a small synthetic
VQVAE through a raw cstorch loop) as the compile canary. We have driven it from
"whole graph lowers to an empty module" through a long series of op/layout fixes.
As of `5e47df6` the graph is **slice-free and past op-lowering**; the last seen
error was in the **layout/lane-selection** stage (positional-embedding arange),
now fixed with a one-hot matmul. Next action: re-run `minprobe-vqvae` and read
the coordinator log.

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

`dbg_crd_0.out` = compile stdout (progress + FATAL). `dbg_crd_0.err` = usually
just the graceful-shutdown-after-client-terminated message (ignore). The `loc(...)`
strings in `cirh.mlir` / `ws_km.mlir` carry the full Python callsite
(`vqvae/....py:line`) — that's how you know which line produced the bad op.

Then: fix → verify numerically + trace locally (see §4) → commit → re-run.

## 3. Discoveries: the failure classes and the rules

Two dominant classes explain almost everything. **When adding any op, apply these
rules; do not reach for stock-PyTorch idioms.**

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
- **Bare `502` with no coordinator FATAL** = transient infra; retry. Confirm the
  cluster is healthy with `bash cerebras/submit_alcf.sh minprobe-attn2` (known-good).
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
- Positional embedding `pe` buffer: `(1,4096,d)` persistent → `(512,d)`
  non-persistent; selected via one-hot matmul.
- AR shift moved to the host (`pre_shifted=True`; dataloader yields `dec_in`/`labels`).

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

1. **Finish the compile of `minprobe-vqvae`** — keep running the §2 loop until it
   prints `Compile was successful!` and sends weights. Fixes so far are all in
   `vqvae/` + `minimal_probe.py`.
2. **Propagate the pre-shift to the real path** (held until the canary compiles):
   - `cerebras/train_cstorch.py` `train_step`: call `model(..., pre_shifted=True)`,
     and `weighted_ce` reads `labels` (no `tgt[:,1:]` slice).
   - `cerebras/opus_input.py`: yield `dec_in` (=tgt[:-1]) and `labels` (=tgt[1:])
     pre-shifted on the host, instead of `tgt_ids`.
   - `cerebras/compile_check.py`: same pre-shift.
3. **Add `cstorch.amp.GradScaler`** for the real fp16 runs (the persistent
   "half dtype float16 but no grad scaler" warning → gradient underflow in M1/M2).
   Held deliberately so it didn't muddy the compile bisection.
4. **`smoke` → `m1`** (needs data staged on ALCF: embedding_table.pt,
   token_weights.pt, opus100 shards; M2 also needs `precompute_opus_semantic.py`).
5. Update `PORT_CS3.md §8/§10` with the final op-rewrite list and the divergences.

## 8. Commit trail (Phase 8 CS-3 bring-up), newest first

```
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

Read the coordinator FATAL, get the op + `loc` from the MLIR (§2). Then classify:
- **`Failed to convert op to WAF`** → fused/specialized kernel missing → rewrite in
  primitives (Class A).
- **`Unable to insert transfer WGT_HOST→WSE` / `split_copy`** → touching a weight →
  F.linear/F.embedding/onehot (Class B).
- **`slice_filter` assert / `ws_km.slice`** → an in-graph slice → move to host or
  express as matmul/compare.
- **`SelectLanesAndLayout` / `no valid layouts` / `act_host_to_wse`** → a streamed
  constant (often i64 arange as an index) → use arange in a compare + matmul, or
  pass the constant from the dataloader as a batched float activation.
- **empty CIRH module** → the step has no fetched output, OR a fused kernel nuked
  the whole graph → check the `step_closure` fetch and Class A ops.

Keep every rewrite numerically identical and verify locally (§4) before spending
wafer time.
