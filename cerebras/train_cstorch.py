"""Nubun VQ-VAE training loop, ported to Cerebras cstorch (SCAFFOLD).

This is the GB10 train_vqvae.py loop re-expressed in the cstorch idiom:
  - cstorch.compile(model) + @cstorch.trace on the step (fwd/bwd/optim)
  - all host-side reads (.item(), logging) moved into @cstorch.step_closure
  - checkpointing via @cstorch.checkpoint_closure + cstorch.save
  - cstorch.optim optimizer + LR scheduler

It is written to the cstorch 2.10.0 API (every call checked against the installed
release on the ALCF user node, Oct 2026). Bring it up in two milestones (see
PORT_CS3.md and cerebras/CS3_BRINGUP.md):

  M1 (recommended first): recon + RVQ losses only, synthetic or real data.
     Proves the custom model trains on the WSE. Run with --no-semantic.
  M2: add the precomputed semantic targets (precompute_semantic_targets.py) and
     the token-weighted CE, matching the GB10 recipe exactly.

Frozen recipe to reproduce (E3@100k winner): RVQ 8x128, tied emb, deep decoder
(d_ff 2048, n_dec 10), lr 3e-4 cosine, warmup 500, lambda_sem 5, weighted CE.
"""
import argparse
import glob
import os
import re
import sys

import torch
from torch.nn import functional as F

# Repo root (this file is cerebras/train_cstorch.py) — appliance workers must be
# able to import `vqvae` from here and mount it, so it seeds the cluster config.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from vqvae.model import VQVAE  # noqa: E402

import cerebras.pytorch as cstorch  # noqa: E402
from cerebras.cs_common import add_cluster_args, make_backend  # noqa: E402


def build_model(args, emb):
    return VQVAE(
        vocab_size=emb.shape[0], n_langs=args.n_langs,
        d_model=args.d_model, d_code=args.d_code, k=args.k, m_max=args.m_max,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        beta_commit=args.beta_commit, pad_token_id=args.pad_token_id,
        embedding_table=emb,
        use_semantic_head=not args.no_semantic,
        use_rvq=True, n_rvq_levels=args.n_rvq_levels,
        decoder_type="ar", tie_embeddings=True,
        # Exact positional-table sizes (required on the wafer — otherwise the
        # forward takes the pe[:T] slice fallback, which slice_filter asserts on).
        # src_ids is (B, seq_len); the decoder input is tgt[:-1] = seq_len-1.
        src_len=args.seq_len, tgt_len=args.seq_len - 1,
    )


def weighted_ce(logits, dec_h, labels, emb_weight, label_mask, token_weight_t):
    """Token-weighted cross-entropy == vqvae.losses.reconstruction_loss.

    CS-3 lookup form. The fused F.cross_entropy does not lower (wgth.cast over
    the vocab dim), and neither does the obvious explicit form: a `.gather()` on
    the (N,V) logits streams an i64 index with no valid layout, and a host-built
    one-hot is ~1 GB/step of bandwidth. Instead:

    - logsumexp with a DETACHED max. torch.logsumexp backprops through amax,
      whose gradient is an ADD-reduce over an i1 tensor, and ReduceOp only
      supports f16/f32/bf16/cb16.
    - target logit by a FROZEN embedding lookup, NOT a gather: because the
      projection is tied and biasless, logits[i,label] == h_i · emb[label], so
      logit_t = (dec_h * emb[label]).sum(-1). emb_weight is the frozen table, so
      the lookup's backward is a no-op (a trainable table would emit a ScatterNd
      with no kernel).

    CRITICAL: `labels` must be the RAW dataloader tensor. F.embedding lowers to
    ws_km.onehot only when its index is an unmodified load_input — any reshape /
    slice / cast on the index first leaves an acth.gather that cannot be placed.
    So look up on `labels` as-is, reshape the RESULT. This is why the AR shift is
    done host-side (pre_shifted) and labels arrive un-sliced.

    label_mask and token_weight_t are (B,T) float activations from the host — the
    pad mask and the per-label token weights, looked up host-side rather than as
    wafer tables. See cerebras/minimal_probe.run_cehead and CS3_BRINGUP §3.
    """
    V = logits.size(-1)
    x = logits.reshape(-1, V).float()
    mx = x.amax(dim=-1, keepdim=True).detach()
    lse = mx.squeeze(-1) + torch.log(torch.exp(x - mx).sum(dim=-1))     # (N,)
    emb_t = F.embedding(labels, emb_weight).reshape(x.size(0), -1)      # (N, d) frozen
    logit_t = (dec_h.reshape(x.size(0), -1) * emb_t).sum(dim=-1)        # (N,)
    nll = lse - logit_t.float()
    m = label_mask.reshape(-1).to(x.dtype)
    w = m if token_weight_t is None else token_weight_t.reshape(-1).to(x.dtype) * m
    return (nll * w).sum() / w.sum().clamp(min=1.0)


def cosine_sim(a, b, eps=1e-8):
    """== F.cosine_similarity(a, b, dim=-1) (vqvae.losses.semantic_loss) in
    primitives: the fused op has never been compiled on the wafer; this is
    sums, a sqrt and a divide (verified fwd+grad identical on CPU)."""
    num = (a * b).sum(dim=-1)
    den = (torch.sqrt((a * a).sum(dim=-1)).clamp(min=eps)
           * torch.sqrt((b * b).sum(dim=-1)).clamp(min=eps))
    return num / den


def exec_steps(args):
    """Steps actually executed in THIS run: --max-steps caps --steps (the schedule
    length), minus the steps already done by a --resume checkpoint."""
    n = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    return max(0, n - getattr(args, "start_step", 0))


def make_synthetic_input_fn(args, vocab):
    """Random batches at the right static shapes — for a no-data dry compile
    (--synthetic-data). Values are meaningless; only shapes matter to the tracer.
    The real pipeline is cerebras/opus_input.py::make_opus_input_fn."""
    B, T, d_sem = args.batch_size, args.seq_len, 384

    def input_fn():
        def gen():
            for _ in range(exec_steps(args) + 2):
                # Host-side AR shift + int32 ids, matching opus_input. tgt is
                # (B,T); dec_in/labels are its (B,T-1) shifted halves — so the
                # decoder input length is seq_len-1, which is what build_model
                # sizes the decoder pe table to (tgt_len=seq_len-1). ids are int32
                # (no i64 on the wafer); label_mask is a float pad mask.
                tgt = torch.randint(4, vocab, (B, T), dtype=torch.int32)
                b = {
                    "src_ids": torch.randint(4, vocab, (B, T), dtype=torch.int32),
                    "dec_in": tgt[:, :-1].contiguous(),
                    "labels": tgt[:, 1:].contiguous(),
                    "label_mask": (tgt[:, 1:] != args.pad_token_id).to(torch.float32),
                    "src_lang_id": torch.randint(0, args.n_langs, (B,), dtype=torch.int32),
                    "tgt_lang_id": torch.randint(0, args.n_langs, (B,), dtype=torch.int32),
                }
                if not args.no_token_weights:
                    # Random positive per-label weights: keeps smoke/compile/m1 ONE
                    # graph (m1 streams the real token_weight[labels] from the host).
                    b["token_weight_t"] = torch.rand(B, T - 1, dtype=torch.float32) + 0.5
                if not args.no_semantic:
                    b["sem_target"] = torch.randn(B, d_sem)
                yield b
        return gen()
    return input_fn


def build_input_fn(args, vocab, token_weight=None):
    """Real opus pipeline by default; synthetic only with --synthetic-data."""
    if args.synthetic_data:
        print("input: SYNTHETIC (random shapes only — no real data)")
        return make_synthetic_input_fn(args, vocab)
    from cerebras.opus_input import make_opus_input_fn
    langs = [s for s in args.langs.split(",") if s] if args.langs else None
    print(f"input: opus100 from {args.opus_dir} "
          f"(seq_len={args.seq_len}, semantic={not args.no_semantic})")
    return make_opus_input_fn(
        opus_dir=args.opus_dir, seq_len=args.seq_len, batch_size=args.batch_size,
        num_steps=exec_steps(args), parallel_corpus=args.parallel_corpus,
        # + start_step: a resumed segment draws a fresh sample stream instead of
        # replaying segment 1's (the graph seed, torch.manual_seed, is unchanged).
        seed=args.seed + getattr(args, "start_step", 0),
        langs=langs, with_semantic=not args.no_semantic,
        sem_dir=args.sem_dir, pad=args.pad_token_id, token_weight=token_weight,
        max_len=args.seq_len if args.fit_seq_len else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-table", default="data/embedding_table.pt")
    ap.add_argument("--token-weights", default="data/token_weights.pt")
    # --- data pipeline (cerebras/opus_input.py) --------------------------------
    ap.add_argument("--opus-dir", default="data/opus100",
                    help="Dir of packed <lang>_en.npz opus shards.")
    ap.add_argument("--parallel-corpus", default="data/parallel_corpus.npz",
                    help="Source of short_codes + pad/bos/eos so lang-ids and "
                         "special tokens match the frozen model.")
    ap.add_argument("--sem-dir", default=None,
                    help="Dir of <lang>_en.sem.npz targets (M2). Defaults to "
                         "--opus-dir. Build with precompute_opus_semantic.py.")
    ap.add_argument("--langs", default=None,
                    help="Comma-separated X-lang subset (default: all shards).")
    ap.add_argument("--synthetic-data", action="store_true",
                    help="Use random batches instead of real opus — for a "
                         "no-data dry compile (pair with --compile-only).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="model_dir")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--lambda-sem", type=float, default=5.0)
    # Optimizer recipe == GB10 train_vqvae.py (AdamW wd 0.01, eps 1e-8, clip 1.0).
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="Global grad-norm clip (0 disables).")
    ap.add_argument("--half-dtype", choices=["bfloat16", "cbfloat16", "float16"],
                    default="bfloat16",
                    help="Mixed-precision activation dtype on the wafer. bfloat16 "
                         "matches the GB10 bf16-autocast recipe and needs no loss "
                         "scaling; cbfloat16/float16 enable dynamic loss scaling.")
    ap.add_argument("--no-semantic", action="store_true", help="M1: drop semantic loss")
    ap.add_argument("--fit-seq-len", action="store_true",
                    help="sample only pairs whose both sides fit --seq-len (no "
                         "truncation); for short-T training phases")
    ap.add_argument("--micro-batch-size", default="auto",
                    help="cstorch.backends.csx.performance.micro_batch_size: 'auto' "
                         "(compiler picks; it chose 64 for B>=128 on Oct 9) or an int "
                         "dividing --batch-size. Part of the compile hash.")
    ap.add_argument("--min-micro-batch", type=int, default=0,
                    help="with --micro-batch-size auto: lower bound on the compiler's "
                         "micro-batch search (INI ws_opt_force_min_grad_accum_batch). "
                         "T=32,B=128 auto ABORTS on a micro-batch-32 floorplan "
                         "('Memory allocation was unsuccessful'); 64 avoids it.")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="make the tied token table trainable, as the GB10 E3 "
                         "recipe (--unfreeze-embeddings) did; EXPERIMENTAL on the "
                         "wafer — its lookup backward is a ScatterNd into the table")
    ap.add_argument("--no-token-weights", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=20000)
    ap.add_argument("--keep-ckpts", type=int, default=0,
                    help="keep only the newest K .mdl checkpoints in --out-dir (0 = keep all)")
    ap.add_argument("--keep-every", type=int, default=0,
                    help="with --keep-ckpts: never prune checkpoints whose step is a "
                         "multiple of this (milestones to roll back to)")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint (.mdl from this script) to continue from: "
                         "restores model, optimizer, LR schedule and step count. "
                         "Use for 24h ALCF job segments; keep --steps identical.")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Execute at most this many steps (DataExecutor num_steps) "
                         "while keeping --steps as the LR-schedule length. The "
                         "schedule constants are traced into the graph, so a 10-step "
                         "smoke with --steps 2000 shares m1's compile.")
    # --- CS-3 appliance / ALCF cluster config (cerebras/cs_common.py) ----------
    # On an ALCF user node mgmt_address + credentials come from
    # /opt/cerebras/config_v2 automatically; we only supply run shape + the dirs
    # the appliance workers must see.
    add_cluster_args(ap, default_label="name=nubun")
    # architecture (frozen E3 recipe defaults)
    ap.add_argument("--vocab-size", type=int, default=250037,
                    help="Only used with --synthetic-data (real vocab comes from "
                         "the embedding table).")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--d-code", type=int, default=256)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--n-enc-layers", type=int, default=4)
    ap.add_argument("--n-dec-layers", type=int, default=10)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--n-rvq-levels", type=int, default=8)
    ap.add_argument("--beta-commit", type=float, default=0.25)
    ap.add_argument("--n-langs", type=int, default=10)
    ap.add_argument("--pad-token-id", type=int, default=1)
    args = ap.parse_args()

    # Seed BEFORE anything draws from torch's RNG. cstorch bakes
    # torch.initial_seed() into the lowered graph as the dropout RandomUniform
    # global_seed (ltc_backend.py set_rng_state), so an unseeded run gets a new
    # graph hash every time — no compile-cache hits, ever (seen Oct 9: smoke and
    # m1 differed ONLY in global_seed). Also makes model init reproducible.
    torch.manual_seed(args.seed)

    # --synthetic-data is a zero-file dry compile: fabricate a random emb table so
    # the graph can be built and traced without any data/ files present.
    if args.synthetic_data and not os.path.exists(args.embedding_table):
        print(f"synthetic-data: fabricating random emb table ({args.vocab_size}, "
              f"{args.d_model}) — no embedding_table.pt needed")
        emb = torch.randn(args.vocab_size, args.d_model)
    else:
        emb = torch.load(args.embedding_table, map_location="cpu")
    vocab = emb.shape[0]
    token_weight = None
    if not args.no_token_weights and not args.synthetic_data:
        token_weight = torch.load(args.token_weights, map_location="cpu").float()
        if token_weight.numel() < vocab:            # 250002-entry table, 250037 vocab
            pad_w = torch.ones(vocab - token_weight.numel())
            token_weight = torch.cat([token_weight, pad_w])

    backend = make_backend(args, compile_only=args.compile_only,
                           validate_only=args.validate_only)
    os.makedirs(args.out_dir, exist_ok=True)

    # Precision. The CSX backend always runs mixed precision (fp32 master
    # weights, half-dtype activations). Default half dtype is float16, which
    # needs loss scaling; we use bfloat16 like the GB10 recipe. Must be set
    # before the model and the dataloader are built.
    cstorch.amp.set_half_dtype(args.half_dtype)
    mbs = args.micro_batch_size
    cstorch.backends.csx.performance.micro_batch_size = (
        mbs if mbs in ("auto", "explore", None) else int(mbs))
    if args.min_micro_batch:
        # Must come AFTER the micro_batch_size setter, which clears these INIs.
        cstorch.backends.csx.debug.ini.ws_opt_force_min_grad_accum_batch = args.min_micro_batch

    model = build_model(args, emb)
    if args.train_embeddings:
        # Tied: encoder.token_emb.weight IS decoder.token_emb.weight.
        model.decoder.token_emb.weight.requires_grad = True
        model.encoder.token_emb.weight.requires_grad = True
    model.train()
    compiled_model = cstorch.compile(model, backend)
    # Only trainable params: the tied 250037x384 token table is frozen by
    # default on the CS-3 (its lookup backward is a ScatterNd — CS3_BRINGUP
    # Class D; --train-embeddings tests whether that lowers), and cstorch AdamW preallocates exp_avg/exp_avg_sq for EVERY
    # param it is given (~770 MB of dead state and checkpoint bloat otherwise).
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params: trainable {n_tr/1e6:.1f}M of {n_all/1e6:.1f}M "
          f"(frozen {(n_all-n_tr)/1e6:.1f}M)")
    optimizer = cstorch.optim.AdamW(trainable, lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    eps=args.adam_eps)
    # Loss scaling only for fp16/cb16 (modelzoo's MixedPrecision does the same:
    # "No need to use loss scaling when half dtype is bfloat16"). Do NOT build a
    # static-scale GradScaler for bf16: it registers on the optimizer but never
    # enters its conditional-update context, so lr_scheduler.step() raises
    # "Failed to mark tensor". The scaler must be built AFTER the optimizer.
    grad_scaler = (None if args.half_dtype == "bfloat16"
                   else cstorch.amp.GradScaler(loss_scale="dynamic"))

    # Linear warmup -> cosine decay, built from native cstorch schedulers.
    # cstorch's LambdaLR is NOT torch's (it takes initial_learning_rate and wants
    # set_value_lambda overridden), so we compose the two-phase schedule with
    # SequentialLR instead. IMPORTANT: CosineDecayLR.total_iters is the ACTUAL
    # remaining step count — do NOT stretch cosine past the run length (that hot-LR
    # tail collapsed the RVQ codebook on the GB10; see PORT_CS3.md / PHASE8.md).
    warmup = cstorch.optim.lr_scheduler.LinearLR(
        optimizer, initial_learning_rate=args.lr / max(1, args.warmup_steps),
        end_learning_rate=args.lr, total_iters=max(1, args.warmup_steps))
    cosine = cstorch.optim.lr_scheduler.CosineDecayLR(
        optimizer, initial_learning_rate=args.lr,
        end_learning_rate=args.lr * args.lr_min_ratio,
        total_iters=max(1, args.steps - args.warmup_steps))
    lr_scheduler = cstorch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

    args.start_step = 0
    if args.resume:
        state = cstorch.load(args.resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        args.start_step = int(state["step"])
        print(f"resumed from {args.resume} at step {args.start_step}; "
              f"{exec_steps(args)} steps remain")

    # token_weight is NOT moved to the wafer device: it is a frozen (V,) constant
    # and the per-label lookup token_weight[labels] happens HOST-side in the
    # dataloader (opus_input), which yields a (B,T) `token_weight_t` activation.
    # A wafer-side (V,1) table would need an unplaceable gather; see weighted_ce.

    @cstorch.trace
    def train_step(batch):
        # pre_shifted: dec_in and labels are shifted on the host (opus_input),
        # so `labels` reaches weighted_ce as an un-sliced load_input — required
        # for the frozen-embedding lookup to lower (see weighted_ce docstring).
        out = compiled_model(batch["src_ids"], batch["dec_in"], batch["tgt_lang_id"],
                             pre_shifted=True)
        recon = weighted_ce(out["logits"], out["dec_h"], batch["labels"],
                            model.decoder.token_emb.weight,
                            batch["label_mask"], batch.get("token_weight_t"))
        loss = recon + out["vq_losses"]["commit"] + out["vq_losses"]["codebook"]
        if not args.no_semantic and out["sem_pred"] is not None:
            cos = cosine_sim(out["sem_pred"].float(), batch["sem_target"].float())
            loss = loss + args.lambda_sem * (1.0 - cos).mean()
        # Same sequence as modelzoo MixedPrecision: [scale] -> backward ->
        # [unscale] -> clip_grad_norm_ -> step -> [update] -> zero_grad -> sched.
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()
        return loss, recon

    @cstorch.step_closure
    def log_step(step, loss, recon):
        # Fetch the returned tensors EVERY step. Fetching a graph output is what
        # anchors the traced step: cstorch dead-code-eliminates a step with no
        # requested outputs down to an empty graph ("Cannot compile empty CIRH
        # module"). The step is compiled once and replayed, so the fetch must be
        # unconditional and identical every step — only the print is throttled.
        # (Host-side reads are legal ONLY inside a step_closure.)
        loss_v = loss.item()
        recon_v = recon.item()
        if step % args.log_every == 0:
            print(f"step {step:>7}  loss={loss_v:.4f}  recon={recon_v:.4f}")

    @cstorch.checkpoint_closure
    def save_ckpt(step):
        state = {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict(), "step": step}
        cstorch.save(state, f"{args.out_dir}/nubun_cs_step{step}.mdl")
        print(f"  checkpoint saved: step {step}")
        if args.keep_ckpts > 0:
            # /home quota is tight (~1.6 GB per checkpoint with AdamW state).
            old = sorted(glob.glob(f"{args.out_dir}/nubun_cs_step*.mdl"),
                         key=lambda f: int(re.search(r"step(\d+)\.mdl$", f).group(1)))
            for f in old[:-args.keep_ckpts]:
                if args.keep_every and int(re.search(r"step(\d+)\.mdl$", f).group(1)) % args.keep_every == 0:
                    continue          # milestone: never pruned
                os.remove(f)
                print(f"  pruned old checkpoint {os.path.basename(f)}")

    dataloader = cstorch.utils.data.DataLoader(
        build_input_fn(args, vocab, token_weight))
    # DataExecutor takes NO backend kwarg in cstorch 2.10.0 — backend is bound
    # globally by cstorch.backend()/cstorch.compile above.
    executor = cstorch.utils.data.DataExecutor(
        dataloader, num_steps=exec_steps(args), checkpoint_steps=args.ckpt_every)

    print(f"Starting cstorch run: steps={exec_steps(args)} (schedule {args.steps}) "
          f"semantic={not args.no_semantic} compile_only={args.compile_only} "
          f"validate_only={args.validate_only} artifact_dir={args.artifact_dir}")
    for step, batch in enumerate(executor, start=args.start_step + 1):
        loss, recon = train_step(batch)
        log_step(step, loss, recon)
        save_ckpt(step)
    print("Done.")


if __name__ == "__main__":
    main()
