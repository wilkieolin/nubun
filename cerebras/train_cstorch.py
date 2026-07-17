"""Nubun VQ-VAE training loop, ported to Cerebras cstorch (SCAFFOLD).

This is the GB10 train_vqvae.py loop re-expressed in the cstorch idiom:
  - cstorch.compile(model) + @cstorch.trace on the step (fwd/bwd/optim)
  - all host-side reads (.item(), logging) moved into @cstorch.step_closure
  - checkpointing via @cstorch.checkpoint_closure + cstorch.save
  - cstorch.optim optimizer + LR scheduler

It is written to the documented cstorch 2.x API. It has NOT been run on hardware
from the authoring machine — treat every `# VERIFY` as a checkpoint against your
installed release. Bring it up in two milestones (see PORT_CS3.md):

  M1 (recommended first): recon + RVQ losses only, synthetic or real data.
     Proves the custom model trains on the WSE. Run with --no-semantic.
  M2: add the precomputed semantic targets (precompute_semantic_targets.py) and
     the token-weighted CE, matching the GB10 recipe exactly.

Frozen recipe to reproduce (E3@100k winner): RVQ 8x128, tied emb, deep decoder
(d_ff 2048, n_dec 10), lr 3e-4 cosine, warmup 500, lambda_sem 5, weighted CE.
"""
import argparse
import os
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, ".")
from vqvae.model import VQVAE  # noqa: E402

import cerebras.pytorch as cstorch  # noqa: E402

# Repo root (this file is cerebras/train_cstorch.py) — appliance workers must be
# able to import `vqvae` from here and mount it, so it seeds the cluster config.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    )


def weighted_ce(logits, targets, pad, token_weight):
    """Token-weighted cross-entropy == vqvae.losses.reconstruction_loss."""
    l2d = logits.reshape(-1, logits.size(-1))
    t1d = targets.reshape(-1)
    if token_weight is None:
        return F.cross_entropy(l2d, t1d, ignore_index=pad)
    ce = F.cross_entropy(l2d, t1d, ignore_index=pad, reduction="none")
    w = token_weight[t1d] * (t1d != pad).to(logits.dtype)
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def make_synthetic_input_fn(args, vocab):
    """Random batches at the right static shapes — for a no-data dry compile
    (--synthetic-data). Values are meaningless; only shapes matter to the tracer.
    The real pipeline is cerebras/opus_input.py::make_opus_input_fn."""
    B, T, d_sem = args.batch_size, args.seq_len, 384

    def input_fn():
        def gen():
            for _ in range(args.steps + 2):
                b = {
                    "src_ids": torch.randint(4, vocab, (B, T)),
                    "tgt_ids": torch.randint(4, vocab, (B, T)),
                    "src_lang_id": torch.randint(0, args.n_langs, (B,)),
                    "tgt_lang_id": torch.randint(0, args.n_langs, (B,)),
                }
                if not args.no_semantic:
                    b["sem_target"] = torch.randn(B, d_sem)
                yield b
        return gen()
    return input_fn


def build_input_fn(args, vocab):
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
        num_steps=args.steps, parallel_corpus=args.parallel_corpus,
        seed=args.seed, langs=langs, with_semantic=not args.no_semantic,
        sem_dir=args.sem_dir)


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
    ap.add_argument("--no-semantic", action="store_true", help="M1: drop semantic loss")
    ap.add_argument("--no-token-weights", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=20000)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    # --- CS-3 appliance / ALCF cluster config (see PORT_CS3.md §11) ------------
    # On an ALCF user node mgmt_address + credentials come from
    # /opt/cerebras/config_v2 automatically; we only supply run shape + the dirs
    # the appliance workers must see. Off an ALCF node these are harmless — the
    # backend still stops at ClusterConfigError (no mgmt_address), as documented.
    ap.add_argument("--num-csx", type=int, default=1,
                    help="Number of CS-3 systems to request (ALCF has 4).")
    ap.add_argument("--job-time-sec", type=int, default=82800,
                    help="Wall-clock cap in seconds (ALCF hard limit is 24h).")
    ap.add_argument("--job-label", default="name=nubun",
                    help="label=value tag for `csctl get jobs` filtering.")
    ap.add_argument("--mount-dirs", default=None,
                    help="Comma-separated dirs the workers mount (repo + data). "
                         "Defaults to the repo root; add your ALCF data dir.")
    ap.add_argument("--python-paths", default=None,
                    help="Comma-separated PYTHONPATH dirs for the workers "
                         "(must include the repo root so `import vqvae` works).")
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
        token_weight = torch.load(args.token_weights, map_location="cpu")

    # Cluster config: what to run and which dirs the appliance workers see. The
    # cluster address/credentials are resolved from the ALCF user node's
    # /opt/cerebras/config_v2, so we deliberately do NOT set mgmt_address here.
    mount_dirs = ([d for d in args.mount_dirs.split(",") if d]
                  if args.mount_dirs else [REPO_ROOT])
    python_paths = ([d for d in args.python_paths.split(",") if d]
                    if args.python_paths else [REPO_ROOT])
    cluster_config = cstorch.distributed.ClusterConfig(  # VERIFY: 2.10.0 fields
        num_csx=args.num_csx,
        job_labels=[args.job_label],
        job_time_sec=args.job_time_sec,
        mount_dirs=mount_dirs,
        python_paths=python_paths,
    )
    # VERIFY: backend construction / kwargs may differ by release.
    backend = cstorch.backend(
        "CSX", compile_only=args.compile_only, validate_only=args.validate_only,
        cluster_config=cluster_config)

    model = build_model(args, emb)
    model.train()
    compiled_model = cstorch.compile(model, backend)              # VERIFY
    optimizer = cstorch.optim.AdamW(model.parameters(), lr=args.lr)  # VERIFY

    # Linear warmup -> cosine decay, built from native cstorch schedulers.
    # cstorch's LambdaLR is NOT torch's (it takes initial_learning_rate and wants
    # set_value_lambda overridden), so we compose the two-phase schedule with
    # SequentialLR instead. IMPORTANT: CosineDecayLR.total_iters is the ACTUAL
    # remaining step count — do NOT stretch cosine past the run length (that hot-LR
    # tail collapsed the RVQ codebook on the GB10; see PORT_CS3.md / PHASE8.md).
    sched = cstorch.optim.lr_scheduler
    warmup = cstorch.optim.lr_scheduler.LinearLR(
        optimizer, initial_learning_rate=0.0, end_learning_rate=args.lr,
        total_iters=max(1, args.warmup_steps))
    cosine = cstorch.optim.lr_scheduler.CosineDecayLR(
        optimizer, initial_learning_rate=args.lr,
        end_learning_rate=args.lr * args.lr_min_ratio,
        total_iters=max(1, args.steps - args.warmup_steps))
    lr_scheduler = cstorch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

    if token_weight is not None:
        token_weight = token_weight.to(cstorch.current_torch_device())  # VERIFY

    @cstorch.trace
    def train_step(batch):
        out = compiled_model(batch["src_ids"], batch["tgt_ids"], batch["tgt_lang_id"])
        recon = weighted_ce(out["logits"], batch["tgt_ids"][:, 1:],
                            args.pad_token_id, token_weight)
        loss = recon + out["vq_losses"]["commit"] + out["vq_losses"]["codebook"]
        if not args.no_semantic and out["sem_pred"] is not None:
            cos = F.cosine_similarity(out["sem_pred"].float(),
                                      batch["sem_target"].float(), dim=-1)
            loss = loss + args.lambda_sem * (1.0 - cos).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        return loss, recon

    @cstorch.step_closure
    def log_step(step, loss, recon):
        # Host-side read is legal ONLY inside a step_closure.
        if step % args.log_every == 0:
            print(f"step {step:>7}  loss={loss.item():.4f}  recon={recon.item():.4f}")

    @cstorch.checkpoint_closure
    def save_ckpt(step):
        state = {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": step}
        cstorch.save(state, f"{args.out_dir}/nubun_cs_step{step}.mdl")  # VERIFY
        print(f"  checkpoint saved: step {step}")

    dataloader = cstorch.utils.data.DataLoader(build_input_fn(args, vocab))  # VERIFY
    # DataExecutor takes NO backend kwarg in cstorch 2.10.0 — backend is bound
    # globally by cstorch.backend()/cstorch.compile above.
    executor = cstorch.utils.data.DataExecutor(
        dataloader, num_steps=args.steps, checkpoint_steps=args.ckpt_every)

    print(f"Starting cstorch run: steps={args.steps} semantic={not args.no_semantic} "
          f"compile_only={args.compile_only}")
    for step, batch in enumerate(executor, start=1):
        loss, recon = train_step(batch)
        log_step(step, loss, recon)
        save_ckpt(step)
    print("Done.")


if __name__ == "__main__":
    main()
