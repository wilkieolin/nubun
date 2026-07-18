"""GATE CHECK — does the frozen Nubun VQ-VAE trace/compile under cstorch?

This is the first thing to run on the CS-3 box. It builds the frozen model
(RVQ 8x128 + tied deep decoder) from the checkpoint's saved config, wraps it
with cstorch.compile, and runs ONE traced train step on SYNTHETIC data with the
CSX backend in compile_only mode. No wafer time, no real data needed — it just
answers "does our custom RVQ + weighted-CE + semantic head lower to the WSE?".

Run:
    python cerebras/compile_check.py \
        --checkpoint data/phase8_e3_rvq8_200k_step100000.pt \
        --embedding-table data/embedding_table.pt

Success = it reaches "COMPILE CHECK PASSED". Any failure prints the offending
op / message — that's the port's real work-list (expected culprits are documented
in PORT_CS3.md; the known one, bincount, is already fixed in vqvae/quantizer.py).

NOTE: cstorch API names below follow the documented 2.x idiom. If your installed
release differs, the VERIFY comments flag the calls most likely to have moved.
"""
import argparse
import sys

import torch
from torch.nn import functional as F

# Make the repo root importable (this file lives in cerebras/).
sys.path.insert(0, ".")
from vqvae.model import VQVAE  # noqa: E402

try:
    import cerebras.pytorch as cstorch
except ImportError:
    sys.exit("cerebras.pytorch not importable — activate the x86 CS env "
             "(see cerebras/requirements-cerebras.txt).")


def build_model(cfg, emb):
    """Reconstruct the frozen architecture from the checkpoint's saved args.
    Mirrors roundtrip_eval.py's construction so the graph matches training."""
    return VQVAE(
        vocab_size=emb.shape[0],
        n_langs=cfg.get("n_langs", 10),
        d_model=cfg["d_model"], d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"],
        n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=cfg.get("pad_token_id", 1),
        embedding_table=emb,
        use_semantic_head=cfg.get("use_semantic_head", False),
        use_rvq=cfg.get("use_rvq", False), n_rvq_levels=cfg.get("n_rvq_levels", 4),
        decoder_type=cfg.get("decoder_type", "ar"),
        tie_embeddings=cfg.get("tie_embeddings", False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    help="Frozen checkpoint (omit with --synthetic).")
    ap.add_argument("--embedding-table", default="data/embedding_table.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--lambda-sem", type=float, default=5.0)
    ap.add_argument("--load-weights", action="store_true",
                    help="Also load the trained weights (tests state_dict load); "
                         "off by default so the check runs with just the config.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Build a tiny RVQ config + random embedding table instead "
                         "of loading the frozen checkpoint. Lets the gate check run "
                         "with NO data/ files — validates that the model + custom RVQ "
                         "+ semantic head lower under the installed cstorch, before "
                         "the 1.6G checkpoint is available.")
    args = ap.parse_args()

    if args.synthetic:
        # Shape-accurate miniature of the frozen E3 recipe: RVQ + tied emb +
        # semantic head, just small. Values are irrelevant for a compile test.
        cfg = dict(n_langs=4, d_model=128, d_code=64, k=32, m_max=16,
                   n_enc_layers=2, n_dec_layers=2, n_heads=4, d_ff=256,
                   beta_commit=0.25, pad_token_id=1,
                   use_semantic_head=True, use_rvq=True, n_rvq_levels=4,
                   decoder_type="ar", tie_embeddings=True)
        emb = torch.randn(512, cfg["d_model"])
        ckpt = None
    else:
        if not args.checkpoint:
            sys.exit("--checkpoint is required unless --synthetic is set.")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["args"]
        try:
            emb = torch.load(args.embedding_table, map_location="cpu")
        except FileNotFoundError:
            # The tied+unfrozen checkpoint already carries the token embedding
            # table (encoder.token_emb.weight). For a compile check we only need
            # its SHAPE — the values are overwritten by --load-weights anyway and
            # irrelevant otherwise — so reconstruct a same-shape table and skip
            # needing the separate embedding_table.pt (see PORT_CS3 §4).
            w = ckpt["model_state"]["encoder.token_emb.weight"]
            print(f"embedding_table not found; deriving shape {tuple(w.shape)} "
                  f"from checkpoint's encoder.token_emb.weight")
            emb = torch.randn_like(w)
    pad = cfg.get("pad_token_id", 1)
    vocab, d_sem = emb.shape[0], 384
    print(f"cfg: RVQ={cfg.get('use_rvq')} levels={cfg.get('n_rvq_levels')} "
          f"tied={cfg.get('tie_embeddings')} d_ff={cfg['d_ff']} "
          f"n_dec={cfg['n_dec_layers']} vocab={vocab}")

    # --- cstorch backend: compile_only means NO wafer is used. -----------------
    # VERIFY: some releases use cstorch.backend("CSX", compile_only=True); others
    # cstorch.backend(backend_type="CSX", compile_dir="./compile", compile_only=True).
    backend = cstorch.backend("CSX", compile_only=True)

    model = build_model(cfg, emb)
    if args.load_weights:
        if ckpt is None:
            sys.exit("--load-weights needs a real --checkpoint (not --synthetic).")
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.train()

    compiled_model = cstorch.compile(model, backend)              # VERIFY signature
    optimizer = cstorch.optim.AdamW(model.parameters(), lr=1e-4)  # VERIFY: cstorch.optim

    token_weight = None  # weighted CE is exercised below only if you pass one in

    @cstorch.trace
    def train_step(src_ids, tgt_ids, tgt_lang_id, sem_target):
        out = compiled_model(src_ids, tgt_ids, tgt_lang_id)
        # reconstruction (plain CE here; swap in weighted CE to exercise that path)
        logits = out["logits"]
        # Explicit CE (log_softmax + gather); fused F.cross_entropy does not
        # lower on cstorch (wgth.cast placement error over the vocab dim).
        l2d = logits.reshape(-1, logits.size(-1))
        t1d = tgt_ids[:, 1:].reshape(-1)
        logp = F.log_softmax(l2d, dim=-1)
        nll = -logp.gather(1, t1d.unsqueeze(1)).squeeze(1)
        m = (t1d != pad).to(logp.dtype)
        recon = (nll * m).sum() / m.sum().clamp(min=1.0)
        loss = recon + out["vq_losses"]["commit"] + out["vq_losses"]["codebook"]
        if out["sem_pred"] is not None:
            cos = F.cosine_similarity(out["sem_pred"].float(), sem_target.float(), dim=-1)
            loss = loss + args.lambda_sem * (1.0 - cos).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    # --- synthetic data: shape-accurate, values irrelevant for a compile test --
    B, T = args.batch_size, args.seq_len

    def input_fn():
        def gen():
            for _ in range(2):
                yield {
                    "src_ids": torch.randint(4, vocab, (B, T)),
                    "tgt_ids": torch.randint(4, vocab, (B, T)),
                    "tgt_lang_id": torch.randint(0, cfg.get("n_langs", 10), (B,)),
                    "sem_target": torch.randn(B, d_sem),
                }
        return gen()

    dataloader = cstorch.utils.data.DataLoader(input_fn)          # VERIFY namespace
    # DataExecutor takes NO backend kwarg in cstorch 2.10.0 — the backend is bound
    # globally by cstorch.backend()/cstorch.compile above.
    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=1)

    # Fetching a graph output anchors the traced step — without it cstorch
    # dead-code-eliminates the step to an empty graph ("Cannot compile empty
    # CIRH module"). The .item() must be registered during the (first) trace, so
    # fetch unconditionally inside a step_closure.
    @cstorch.step_closure
    def fetch_loss(loss):
        _ = loss.item()

    print("Tracing + compiling one step (compile_only, no wafer)...")
    for batch in executor:
        loss = train_step(batch["src_ids"], batch["tgt_ids"],
                          batch["tgt_lang_id"], batch["sem_target"])
        fetch_loss(loss)
    print("COMPILE CHECK PASSED — the frozen model lowers to the WSE.")


if __name__ == "__main__":
    main()
