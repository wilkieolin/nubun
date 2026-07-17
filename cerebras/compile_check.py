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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embedding-table", default="data/embedding_table.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--lambda-sem", type=float, default=5.0)
    ap.add_argument("--load-weights", action="store_true",
                    help="Also load the trained weights (tests state_dict load); "
                         "off by default so the check runs with just the config.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    emb = torch.load(args.embedding_table, map_location="cpu")
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
        recon = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_ids[:, 1:].reshape(-1),
            ignore_index=pad,
        )
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
    executor = cstorch.utils.data.DataExecutor(                   # VERIFY namespace
        dataloader, num_steps=1, backend=backend)

    print("Tracing + compiling one step (compile_only, no wafer)...")
    for batch in executor:
        loss = train_step(batch["src_ids"], batch["tgt_ids"],
                          batch["tgt_lang_id"], batch["sem_target"])
    print("COMPILE CHECK PASSED — the frozen model lowers to the WSE.")


if __name__ == "__main__":
    main()
