"""Convert a cstorch .mdl checkpoint into a train_vqvae-style dict for
roundtrip_eval.py (PORT_CS3 §9), and verify it on the CPU.

    python cerebras/export_ckpt.py model_dir/m1-<stamp>/nubun_cs_step500.mdl \
        --embedding-table <emb.pt> [--opus-dir <opus100_npy> --token-weights <tw.pt>]

Writes <mdl>.eval.pt next to the input. The model is rebuilt exactly as
roundtrip_eval.py builds it (no fixed src_len/tgt_len) and loaded strict=True.
With --opus-dir, also scores a few real batches on the CPU so the number can be
compared with the wafer's logged loss at that step.
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import cerebras.pytorch as cstorch  # noqa: E402
from vqvae.model import VQVAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mdl")
    ap.add_argument("--embedding-table", required=True)
    ap.add_argument("--opus-dir", default=None)
    ap.add_argument("--token-weights", default=None)
    ap.add_argument("--parallel-corpus", default=None)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--out", default=None)
    # Must match the train_cstorch.py run that produced the checkpoint.
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
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    state = cstorch.load(args.mdl)
    sd = {k: (v.to(torch.float32) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in state["model"].items()}
    sd = {k: (torch.as_tensor(v) if not torch.is_tensor(v) else v) for k, v in sd.items()}
    use_sem = any(k.startswith("semantic_head.") for k in sd)
    step = int(state["step"])

    emb = torch.load(args.embedding_table, map_location="cpu")
    cfg = {"d_model": args.d_model, "d_code": args.d_code, "k": args.k,
           "m_max": args.m_max, "n_enc_layers": args.n_enc_layers,
           "n_dec_layers": args.n_dec_layers, "n_heads": args.n_heads,
           "d_ff": args.d_ff, "beta_commit": args.beta_commit,
           "use_stop_mask": False, "use_semantic_head": use_sem,
           "decoder_type": "ar", "use_rvq": True,
           "n_rvq_levels": args.n_rvq_levels, "source": "cerebras-cs3",
           "cs_checkpoint": os.path.abspath(args.mdl)}
    # Same constructor call as roundtrip_eval.py.
    model = VQVAE(
        vocab_size=emb.shape[0], n_langs=args.n_langs,
        d_model=cfg["d_model"], d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"],
        n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=args.pad_token_id,
        embedding_table=emb, use_stop_mask=False,
        use_semantic_head=use_sem, decoder_type="ar",
        use_rvq=True, n_rvq_levels=cfg["n_rvq_levels"],
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"step {step}  semantic_head={use_sem}  tensors={len(sd)}")
    print(f"missing={missing}\nunexpected={unexpected}")
    model.load_state_dict(sd, strict=True)
    bad = [k for k, v in model.state_dict().items()
           if v.is_floating_point() and not torch.isfinite(v).all()]
    assert not bad, f"non-finite tensors: {bad[:5]}"

    out = args.out or os.path.splitext(args.mdl)[0] + ".eval.pt"
    torch.save({"step": step, "args": cfg, "model_state": model.state_dict()}, out)
    print(f"wrote {out}")

    if args.opus_dir:
        from cerebras.opus_input import make_opus_input_fn
        from cerebras.train_cstorch import weighted_ce
        tw = torch.load(args.token_weights, map_location="cpu").float()
        if tw.numel() < emb.shape[0]:
            tw = torch.cat([tw, torch.ones(emb.shape[0] - tw.numel())])
        gen = iter(make_opus_input_fn(
            opus_dir=args.opus_dir, seq_len=args.seq_len, batch_size=32,
            num_steps=args.batches, parallel_corpus=args.parallel_corpus,
            seed=1234, pad=args.pad_token_id, token_weight=tw)())
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(args.batches):
                b = next(gen)
                o = model(b["src_ids"].long(), b["dec_in"].long(),
                          b["tgt_lang_id"].long(), pre_shifted=True)
                r = weighted_ce(o["logits"], o["dec_h"], b["labels"].long(),
                                model.decoder.token_emb.weight,
                                b["label_mask"].float(), b["token_weight_t"].float())
                tot += r.item()
                print(f"  cpu batch {i}: recon={r.item():.4f}")
        print(f"CPU mean recon over {args.batches} batches: {tot / args.batches:.4f}")


if __name__ == "__main__":
    main()
