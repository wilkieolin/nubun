"""diagnose_ablation.py — Step 0 diagnostic: does the decoder rely on the codes?

Reconstructs the held-out val set under three bottleneck conditions and reports
content-weighted accuracy (punctuation ~0, content ~1) for each:

  real    : normal quantized bottleneck
  shuffle : each target reconstructed from ANOTHER sentence's bottleneck
            (roll the batch by 1) — same code distribution, wrong content
  zero    : bottleneck zeroed entirely

Interpretation:
  shuffle/zero content-acc ~= real  -> decoder IGNORES codes (posterior collapse)
                                       => word dropout / NAT are the right fix
  shuffle/zero content-acc << real  -> codes are ESSENTIAL but still low real acc
                                       => capacity/quantization limit; forcing more
                                          code-reliance (dropout/NAT) would BACKFIRE

The "reliance gap" = real - shuffle. Large gap = codes carry the content.

Replicates VQVAE.forward (incl. the training hard length cap) so the "real"
number is a faithful capped eval; only z_q/mem_mask are swapped for ablation.
"""

import argparse

import numpy as np
import torch

from vqvae.data import combined_split, load_corpus
from vqvae.model import VQVAE


def compute_target_len(src_ids, pad, ratio, slack, m_max):
    src_lens = (src_ids != pad).sum(dim=1).float()
    return (src_lens * ratio + slack).ceil().long().clamp(min=1, max=m_max)


@torch.no_grad()
def reconstruct(model, src_t, tgt_t, lang_t, pad, ratio, slack, cond):
    """One forward pass under an ablation condition. Returns logits (B, T-1, V)."""
    z_e = model.encoder(src_t)
    z_q, indices, _, _ = model.quantizer(z_e)
    if ratio > 0:
        tl = compute_target_len(src_t, pad, ratio, slack, model.encoder.m_max)
        indices = model.quantizer.force_stop_at(indices, tl)
        z_q_forced = model.quantizer.codebook[indices]
        mb = (indices != model.quantizer.stop_index).unsqueeze(-1)
        z_q = torch.where(mb, z_q, z_q_forced)
    if model.use_stop_mask:
        mem_mask = model.quantizer.get_stop_mask(indices)
    else:
        mem_mask = torch.ones_like(indices, dtype=torch.bool)

    if cond == "zero":
        z_q = torch.zeros_like(z_q)
    elif cond == "shuffle":
        B = z_q.size(0)
        perm = torch.roll(torch.arange(B, device=z_q.device), shifts=1)
        z_q = z_q[perm]
        mem_mask = mem_mask[perm]
    elif cond != "real":
        raise ValueError(cond)

    return model.decoder(z_q, mem_mask, tgt_t[:, :-1], lang_t)


@torch.no_grad()
def eval_condition(model, val_ids, langs, meta, token_weight, device,
                   ratio, slack, cond, batch_size=32):
    pad = meta.pad_token_id
    agg = {"same": [0.0, 0.0, 0, 0], "cross": [0.0, 0.0, 0, 0]}  # cwt_correct, cwt, correct, count
    n = val_ids.shape[0]
    for s in langs:
        for t in langs:
            bucket = "same" if s == t else "cross"
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                B = end - start
                if B < 2:
                    continue  # shuffle needs >=2
                src_seqs = [val_ids[i, s] for i in range(start, end)]
                tgt_seqs = [val_ids[i, t] for i in range(start, end)]
                T_src = max(len(x) for x in src_seqs)
                T_tgt = max(len(x) for x in tgt_seqs)
                src = np.full((B, T_src), pad, dtype=np.int64)
                tgt = np.full((B, T_tgt), pad, dtype=np.int64)
                for i, (a, b) in enumerate(zip(src_seqs, tgt_seqs)):
                    src[i, : len(a)] = a
                    tgt[i, : len(b)] = b
                src_t = torch.from_numpy(src).to(device)
                tgt_t = torch.from_numpy(tgt).to(device)
                lang_t = torch.full((B,), t, dtype=torch.int64, device=device)

                logits = reconstruct(model, src_t, tgt_t, lang_t, pad, ratio, slack, cond)
                gold = tgt_t[:, 1:]
                mask = gold != pad
                correct = (logits.argmax(-1) == gold) & mask
                w = token_weight[gold] * mask.to(token_weight.dtype)
                a = agg[bucket]
                a[0] += (correct.to(w.dtype) * w).sum().item()
                a[1] += w.sum().item()
                a[2] += correct.sum().item()
                a[3] += mask.sum().item()
    out = {}
    for k, (cwc, cw, c, cnt) in agg.items():
        out[k] = {
            "content_acc": cwc / cw if cw > 0 else float("nan"),
            "raw_acc": c / cnt if cnt > 0 else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", default="data/parallel_corpus.npz")
    ap.add_argument("--embedding-table", default="data/embedding_table.pt")
    ap.add_argument("--token-weights", default="data/token_weights.pt")
    ap.add_argument("--combine-splits", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--langs", default="en,zh,es,fr,ar,ru",
                    help="Comma-separated subset to keep runtime bounded")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"Checkpoint: {args.checkpoint}  ({ckpt['step']} steps)")

    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.corpus)
    if args.combine_splits:
        _, val_ids, _, _ = combined_split(dev_ids, devtest_ids, dev_lens, devtest_lens,
                                          val_fraction=args.val_fraction, seed=args.seed)
    else:
        val_ids = devtest_ids

    emb = torch.load(args.embedding_table, map_location="cpu")
    model = VQVAE(
        vocab_size=emb.shape[0], n_langs=len(meta.short_codes),
        d_model=cfg["d_model"], d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"],
        n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=meta.pad_token_id,
        embedding_table=emb, use_stop_mask=cfg["use_stop_mask"],
        use_ema=cfg.get("use_ema", False), ema_decay=cfg.get("ema_decay", 0.99),
        use_semantic_head=cfg.get("use_semantic_head", False),
        use_length_head=cfg.get("use_length_head", False),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(args.device).eval()
    token_weight = torch.load(args.token_weights, map_location="cpu").to(args.device)

    langs = [meta.short_codes.index(x) for x in args.langs.split(",")]
    ratio = float(cfg.get("compression_ratio", 0.0) or 0.0)
    slack = int(cfg.get("length_slack", 4))
    print(f"langs={args.langs}  cap ratio={ratio} slack={slack}  val={val_ids.shape}\n")

    results = {}
    for cond in ("real", "shuffle", "zero"):
        results[cond] = eval_condition(model, val_ids, langs, meta, token_weight,
                                       args.device, ratio, slack, cond)
        r = results[cond]
        print(f"  {cond:>7s}:  same content-acc={r['same']['content_acc']:.4f} "
              f"(raw {r['same']['raw_acc']:.4f})   "
              f"cross content-acc={r['cross']['content_acc']:.4f} "
              f"(raw {r['cross']['raw_acc']:.4f})")

    # Reliance gap: how much content-acc collapses when codes are wrong/absent.
    print("\n=== RELIANCE GAP (real - ablated), content-acc ===")
    for cond in ("shuffle", "zero"):
        for b in ("same", "cross"):
            gap = results["real"][b]["content_acc"] - results[cond][b]["content_acc"]
            frac = gap / results["real"][b]["content_acc"] if results["real"][b]["content_acc"] else float("nan")
            print(f"  {b:>5s}  real - {cond:<7s} = {gap:+.4f}  ({frac*100:5.1f}% of real)")
    print("\nSmall gap => decoder ignores codes (posterior collapse) => word dropout/NAT.")
    print("Large gap => codes essential but low => capacity/quantization limit.")


if __name__ == "__main__":
    main()
