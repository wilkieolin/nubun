"""evaluate_vqvae.py — Held-out evaluation for VQ-VAE checkpoints.

Reports:
  - Same-language token accuracy (chrF on decoded text per language)
  - Cross-lingual translation token accuracy
  - Compression ratio (avg bottleneck length / source token count)
  - Codebook usage entropy and # codes used
  - Compositional probe (single-word inputs, see how their bottlenecks overlap)
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from vqvae.data import combined_split, load_corpus
from vqvae.model import VQVAE


@torch.no_grad()
def evaluate_pair(
    model: VQVAE, val_ids: np.ndarray, src_lang: int, tgt_lang: int,
    pad_token_id: int, bos_token_id: int, eos_token_id: int,
    batch_size: int = 32, device: str = "cuda",
    token_weight: torch.Tensor | None = None,
) -> dict:
    """Compute teacher-forced token accuracy + bottleneck stats for (src, tgt).

    If token_weight is given, also computes content_accuracy: token correctness
    weighted by token_weight[target] (punctuation ~0, content ~1). This is the
    fair reconstruction measure when the model was trained with weighted CE and
    deliberately ignores punctuation — raw token_accuracy understates it.
    """
    n = val_ids.shape[0]
    total_correct = 0
    total_count = 0
    total_recon = 0.0
    total_batches = 0
    content_correct = 0.0
    content_weight = 0.0
    bn_lengths = []
    code_counts = torch.zeros(model.quantizer.k, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B = end - start
        src_seqs = [val_ids[i, src_lang] for i in range(start, end)]
        tgt_seqs = [val_ids[i, tgt_lang] for i in range(start, end)]
        T_src = max(len(s) for s in src_seqs)
        T_tgt = max(len(s) for s in tgt_seqs)
        src = np.full((B, T_src), pad_token_id, dtype=np.int64)
        tgt = np.full((B, T_tgt), pad_token_id, dtype=np.int64)
        for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
            src[i, : len(s)] = s
            tgt[i, : len(t)] = t

        src_t = torch.from_numpy(src).to(device)
        tgt_t = torch.from_numpy(tgt).to(device)
        lang_t = torch.full((B,), tgt_lang, dtype=torch.int64, device=device)

        out = model(src_t, tgt_t, lang_t)
        logits = out["logits"]
        recon = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_t[:, 1:].reshape(-1), ignore_index=pad_token_id)
        total_recon += recon.item()
        preds = logits.argmax(dim=-1)
        gold = tgt_t[:, 1:]
        mask = gold != pad_token_id
        correct = (preds == gold) & mask
        total_correct += correct.sum().item()
        total_count += mask.sum().item()
        total_batches += 1

        if token_weight is not None:
            w = token_weight[gold] * mask.to(token_weight.dtype)   # (B, T-1)
            content_correct += (correct.to(w.dtype) * w).sum().item()
            content_weight += w.sum().item()

        # Bottleneck lengths (positions before first <stop>)
        if model.use_stop_mask:
            mem_mask = out["mem_mask"]
            bn_lengths.extend(mem_mask.sum(dim=1).tolist())
        code_counts += out["usage"]

    p = code_counts / (code_counts.sum() + 1e-8)
    perplexity = float(torch.exp(-(p * (p + 1e-8).log()).sum()))
    used = int((code_counts > 0).sum())
    return {
        "n_examples": n,
        "token_accuracy": total_correct / max(1, total_count),
        "content_accuracy": (content_correct / content_weight)
                            if content_weight > 0 else float("nan"),
        "recon_loss": total_recon / max(1, total_batches),
        "avg_bottleneck_len": float(np.mean(bn_lengths)) if bn_lengths else float(model.encoder.m_max),
        "codes_used": used,
        "code_perplexity": perplexity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", default="data/parallel_corpus.npz")
    parser.add_argument("--embedding-table", default="data/embedding_table.pt")
    parser.add_argument("--combine-splits", action="store_true",
                        help="Use the train_vqvae split (combined dev+devtest, 90/10) "
                             "to evaluate on the same val set the model didn't see.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/vqvae_eval.txt")
    parser.add_argument("--token-weights", default=None,
                        help="Optional (vocab,) weights from build_token_weights.py; "
                             "enables content-weighted accuracy (fair metric when the "
                             "model was trained with weighted CE).")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"  trained for {ckpt['step']} steps")

    print(f"Loading corpus: {args.corpus}")
    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.corpus)

    if args.combine_splits:
        _, val_ids, _, val_lens = combined_split(
            dev_ids, devtest_ids, dev_lens, devtest_lens,
            val_fraction=args.val_fraction, seed=args.seed)
    else:
        val_ids = devtest_ids
    print(f"  val set: {val_ids.shape}")

    print(f"Loading embedding table: {args.embedding_table}")
    emb = torch.load(args.embedding_table, map_location="cpu")
    vocab_size = emb.shape[0]

    model = VQVAE(
        vocab_size=vocab_size, n_langs=len(meta.short_codes),
        d_model=cfg["d_model"], d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"],
        n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=meta.pad_token_id,
        embedding_table=emb, use_stop_mask=cfg["use_stop_mask"],
        use_ema=cfg.get("use_ema", False), ema_decay=cfg.get("ema_decay", 0.99),
        use_semantic_head=cfg.get("use_semantic_head", False),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(args.device).eval()

    token_weight = None
    if args.token_weights:
        print(f"Loading token weights: {args.token_weights}")
        token_weight = torch.load(args.token_weights, map_location="cpu").to(args.device)

    n_lang = len(meta.short_codes)
    rows = []

    print(f"\nEvaluating {n_lang} same-lang + {n_lang*(n_lang-1)} cross-lang pairs...")
    for s in range(n_lang):
        for t in range(n_lang):
            print(f"  {meta.short_codes[s]} -> {meta.short_codes[t]} ", end="", flush=True)
            metrics = evaluate_pair(
                model, val_ids, s, t,
                pad_token_id=meta.pad_token_id,
                bos_token_id=meta.bos_token_id,
                eos_token_id=meta.eos_token_id,
                batch_size=args.batch_size, device=args.device,
                token_weight=token_weight)
            metrics["src"] = meta.short_codes[s]
            metrics["tgt"] = meta.short_codes[t]
            metrics["same_lang"] = (s == t)
            rows.append(metrics)
            print(f"acc={metrics['token_accuracy']:.3f}  "
                  f"c_acc={metrics['content_accuracy']:.3f}  "
                  f"perp={metrics['code_perplexity']:.1f}  "
                  f"used={metrics['codes_used']}")

    same_acc = np.mean([r["token_accuracy"] for r in rows if r["same_lang"]])
    cross_acc = np.mean([r["token_accuracy"] for r in rows if not r["same_lang"]])
    same_cacc = np.nanmean([r["content_accuracy"] for r in rows if r["same_lang"]])
    cross_cacc = np.nanmean([r["content_accuracy"] for r in rows if not r["same_lang"]])
    print(f"\nSummary:")
    print(f"  same-lang acc avg:  {same_acc:.3f}   content-acc: {same_cacc:.3f}")
    print(f"  cross-lang acc avg: {cross_acc:.3f}   content-acc: {cross_cacc:.3f}")
    if rows[0]["avg_bottleneck_len"] != model.encoder.m_max:
        avg_bn = np.mean([r["avg_bottleneck_len"] for r in rows])
        print(f"  avg bottleneck length: {avg_bn:.1f} / {model.encoder.m_max}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"VQ-VAE evaluation: {args.checkpoint}\n")
        f.write(f"trained for {ckpt['step']} steps\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"{'src':>4s} {'tgt':>4s} {'acc':>8s} {'c_acc':>8s} {'recon':>8s} "
                f"{'codes':>8s} {'perp':>8s} {'avg_bn':>8s}\n")
        for r in rows:
            f.write(f"{r['src']:>4s} {r['tgt']:>4s} {r['token_accuracy']:>8.3f} "
                    f"{r['content_accuracy']:>8.3f} "
                    f"{r['recon_loss']:>8.3f} {r['codes_used']:>8d} "
                    f"{r['code_perplexity']:>8.1f} {r['avg_bottleneck_len']:>8.2f}\n")
        f.write(f"\nsame-lang  avg acc: {same_acc:.4f}   content-acc: {same_cacc:.4f}\n")
        f.write(f"cross-lang avg acc: {cross_acc:.4f}   content-acc: {cross_cacc:.4f}\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
