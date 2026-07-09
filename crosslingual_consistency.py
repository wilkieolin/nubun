"""crosslingual_consistency.py — Phase 6: the interlingua test we never ran.

The thesis claims the characters are *language-independent*: the same meaning, in
any source language, should encode to (nearly) the same codes. We have inferred
this from a pooled semantic loss; we have never measured it directly.

For each parallel sentence we encode it separately in every language and compare
the resulting code multiset across language pairs. Order-free (bag-of-codes)
Jaccard is the primary measure — robust to the fact that nothing constrains slot
order to align across languages.

Chance baseline: compare language A of sentence i against language B of sentence
j != i (roll by 1). If real agreement >> chance, the codes are genuinely shared
across languages (interlingua holds at the bag level). If real ~ chance, the
"language-independent character" claim fails.
"""

import argparse
from itertools import combinations

import numpy as np
import torch

from vqvae.data import combined_split, load_corpus
from vqvae.model import VQVAE


def compute_target_len(src_ids, pad, ratio, slack, m_max):
    src_lens = (src_ids != pad).sum(dim=1).float()
    return (src_lens * ratio + slack).ceil().long().clamp(min=1, max=m_max)


@torch.no_grad()
def code_sets(model, src_t, pad, ratio, slack):
    """Return a list (len B) of python sets of active (non-stop) code indices."""
    z_e = model.encoder(src_t)
    _, indices, _, _ = model.quantizer(z_e)
    if ratio > 0:
        tl = compute_target_len(src_t, pad, ratio, slack, model.encoder.m_max)
        indices = model.quantizer.force_stop_at(indices, tl)
    if model.use_stop_mask:
        mem_mask = model.quantizer.get_stop_mask(indices)
    else:
        mem_mask = torch.ones_like(indices, dtype=torch.bool)
    out = []
    idx_cpu = indices.cpu().numpy()
    msk_cpu = mem_mask.cpu().numpy()
    for i in range(idx_cpu.shape[0]):
        out.append(set(int(c) for c in idx_cpu[i][msk_cpu[i]]))
    return out


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--langs", default="en,zh,es,fr,ar,ru")
    p.add_argument("--combine-splits", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"Checkpoint: {args.checkpoint} ({ckpt['step']} steps)")

    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.corpus)
    if args.combine_splits:
        _, val_ids, _, _ = combined_split(dev_ids, devtest_ids, dev_lens,
                                          devtest_lens, args.val_fraction, args.seed)
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

    ratio = float(cfg.get("compression_ratio", 0.0) or 0.0)
    slack = int(cfg.get("length_slack", 4))
    pad = meta.pad_token_id
    langs = [meta.short_codes.index(c) for c in args.langs.split(",")]
    n = val_ids.shape[0]

    # Per-language code sets for every sentence.
    sets = {l: [] for l in langs}
    for l in langs:
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            seqs = [val_ids[i, l] for i in range(start, end)]
            T = max(len(x) for x in seqs)
            bt = torch.full((len(seqs), T), pad, dtype=torch.long)
            for i, x in enumerate(seqs):
                bt[i, :len(x)] = torch.as_tensor(x)
            sets[l].extend(code_sets(model, bt.to(args.device), pad, ratio, slack))

    print(f"\nBag-of-codes Jaccard across languages ({args.langs}), N={n} sentences")
    real_all, chance_all = [], []
    for a, b in combinations(langs, 2):
        real = np.mean([jaccard(sets[a][i], sets[b][i]) for i in range(n)])
        # chance: same lang pair, but sentence i of A vs sentence (i+1) of B
        chance = np.mean([jaccard(sets[a][i], sets[b][(i + 1) % n]) for i in range(n)])
        real_all.append(real)
        chance_all.append(chance)
        ca, cb = meta.short_codes[a], meta.short_codes[b]
        print(f"  {ca}-{cb}:  real={real:.3f}  chance={chance:.3f}  lift={real - chance:+.3f}")

    print(f"\nMean real Jaccard : {np.mean(real_all):.3f}")
    print(f"Mean chance       : {np.mean(chance_all):.3f}")
    print(f"Mean lift         : {np.mean(real_all) - np.mean(chance_all):+.3f}")
    print("  (lift >> 0 => codes are shared across languages = interlingua holds;")
    print("   lift ~ 0  => codes are language-specific, thesis fails at the bag level.)")


if __name__ == "__main__":
    main()
