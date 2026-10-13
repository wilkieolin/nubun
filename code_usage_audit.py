"""code_usage_audit.py — how many bits does the bottleneck really spend?

Encoder-only (no decoding, CPU is fine). Encodes the seeded validation split in
every language through the encoder + RVQ and reports, per RVQ level, the number
of active codes and the marginal entropy; per sentence, the number of DISTINCT
level-0 codes and distinct full code stacks among the fixed m_max slots; and
how many level-0 codes are shared by all languages of the same sentence.

This is the audit that showed (PHASE9.md §2) the 64-slot bottleneck is far from
sparse: the 500k CS-3 model uses ~8.5 distinct level-0 codes and ~17 distinct
stacks per sentence, with the other slots duplicating them.

Usage:
    python code_usage_audit.py --checkpoint <ckpt> [--corpus data/parallel_corpus.npz]
        [--embedding-table data/embedding_table.pt] [--langs en,zh,...] [--max-sents 400]
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from vqvae.data import combined_split, load_corpus
from vqvae.model import VQVAE


def build_model(cfg, emb, meta):
    kw = dict(
        vocab_size=emb.shape[0], n_langs=len(meta.short_codes), d_model=cfg["d_model"],
        d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"], n_enc_layers=cfg["n_enc_layers"],
        n_dec_layers=cfg["n_dec_layers"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=meta.pad_token_id, embedding_table=emb,
        use_stop_mask=cfg["use_stop_mask"], use_semantic_head=cfg.get("use_semantic_head", False),
        use_rvq=cfg.get("use_rvq", False), n_rvq_levels=cfg.get("n_rvq_levels", 1),
    )
    if "tie_embeddings" in cfg:
        kw["tie_embeddings"] = cfg["tie_embeddings"]
    try:
        return VQVAE(**kw)
    except TypeError:  # older/newer model signatures
        kw.pop("tie_embeddings", None)
        return VQVAE(**kw)


@torch.no_grad()
def encode_all_levels(model, src):
    """Greedy RVQ assignment at every level -> (B, M, L) int64 (argmin, no ties)."""
    q = model.quantizer
    z = model.encoder(src).reshape(-1, q.d_code)
    res, idx = z, []
    for cb in q.codebooks:
        d = (res ** 2).sum(1, keepdim=True) + (cb ** 2).sum(1)[None] - 2 * F.linear(res, cb)
        i = d.argmin(1)
        idx.append(i)
        res = res - cb[i]
    return torch.stack(idx, -1).view(src.shape[0], model.encoder.m_max, len(q.codebooks))


def entropy_bits(x, k):
    c = np.bincount(x, minlength=k).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()), int((c > 0).sum()), c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--langs", default="en,zh,es,fr,ar,ru,hi,ja,pt,de")
    p.add_argument("--max-sents", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(a.corpus)
    _, val_ids, _, _ = combined_split(dev_ids, devtest_ids, dev_lens, devtest_lens, 0.1, a.seed)
    emb = torch.load(a.embedding_table, map_location="cpu")
    model = build_model(cfg, emb, meta).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    if not hasattr(model.quantizer, "codebooks"):
        raise SystemExit("this audit is for RVQ checkpoints (quantizer.codebooks)")

    q = model.quantizer
    L, K, M = len(q.codebooks), q.k, model.encoder.m_max
    pad = meta.pad_token_id
    langs = [meta.short_codes.index(c) for c in a.langs.split(",")]
    n = min(val_ids.shape[0], a.max_sents)

    codes, ntok = [], []
    for l in langs:
        for s in range(0, n, a.batch_size):
            seqs = [val_ids[i, l] for i in range(s, min(s + a.batch_size, n))]
            T = max(len(x) for x in seqs)
            bt = torch.full((len(seqs), T), pad, dtype=torch.long)
            for i, x in enumerate(seqs):
                bt[i, :len(x)] = torch.as_tensor(x)
            codes.append(encode_all_levels(model, bt))
            ntok += [len(x) for x in seqs]
    C = torch.cat(codes).numpy()
    N = C.shape[0]
    ntok = np.array(ntok)

    print(f"Checkpoint: {a.checkpoint} ({ck['step']} steps)")
    print(f"{N} encodings ({n} sentences x {len(langs)} langs), mean source tokens {ntok.mean():.1f}")
    print(f"raw bottleneck: {M} slots x {L} levels x log2({K})={math.log2(K):.0f} bits "
          f"= {M * L * math.log2(K):.0f} bits/sentence")
    print("level  active/%d  H(bits)  perplexity" % K)
    tot = 0.0
    for lv in range(L):
        h, act, _ = entropy_bits(C[:, :, lv].ravel(), K)
        tot += h
        print(f"  L{lv}    {act:3d}        {h:5.2f}    {2 ** h:6.1f}")
    print(f"sum of marginal entropies x {M} slots = {tot * M:.0f} bits/sentence (independence upper bound)")

    hs = [entropy_bits(C[:, m, 0], K)[0] for m in range(M)]
    print(f"L0 per-slot entropy: mean {np.mean(hs):.2f} bits, min {min(hs):.2f}, max {max(hs):.2f}")
    d0 = np.array([len(set(C[i, :, 0])) for i in range(N)])
    stacks = np.array([len({tuple(C[i, m]) for m in range(M)}) for i in range(N)])
    print(f"distinct L0 codes per sentence: mean {d0.mean():.1f} of {M}; "
          f"distinct full stacks {stacks.mean():.1f} of {M}")

    _, _, c0 = entropy_bits(C[:, :, 0].ravel(), K)
    o = np.argsort(-c0)
    cum = np.cumsum(c0[o]) / c0.sum()
    print("top-10 L0 codes by usage (code: share):",
          ", ".join(f"{int(i)}:{c0[i] / c0.sum():.3f}" for i in o[:10]))
    print(f"L0 codes covering 50% / 90% of slot mass: {np.searchsorted(cum, .5) + 1} / {np.searchsorted(cum, .9) + 1}")
    # Zipf exponent of the L0 rank-frequency curve (least squares in log-log over active codes)
    r = np.arange(1, (c0[o] > 0).sum() + 1)
    slope = np.polyfit(np.log(r), np.log(c0[o][c0[o] > 0]), 1)[0]
    print(f"L0 rank-frequency log-log slope: {slope:.2f} (Zipf = -1)")

    per_sent = {}
    for li, _ in enumerate(langs):
        for i in range(n):
            per_sent.setdefault(i, []).append(set(C[li * n + i, :, 0]))
    shared = np.mean([len(set.intersection(*v)) for v in per_sent.values()])
    union = np.mean([len(set.union(*v)) for v in per_sent.values()])
    print(f"L0 codes shared by ALL {len(langs)} languages of the same sentence: "
          f"mean {shared:.1f} (union {union:.1f})")


if __name__ == "__main__":
    main()
