"""Code distance between minimal-pair members (PHASE9 §8 branch 2 readout).

For each pair in augment/minpairs.py package shards, encode both members (the
non-English side by default) and count the slots whose RVQ stack over levels
0..d-1 differs, against random same-language sentence pairs. The penalty in
vqvae/pairs.py pushes the level-0 count to <= its budget; this measures where
a model sits, with or without it.

  python scripts/pair_hamming.py --checkpoint model_dir/<run>/nubun_cs_step50000.eval.pt \\
      --minpairs-dir data/minpairs_medium [--side en] [--depths 1,2,8]

Baseline, converged seed 2 (500k, no pairs in training), pilot pairs, X side,
level 0: tense median 2 slots (random 62), negation median 47 (bimodal: an
inserted negator shifts the positional code); all 8 levels: 63 of 64.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vqvae_select  # noqa: E402
from code_usage_audit import build_model, encode_all_levels  # noqa: E402
from vqvae.data import load_corpus  # noqa: E402

FACTORS = ("negation", "tense")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--minpairs-dir", required=True)
    p.add_argument("--side", choices=["x", "en"], default="x")
    p.add_argument("--depths", default="1,2,8")
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ck = vqvae_select.load_checkpoint(a.checkpoint)
    _, _, _, _, meta = load_corpus(a.corpus)
    emb = torch.load(a.embedding_table, map_location="cpu")
    model = build_model(ck["args"], emb, meta).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    model = model.to(a.device)
    pad = meta.pad_token_id

    def codes(seqs):
        out = []
        for s in range(0, len(seqs), a.batch_size):
            ch = seqs[s:s + a.batch_size]
            bt = torch.full((len(ch), max(map(len, ch))), pad, dtype=torch.long)
            for j, x in enumerate(ch):
                bt[j, :len(x)] = torch.from_numpy(np.asarray(x))
            out.append(encode_all_levels(model, bt.to(a.device)).cpu().numpy())
        return np.concatenate(out)                  # (N, M, L)

    depths = [int(d) for d in a.depths.split(",")]
    rng = np.random.default_rng(a.seed)
    res = {}
    # packed <lang>_en.npz or unpacked <lang>_en/ (cerebras/unpack_opus.sh)
    for path in sorted(glob.glob(os.path.join(a.minpairs_dir, "*_en.npz"))
                       + [p for p in glob.glob(os.path.join(a.minpairs_dir, "*_en"))
                          if os.path.isdir(p)]):
        d = (np.load(path) if path.endswith(".npz") else
             {k[:-4]: np.load(os.path.join(path, k)) for k in os.listdir(path)})
        tok, off = ((d["src_tokens"], d["src_offsets"]) if a.side == "x"
                    else (d["tgt_tokens"], d["tgt_offsets"]))
        mem, fac = d["member"], d["factor"]
        c = codes([tok[off[i]:off[i + 1]] for i in range(len(mem))])
        first = np.flatnonzero(mem == 0)
        perm = rng.permutation(first)
        for depth in depths:
            for kind, other in (("pair", first + 1), ("random", perm)):
                h = (c[first][..., :depth] != c[other][..., :depth]).any(-1).sum(-1)
                for k, f in enumerate(FACTORS):
                    res.setdefault((depth, f, kind), []).extend(h[fac[first] == k].tolist())
    M = model.encoder.m_max
    print(f"{a.checkpoint}  side={a.side}  slots differing (of {M})")
    print(f"{'levels':<8s}{'factor':<10s}{'kind':<8s}{'n':>6s}{'mean':>7s}{'median':>7s}{'p25':>5s}{'=0':>7s}{'<=2':>7s}{'<=4':>7s}")
    for (depth, f, kind), v in sorted(res.items()):
        v = np.array(v)
        print(f"0-{depth - 1:<6d}{f:<10s}{kind:<8s}{len(v):6d}{v.mean():7.1f}{np.median(v):7.0f}"
              f"{np.percentile(v, 25):5.0f}{np.mean(v == 0):7.1%}{np.mean(v <= 2):7.1%}{np.mean(v <= 4):7.1%}")


if __name__ == "__main__":
    main()
