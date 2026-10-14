"""codebook_stability.py — Phase 9 A4: do two training runs learn the same radicals?

Locatello et al. (2019) showed that "disentangled" factor inventories are
largely a function of the random seed. Before we read our level-0 codes as a
radical inventory, we need to know how much of that inventory recurs when
only the seed changes.

Two checkpoints A and B encode the same held-out sentences (seeded split, all
10 FLORES languages). Codes are compared per RVQ level in two views:

  slot view     A's code at slot i of sentence s vs B's code at the same slot.
                Contingency table (k x k) -> Hungarian matching maximising
                agreement -> fraction of slots that agree under the best
                one-to-one relabelling. Also NMI (relabelling-invariant).
  sentence view slot-agnostic: each code's footprint is the SET of sentences
                in which it appears. Pairwise Jaccard between A-codes and
                B-codes -> Hungarian -> matched Jaccard per code. A radical
                "recurs" if its matched partner shares >= tau of its sentences.

Chance for every number: the same statistic with B's sentences rolled by one
(sentence s of A against sentence s+1 of B), so the labels keep their
marginals but lose their content.

Controls worth running first: the same run at two late steps (upper bound,
e.g. 480k vs 500k) and the same run early vs late (100k vs 500k).

  python codebook_stability.py --a ckptA.eval.pt --b ckptB.eval.pt [--out pairs.json]
"""

import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as nmi

import vqvae_select
vqvae_select.select_from_argv("--a")   # pre-port checkpoints need pre-port vqvae/
vqvae_select.select_from_argv("--b")
from code_usage_audit import build_model, encode_all_levels  # noqa: E402
from vqvae.data import combined_split, load_corpus  # noqa: E402


def load_model(path, emb, meta):
    ck = vqvae_select.load_checkpoint(path)
    model = build_model(ck["args"], emb, meta).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    if not hasattr(model.quantizer, "codebooks"):
        raise SystemExit(f"{path}: not an RVQ checkpoint")
    return model, int(ck.get("step", -1))


@torch.no_grad()
def encode(model, val_ids, langs, n, bs, pad):
    """-> (S, M, L) int64, sentences ordered language-major."""
    out = []
    for l in langs:
        for s in range(0, n, bs):
            seqs = [val_ids[i, l] for i in range(s, min(s + bs, n))]
            T = max(len(x) for x in seqs)
            bt = torch.full((len(seqs), T), pad, dtype=torch.long)
            for i, x in enumerate(seqs):
                bt[i, :len(x)] = torch.as_tensor(x)
            out.append(encode_all_levels(model, bt))
    return torch.cat(out).numpy()


def contingency(a, b, k):
    C = np.zeros((k, k))
    np.add.at(C, (a.ravel(), b.ravel()), 1.0)
    return C


def hungarian(sim):
    """Maximise sum sim[r, c] over a permutation. -> (rows, cols)."""
    return linear_sum_assignment(-sim)


def slot_view(a, b, k):
    """a, b: (S, M) codes at one level. -> agreement under best relabelling, NMI."""
    C = contingency(a, b, k)
    r, c = hungarian(C)
    agree = C[r, c].sum() / C.sum()
    return agree, nmi(a.ravel(), b.ravel())


def sentence_sets(a, k):
    """a: (S, M) -> (k, S) bool: code j appears in sentence s."""
    S = a.shape[0]
    P = np.zeros((k, S), dtype=bool)
    for s in range(S):
        P[np.unique(a[s]), s] = True
    return P


def sentence_view(a, b, k, usage_min):
    """Slot-agnostic matching on sentence footprints.
    -> dict with per-code matched Jaccard for A's active codes."""
    PA, PB = sentence_sets(a, k), sentence_sets(b, k)
    inter = (PA.astype(np.int32) @ PB.T.astype(np.int32)).astype(float)   # (k, k)
    na, nb = PA.sum(1).astype(float), PB.sum(1).astype(float)
    union = na[:, None] + nb[None, :] - inter
    J = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    r, c = hungarian(J)
    perm = np.full(k, -1)
    perm[r] = c
    jm = np.zeros(k)
    jm[r] = J[r, c]
    usage_a = np.bincount(a.ravel(), minlength=k).astype(float)
    active = usage_a >= usage_min
    return dict(perm=perm, jacc=jm, active=active, usage_a=usage_a,
                usage_b=np.bincount(b.ravel(), minlength=k).astype(float))


def stack_labels(codes, upto):
    """codes: (S, M, L) -> (S*M,) int labels for the joint stack of levels 0..upto."""
    x = codes[:, :, :upto + 1].reshape(-1, upto + 1)
    _, inv = np.unique(x, axis=0, return_inverse=True)
    return inv.ravel()


def roll_sentences(codes, langs_n, n):
    """Shift sentence index by one within each language block -> chance pairing."""
    out = codes.copy()
    for l in range(langs_n):
        blk = slice(l * n, (l + 1) * n)
        out[blk] = np.roll(codes[blk], 1, axis=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--langs", default="en,zh,es,fr,ar,ru,hi,ja,pt,de")
    p.add_argument("--max-sents", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--usage-min-frac", type=float, default=1e-3,
                   help="a code is 'active' if it takes >= this fraction of slots")
    p.add_argument("--tau", type=float, default=0.5,
                   help="matched sentence-Jaccard at which a radical counts as recurring")
    p.add_argument("--out", default=None, help="JSON of level-0 matched pairs")
    args = p.parse_args()

    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.corpus)
    _, val_ids, _, _ = combined_split(dev_ids, devtest_ids, dev_lens, devtest_lens, 0.1, args.seed)
    emb = torch.load(args.embedding_table, map_location="cpu")
    pad = meta.pad_token_id
    langs = [meta.short_codes.index(c) for c in args.langs.split(",")]
    n = min(val_ids.shape[0], args.max_sents)

    ckA, ckB = vqvae_select.load_checkpoint(args.a), vqvae_select.load_checkpoint(args.b)
    if vqvae_select.is_preport(ckA["model_state"]) != vqvae_select.is_preport(ckB["model_state"]):
        raise SystemExit("A and B must both be pre-port or both post-port checkpoints: "
                         "they need different vqvae/ code and cannot share a process")
    mA, stepA = load_model(args.a, emb, meta)
    mB, stepB = load_model(args.b, emb, meta)
    k = mA.quantizer.k
    assert k == mB.quantizer.k and mA.encoder.m_max == mB.encoder.m_max
    print(f"A: {args.a} (step {stepA})\nB: {args.b} (step {stepB})")
    print(f"{n} sentences x {len(langs)} langs, k={k}, m_max={mA.encoder.m_max}")

    A = encode(mA, val_ids, langs, n, args.batch_size, pad)      # (S, M, L)
    B = encode(mB, val_ids, langs, n, args.batch_size, pad)
    Bc = roll_sentences(B, len(langs), n)                        # chance partner
    S, M, L = A.shape
    usage_min = args.usage_min_frac * S * M

    print(f"\n{'lvl':>3} {'actA':>4} {'actB':>4} | {'slot-agree':>17} | {'NMI':>15} | "
          f"{'sent-Jacc(mean)':>19} | {'recur@tau':>13} | {'mass':>5}")
    print(f"{'':>3} {'':>4} {'':>4} | {'real':>8} {'chance':>8} | {'real':>7} {'chance':>7} | "
          f"{'real':>9} {'chance':>9} | {'real':>6} {'chance':>6} | ")
    level0 = None
    for l in range(L):
        a, b, bc = A[:, :, l], B[:, :, l], Bc[:, :, l]
        ag, mi = slot_view(a, b, k)
        agc, mic = slot_view(a, bc, k)
        sv, svc = sentence_view(a, b, k, usage_min), sentence_view(a, bc, k, usage_min)
        act = sv["active"]
        actB = (sv["usage_b"] >= usage_min)
        jm, jmc = sv["jacc"][act].mean(), svc["jacc"][act].mean()
        rec = (sv["jacc"][act] >= args.tau).mean()
        recc = (svc["jacc"][act] >= args.tau).mean()
        mass = sv["usage_a"][act & (sv["jacc"] >= args.tau)].sum() / sv["usage_a"].sum()
        print(f"{l:>3} {act.sum():>4} {actB.sum():>4} | {ag:8.3f} {agc:8.3f} | {mi:7.3f} {mic:7.3f} | "
              f"{jm:9.3f} {jmc:9.3f} | {rec:6.3f} {recc:6.3f} | {mass:5.2f}")
        if l == 0:
            level0 = sv

    print("\nJoint stacks (levels 0..l), NMI between A's and B's stack labels")
    print("  (read the real-chance GAP: past level 2 most stacks are unique, so NMI"
          " saturates toward 1 for any pairing and the gap shrinks):")
    for l in range(L):
        la, lb, lbc = stack_labels(A, l), stack_labels(B, l), stack_labels(Bc, l)
        print(f"  0..{l}: NMI real={nmi(la, lb):.3f}  chance={nmi(la, lbc):.3f}  "
              f"(distinct stacks A={len(np.unique(la))}, B={len(np.unique(lb))})")

    # Level-0 detail: the radicals, sorted by A usage, with their matched partner.
    act = level0["active"]
    order = np.argsort(-level0["usage_a"])
    print(f"\nLevel-0 radicals of A (usage >= {args.usage_min_frac:.1e} of slots), matched partner in B:")
    print(f"{'A':>4} {'useA%':>6} -> {'B':>4} {'useB%':>6}  {'sentJ':>5}")
    tot_a, tot_b = level0["usage_a"].sum(), level0["usage_b"].sum()
    for j in order:
        if not act[j]:
            break
        pj = level0["perm"][j]
        print(f"{j:>4} {100*level0['usage_a'][j]/tot_a:6.2f} -> {pj:>4} "
              f"{100*level0['usage_b'][pj]/tot_b:6.2f}  {level0['jacc'][j]:5.2f}")

    if args.out:
        pairs = [dict(a=int(j), b=int(level0["perm"][j]), jacc=float(level0["jacc"][j]),
                      use_a=float(level0["usage_a"][j] / tot_a),
                      use_b=float(level0["usage_b"][level0["perm"][j]] / tot_b))
                 for j in order if act[j]]
        with open(args.out, "w") as f:
            json.dump(dict(a=args.a, b=args.b, step_a=stepA, step_b=stepB, tau=args.tau,
                           level0_pairs=pairs), f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
