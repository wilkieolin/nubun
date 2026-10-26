"""three_way_core.py — PHASE9 A4, branch 1: is the cross-seed core a fixed set?

Takes three codebook_stability.py pair lists (A-B, A-C, B-C; field `a` is a
code of the first model, `b` its Hungarian partner in the second) and reports
the level-0 codes of A whose partners in B and C are themselves matched to each
other (the triangle closes) with every edge at Jaccard >= tau. Also reports the
pairwise recurring fractions and the two-of-three sets, so "fixed set" vs
"pairwise coincidence" can be read off directly.

  python scripts/three_way_core.py --ab pairs_seed3_seed42_500k.json \
      --ac pairs_seed3_seed2_500k.json --bc pairs_seed42_seed2_500k.json
(bc may be given in either orientation; use --bc-reversed if its `a` is C.)
"""
import argparse
import json


def load(path, reverse=False):
    d = json.load(open(path))
    m = {}
    for p in d["level0_pairs"]:
        a, b = (p["b"], p["a"]) if reverse else (p["a"], p["b"])
        m[a] = (b, p["jacc"], p["use_a"] if not reverse else p["use_b"])
    return d, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--ac", required=True)
    ap.add_argument("--bc", required=True)
    ap.add_argument("--bc-reversed", action="store_true", help="bc file's `a` is model C")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--names", default="A,B,C")
    args = ap.parse_args()
    nA, nB, nC = args.names.split(",")
    dab, AB = load(args.ab)
    dac, AC = load(args.ac)
    dbc, BC = load(args.bc, reverse=args.bc_reversed)
    t = args.tau

    def recur(m):
        act = list(m.values())
        return sum(j >= t for _, j, _ in act), len(act)

    for lbl, m in ((f"{nA}-{nB}", AB), (f"{nA}-{nC}", AC), (f"{nB}-{nC}", BC)):
        r, n = recur(m)
        print(f"{lbl}: {r}/{n} active level-0 codes recur at J >= {t} ({r / n:.3f})")

    print(f"\n{nA} codes recurring in both {nB} and {nC}, and whether the triangle closes:")
    print(f"{nA:>4} {'use%':>5} -> {nB:>4} {'J':>5} -> {nC:>4} {'J':>5} | {nB}->{nC} {'J':>5}  closed")
    both, closed = [], []
    for a, (b, jab, ua) in sorted(AB.items(), key=lambda kv: -kv[1][2]):
        if a not in AC or jab < t:
            continue
        c, jac, _ = AC[a]
        if jac < t:
            continue
        both.append(a)
        bc = BC.get(b)
        ok = bc is not None and bc[0] == c and bc[1] >= t
        closed.append(a) if ok else None
        bcs = f"{bc[0]:>4} {bc[1]:5.2f}" if bc else f"{'-':>4} {'-':>5}"
        print(f"{a:>4} {100 * ua:5.2f} -> {b:>4} {jab:5.2f} -> {c:>4} {jac:5.2f} | {bcs}  {'yes' if ok else 'no'}")
    onlyB = [a for a, (b, j, _) in AB.items() if j >= t and a not in both]
    onlyC = [a for a, (c, j, _) in AC.items() if j >= t and a not in both]
    print(f"\n{nA} codes recurring with {nB} only: {len(onlyB)}, with {nC} only: {len(onlyC)}, "
          f"with both: {len(both)}, three-way closed: {len(closed)}")
    print(f"three-way core ({nA} codes): {closed}")
    print(f"(pairwise A-B recurring {recur(AB)[0]}, A-C {recur(AC)[0]}: a fixed set shows up as "
          f"'both' close to the pairwise counts; a coincidence as 'both' near their product / n)")


if __name__ == "__main__":
    main()
