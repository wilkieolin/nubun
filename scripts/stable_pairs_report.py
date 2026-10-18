"""stable_pairs_report.py — PHASE9 A4 follow-up: do cross-seed-stable radicals mean the same thing?

Takes the level-0 pair list written by codebook_stability.py --out (field `a` =
code in model A, `b` = matched code in model B, `jacc` = sentence Jaccard) and
the two analyze_codebook.py level-0 reports, and prints, for every pair with
jacc >= tau, the AGGREGATE top tokens of both codes side by side, plus the
per-language lines on request.

  python scripts/stable_pairs_report.py --pairs results/a4/pairs_seed2_seed42_500k.json \
      --a results/a4/codebook_L0_seed2_500k.txt --b results/cs3_500k_codebook_L0.txt
"""
import argparse
import json
import re

HEAD = re.compile(r"^(?:L\d+ )?Char\s+(\d+)\s*$")


def parse_report(path):
    """-> {code: {"AGGREGATE": str, "en": str, ...}}"""
    out, cur = {}, None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = HEAD.match(line)
            if m:
                cur = out.setdefault(int(m.group(1)), {})
                continue
            if cur is not None and line.startswith("  ") and ":" in line:
                key, val = line.strip().split(":", 1)
                cur[key.strip()] = val.strip()
    return out


def words(agg, n):
    """'water(+3.1), agua(+2.0)' -> 'water agua'"""
    return " ".join(re.findall(r"(\S+?)\(\+?-?\d", agg)[:n]) or "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--a", required=True, help="analyze_codebook L0 report of model A (pairs' `a`)")
    p.add_argument("--b", required=True, help="analyze_codebook L0 report of model B (pairs' `b`)")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--langs", default="", help="also print these per-language lines, e.g. en,zh")
    args = p.parse_args()

    d = json.load(open(args.pairs))
    ra, rb = parse_report(args.a), parse_report(args.b)
    pairs = [x for x in d["level0_pairs"] if x["jacc"] >= args.tau]
    pairs.sort(key=lambda x: -x["jacc"])
    print(f"A = {d['a']} (step {d.get('step_a')})\nB = {d['b']} (step {d.get('step_b')})")
    print(f"{len(pairs)} level-0 pairs with sentence-Jaccard >= {args.tau}; "
          f"decoded codes: A {len(ra)}, B {len(rb)}\n")
    print(f"{'J':>5} {'A':>4} {'useA%':>6} {'B':>4} {'useB%':>6}  aggregate tokens (A | B)")
    missing = 0
    for x in pairs:
        a, b = x["a"], x["b"]
        wa = words(ra.get(a, {}).get("AGGREGATE", ""), args.top)
        wb = words(rb.get(b, {}).get("AGGREGATE", ""), args.top)
        missing += (a not in ra) + (b not in rb)
        print(f"{x['jacc']:5.2f} {a:>4} {100*x['use_a']:6.2f} {b:>4} {100*x['use_b']:6.2f}  {wa}  |  {wb}")
        for l in filter(None, args.langs.split(",")):
            print(f"{'':>29}  {l} A: {words(ra.get(a, {}).get(l, ''), args.top)}")
            print(f"{'':>29}  {l} B: {words(rb.get(b, {}).get(l, ''), args.top)}")
    if missing:
        print(f"\n{missing} code(s) absent from a report (analyze_codebook --active-only or --max-codes?)")


if __name__ == "__main__":
    main()
