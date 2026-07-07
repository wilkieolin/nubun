"""summarize_sweep.py — Aggregate results from a Phase 4 sweep.

Reads results/{config}_eval.txt for each config in scripts/sweep.yaml,
prints a comparison table, identifies the winning config (highest
cross-lingual avg acc), and writes a markdown summary.
"""

import argparse
import glob
import os
import re

import yaml


EVAL_HEAD = re.compile(r"VQ-VAE evaluation:\s*(.+)")
SAME_RE = re.compile(r"same-lang avg acc:\s*([\d.]+)")
CROSS_RE = re.compile(r"cross-lang avg acc:\s*([\d.]+)")


def parse_eval(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    same = SAME_RE.search(text)
    cross = CROSS_RE.search(text)
    head = EVAL_HEAD.search(text)
    if not same or not cross:
        return None

    # Aggregate codes_used and avg_bn across all rows
    rows = []
    for line in text.splitlines():
        parts = line.split()
        # Expected per-row format from evaluate_vqvae.py:
        # src tgt acc recon codes perp avg_bn
        if len(parts) == 7 and parts[0].isalpha() and parts[1].isalpha():
            try:
                rows.append({
                    "src": parts[0], "tgt": parts[1],
                    "acc": float(parts[2]), "recon": float(parts[3]),
                    "codes": int(parts[4]), "perp": float(parts[5]),
                    "avg_bn": float(parts[6]),
                })
            except ValueError:
                pass

    return {
        "checkpoint": head.group(1).strip() if head else "?",
        "same_lang_acc": float(same.group(1)),
        "cross_lang_acc": float(cross.group(1)),
        "n_rows": len(rows),
        "avg_codes_used": sum(r["codes"] for r in rows) / max(1, len(rows)),
        "avg_bottleneck_len": sum(r["avg_bn"] for r in rows) / max(1, len(rows)),
        "avg_recon": sum(r["recon"] for r in rows) / max(1, len(rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", default="scripts/sweep.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--remote-host", default="spark2",
                        help="Pull missing eval files from this host via scp")
    parser.add_argument("--output", default="results/sweep_summary.md")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)
    config_names = [c["name"] for c in sweep["configs"]]
    config_args = {c["name"]: c["args"] for c in sweep["configs"]}

    rows = []
    for name in config_names:
        local_path = os.path.join(args.results_dir, f"{name}_eval.txt")
        if not os.path.exists(local_path):
            # Try to pull from remote
            if args.remote_host:
                cmd = (f"scp {args.remote_host}:code/nubun/{local_path} "
                       f"{local_path} 2>/dev/null")
                os.system(cmd)
        result = parse_eval(local_path)
        if result is None:
            print(f"[skip] {name}: no eval file at {local_path}")
            continue
        result["name"] = name
        result["config"] = config_args[name]
        rows.append(result)

    if not rows:
        print("no results to summarize")
        return

    # Sort by cross-lang acc descending
    rows.sort(key=lambda r: -r["cross_lang_acc"])

    print(f"\n{'='*92}")
    print(f"Sweep summary  ({len(rows)} configs)")
    print(f"{'='*92}")
    print(f"{'config':>20s} {'K':>5s} {'tgtL':>5s} {'cmp':>5s} "
          f"{'same':>7s} {'cross':>7s} {'codes':>7s} {'avg_bn':>7s} {'recon':>7s}")
    print("-" * 92)
    for r in rows:
        cfg = r["config"]
        print(f"{r['name']:>20s} {cfg.get('k', '?'):>5} "
              f"{cfg.get('target_avg_len', '?'):>5} "
              f"{cfg.get('compression_ratio', '?'):>5} "
              f"{r['same_lang_acc']:>7.4f} {r['cross_lang_acc']:>7.4f} "
              f"{r['avg_codes_used']:>7.1f} {r['avg_bottleneck_len']:>7.2f} "
              f"{r['avg_recon']:>7.3f}")

    winner = rows[0]
    print(f"\nWinner: {winner['name']}  (cross-lang acc {winner['cross_lang_acc']:.4f})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# Sweep summary ({len(rows)} configs)\n\n")
        f.write(f"| config | K | target_len | compression | same-lang | cross-lang | "
                f"codes used | avg bn | recon |\n")
        f.write(f"|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            cfg = r["config"]
            f.write(f"| {r['name']} | {cfg.get('k')} | {cfg.get('target_avg_len')} "
                    f"| {cfg.get('compression_ratio')} | {r['same_lang_acc']:.4f} "
                    f"| {r['cross_lang_acc']:.4f} | {r['avg_codes_used']:.1f} "
                    f"| {r['avg_bottleneck_len']:.2f} | {r['avg_recon']:.3f} |\n")
        f.write(f"\n**Winner: {winner['name']}** "
                f"(cross-lang acc {winner['cross_lang_acc']:.4f})\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
