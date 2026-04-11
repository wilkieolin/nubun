"""
build_radical_hierarchy.py — Radical Hierarchy & Interpretability

Given a radical dictionary (from discover_radicals.py), this script:
1. Clusters radicals into ~50-80 "super-radical" semantic families
2. Scores each radical for interpretability (coherence, specificity, usage)
3. Identifies core vs extended radicals
4. Evaluates coverage with core radicals only

Input:  data/concepts.npz, data/radicals_K*.npz
Output: data/hierarchy.npz, results/hierarchy_report.txt
"""

import argparse
import glob
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def load_data(concepts_path: str, radicals_path: str):
    concepts = np.load(concepts_path, allow_pickle=True)
    radicals = np.load(radicals_path, allow_pickle=True)
    return {
        "concept_embs": concepts["embeddings"],
        "concept_freqs": concepts["frequencies"],
        "concept_labels": list(concepts["labels"]),
        "R": radicals["radicals"],
        "C": radicals["compositions"],
    }


def cluster_radicals(R: np.ndarray, n_super: int = 60,
                     ) -> tuple[np.ndarray, int]:
    """Cluster K radical vectors into super-radical families."""
    clustering = AgglomerativeClustering(
        n_clusters=n_super,
        metric="cosine",
        linkage="average",
    )
    super_labels = clustering.fit_predict(R)
    return super_labels, n_super


def label_super_radicals(R: np.ndarray, super_labels: np.ndarray,
                         concept_embs: np.ndarray,
                         concept_labels: list[str],
                         ) -> list[dict]:
    """Auto-label each super-radical group by its semantic theme."""
    n_super = super_labels.max() + 1
    sims = cosine_similarity(R, concept_embs)  # (K, N_concepts)

    super_info = []
    for sid in range(n_super):
        member_mask = super_labels == sid
        member_indices = np.where(member_mask)[0]
        n_members = len(member_indices)

        # Aggregate similarity: average similarity of member radicals to each concept
        avg_sims = sims[member_mask].mean(axis=0)
        top_concept_idx = np.argsort(-avg_sims)[:10]
        top_words = [concept_labels[i] for i in top_concept_idx]
        top_scores = [float(avg_sims[i]) for i in top_concept_idx]

        # Auto-label: use the top concept word
        label = top_words[0]

        super_info.append({
            "id": sid,
            "label": label,
            "n_members": n_members,
            "member_radical_ids": member_indices.tolist(),
            "top_concepts": top_words,
            "top_scores": top_scores,
        })

    return super_info


def score_radical_interpretability(R: np.ndarray, C: np.ndarray,
                                   concept_embs: np.ndarray,
                                   concept_freqs: np.ndarray,
                                   concept_labels: list[str],
                                   ) -> list[dict]:
    """Score each radical for coherence, specificity, and usage."""
    K = R.shape[0]
    N = concept_embs.shape[0]

    # Dominant radical for each concept (by absolute coefficient magnitude)
    dominant = np.argmax(np.abs(C), axis=1)

    # Active radical mask (nonzero coefficients)
    active = np.abs(C) > 0.01 * np.abs(C).max()

    # Per-concept cosine similarity to its dominant radical
    sims_to_radical = cosine_similarity(R, concept_embs)  # (K, N)

    scores = []
    for k in range(K):
        # Usage: how many concepts use this radical (active, not just dominant)
        usage_count = int(active[:, k].sum())
        usage_freq = float(concept_freqs[active[:, k]].sum()) if usage_count > 0 else 0.0

        # Coherence: among concepts where this radical is dominant,
        # how similar are they to each other?
        dom_mask = dominant == k
        n_dominant = int(dom_mask.sum())
        if n_dominant >= 2:
            dom_embs = concept_embs[dom_mask]
            pairwise = cosine_similarity(dom_embs)
            # Average off-diagonal similarity
            coherence = float((pairwise.sum() - n_dominant) / (n_dominant * (n_dominant - 1)))
        elif n_dominant == 1:
            coherence = 1.0
        else:
            coherence = 0.0

        # Specificity: how peaked is the radical's similarity distribution?
        # High specificity = radical is close to a few concepts and far from most
        radical_sims = sims_to_radical[k]
        specificity = float(np.std(radical_sims))

        # Nearest concept label for this radical
        nearest_idx = np.argmax(radical_sims)
        nearest_label = concept_labels[nearest_idx]
        nearest_sim = float(radical_sims[nearest_idx])

        scores.append({
            "radical_id": k,
            "label": nearest_label,
            "nearest_sim": nearest_sim,
            "coherence": coherence,
            "specificity": specificity,
            "usage_count": usage_count,
            "usage_freq": usage_freq,
            "n_dominant": n_dominant,
        })

    return scores


def identify_core_radicals(scores: list[dict],
                           n_core: int = 80,
                           ) -> tuple[list[int], list[int]]:
    """Split radicals into core (learn first) and extended sets.

    Core radicals have high usage frequency AND high coherence.
    """
    # Combined score: usage_freq * coherence * specificity
    for s in scores:
        s["core_score"] = s["usage_freq"] * s["coherence"] * (1 + s["specificity"])

    ranked = sorted(scores, key=lambda s: s["core_score"], reverse=True)
    core_ids = [s["radical_id"] for s in ranked[:n_core]]
    extended_ids = [s["radical_id"] for s in ranked[n_core:]]

    return core_ids, extended_ids


def evaluate_core_only(C: np.ndarray, R: np.ndarray,
                       concept_embs: np.ndarray,
                       concept_freqs: np.ndarray,
                       core_ids: list[int],
                       max_nonzero: int = 4,
                       ) -> dict:
    """Evaluate coverage using ONLY core radicals."""
    from sklearn.linear_model import OrthogonalMatchingPursuit

    R_core = R[core_ids]

    C_core = np.zeros((concept_embs.shape[0], len(core_ids)))
    for i in range(concept_embs.shape[0]):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max_nonzero)
        omp.fit(R_core.T, concept_embs[i])
        C_core[i] = omp.coef_

    X_recon = C_core @ R_core
    X_norm = concept_embs / (np.linalg.norm(concept_embs, axis=1, keepdims=True) + 1e-8)
    R_norm = X_recon / (np.linalg.norm(X_recon, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(X_norm * R_norm, axis=1)

    # Top-1000 coverage
    top1k_idx = np.argsort(-concept_freqs)[:1000]
    top1k_coverage = float((cos_sims[top1k_idx] > 0.8).mean())

    return {
        "n_core": len(core_ids),
        "overall_coverage_80": float((cos_sims > 0.8).mean()),
        "overall_mean_cosine": float(cos_sims.mean()),
        "top1k_coverage_80": top1k_coverage,
        "top1k_mean_cosine": float(cos_sims[top1k_idx].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Build radical hierarchy")
    parser.add_argument("--concepts", type=str, default="data/concepts.npz")
    parser.add_argument("--radicals", type=str, default=None,
                        help="Path to radicals .npz (auto-detected if omitted)")
    parser.add_argument("--n-super", type=int, default=60,
                        help="Number of super-radical groups (default: 60)")
    parser.add_argument("--n-core", type=int, default=80,
                        help="Number of core radicals (default: 80)")
    parser.add_argument("--output", type=str, default="data/hierarchy.npz")
    parser.add_argument("--report-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    # Find radicals file
    if args.radicals is None:
        candidates = sorted(glob.glob("data/radicals_K*.npz"))
        if not candidates:
            raise FileNotFoundError("No radical files found in data/")
        args.radicals = candidates[-1]

    print(f"Loading data...")
    data = load_data(args.concepts, args.radicals)
    R, C = data["R"], data["C"]
    K = R.shape[0]
    print(f"  {K} radicals, {len(data['concept_labels'])} concepts")

    # 1. Cluster radicals into super-radical families
    print(f"\n1. Clustering {K} radicals into {args.n_super} super-radical families...")
    super_labels, n_super = cluster_radicals(R, n_super=args.n_super)
    super_info = label_super_radicals(
        R, super_labels, data["concept_embs"], data["concept_labels"])

    print(f"  Super-radical families:")
    for si in super_info[:20]:
        print(f"    [{si['id']:2d}] {si['label']:<20s} ({si['n_members']} radicals) "
              f"— {', '.join(si['top_concepts'][:5])}")
    if n_super > 20:
        print(f"    ... ({n_super - 20} more)")

    # 2. Score radical interpretability
    print(f"\n2. Scoring radical interpretability...")
    scores = score_radical_interpretability(
        R, C, data["concept_embs"], data["concept_freqs"], data["concept_labels"])

    # Sort by coherence for display
    by_coherence = sorted(scores, key=lambda s: s["coherence"], reverse=True)
    print(f"  Most coherent radicals:")
    for s in by_coherence[:15]:
        print(f"    R{s['radical_id']:3d} '{s['label']:<15s}' "
              f"coherence={s['coherence']:.3f} "
              f"specificity={s['specificity']:.3f} "
              f"usage={s['usage_count']}")

    # Junk radicals (low coherence + low usage)
    junk = [s for s in scores if s["coherence"] < 0.3 and s["usage_count"] < 10]
    print(f"\n  Potential junk radicals (low coherence + low usage): {len(junk)}/{K}")

    # 3. Identify core vs extended
    print(f"\n3. Identifying core ({args.n_core}) vs extended radicals...")
    core_ids, extended_ids = identify_core_radicals(scores, n_core=args.n_core)

    core_labels = [scores[i]["label"] for i in core_ids[:20]]
    print(f"  Top 20 core radicals: {core_labels}")

    # 4. Evaluate core-only coverage
    print(f"\n4. Evaluating core-only coverage...")
    core_metrics = evaluate_core_only(
        C, R, data["concept_embs"], data["concept_freqs"], core_ids)
    print(f"  Core-only ({core_metrics['n_core']} radicals):")
    print(f"    Overall coverage (>0.8): {core_metrics['overall_coverage_80']:.1%}")
    print(f"    Overall mean cosine: {core_metrics['overall_mean_cosine']:.4f}")
    print(f"    Top-1000 coverage (>0.8): {core_metrics['top1k_coverage_80']:.1%}")
    print(f"    Top-1000 mean cosine: {core_metrics['top1k_mean_cosine']:.4f}")

    # Save hierarchy
    np.savez(
        args.output,
        super_labels=super_labels,
        core_ids=np.array(core_ids),
        extended_ids=np.array(extended_ids),
        **core_metrics,
    )
    print(f"\n  Hierarchy saved to {args.output}")

    # Write detailed report
    report_path = os.path.join(args.report_dir, "hierarchy_report.txt")
    with open(report_path, "w") as f:
        f.write("RADICAL HIERARCHY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total radicals: {K}\n")
        f.write(f"Super-radical families: {n_super}\n")
        f.write(f"Core radicals: {len(core_ids)}\n")
        f.write(f"Extended radicals: {len(extended_ids)}\n")
        f.write(f"Potential junk radicals: {len(junk)}\n\n")

        f.write("SUPER-RADICAL FAMILIES\n")
        f.write("-" * 60 + "\n")
        for si in super_info:
            f.write(f"[{si['id']:2d}] {si['label']:<20s} ({si['n_members']} members)\n")
            f.write(f"     Concepts: {', '.join(si['top_concepts'][:8])}\n")
            f.write(f"     Radicals: {si['member_radical_ids']}\n\n")

        f.write("\nRADICAL INTERPRETABILITY SCORES\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'ID':>4s} {'Label':<20s} {'Coherence':>10s} {'Specificity':>12s} "
                f"{'Usage':>8s} {'Core':>6s}\n")
        for s in sorted(scores, key=lambda s: s["core_score"], reverse=True):
            is_core = "CORE" if s["radical_id"] in core_ids else ""
            f.write(f"{s['radical_id']:4d} {s['label']:<20s} {s['coherence']:10.3f} "
                    f"{s['specificity']:12.3f} {s['usage_count']:8d} {is_core:>6s}\n")

        f.write(f"\nCORE-ONLY EVALUATION\n")
        f.write("-" * 60 + "\n")
        for k, v in core_metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
