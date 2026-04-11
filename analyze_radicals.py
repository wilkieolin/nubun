"""
analyze_radicals.py — Inspection & Visualization

Analyzes the discovered radical dictionary: semantic labeling, composition
examples, Pareto frontier, UMAP visualization, and coverage gap analysis.

Input:  data/concepts.npz, data/sweep_results.csv, data/radicals_K*.npz
Output: results/ directory with plots and text reports
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def load_concepts(path: str = "data/concepts.npz"):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["frequencies"], data["labels"]


def find_best_radicals(data_dir: str = "data") -> str:
    """Find the best radicals file in the data directory."""
    candidates = sorted(glob.glob(os.path.join(data_dir, "radicals_K*.npz")))
    if not candidates:
        raise FileNotFoundError(f"No radical files found in {data_dir}")
    return candidates[-1]  # take the last one (alphabetically)


def label_radicals(R: np.ndarray, concept_embs: np.ndarray,
                   concept_labels: list[str], top_k: int = 10,
                   ) -> list[tuple[int, list[str], list[float]]]:
    """For each radical, find the nearest concept words."""
    # R: (K, d), concept_embs: (N, d)
    sims = cosine_similarity(R, concept_embs)  # (K, N)
    results = []
    for i in range(len(R)):
        top_idx = np.argsort(-sims[i])[:top_k]
        top_words = [concept_labels[j] for j in top_idx]
        top_sims = [float(sims[i, j]) for j in top_idx]
        results.append((i, top_words, top_sims))
    return results


def test_compositions(R: np.ndarray, C: np.ndarray,
                      concept_embs: np.ndarray, concept_labels: list[str],
                      ) -> list[dict]:
    """Test whether natural compositions emerge from the radical structure."""
    test_cases = [
        ("fire", "mountain", "volcano"),
        ("water", "fall", "waterfall"),
        ("sun", "flower", "sunflower"),
        ("book", "house", "library"),
        ("fire", "fight", "firefighter"),
        ("rain", "forest", "rainforest"),
        ("sea", "food", "seafood"),
        ("air", "port", "airport"),
        ("foot", "ball", "football"),
        ("eye", "glass", "glasses"),
        ("moon", "light", "moonlight"),
        ("earth", "quake", "earthquake"),
        ("snow", "man", "snowman"),
        ("hand", "write", "handwriting"),
        ("star", "fish", "starfish"),
    ]

    label_to_idx = {label: i for i, label in enumerate(concept_labels)}
    results = []

    for a_word, b_word, expected in test_cases:
        a_idx = label_to_idx.get(a_word)
        b_idx = label_to_idx.get(b_word)
        exp_idx = label_to_idx.get(expected)

        if a_idx is None or b_idx is None:
            results.append({
                "a": a_word, "b": b_word, "expected": expected,
                "status": "missing_inputs",
                "predicted": None, "cosine_sim": None,
            })
            continue

        # Compose: add the radical representations
        composed_radicals = C[a_idx] + C[b_idx]
        composed_emb = composed_radicals @ R
        composed_emb = composed_emb / (np.linalg.norm(composed_emb) + 1e-8)

        # Find nearest concept
        sims = cosine_similarity(composed_emb.reshape(1, -1), concept_embs)[0]
        predicted_idx = np.argmax(sims)
        predicted_word = concept_labels[predicted_idx]

        # Also check similarity to expected
        exp_sim = float(sims[exp_idx]) if exp_idx is not None else None

        results.append({
            "a": a_word, "b": b_word, "expected": expected,
            "status": "ok",
            "predicted": predicted_word,
            "predicted_sim": float(sims[predicted_idx]),
            "expected_sim": exp_sim,
            "match": predicted_word == expected,
        })

    return results


def plot_pareto_frontier(csv_path: str, output_dir: str):
    """Plot K vs coverage vs sparsity Pareto frontier."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for method, color in [("nmf", "#2196F3"), ("kmeans_omp", "#4CAF50"), ("dictlearn", "#FF5722")]:
        mask = df["method"] == method
        sub = df[mask].sort_values("K")
        if sub.empty:
            continue

        axes[0].plot(sub["K"], sub["coverage_80"], "o-", color=color, label=method)
        axes[0].set_xlabel("K (number of radicals)")
        axes[0].set_ylabel("Coverage (cosine > 0.8)")
        axes[0].set_title("Coverage vs. Radical Count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(sub["K"], sub["mean_active_radicals"], "o-", color=color, label=method)
        axes[1].set_xlabel("K (number of radicals)")
        axes[1].set_ylabel("Mean active radicals per concept")
        axes[1].set_title("Sparsity vs. Radical Count")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(sub["K"], sub["discriminability"], "o-", color=color, label=method)
        axes[2].set_xlabel("K (number of radicals)")
        axes[2].set_ylabel("Discriminability")
        axes[2].set_title("Discriminability vs. Radical Count")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_frontier.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_umap(concept_embs: np.ndarray, concept_labels: list[str],
              C: np.ndarray, output_dir: str, n_samples: int = 2000):
    """UMAP visualization of concepts colored by dominant radical."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP plot")
        return

    # Subsample for speed
    n = min(n_samples, len(concept_labels))
    idx = np.random.RandomState(42).choice(len(concept_labels), n, replace=False)
    X_sub = concept_embs[idx]
    C_sub = C[idx]
    labels_sub = [concept_labels[i] for i in idx]

    # Dominant radical for each concept
    dominant = np.argmax(np.abs(C_sub), axis=1)

    print("  Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    proj = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=dominant, cmap="tab20",
                         s=8, alpha=0.6)
    ax.set_title(f"UMAP of {n} concepts (colored by dominant radical)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Annotate a few high-frequency concepts
    for i in range(min(50, n)):
        ax.annotate(labels_sub[i], (proj[i, 0], proj[i, 1]),
                    fontsize=5, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, "umap_concepts.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def coverage_gap_analysis(concept_embs: np.ndarray, concept_labels: list[str],
                          freqs: np.ndarray, C: np.ndarray, R: np.ndarray,
                          top_n: int = 50) -> list[dict]:
    """Find the worst-reconstructed important concepts."""
    X_recon = C @ R
    X_norm = concept_embs / (np.linalg.norm(concept_embs, axis=1, keepdims=True) + 1e-8)
    R_norm = X_recon / (np.linalg.norm(X_recon, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(X_norm * R_norm, axis=1)

    # Score = frequency * (1 - cosine_sim) — high = important AND poorly reconstructed
    gap_score = freqs * (1 - cos_sims)
    worst_idx = np.argsort(-gap_score)[:top_n]

    gaps = []
    for i in worst_idx:
        gaps.append({
            "concept": concept_labels[i],
            "frequency": float(freqs[i]),
            "cosine_sim": float(cos_sims[i]),
            "gap_score": float(gap_score[i]),
            "n_active_radicals": int((np.abs(C[i]) > 0.01 * np.abs(C[i]).max()).sum()),
        })
    return gaps


def main():
    parser = argparse.ArgumentParser(description="Analyze discovered radicals")
    parser.add_argument("--concepts", type=str, default="data/concepts.npz")
    parser.add_argument("--radicals", type=str, default=None,
                        help="Path to radicals .npz (auto-detected if omitted)")
    parser.add_argument("--sweep-csv", type=str, default="data/sweep_results.csv")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading concepts...")
    concept_embs, freqs, concept_labels = load_concepts(args.concepts)
    concept_labels = list(concept_labels)

    # Find radicals file
    radicals_path = args.radicals or find_best_radicals()
    print(f"Loading radicals from {radicals_path}...")
    rad_data = np.load(radicals_path, allow_pickle=True)
    R = rad_data["radicals"]
    C = rad_data["compositions"]
    K = R.shape[0]
    print(f"  K={K} radicals, {R.shape[1]}-dimensional")

    # 1. Radical semantics
    print("\n--- Radical Semantics ---")
    radical_info = label_radicals(R, concept_embs, concept_labels)
    sem_path = os.path.join(args.output_dir, "radical_semantics.txt")
    with open(sem_path, "w") as f:
        for idx, words, sims in radical_info:
            line = f"Radical {idx:3d}: {', '.join(f'{w} ({s:.2f})' for w, s in zip(words, sims))}"
            f.write(line + "\n")
            if idx < 20:
                print(f"  {line}")
        if K > 20:
            print(f"  ... ({K - 20} more radicals in {sem_path})")
    print(f"  Full list saved to {sem_path}")

    # 2. Composition examples
    print("\n--- Composition Examples ---")
    comp_results = test_compositions(R, C, concept_embs, concept_labels)
    comp_path = os.path.join(args.output_dir, "composition_examples.txt")
    with open(comp_path, "w") as f:
        for r in comp_results:
            if r["status"] == "missing_inputs":
                line = f"  {r['a']} + {r['b']} -> {r['expected']}  [SKIPPED: input not in vocabulary]"
            else:
                match_str = "MATCH" if r["match"] else "miss"
                line = (f"  {r['a']} + {r['b']} -> expected: {r['expected']}, "
                        f"predicted: {r['predicted']} (sim={r['predicted_sim']:.3f})  "
                        f"[{match_str}]")
                if r["expected_sim"] is not None:
                    line += f"  (expected sim={r['expected_sim']:.3f})"
            f.write(line + "\n")
            print(line)
    n_testable = sum(1 for r in comp_results if r["status"] == "ok")
    n_match = sum(1 for r in comp_results if r.get("match"))
    print(f"  Exact matches: {n_match}/{n_testable}")
    print(f"  Full results saved to {comp_path}")

    # 3. Pareto frontier plot
    print("\n--- Pareto Frontier ---")
    if os.path.exists(args.sweep_csv):
        plot_pareto_frontier(args.sweep_csv, args.output_dir)
    else:
        print(f"  Sweep CSV not found at {args.sweep_csv}, skipping")

    # 4. UMAP visualization
    print("\n--- UMAP Visualization ---")
    plot_umap(concept_embs, concept_labels, C, args.output_dir)

    # 5. Coverage gap analysis
    print("\n--- Coverage Gap Analysis ---")
    gaps = coverage_gap_analysis(concept_embs, concept_labels, freqs, C, R)
    gap_path = os.path.join(args.output_dir, "coverage_gaps.txt")
    with open(gap_path, "w") as f:
        f.write("Top poorly-reconstructed important concepts:\n")
        f.write(f"{'Concept':<20} {'Freq':>8} {'CosSim':>8} {'GapScore':>10} {'Radicals':>10}\n")
        f.write("-" * 60 + "\n")
        for g in gaps:
            line = (f"{g['concept']:<20} {g['frequency']:>8.2f} {g['cosine_sim']:>8.3f} "
                    f"{g['gap_score']:>10.3f} {g['n_active_radicals']:>10d}")
            f.write(line + "\n")
            if gaps.index(g) < 20:
                print(f"  {line}")
    print(f"  Full gap analysis saved to {gap_path}")

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"  Results saved to {args.output_dir}/")
    print(f"  Files: radical_semantics.txt, composition_examples.txt,")
    print(f"         pareto_frontier.png, umap_concepts.png, coverage_gaps.txt")


if __name__ == "__main__":
    main()
