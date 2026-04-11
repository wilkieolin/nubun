"""
discover_radicals.py — Radical Dictionary Learning

Sweeps over different numbers of radicals (K) using NMF and sparse dictionary
learning, computes quality metrics, and saves the best radical set.

Input:  data/concepts.npz
Output: data/sweep_results.csv, data/radicals_K{best}.npz
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.metrics.pairwise import cosine_similarity


def load_concepts(path: str = "data/concepts.npz"):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["frequencies"], data["labels"]


def make_nonnegative(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift embeddings to be non-negative for NMF."""
    shift = -X.min()
    return X + shift, shift


def frequency_weight(X: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Weight rows by sqrt(frequency) so common concepts matter more."""
    weights = np.sqrt(freqs / freqs.max())
    return X * weights[:, np.newaxis], weights


def compute_metrics(X_orig: np.ndarray, X_recon: np.ndarray,
                    C: np.ndarray, freqs: np.ndarray,
                    sparsity_threshold: float = 0.01,
                    ) -> dict:
    """Compute coverage, reconstruction error, sparsity, discriminability."""
    # Normalize both for cosine comparison
    X_norm = X_orig / (np.linalg.norm(X_orig, axis=1, keepdims=True) + 1e-8)
    R_norm = X_recon / (np.linalg.norm(X_recon, axis=1, keepdims=True) + 1e-8)

    cos_sims = np.sum(X_norm * R_norm, axis=1)

    # Frequency weights for weighted metrics
    weights = freqs / freqs.sum()

    # Coverage: % of concepts with cosine sim > 0.8
    coverage_80 = (cos_sims > 0.8).mean()
    coverage_90 = (cos_sims > 0.9).mean()

    # Weighted mean reconstruction error
    recon_error = np.mean(weights * (1 - cos_sims))

    # Sparsity: mean number of active radicals per concept
    active = np.abs(C) > sparsity_threshold * np.abs(C).max()
    mean_active = active.sum(axis=1).mean()
    median_active = np.median(active.sum(axis=1))

    # Discriminability: % of unique radical combinations
    # Quantize: zero out small weights, then hash the pattern
    C_quantized = active.astype(int)
    patterns = set()
    for row in C_quantized:
        patterns.add(tuple(row.nonzero()[0]))
    discriminability = len(patterns) / len(C_quantized)

    return {
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "weighted_recon_error": recon_error,
        "mean_cosine_sim": cos_sims.mean(),
        "mean_active_radicals": mean_active,
        "median_active_radicals": median_active,
        "discriminability": discriminability,
    }


def run_nmf(X: np.ndarray, freqs: np.ndarray, K: int,
            max_iter: int = 500) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run NMF decomposition."""
    X_nn, shift = make_nonnegative(X)
    X_weighted, weights = frequency_weight(X_nn, freqs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NMF(n_components=K, init="nndsvda", max_iter=max_iter,
                     random_state=42)
        C = model.fit_transform(X_weighted)
        R = model.components_

    # Unweight the reconstruction for evaluation
    X_recon_weighted = C @ R
    X_recon = X_recon_weighted / weights[:, np.newaxis] - shift

    metrics = compute_metrics(X, X_recon, C, freqs)
    return R, C, metrics


def run_dictionary_learning(X: np.ndarray, freqs: np.ndarray, K: int,
                            max_nonzero: int = 4,
                            max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run sparse dictionary learning with PCA pre-reduction."""
    from sklearn.decomposition import PCA

    # Reduce dimensionality with PCA first — dictionary learning struggles
    # on unit-normalized high-dimensional data. Keep enough variance (~95%).
    n_pca = min(X.shape[0] - 1, X.shape[1], max(K * 2, 100))
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)

    model = MiniBatchDictionaryLearning(
        n_components=K,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=max_nonzero,
        max_iter=max(max_iter, 500),
        random_state=42,
        batch_size=min(256, X_pca.shape[0]),
    )
    model.fit(X_pca)
    C = model.transform(X_pca)
    R_pca = model.components_

    # Project dictionary back to full embedding space for evaluation
    R = R_pca @ pca.components_[:n_pca]
    X_recon = C @ R

    metrics = compute_metrics(X, X_recon, C, freqs)
    return R, C, metrics


def run_kmeans_omp(X: np.ndarray, freqs: np.ndarray, K: int,
                   max_nonzero: int = 4,
                   **kwargs) -> tuple[np.ndarray, np.ndarray, dict]:
    """K-means centroids as dictionary + OMP sparse coding.

    Uses k-means to find K cluster centroids as radical directions, then
    encodes each concept as a sparse combination via Orthogonal Matching
    Pursuit. Works directly on signed embeddings — no shift needed.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import OrthogonalMatchingPursuit

    # Find K radical directions via k-means
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=256,
                             n_init=3)
    kmeans.fit(X)
    R = kmeans.cluster_centers_  # (K, d)

    # Normalize radical directions
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)

    # Sparse coding via OMP: for each concept, find best max_nonzero radicals
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max_nonzero)
    omp.fit(R.T, X[0])  # just to init, we'll use it per-sample

    C = np.zeros((X.shape[0], K))
    for i in range(X.shape[0]):
        omp_i = OrthogonalMatchingPursuit(n_nonzero_coefs=max_nonzero)
        omp_i.fit(R.T, X[i])
        C[i] = omp_i.coef_

    X_recon = C @ R
    metrics = compute_metrics(X, X_recon, C, freqs)
    return R, C, metrics


def main():
    parser = argparse.ArgumentParser(description="Discover radical dictionary")
    parser.add_argument("--input", type=str, default="data/concepts.npz")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--k-values", type=str, default="50,100,150,200,250,300,400,500",
                        help="Comma-separated K values to sweep")
    parser.add_argument("--max-nonzero", type=int, default=4,
                        help="Max active radicals per concept for dict learning")
    parser.add_argument("--methods", type=str, default="nmf,kmeans_omp,dictlearn",
                        help="Comma-separated methods to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k_values = [int(k) for k in args.k_values.split(",")]
    methods = args.methods.split(",")

    print(f"Loading concepts from {args.input}...")
    X, freqs, labels = load_concepts(args.input)
    print(f"  {X.shape[0]} concepts, {X.shape[1]}-dimensional embeddings")

    results = []
    best_score = -1
    best_result = None

    for K in k_values:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"K={K}, method={method}")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                if method == "nmf":
                    R, C, metrics = run_nmf(X, freqs, K)
                elif method == "kmeans_omp":
                    R, C, metrics = run_kmeans_omp(
                        X, freqs, K, max_nonzero=args.max_nonzero)
                elif method == "dictlearn":
                    R, C, metrics = run_dictionary_learning(
                        X, freqs, K, max_nonzero=args.max_nonzero)
                else:
                    print(f"  Unknown method: {method}, skipping")
                    continue
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

            elapsed = time.time() - t0

            row = {"K": K, "method": method, "time_s": elapsed, **metrics}
            results.append(row)

            print(f"  Time: {elapsed:.1f}s")
            print(f"  Coverage (>0.8): {metrics['coverage_80']:.1%}")
            print(f"  Coverage (>0.9): {metrics['coverage_90']:.1%}")
            print(f"  Mean cosine sim: {metrics['mean_cosine_sim']:.4f}")
            print(f"  Mean active radicals: {metrics['mean_active_radicals']:.1f}")
            print(f"  Discriminability: {metrics['discriminability']:.1%}")

            # Track best by combined score: coverage * discriminability
            score = metrics["coverage_80"] * metrics["discriminability"]
            if score > best_score:
                best_score = score
                best_result = (K, method, R, C, metrics)

    # Save sweep results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved to {csv_path}")
    print(df.to_string(index=False))

    # Save best radical set
    if best_result:
        K, method, R, C, metrics = best_result
        best_path = os.path.join(args.output_dir, f"radicals_K{K}_{method}.npz")
        np.savez(best_path, radicals=R, compositions=C, labels=labels,
                 K=K, method=method, **metrics)
        print(f"\nBest result: K={K}, method={method}")
        print(f"  Coverage: {metrics['coverage_80']:.1%}, "
              f"Discriminability: {metrics['discriminability']:.1%}")
        print(f"  Saved to {best_path}")


if __name__ == "__main__":
    main()
