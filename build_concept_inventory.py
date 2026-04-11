"""
build_concept_inventory.py — Multilingual Concept Extraction

Gathers top words from 10 languages via wordfreq, embeds them in a shared
multilingual space using sentence-transformers, clusters translations of the
same concept, and selects the top N concepts by aggregate frequency.

Output: data/concepts.npz
  - embeddings: (N, 384) concept embedding matrix
  - frequencies: (N,) aggregate Zipf frequency vector
  - labels: (N,) English label for each concept
"""

import argparse
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from wordfreq import top_n_list, zipf_frequency


LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "hi": "Hindi",
    "ar": "Arabic",
    "fr": "French",
    "ru": "Russian",
    "ja": "Japanese",
    "pt": "Portuguese",
    "de": "German",
}

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def gather_words(words_per_lang: int = 8000) -> tuple[list[str], list[str], list[float]]:
    """Collect top words from each language with their frequencies.

    Returns (words, lang_codes, zipf_freqs).
    """
    words = []
    lang_codes = []
    zipf_freqs = []

    for lang in LANGUAGES:
        print(f"  Gathering top {words_per_lang} words for {LANGUAGES[lang]} ({lang})...")
        try:
            top_words = top_n_list(lang, words_per_lang)
        except Exception as e:
            print(f"    Warning: could not get words for {lang}: {e}")
            continue

        for w in top_words:
            freq = zipf_frequency(w, lang)
            if freq > 0:
                words.append(w)
                lang_codes.append(lang)
                zipf_freqs.append(freq)

    print(f"  Total words collected: {len(words)}")
    return words, lang_codes, zipf_freqs


def embed_words(words: list[str], model_name: str = MODEL_NAME,
                batch_size: int = 512) -> np.ndarray:
    """Embed all words using a multilingual sentence-transformer."""
    print(f"  Loading model {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"  Encoding {len(words)} words (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(words, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    elapsed = time.time() - t0
    print(f"  Encoding done in {elapsed:.1f}s — shape: {embeddings.shape}")
    return embeddings


def cluster_concepts(embeddings: np.ndarray, words: list[str],
                     lang_codes: list[str], zipf_freqs: list[float],
                     distance_threshold: float = 0.4,
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cluster word embeddings to identify unique concepts.

    Translations of the same concept land near each other in the multilingual
    space. We cluster them and pick a representative embedding, aggregate
    frequency, and English label for each cluster.

    Returns (concept_embeddings, concept_freqs, concept_labels).
    """
    print(f"  Computing cosine distance matrix for {len(words)} words...")
    t0 = time.time()

    # For large word counts, cluster in batches or use a connectivity constraint.
    # AgglomerativeClustering with precomputed distances works up to ~50-80K words
    # on a machine with 32GB+ RAM. For safety, subsample if needed.
    n = len(words)
    if n > 60000:
        print(f"  Subsampling from {n} to 60000 for clustering feasibility...")
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 60000, replace=False)
        embeddings = embeddings[idx]
        words = [words[i] for i in idx]
        lang_codes = [lang_codes[i] for i in idx]
        zipf_freqs = [zipf_freqs[i] for i in idx]
        n = 60000

    dist_matrix = cosine_distances(embeddings)
    elapsed = time.time() - t0
    print(f"  Distance matrix computed in {elapsed:.1f}s")

    print(f"  Running agglomerative clustering (threshold={distance_threshold})...")
    t0 = time.time()
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)
    n_clusters = labels.max() + 1
    elapsed = time.time() - t0
    print(f"  Found {n_clusters} concept clusters in {elapsed:.1f}s")

    # Build concept representations
    concept_embeddings = []
    concept_freqs = []
    concept_labels = []

    words_arr = np.array(words)
    langs_arr = np.array(lang_codes)
    freqs_arr = np.array(zipf_freqs)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_embs = embeddings[mask]
        cluster_words = words_arr[mask]
        cluster_langs = langs_arr[mask]
        cluster_freqs = freqs_arr[mask]

        # Centroid embedding (re-normalize)
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        concept_embeddings.append(centroid)

        # Aggregate frequency: sum of Zipf frequencies across languages
        concept_freqs.append(float(cluster_freqs.sum()))

        # Label: prefer English word if available, else most frequent word
        en_mask = cluster_langs == "en"
        if en_mask.any():
            en_words = cluster_words[en_mask]
            en_freqs = cluster_freqs[en_mask]
            label = en_words[en_freqs.argmax()]
        else:
            label = cluster_words[cluster_freqs.argmax()]
        concept_labels.append(label)

    concept_embeddings = np.array(concept_embeddings)
    concept_freqs = np.array(concept_freqs)

    print(f"  Concept embeddings shape: {concept_embeddings.shape}")
    return concept_embeddings, concept_freqs, concept_labels


def select_top_n(embeddings: np.ndarray, freqs: np.ndarray,
                 labels: list[str], n: int = 5000,
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Select the top N concepts by aggregate frequency."""
    if len(labels) <= n:
        print(f"  Only {len(labels)} concepts found, keeping all.")
        order = np.argsort(-freqs)
        return embeddings[order], freqs[order], [labels[i] for i in order]

    top_idx = np.argsort(-freqs)[:n]
    print(f"  Selected top {n} concepts (freq range: {freqs[top_idx[-1]]:.2f} – {freqs[top_idx[0]]:.2f})")
    return embeddings[top_idx], freqs[top_idx], [labels[i] for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Build multilingual concept inventory")
    parser.add_argument("--words-per-lang", type=int, default=8000,
                        help="Top N words to pull per language (default: 8000)")
    parser.add_argument("--n-concepts", type=int, default=5000,
                        help="Number of concepts to select (default: 5000)")
    parser.add_argument("--distance-threshold", type=float, default=0.4,
                        help="Cosine distance threshold for clustering (default: 0.4)")
    parser.add_argument("--output", type=str, default="data/concepts.npz",
                        help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Step 1/4: Gathering words from multiple languages")
    words, lang_codes, zipf_freqs = gather_words(args.words_per_lang)

    print("\nStep 2/4: Embedding words in shared multilingual space")
    embeddings = embed_words(words)

    print("\nStep 3/4: Clustering to identify unique concepts")
    concept_embs, concept_freqs, concept_labels = cluster_concepts(
        embeddings, words, lang_codes, zipf_freqs,
        distance_threshold=args.distance_threshold,
    )

    print("\nStep 4/4: Selecting top concepts by frequency")
    concept_embs, concept_freqs, concept_labels = select_top_n(
        concept_embs, concept_freqs, concept_labels, n=args.n_concepts,
    )

    # Save
    np.savez(
        args.output,
        embeddings=concept_embs,
        frequencies=concept_freqs,
        labels=np.array(concept_labels, dtype=object),
    )
    print(f"\nSaved {len(concept_labels)} concepts to {args.output}")
    print(f"  Embedding shape: {concept_embs.shape}")
    print(f"  Top 20 concepts: {concept_labels[:20]}")


if __name__ == "__main__":
    main()
