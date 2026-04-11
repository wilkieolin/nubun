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
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer
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

# Mapping from our language codes to NLTK stopword filenames
NLTK_STOPWORD_LANGS = {
    "en": "english", "es": "spanish", "fr": "french", "de": "german",
    "pt": "portuguese", "ru": "russian", "ar": "arabic", "zh": "chinese",
}

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _load_stopwords() -> dict[str, set[str]]:
    """Load multilingual stopword sets from NLTK."""
    from nltk.corpus import stopwords
    result = {}
    for lang_code, nltk_name in NLTK_STOPWORD_LANGS.items():
        try:
            result[lang_code] = set(stopwords.words(nltk_name))
        except Exception:
            result[lang_code] = set()

    # Manual stopwords for languages NLTK doesn't cover well
    # Japanese particles/function words
    result["ja"] = set("の に は を た が で て と し れ さ ある いる も する から な こと "
                       "として い や れる など なっ ない この ため その あっ よう また もの "
                       "という あり まで られ なる へ か だ これ によって により おり "
                       "より による ず なり られる において".split())
    # Hindi postpositions/function words
    result.setdefault("hi", set()).update(
        "का के की है में को से हैं पर ने कि और एक यह भी "
        "तो कर था हो इस पे नहीं हुए हुई कम जो ही".split()
    )
    return result


def _is_content_word(word: str, lang: str) -> bool:
    """Check if a word is likely a content word (not function/noise)."""
    # Filter pure numbers
    if re.match(r'^[\d.,]+$', word):
        return False
    # Filter very short words (except CJK which uses 1-2 char words)
    if lang not in ("zh", "ja") and len(word) < 2:
        return False
    # Filter words with special characters (URLs, codes, etc.)
    if re.search(r'[/@#$%^&*(){}|<>]', word):
        return False
    return True


def gather_words(words_per_lang: int = 8000,
                 filter_stopwords: bool = True,
                 ) -> tuple[list[str], list[str], list[float]]:
    """Collect top content words from each language with their frequencies.

    Filters out stopwords, numbers, and other non-content tokens.
    Returns (words, lang_codes, zipf_freqs).
    """
    stopwords = _load_stopwords() if filter_stopwords else {}

    words = []
    lang_codes = []
    zipf_freqs = []

    for lang in LANGUAGES:
        lang_stops = stopwords.get(lang, set())
        # Pull more words than needed to compensate for filtering
        pull_count = int(words_per_lang * 1.5) if filter_stopwords else words_per_lang
        print(f"  Gathering words for {LANGUAGES[lang]} ({lang})...")
        try:
            top_words = top_n_list(lang, pull_count)
        except Exception as e:
            print(f"    Warning: could not get words for {lang}: {e}")
            continue

        kept = 0
        for w in top_words:
            if kept >= words_per_lang:
                break
            if filter_stopwords:
                if w.lower() in lang_stops or w in lang_stops:
                    continue
                if not _is_content_word(w, lang):
                    continue
            freq = zipf_frequency(w, lang)
            if freq > 0:
                words.append(w)
                lang_codes.append(lang)
                zipf_freqs.append(freq)
                kept += 1

        print(f"    Kept {kept} content words (filtered {pull_count - kept})")

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


def _pick_label(words: np.ndarray, langs: np.ndarray,
                freqs: np.ndarray) -> str:
    """Pick the best human-readable label for a concept cluster.

    Prefers single English words; breaks ties by frequency then shorter length.
    """
    en_mask = langs == "en"
    if en_mask.any():
        candidates = words[en_mask]
        cand_freqs = freqs[en_mask]
        # Score: frequency bonus - length penalty (prefer shorter common words)
        scores = cand_freqs - 0.1 * np.array([len(w) for w in candidates])
        return candidates[scores.argmax()]
    # Fallback: highest frequency word in any language
    return words[freqs.argmax()]


def cluster_concepts(embeddings: np.ndarray, words: list[str],
                     lang_codes: list[str], zipf_freqs: list[float],
                     min_cluster_size: int = 3,
                     ) -> tuple[np.ndarray, np.ndarray, list[str], list[list[str]]]:
    """Cluster word embeddings to identify unique concepts using HDBSCAN.

    HDBSCAN handles varying density and doesn't force every point into a
    cluster — outliers become singleton concepts, preserving atomic words.

    Returns (concept_embeddings, concept_freqs, concept_labels, cluster_members).
    """
    import hdbscan

    n = len(words)
    print(f"  Running HDBSCAN on {n} words (min_cluster_size={min_cluster_size})...")
    t0 = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",  # on normalized vectors, euclidean ~ cosine
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - t0

    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters + {n_noise} noise points in {elapsed:.1f}s")

    # Build concept representations
    concept_embeddings = []
    concept_freqs = []
    concept_labels = []
    cluster_members_list = []

    words_arr = np.array(words)
    langs_arr = np.array(lang_codes)
    freqs_arr = np.array(zipf_freqs)

    # Process proper clusters
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_embs = embeddings[mask]
        cluster_words = words_arr[mask]
        cluster_langs = langs_arr[mask]
        cluster_freqs = freqs_arr[mask]

        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        concept_embeddings.append(centroid)
        concept_freqs.append(float(cluster_freqs.sum()))
        concept_labels.append(_pick_label(cluster_words, cluster_langs, cluster_freqs))
        cluster_members_list.append(
            [f"{w}({l})" for w, l in zip(cluster_words, cluster_langs)]
        )

    # Process noise points as singleton concepts
    noise_mask = labels == -1
    noise_indices = np.where(noise_mask)[0]
    for idx in noise_indices:
        concept_embeddings.append(embeddings[idx])
        concept_freqs.append(float(zipf_freqs[idx]))
        concept_labels.append(words[idx])
        cluster_members_list.append([f"{words[idx]}({lang_codes[idx]})"])

    concept_embeddings = np.array(concept_embeddings)
    concept_freqs = np.array(concept_freqs)

    print(f"  Total concepts: {len(concept_labels)} "
          f"({n_clusters} clustered + {n_noise} singletons)")
    return concept_embeddings, concept_freqs, concept_labels, cluster_members_list


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
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="HDBSCAN min cluster size (default: 3)")
    parser.add_argument("--no-filter-stopwords", action="store_true",
                        help="Disable stopword filtering")
    parser.add_argument("--output", type=str, default="data/concepts.npz",
                        help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Step 1/4: Gathering words from multiple languages")
    words, lang_codes, zipf_freqs = gather_words(
        args.words_per_lang, filter_stopwords=not args.no_filter_stopwords)

    print("\nStep 2/4: Embedding words in shared multilingual space")
    embeddings = embed_words(words)

    print("\nStep 3/4: Clustering to identify unique concepts")
    concept_embs, concept_freqs, concept_labels, cluster_members = cluster_concepts(
        embeddings, words, lang_codes, zipf_freqs,
        min_cluster_size=args.min_cluster_size,
    )

    print("\nStep 4/4: Selecting top concepts by frequency")
    concept_embs, concept_freqs, concept_labels = select_top_n(
        concept_embs, concept_freqs, concept_labels, n=args.n_concepts,
    )

    # Verify key atomic concepts survived
    key_words = ["fire", "water", "mountain", "sun", "moon", "star", "sea",
                 "air", "earth", "tree", "rain", "snow", "hand", "eye"]
    present = [w for w in key_words if w in concept_labels]
    missing = [w for w in key_words if w not in concept_labels]
    print(f"\n  Key concepts present: {present}")
    if missing:
        print(f"  Key concepts MISSING: {missing}")

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
