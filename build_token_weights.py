"""build_token_weights.py — per-token reconstruction weights (Phase 5b).

Plain token-level cross-entropy is dominated by Zipfian high-frequency tokens
(punctuation, articles, copulas, subword glue), so the cheapest way to lower it
is to encode a language-independent *boilerplate skeleton* — which is exactly
why Phase 4 codes captured `==`/`...`/diacritics instead of content words.

This script builds a weight vector w[vocab] applied per target token in the
reconstruction loss so that content tokens dominate the gradient:
  1. Count token frequencies over the opus100 shards (src + tgt sides).
  2. word2vec-style subsampling weight: w(t) = clip(sqrt(thresh / p(t)), 0, 1).
     Rare (content) tokens -> ~1; frequent (function/glue) tokens -> small.
  3. Hard-zero pure punctuation / symbol / whitespace tokens (via tokenizer +
     unicodedata) so they never steer the encoder+codebook.
  4. Keep special tokens (bos/eos/unk/mask/cls/sep) at weight 1.0 so the decoder
     still learns sentence structure and when to stop.

Output: data/token_weights.pt  — (vocab_size,) float32.
"""

import argparse
import glob
import os
import unicodedata

import numpy as np
import torch


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def count_token_frequencies(opus_dir: str, vocab_size: int) -> np.ndarray:
    """Sum token counts across all shards (both src and tgt token streams)."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    shards = sorted(glob.glob(os.path.join(opus_dir, "*_en.npz")))
    if not shards:
        raise FileNotFoundError(f"No *_en.npz shards in {opus_dir}")
    for path in shards:
        with np.load(path, mmap_mode="r", allow_pickle=False) as d:
            for key in ("src_tokens", "tgt_tokens"):
                toks = np.asarray(d[key]).astype(np.int64, copy=False)
                counts += np.bincount(toks, minlength=vocab_size)[:vocab_size]
        print(f"  counted {os.path.basename(path)}  (running total {counts.sum():,} tokens)")
    return counts


def is_punct_like(piece: str) -> bool:
    """True if the token piece is entirely punctuation / symbol / whitespace.

    SentencePiece marks a leading space with '▁'; strip it before judging so
    '▁.' counts as punctuation but '▁the' does not.
    """
    s = piece.replace("▁", "")  # '▁'
    if s == "":
        return True  # bare space marker
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] not in ("P", "S", "Z", "C"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opus-dir", default="data/opus100")
    parser.add_argument("--output", default="data/token_weights.pt")
    parser.add_argument("--thresh", type=float, default=1e-4,
                        help="Subsampling threshold; smaller = downweight frequent "
                             "tokens more aggressively (word2vec default 1e-4).")
    parser.add_argument("--no-zero-punct", action="store_true",
                        help="Skip hard-zeroing punctuation/symbol tokens")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = len(tok)
    print(f"  vocab_size={vocab_size}")

    print(f"\nCounting token frequencies over {args.opus_dir}...")
    counts = count_token_frequencies(args.opus_dir, vocab_size)
    total = counts.sum()
    p = counts.astype(np.float64) / max(1, total)

    # word2vec subsampling weight: rare -> ~1, frequent -> small.
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt(args.thresh / np.where(p > 0, p, np.inf))
    w = np.clip(w, 0.0, 1.0).astype(np.float32)

    n_zeroed = 0
    if not args.no_zero_punct:
        print("\nZeroing punctuation/symbol/whitespace tokens...")
        pieces = tok.convert_ids_to_tokens(list(range(vocab_size)))
        for i, piece in enumerate(pieces):
            if piece is None:
                continue
            if is_punct_like(piece):
                w[i] = 0.0
                n_zeroed += 1
        print(f"  zeroed {n_zeroed} punctuation/symbol tokens")

    # Keep special tokens at full weight (structure + stop signal).
    special = [t for t in tok.all_special_ids if t is not None and 0 <= t < vocab_size]
    for t in special:
        w[t] = 1.0
    print(f"  restored {len(special)} special tokens to weight 1.0: {special}")

    # Diagnostics: show what the most-frequent tokens got weighted to.
    top = np.argsort(counts)[::-1][:20]
    print("\nTop-20 most frequent tokens -> weight:")
    for i in top:
        piece = tok.convert_ids_to_tokens(int(i))
        print(f"  {int(i):>7d}  {repr(piece):<14}  count={int(counts[i]):>12,}  w={w[i]:.4f}")

    nonzero = (w > 0).sum()
    print(f"\nweight stats: nonzero={nonzero}/{vocab_size}  "
          f"mean={w.mean():.4f}  "
          f"mean_over_seen={w[counts > 0].mean():.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(torch.from_numpy(w), args.output)
    print(f"\nsaved {args.output}  ({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
