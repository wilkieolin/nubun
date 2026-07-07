"""build_opus100_corpus.py — Pre-tokenize Helsinki-NLP/opus-100 for training.

Downloads 9 English-pivoted parquet configs (one per non-English target language),
tokenizes both sides with the MiniLM tokenizer, drops too-long pairs, and saves
one .npz shard per config under data/opus100/.

Output layout (memmap-friendly: packed int32 + offsets):
  data/opus100/{lang}_en.npz  containing:
    src_tokens    : (sum_src_lens,) int32 — concatenated token IDs
    src_offsets   : (N+1,)         int32 — start of pair i (prefix sums); pair i is
                                            src_tokens[offsets[i]:offsets[i+1]]
    tgt_tokens    : (sum_tgt_lens,) int32
    tgt_offsets   : (N+1,)         int32
    n_pairs       : int
    lang_code     : str — short code ('zh', 'es', ...)
"""

import argparse
import os
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# All 9 non-English short codes (same as Phase 1/3)
ALL_LANGS = ["ar", "de", "es", "fr", "hi", "ja", "pt", "ru", "zh"]


def build_one_shard(lang: str, tokenizer, max_len: int, output_dir: str,
                    progress_every: int = 50000) -> tuple[int, int]:
    """Tokenize one en-X opus-100 config. Returns (n_kept, n_dropped)."""
    out_path = os.path.join(output_dir, f"{lang}_en.npz")
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=False)
        if "src_offsets" in d:
            n = int(d["n_pairs"])
            print(f"  [{lang}] new-format shard exists at {out_path} ({n} pairs); skipping")
            return n, 0
        else:
            print(f"  [{lang}] OLD-format shard at {out_path}; will rebuild")

    config = f"{lang}-en"
    print(f"  [{lang}] loading {config}...")
    t0 = time.time()
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", config, split="train")
    except Exception as e:
        # Try reversed config (some pairs are en-X instead of X-en)
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", f"en-{lang}", split="train")
            print(f"  [{lang}] note: dataset config is en-{lang} not {lang}-en")
        except Exception as e2:
            print(f"  [{lang}] FAILED to load either {config} or en-{lang}: {e2}")
            return 0, 0

    n_total = len(ds)
    print(f"  [{lang}] loaded {n_total} pairs in {time.time() - t0:.1f}s")

    # Inspect first row to determine src/tgt key layout
    sample = ds[0]
    if "translation" in sample:
        # Standard opus-100 schema: {"translation": {"en": "...", "X": "..."}}
        keys = list(sample["translation"].keys())
        assert "en" in keys, f"expected 'en' key in {keys}"
        # The non-en key is the source language
        src_key_lang = [k for k in keys if k != "en"][0]
        def get_src(row): return row["translation"][src_key_lang]
        def get_tgt(row): return row["translation"]["en"]
    else:
        # Flat schema: src/tgt or lang codes as direct keys
        if lang in sample and "en" in sample:
            def get_src(row): return row[lang]
            def get_tgt(row): return row["en"]
        else:
            print(f"  [{lang}] unknown schema: keys={list(sample.keys())}")
            return 0, 0

    print(f"  [{lang}] tokenizing...")
    t0 = time.time()
    src_ids_list = []
    tgt_ids_list = []
    src_lens = []
    tgt_lens = []
    n_dropped = 0

    for i in range(n_total):
        row = ds[i]
        try:
            src_text = get_src(row)
            tgt_text = get_tgt(row)
        except (KeyError, TypeError):
            n_dropped += 1
            continue
        if not src_text or not tgt_text:
            n_dropped += 1
            continue

        src_ids = tokenizer.encode(src_text, add_special_tokens=True,
                                   truncation=True, max_length=max_len + 1)
        tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=True,
                                   truncation=True, max_length=max_len + 1)
        # Drop pairs that hit the truncation cap (approximation for "too long")
        if len(src_ids) > max_len or len(tgt_ids) > max_len:
            n_dropped += 1
            continue

        src_ids_list.append(np.array(src_ids, dtype=np.int32))
        tgt_ids_list.append(np.array(tgt_ids, dtype=np.int32))
        src_lens.append(len(src_ids))
        tgt_lens.append(len(tgt_ids))

        if (i + 1) % progress_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{lang}]   {i+1}/{n_total}  ({rate:.0f} sent/s, "
                  f"kept {len(src_ids_list)}, dropped {n_dropped})")

    n_kept = len(src_ids_list)
    elapsed = time.time() - t0
    print(f"  [{lang}] tokenized {n_kept} kept / {n_dropped} dropped in {elapsed:.0f}s "
          f"({n_kept / elapsed:.0f} sent/s kept)")

    # Pack into concatenated int32 arrays + offsets (memmap-friendly)
    src_lens_arr = np.array(src_lens, dtype=np.int32)
    tgt_lens_arr = np.array(tgt_lens, dtype=np.int32)
    src_offsets = np.zeros(n_kept + 1, dtype=np.int32)
    tgt_offsets = np.zeros(n_kept + 1, dtype=np.int32)
    np.cumsum(src_lens_arr, out=src_offsets[1:])
    np.cumsum(tgt_lens_arr, out=tgt_offsets[1:])

    src_tokens = np.empty(int(src_offsets[-1]), dtype=np.int32)
    tgt_tokens = np.empty(int(tgt_offsets[-1]), dtype=np.int32)
    for i in range(n_kept):
        src_tokens[src_offsets[i]:src_offsets[i+1]] = src_ids_list[i]
        tgt_tokens[tgt_offsets[i]:tgt_offsets[i+1]] = tgt_ids_list[i]

    np.savez(out_path,
             src_tokens=src_tokens, src_offsets=src_offsets,
             tgt_tokens=tgt_tokens, tgt_offsets=tgt_offsets,
             n_pairs=np.array(n_kept, dtype=np.int32),
             lang_code=np.array(lang))
    sz = os.path.getsize(out_path) / 1e6
    print(f"  [{lang}] saved to {out_path} ({sz:.0f} MB) "
          f"[{src_tokens.size + tgt_tokens.size} tokens]")
    return n_kept, n_dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default=",".join(ALL_LANGS),
                        help="Comma-separated short codes (subset of " + ",".join(ALL_LANGS) + ")")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--output-dir", default="data/opus100")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    print(f"Languages to build: {langs}")
    print(f"Output: {args.output_dir}/")

    print(f"\nLoading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  vocab_size={tokenizer.vocab_size}, max_len={args.max_len}")

    grand_t0 = time.time()
    summary = []
    for lang in langs:
        print(f"\n=== {lang} ===")
        n_kept, n_dropped = build_one_shard(
            lang, tokenizer, args.max_len, args.output_dir)
        summary.append((lang, n_kept, n_dropped))

    elapsed = (time.time() - grand_t0) / 60
    total_kept = sum(s[1] for s in summary)
    total_dropped = sum(s[2] for s in summary)

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f} min. Summary:")
    print(f"{'lang':>6s} {'kept':>10s} {'dropped':>10s}")
    for lang, k, d in summary:
        print(f"{lang:>6s} {k:>10d} {d:>10d}")
    print(f"{'TOTAL':>6s} {total_kept:>10d} {total_dropped:>10d}")


if __name__ == "__main__":
    main()
