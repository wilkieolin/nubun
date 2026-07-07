"""
build_parallel_corpus.py — Multilingual parallel sentence corpus

Loads FLORES-200 (via mteb/flores on HF Hub), restricts to the 10 Phase-1
languages, tokenizes every sentence with the MiniLM tokenizer, and saves a
compact .npz containing parallel token-id arrays for both splits.

Output: data/parallel_corpus.npz
  lang_codes          : (10,)   FLORES language codes
  short_codes         : (10,)   Phase-1 codes (en, zh, ...)
  tokenizer_name      : str
  dev_token_ids       : (997, 10) object — each cell is np.int32 token IDs
  devtest_token_ids   : (1012, 10) object — same
  dev_lengths         : (997, 10) int32
  devtest_lengths     : (1012, 10) int32
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Phase-1 short code → FLORES-200 language_script code
LANG_MAP = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "pt": "por_Latn",
    "de": "deu_Latn",
}


def tokenize_split(ds, tokenizer, flores_codes: list[str], max_len: int = 128):
    n = len(ds)
    L = len(flores_codes)
    token_ids = np.empty((n, L), dtype=object)
    lengths = np.zeros((n, L), dtype=np.int32)
    truncated = 0
    for i in range(n):
        row = ds[i]
        for j, fc in enumerate(flores_codes):
            ids = tokenizer.encode(row[fc], add_special_tokens=True,
                                   truncation=True, max_length=max_len)
            token_ids[i, j] = np.array(ids, dtype=np.int32)
            lengths[i, j] = len(ids)
            if len(ids) >= max_len:
                truncated += 1
    return token_ids, lengths, truncated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/parallel_corpus.npz")
    parser.add_argument("--max-len", type=int, default=128,
                        help="Max tokens per sentence (truncated)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    short_codes = list(LANG_MAP.keys())
    flores_codes = [LANG_MAP[c] for c in short_codes]

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  vocab size: {tokenizer.vocab_size}")
    print(f"  pad: {tokenizer.pad_token_id}, bos: {tokenizer.cls_token_id}, "
          f"eos: {tokenizer.sep_token_id}")

    print("\nLoading FLORES-200 (mteb/flores)...")
    dev = load_dataset("mteb/flores", "default", split="dev")
    devtest = load_dataset("mteb/flores", "default", split="devtest")
    print(f"  dev: {len(dev)} sentences; devtest: {len(devtest)} sentences")

    # Verify all our languages are present
    missing = [c for c in flores_codes if c not in dev.features]
    if missing:
        raise RuntimeError(f"FLORES missing language codes: {missing}")

    print("\nTokenizing dev split...")
    dev_ids, dev_lens, dev_trunc = tokenize_split(
        dev, tokenizer, flores_codes, max_len=args.max_len)
    print(f"  done. truncated: {dev_trunc}/{dev_ids.size} cells")
    print(f"  length stats: mean={dev_lens.mean():.1f}, "
          f"median={np.median(dev_lens):.0f}, max={dev_lens.max()}")

    print("\nTokenizing devtest split...")
    devtest_ids, devtest_lens, devtest_trunc = tokenize_split(
        devtest, tokenizer, flores_codes, max_len=args.max_len)
    print(f"  done. truncated: {devtest_trunc}/{devtest_ids.size} cells")
    print(f"  length stats: mean={devtest_lens.mean():.1f}, "
          f"median={np.median(devtest_lens):.0f}, max={devtest_lens.max()}")

    # Save
    print(f"\nSaving to {args.output}...")
    np.savez(
        args.output,
        lang_codes=np.array(flores_codes),
        short_codes=np.array(short_codes),
        tokenizer_name=np.array(MODEL_NAME),
        pad_token_id=np.array(tokenizer.pad_token_id),
        bos_token_id=np.array(tokenizer.cls_token_id),
        eos_token_id=np.array(tokenizer.sep_token_id),
        vocab_size=np.array(tokenizer.vocab_size),
        max_len=np.array(args.max_len),
        dev_token_ids=dev_ids,
        devtest_token_ids=devtest_ids,
        dev_lengths=dev_lens,
        devtest_lengths=devtest_lens,
    )
    sz_mb = os.path.getsize(args.output) / 1e6
    print(f"  saved ({sz_mb:.1f} MB)")

    # Smoke test
    print("\nSmoke test: round-trip a parallel pair from disk")
    d = np.load(args.output, allow_pickle=True)
    sid = 0
    for j, sc in enumerate(d["short_codes"]):
        ids = d["dev_token_ids"][sid, j]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  {sc} ({d['lang_codes'][j]}): [{len(ids)} toks] {text[:80]}")


if __name__ == "__main__":
    main()
