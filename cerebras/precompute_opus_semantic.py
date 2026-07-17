"""Precompute opus-100 semantic targets for CS-3 Milestone 2 (offline, one-time).

For each packed shard data/opus100/<lang>_en.npz this reconstructs BOTH sides of
every pair (the X sentence and its English pair), mean-pools frozen MiniLM over
them (identical pooling to train_vqvae.embed_sentences), and writes
data/opus100/<lang>_en.sem.npz with:

    src_sem  (n_pairs, 384) float16   MiniLM(X sentence)   — aligned to pair order
    tgt_sem  (n_pairs, 384) float16   MiniLM(en sentence)

Both sides are embedded because the pipeline uses either side as "source"
(50/50 swap), and the semantic target is always MiniLM(source). cerebras/
opus_input.py then looks up src_sem[row] or tgt_sem[row] per sample with no
frozen model in the traced step.

Run on the GB10 (`nubun` env, has CUDA) or the x86 CS env:

    python cerebras/precompute_opus_semantic.py --opus-dir data/opus100
    python cerebras/precompute_opus_semantic.py --opus-dir data/opus100 --langs zh,es,fr

Deterministic (frozen model) — safe to resume; already-done shards are skipped
unless --overwrite.
"""
import argparse
import glob
import os

import numpy as np
import torch
from transformers import AutoModel

# Reuse the exact pooling used for the parallel-corpus targets.
from precompute_semantic_targets import SEM_MODEL, embed_token_ids

try:
    from tqdm import tqdm
except ImportError:                         # progress bar optional
    tqdm = None


def _rows_to_matrix(tokens, offsets, n_pairs, pad):
    """Slice a packed (concatenated) token array into a padded (n_pairs, Tmax)
    int64 rectangle using the per-sentence offsets."""
    tmax = int(np.max(offsets[1:n_pairs + 1] - offsets[:n_pairs]))
    mat = np.full((n_pairs, tmax), pad, dtype=np.int64)
    for i in range(n_pairs):
        seq = tokens[offsets[i]:offsets[i + 1]]
        mat[i, :len(seq)] = seq
    return mat


def process_shard(path, model, device, pad, batch_size, overwrite):
    out_path = path.replace("_en.npz", "_en.sem.npz")
    if os.path.exists(out_path) and not overwrite:
        print(f"skip (exists): {out_path}")
        return
    with np.load(path, allow_pickle=False) as d:
        n = int(d["n_pairs"])
        src_mat = _rows_to_matrix(np.asarray(d["src_tokens"]),
                                  np.asarray(d["src_offsets"]), n, pad)
        tgt_mat = _rows_to_matrix(np.asarray(d["tgt_tokens"]),
                                  np.asarray(d["tgt_offsets"]), n, pad)
    lang = os.path.basename(path).split("_en.npz")[0]
    print(f"{os.path.basename(path)}: {n} pairs, "
          f"src Tmax={src_mat.shape[1]} tgt Tmax={tgt_mat.shape[1]}")
    src_sem = embed_token_ids(model, src_mat, pad, device, batch_size,
                              desc=f"{lang} src")
    tgt_sem = embed_token_ids(model, tgt_mat, pad, device, batch_size,
                              desc=f"{lang} tgt")
    np.savez(out_path, src_sem=src_sem, tgt_sem=tgt_sem)
    print(f"  wrote {src_sem.shape} src + {tgt_sem.shape} tgt -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opus-dir", default="data/opus100")
    ap.add_argument("--langs", default=None,
                    help="Comma-separated subset (e.g. zh,es,fr); default all.")
    ap.add_argument("--pad-token-id", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.opus_dir, "*_en.npz")))
    shards = [s for s in shards if not s.endswith(".sem.npz")]
    if args.langs:
        keep = set(args.langs.split(","))
        shards = [s for s in shards
                  if os.path.basename(s).split("_en.npz")[0] in keep]
    if not shards:
        raise FileNotFoundError(f"No *_en.npz shards in {args.opus_dir}")

    print(f"loading {SEM_MODEL} on {args.device}...")
    model = AutoModel.from_pretrained(SEM_MODEL).eval().to(args.device)
    shard_iter = tqdm(shards, desc="shards", unit="shard") if tqdm is not None else shards
    for path in shard_iter:
        process_shard(path, model, args.device, args.pad_token_id,
                      args.batch_size, args.overwrite)
    print("done.")


if __name__ == "__main__":
    main()
