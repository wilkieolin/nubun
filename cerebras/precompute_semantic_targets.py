"""Milestone 2 — precompute the frozen-MiniLM semantic targets OFFLINE.

The GB10 training runs MiniLM on every batch to make the semantic-loss target
(train_vqvae.embed_sentences). On CS-3 you do NOT want a second frozen model in
the traced step — precompute the targets once, save them, and have the cstorch
dataloader yield each sentence's target alongside its tokens.

The targets are deterministic (frozen model), so this is a one-time job. Run it
either here (x86 CS env) or on the GB10 in the `nubun` env — the .npy is portable.

    python cerebras/precompute_semantic_targets.py \
        --token-ids data/parallel_corpus.npz \
        --out data/sem_targets_parallel.npy

For opus-scale training targets, point --token-ids at a packed shard's SOURCE
side (see the "opus shards" note at the bottom) and run once per shard.

The pooling here is byte-for-byte the same as train_vqvae.embed_sentences:
mean over non-pad tokens of MiniLM last_hidden_state. Keep it identical or the
loss target shifts.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModel

SEM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@torch.no_grad()
def embed_token_ids(model, ids, pad_token_id, device, batch_size=256, fp16=True):
    """ids: (N, T) int64 already tokenized with the MiniLM/XLM-R tokenizer.
    Returns (N, D) float array. Mean-pool over non-pad tokens (== embed_sentences)."""
    out = np.empty((ids.shape[0], model.config.hidden_size),
                   dtype=np.float16 if fp16 else np.float32)
    for i in range(0, ids.shape[0], batch_size):
        chunk = torch.as_tensor(ids[i:i + batch_size], dtype=torch.long, device=device)
        attn = (chunk != pad_token_id).long()
        h = model(input_ids=chunk, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        out[i:i + batch_size] = pooled.cpu().numpy()
    return out


def load_token_ids(path, split_key):
    """Load a (N, T) int array of token ids. .npz with the given key, or .npy."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        arr = d[split_key]
        # parallel_corpus.npz stores (N, n_langs) object arrays of ragged lists;
        # flatten langs into rows and pad to a rectangle.
        if arr.dtype == object:
            rows = [seq for row in arr for seq in row]
            T = max(len(s) for s in rows)
            pad = int(d["pad_token_id"])
            mat = np.full((len(rows), T), pad, np.int64)
            for r, s in enumerate(rows):
                mat[r, :len(s)] = s
            return mat, pad
        return arr, int(d.get("pad_token_id", 1))
    return np.load(path), 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-ids", required=True)
    ap.add_argument("--split-key", default="dev_token_ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ids, pad = load_token_ids(args.token_ids, args.split_key)
    print(f"loaded {ids.shape} token-id matrix (pad={pad})")
    model = AutoModel.from_pretrained(SEM_MODEL).eval().to(args.device)
    emb = embed_token_ids(model, ids, pad, args.device, args.batch_size)
    np.save(args.out, emb)
    print(f"wrote {emb.shape} {emb.dtype} -> {args.out}")


if __name__ == "__main__":
    main()

# --- opus shards note --------------------------------------------------------
# data/opus100/<lang>_en.npz packs src_tokens/src_offsets (+tgt_*) as flat int32
# with per-sentence offsets. To make training targets, reconstruct each source
# sentence (offsets[i]:offsets[i+1]), pad to a rectangle in chunks, run
# embed_token_ids, and save aligned to the shard's pair order. Because the data
# pipeline can use either side as "source" (src_langs/tgt_langs = all), embed
# BOTH src and tgt sides if you train bidirectionally. The cstorch dataloader
# then yields sem_target[i] next to (src_ids[i], tgt_ids[i]).
