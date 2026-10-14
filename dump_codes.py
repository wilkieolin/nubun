"""dump_codes.py — PHASE9 A5: encode a corpus sample to RVQ code stacks.

Samples opus100 pairs from every shard, encodes the sentences with a
checkpoint's encoder (greedy RVQ, as in code_usage_audit.py), and writes:

  <out>/codes.npy    (N, M, L) uint8   code stack per sentence
  <out>/lang.npy     (N,)      uint8   index into CorpusMeta.short_codes
  <out>/split.npy    (N,)      uint8   0 = train, 1 = test (opus100 held out)
                                       2 = FLORES (out-of-domain test)
  <out>/tokens.npy, offsets.npy        packed XLM-R ids of the same sentences
                                       (for the text-LM baseline)
  <out>/meta.json

Sampling: per shard, --pairs-per-shard pairs; the X side is always kept, the
English side with probability --en-frac and de-duplicated across shards
(opus100's English sides repeat). The train/test split is by PAIR, so a
sentence and its translation never fall on different sides: the codes are
language-independent, and a translation in train would leak the test meaning.
FLORES (dev + devtest, all 10 languages) is appended as split 2.

Usage:
    python dump_codes.py --checkpoint ~/data/nubun_cs_step500000.eval.pt \
        --out data/codes_cs3_500k
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import torch

import vqvae_select
vqvae_select.select_from_argv()  # pre-port checkpoints need pre-port vqvae/
from vqvae.data import load_corpus
from code_usage_audit import build_model, encode_all_levels


def shard_rows(path):
    d = np.load(path, allow_pickle=False)
    return (str(d["lang_code"]), d["src_tokens"], d["src_offsets"],
            d["tgt_tokens"], d["tgt_offsets"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opus-dir", default="data/opus100")
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--pairs-per-shard", type=int, default=200000)
    p.add_argument("--en-frac", type=float, default=0.25)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(a.seed)

    ck = vqvae_select.load_checkpoint(a.checkpoint)
    dev_ids, devtest_ids, _, _, meta = load_corpus(a.corpus)
    codes_of = [str(c) for c in meta.short_codes]
    en = codes_of.index("en")

    # ---- sample sentences
    seqs, langs, splits, seen_en = [], [], [], set()
    for path in sorted(glob.glob(os.path.join(a.opus_dir, "*_en.npz"))):
        lc, st, so, tt, to = shard_rows(path)
        n = len(so) - 1
        rows = rng.choice(n, min(a.pairs_per_shard, n), replace=False)
        test = rng.random(len(rows)) < a.test_frac
        keep_en = rng.random(len(rows)) < a.en_frac
        n_en = 0
        for r, te, ke in zip(rows, test, keep_en):
            seqs.append(st[so[r]:so[r + 1]].astype(np.int32))
            langs.append(codes_of.index(lc))
            splits.append(int(te))
            if ke:
                t = tt[to[r]:to[r + 1]].astype(np.int32)
                h = hashlib.blake2b(t.tobytes(), digest_size=8).digest()
                if h not in seen_en:
                    seen_en.add(h)
                    seqs.append(t)
                    langs.append(en)
                    splits.append(int(te))
                    n_en += 1
        print(f"{lc}: {len(rows)} pairs sampled, +{n_en} English sides", flush=True)
    flores = np.concatenate([dev_ids, devtest_ids], 0)
    for l in range(flores.shape[1]):
        for i in range(flores.shape[0]):
            seqs.append(np.asarray(flores[i, l], dtype=np.int32))
            langs.append(l)
            splits.append(2)
    N = len(seqs)
    print(f"{N} sentences total", flush=True)

    # ---- encode, length-sorted batches, stored in sample order
    emb = torch.load(a.embedding_table, map_location="cpu")
    model = build_model(ck["args"], emb, meta).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    model = model.to(a.device)
    M, L = model.encoder.m_max, len(model.quantizer.codebooks)
    assert model.quantizer.k <= 256, "uint8 storage needs k <= 256"
    os.makedirs(a.out, exist_ok=True)
    codes = np.lib.format.open_memmap(os.path.join(a.out, "codes.npy"), mode="w+",
                                      dtype=np.uint8, shape=(N, M, L))
    lens = np.array([len(s) for s in seqs])
    order = np.argsort(lens, kind="stable")
    pad = meta.pad_token_id
    for b, s in enumerate(range(0, N, a.batch_size)):
        idx = order[s:s + a.batch_size]
        T = int(lens[idx].max())
        bt = torch.full((len(idx), T), pad, dtype=torch.long)
        for j, i in enumerate(idx):
            bt[j, :lens[i]] = torch.from_numpy(seqs[i])
        codes[idx] = encode_all_levels(model, bt.to(a.device)).cpu().numpy().astype(np.uint8)
        if b % 500 == 0:
            print(f"  encoded {s + len(idx)}/{N}", flush=True)
    codes.flush()

    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    np.save(os.path.join(a.out, "tokens.npy"), np.concatenate(seqs))
    np.save(os.path.join(a.out, "offsets.npy"), offsets)
    np.save(os.path.join(a.out, "lang.npy"), np.array(langs, dtype=np.uint8))
    np.save(os.path.join(a.out, "split.npy"), np.array(splits, dtype=np.uint8))
    splits = np.array(splits)
    info = dict(checkpoint=a.checkpoint, step=ck.get("step"), N=N, M=M, L=L,
                k=model.quantizer.k, short_codes=codes_of, pad=pad,
                bos=meta.bos_token_id, eos=meta.eos_token_id, vocab_size=int(emb.shape[0]),
                n_train=int((splits == 0).sum()), n_test=int((splits == 1).sum()),
                n_flores=int((splits == 2).sum()), args=vars(a))
    json.dump(info, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "args"}))


if __name__ == "__main__":
    main()
