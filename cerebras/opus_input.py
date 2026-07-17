"""Real opus-100 input pipeline for the CS-3 cstorch training loop.

This replaces train_cstorch.py's synthetic `make_input_fn`. It streams the same
packed-int32 opus shards that the GB10 loop uses (vqvae/data.py::Opus100Dataset),
but adapts them to the two hard cstorch requirements:

  1. STATIC SHAPES. Every batch is exactly (B, seq_len) — sequences longer than
     seq_len are truncated (terminal EOS preserved), shorter ones pad to seq_len.
     The GB10 loader pads to the per-batch max (a shape that changes every step);
     the wafer needs one fixed graph, so we pad/truncate to a constant T here.
  2. NO FROZEN MiniLM IN THE TRACE. The semantic target is the MiniLM embedding
     of the SOURCE sentence. We never run MiniLM here — precompute_opus_semantic.py
     writes per-shard `<lang>_en.sem.npz` (src_sem, tgt_sem aligned to pair order)
     offline, and we look the vector up by (shard, row, which-side-is-source).

The pipeline is otherwise identical to Opus100Dataset: pick a shard uniformly,
a row uniformly, then 50/50 swap direction (X->en vs en->X). Kept self-contained
in cerebras/ so the shared GB10 data code stays untouched.

Contract: make_opus_input_fn(...) returns a zero-arg `input_fn` that yields
pre-batched CPU-tensor dicts {src_ids, tgt_ids, src_lang_id, tgt_lang_id
[, sem_target]}, matching what train_cstorch.py's train_step consumes.
"""
import glob
import os

import numpy as np
import torch

# XLM-R / MiniLM special tokens — fallbacks if no parallel_corpus.npz meta given.
# (<s>=0 bos, <pad>=1, </s>=2 eos, <unk>=3). Order matches the pretrained table.
_XLMR_PAD, _XLMR_BOS, _XLMR_EOS = 1, 0, 2
# Canonical 10-lang order used throughout Nubun; overridden by corpus meta.
_DEFAULT_SHORT_CODES = ["en", "zh", "es", "fr", "ar", "ru", "de", "ja", "ko", "hi"]


def load_meta(parallel_corpus_path):
    """Pull (short_codes, pad, bos, eos) from parallel_corpus.npz if present, so
    lang-id order and special tokens match the frozen model exactly. Falls back
    to the XLM-R defaults when the file is absent (e.g. a data-light dry run)."""
    if parallel_corpus_path and os.path.exists(parallel_corpus_path):
        d = np.load(parallel_corpus_path, allow_pickle=True)
        return (list(d["short_codes"]), int(d["pad_token_id"]),
                int(d["bos_token_id"]), int(d["eos_token_id"]))
    return _DEFAULT_SHORT_CODES, _XLMR_PAD, _XLMR_BOS, _XLMR_EOS


def _scan_shards(root_dir, short_codes, langs):
    """Return [(lang_short, n_pairs, path)] for each *_en.npz shard, validated
    against the packed-int32 format and the known language set."""
    shards = sorted(glob.glob(os.path.join(root_dir, "*_en.npz")))
    if langs is not None:
        keep = set(langs)
        shards = [s for s in shards
                  if os.path.basename(s).split("_en.npz")[0] in keep]
    if not shards:
        raise FileNotFoundError(f"No *_en.npz shards found in {root_dir}")
    lang_to_idx = {c: i for i, c in enumerate(short_codes)}
    meta = []
    for path in shards:
        with np.load(path, allow_pickle=False) as d:
            if "src_offsets" not in d:
                raise ValueError(f"{path} is old object-array format; re-run "
                                 f"build_opus100_corpus.py for packed int32.")
            lang = str(d["lang_code"])
            n = int(d["n_pairs"])
        if lang not in lang_to_idx:
            raise ValueError(f"shard lang '{lang}' not in short_codes {short_codes}")
        meta.append((lang, n, path))
    return meta, lang_to_idx


def _fix_len(seq, T, pad, eos):
    """Pad/truncate a 1-D id sequence to exactly length T (static shape).
    On truncation keep the final position as EOS so the target still terminates
    (the decoder is trained on tgt_ids[:, 1:], so a lost EOS hurts recon)."""
    row = np.full(T, pad, dtype=np.int64)
    n = len(seq)
    if n >= T:
        row[:] = seq[:T]
        row[T - 1] = eos
    else:
        row[:n] = seq
    return row


def make_opus_input_fn(opus_dir, seq_len, batch_size, num_steps,
                       parallel_corpus="data/parallel_corpus.npz",
                       seed=42, langs=None, with_semantic=False,
                       sem_dir=None, d_sem=384):
    """Build the cstorch input_fn. `num_steps` batches are produced (plus a small
    buffer). When with_semantic, `<lang>_en.sem.npz` (from precompute_opus_semantic.py)
    must sit in sem_dir (defaults to opus_dir) and carry src_sem/tgt_sem."""
    short_codes, pad, bos, eos = load_meta(parallel_corpus)
    shard_meta, lang_to_idx = _scan_shards(opus_dir, short_codes, langs)
    en_idx = lang_to_idx["en"]
    sem_dir = sem_dir or opus_dir

    # Lazy per-shard caches (mmap'd token arrays; loaded sem arrays).
    tok_cache, sem_cache = {}, {}

    def _tokens(path):
        if path not in tok_cache:
            tok_cache[path] = dict(np.load(path, mmap_mode="r", allow_pickle=False))
        return tok_cache[path]

    def _sem(path):
        if path not in sem_cache:
            base = os.path.basename(path).replace("_en.npz", "_en.sem.npz")
            spath = os.path.join(sem_dir, base)
            if not os.path.exists(spath):
                raise FileNotFoundError(
                    f"semantic targets {spath} missing — run "
                    f"cerebras/precompute_opus_semantic.py (with_semantic=True).")
            with np.load(spath, allow_pickle=False) as d:
                sem_cache[path] = (np.asarray(d["src_sem"], dtype=np.float32),
                                   np.asarray(d["tgt_sem"], dtype=np.float32))
        return sem_cache[path]

    n_shards = len(shard_meta)

    def input_fn():
        rng = np.random.default_rng(seed)

        def gen():
            B, T = batch_size, seq_len
            for _ in range(num_steps + 2):          # small buffer past num_steps
                src = np.full((B, T), pad, dtype=np.int64)
                tgt = np.full((B, T), pad, dtype=np.int64)
                slang = np.empty(B, dtype=np.int64)
                tlang = np.empty(B, dtype=np.int64)
                sem = np.empty((B, d_sem), dtype=np.float32) if with_semantic else None
                for i in range(B):
                    si = int(rng.integers(0, n_shards))
                    lang, n_pairs, path = shard_meta[si]
                    d = _tokens(path)
                    row = int(rng.integers(0, n_pairs))
                    so, to = d["src_offsets"], d["tgt_offsets"]
                    x_ids = np.asarray(d["src_tokens"][so[row]:so[row + 1]])
                    en_ids = np.asarray(d["tgt_tokens"][to[row]:to[row + 1]])
                    swap = rng.random() < 0.5           # en->X instead of X->en
                    if not swap:
                        s_ids, t_ids = x_ids, en_ids
                        slang[i], tlang[i] = lang_to_idx[lang], en_idx
                    else:
                        s_ids, t_ids = en_ids, x_ids
                        slang[i], tlang[i] = en_idx, lang_to_idx[lang]
                    src[i] = _fix_len(s_ids, T, pad, eos)
                    tgt[i] = _fix_len(t_ids, T, pad, eos)
                    if with_semantic:                   # target = MiniLM(source)
                        src_sem, tgt_sem = _sem(path)
                        sem[i] = tgt_sem[row] if swap else src_sem[row]
                batch = {
                    "src_ids": torch.from_numpy(src),
                    "tgt_ids": torch.from_numpy(tgt),
                    "src_lang_id": torch.from_numpy(slang),
                    "tgt_lang_id": torch.from_numpy(tlang),
                }
                if with_semantic:
                    batch["sem_target"] = torch.from_numpy(sem)
                yield batch

        return gen()

    return input_fn
