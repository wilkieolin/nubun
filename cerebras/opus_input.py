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
pre-batched CPU-tensor dicts {src_ids, dec_in, labels, label_mask, src_lang_id,
tgt_lang_id [, token_weight_t] [, sem_target]}, matching what
train_cstorch.py's train_step consumes.

Appliance-worker notes (the closure returned here is pickled and runs on the
worker pods, whose cwd is NOT the repo and whose memory is limited):
  - every path is made absolute at construction time;
  - shards are read from UNPACKED `.npy` members so they are memory-mapped:
    `<opus_dir>/<lang>_en/{src_tokens,src_offsets,tgt_tokens,tgt_offsets,n_pairs,lang_code}.npy`
    and `<sem_dir>/<lang>_en.sem/{src_sem,tgt_sem}.npy` (fp16, cast per row).
    Unpack once with cerebras/unpack_opus.sh (an .npz is a zip of stored
    .npy files, and numpy ignores mmap_mode for .npz: loading the .npz form
    reads ~1.5 GB per semantic shard into RAM per worker). The .npz form is
    still accepted as a fallback for small/dry runs;
  - each streaming worker offsets the RNG seed by its rank so multiple
    workers / CS-X systems do not replay the same sample stream.
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


_TOKEN_KEYS = ("src_tokens", "src_offsets", "tgt_tokens", "tgt_offsets")


def _open_tokens(path):
    """Open one token shard. `path` is either an unpacked directory (members
    memory-mapped, O(1) RAM) or a legacy .npz (fully read into RAM)."""
    if os.path.isdir(path):
        d = {k: np.load(os.path.join(path, k + ".npy"), mmap_mode="r",
                        allow_pickle=False) for k in _TOKEN_KEYS}
        d["n_pairs"] = np.load(os.path.join(path, "n_pairs.npy"), allow_pickle=False)
        d["lang_code"] = np.load(os.path.join(path, "lang_code.npy"), allow_pickle=False)
        return d
    with np.load(path, allow_pickle=False) as z:
        if "src_offsets" not in z:
            raise ValueError(f"{path} is old object-array format; re-run "
                             f"build_opus100_corpus.py for packed int32.")
        return {k: np.asarray(z[k]) for k in _TOKEN_KEYS + ("n_pairs", "lang_code")}


def _open_sem(sem_dir, lang):
    """(src_sem, tgt_sem) for one language: unpacked fp16 .npy (mmap) preferred,
    `<lang>_en.sem.npz` fallback (read into RAM)."""
    d = os.path.join(sem_dir, f"{lang}_en.sem")
    if os.path.isdir(d):
        return (np.load(os.path.join(d, "src_sem.npy"), mmap_mode="r", allow_pickle=False),
                np.load(os.path.join(d, "tgt_sem.npy"), mmap_mode="r", allow_pickle=False))
    spath = os.path.join(sem_dir, f"{lang}_en.sem.npz")
    if not os.path.exists(spath):
        raise FileNotFoundError(
            f"semantic targets {d}/ or {spath} missing — run "
            f"cerebras/precompute_opus_semantic.py, then cerebras/unpack_opus.sh.")
    with np.load(spath, allow_pickle=False) as z:
        return np.asarray(z["src_sem"]), np.asarray(z["tgt_sem"])


def _scan_shards(root_dir, short_codes, langs):
    """Return [(lang_short, n_pairs, path)] for each shard — an unpacked
    `<lang>_en/` directory (preferred) or a `<lang>_en.npz` — validated against
    the packed-int32 format and the known language set."""
    dirs = sorted(d for d in glob.glob(os.path.join(root_dir, "*_en"))
                  if os.path.isdir(d))
    npzs = sorted(glob.glob(os.path.join(root_dir, "*_en.npz")))
    have_dir = {os.path.basename(d) for d in dirs}
    shards = dirs + [z for z in npzs
                     if os.path.basename(z)[:-len(".npz")] not in have_dir]
    if langs is not None:
        keep = set(langs)
        shards = [s for s in shards
                  if os.path.basename(s).split("_en")[0] in keep]
    if not shards:
        raise FileNotFoundError(f"No *_en/ dirs or *_en.npz shards found in {root_dir}")
    lang_to_idx = {c: i for i, c in enumerate(short_codes)}
    meta = []
    for path in shards:
        d = _open_tokens(path)
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
                       sem_dir=None, d_sem=384, pad=None, token_weight=None,
                       max_len=None):
    """Build the cstorch input_fn. `num_steps` batches are produced (plus a small
    buffer). When with_semantic, `<lang>_en.sem.npz` (from precompute_opus_semantic.py)
    must sit in sem_dir (defaults to opus_dir) and carry src_sem/tgt_sem.

    CS-3 host-side prep (matches cerebras/train_cstorch.weighted_ce):
      - ids are int32 (no i64 on the wafer);
      - the AR shift is done here (pre_shifted): yields dec_in = tgt[:-1] and
        labels = tgt[1:], so `labels` reaches the CE as an un-sliced load_input
        (required for the frozen-embedding lookup to lower);
      - label_mask (float pad mask) and, if token_weight is given, token_weight_t
        (= token_weight[labels]) are looked up HOST-side and shipped as (B,T-1)
        float activations rather than wafer-side (V,·) tables.

    max_len: if set, only pairs whose src AND tgt (ids incl. BOS/EOS) are both
    <= max_len are sampled — used to train at a short --seq-len without
    truncating (~70% of every 64-slot row is padding; 82% of pairs fit 32).
    """
    # Absolute paths: this closure is pickled to the appliance worker pods,
    # whose working directory is not the repo.
    opus_dir = os.path.abspath(opus_dir)
    sem_dir = os.path.abspath(sem_dir or opus_dir)
    parallel_corpus = os.path.abspath(parallel_corpus) if parallel_corpus else None
    meta_short, meta_pad, bos, eos = load_meta(parallel_corpus)
    short_codes = meta_short
    if pad is None:
        pad = meta_pad
    token_weight_np = (token_weight.detach().cpu().numpy().astype(np.float32)
                       if token_weight is not None else None)
    shard_meta, lang_to_idx = _scan_shards(opus_dir, short_codes, langs)
    en_idx = lang_to_idx["en"]
    if with_semantic:                      # fail at construction, not on a worker
        for lang, _, _ in shard_meta:
            _open_sem(sem_dir, lang)

    # Lazy per-shard caches (mmap'd views; built inside the worker process).
    tok_cache, sem_cache = {}, {}

    def _tokens(path):
        if path not in tok_cache:
            tok_cache[path] = _open_tokens(path)
        return tok_cache[path]

    elig_cache = {}

    def _eligible(path, d):
        """Row indices of this shard that fit max_len on both sides (cached)."""
        if path not in elig_cache:
            so, to = np.asarray(d["src_offsets"]), np.asarray(d["tgt_offsets"])
            longest = np.maximum(np.diff(so), np.diff(to))
            elig_cache[path] = np.flatnonzero(longest <= max_len)
        return elig_cache[path]

    def _sem(lang):
        if lang not in sem_cache:
            sem_cache[lang] = _open_sem(sem_dir, lang)
        return sem_cache[lang]

    n_shards = len(shard_meta)

    def _worker_rank():
        """Rank among the appliance's streaming workers (0 off-appliance)."""
        try:
            import cerebras.pytorch.distributed as cdist
            return int(cdist.get_streaming_rank())
        except Exception:
            return 0

    def input_fn():
        rng = np.random.default_rng(seed + 1000 * _worker_rank())

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
                    if max_len is None:
                        row = int(rng.integers(0, n_pairs))
                    else:
                        elig = _eligible(path, d)
                        row = int(elig[rng.integers(0, len(elig))])
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
                        src_sem, tgt_sem = _sem(lang)
                        sem[i] = tgt_sem[row] if swap else src_sem[row]  # fp16 -> fp32
                # Host-side AR shift + int32. dec_in/labels are (B, T-1); labels
                # stays un-sliced downstream so its embedding lookup lowers.
                dec_in = tgt[:, :-1].astype(np.int32)
                labels = tgt[:, 1:].astype(np.int32)
                # Host-side floats are fp32: the graph casts to its half dtype
                # (bf16 on the CS-3; see train_cstorch --half-dtype).
                label_mask = (labels != pad).astype(np.float32)
                batch = {
                    "src_ids": torch.from_numpy(src.astype(np.int32)),
                    "dec_in": torch.from_numpy(np.ascontiguousarray(dec_in)),
                    "labels": torch.from_numpy(np.ascontiguousarray(labels)),
                    "label_mask": torch.from_numpy(np.ascontiguousarray(label_mask)),
                    "src_lang_id": torch.from_numpy(slang.astype(np.int32)),
                    "tgt_lang_id": torch.from_numpy(tlang.astype(np.int32)),
                }
                if token_weight_np is not None:
                    tw = token_weight_np[labels].astype(np.float32)  # host lookup
                    batch["token_weight_t"] = torch.from_numpy(np.ascontiguousarray(tw))
                if with_semantic:
                    batch["sem_target"] = torch.from_numpy(sem)
                yield batch

        return gen()

    return input_fn
