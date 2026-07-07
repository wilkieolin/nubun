"""Parallel-sentence batch samplers for VQ-VAE training.

Two corpora supported:
  - FLORES-200 (loaded via load_corpus + ParallelDataset). 2009 sentences,
    truly parallel across all 10 languages. Cheap to keep in memory.
  - Opus-100 (loaded via Opus100Dataset). ~9M (en, X) sentence pairs across
    9 X languages. English-pivoted: each shard gives true (en, X) parallel
    pairs but sentences differ across X. IterableDataset that streams from
    sharded npz on disk.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class CorpusMeta:
    short_codes: list[str]      # ['en', 'zh', ...]
    lang_codes: list[str]       # ['eng_Latn', 'zho_Hans', ...]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    vocab_size: int


def load_corpus(path: str = "data/parallel_corpus.npz",
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CorpusMeta]:
    """Returns (dev_token_ids, devtest_token_ids, dev_lengths, devtest_lengths, meta).
    Token-id arrays are object arrays of shape (N_sent, N_lang)."""
    d = np.load(path, allow_pickle=True)
    meta = CorpusMeta(
        short_codes=list(d["short_codes"]),
        lang_codes=list(d["lang_codes"]),
        pad_token_id=int(d["pad_token_id"]),
        bos_token_id=int(d["bos_token_id"]),
        eos_token_id=int(d["eos_token_id"]),
        vocab_size=int(d["vocab_size"]),
    )
    return (
        d["dev_token_ids"],
        d["devtest_token_ids"],
        d["dev_lengths"],
        d["devtest_lengths"],
        meta,
    )


def combined_split(
    dev_ids: np.ndarray, devtest_ids: np.ndarray,
    dev_lens: np.ndarray, devtest_lens: np.ndarray,
    val_fraction: float = 0.1, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate dev + devtest, then split into train/val deterministically.

    With cross-lingual any-to-any sampling, the original 997/1012 dev/devtest
    split leaves training data-starved. Combining + custom-splitting gives more
    sentences per language for training while still keeping a held-out set.

    Returns (train_ids, val_ids, train_lens, val_lens).
    """
    all_ids = np.concatenate([dev_ids, devtest_ids], axis=0)
    all_lens = np.concatenate([dev_lens, devtest_lens], axis=0)
    n = all_ids.shape[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return (
        all_ids[train_idx], all_ids[val_idx],
        all_lens[train_idx], all_lens[val_idx],
    )


class ParallelDataset(Dataset):
    """One example = one parallel sentence (all 10 languages).

    Sampling (src, tgt) language pair happens in the collate function.
    """

    def __init__(self, token_ids: np.ndarray, lengths: np.ndarray):
        self.token_ids = token_ids       # (N, L) object
        self.lengths = lengths           # (N, L) int
        self.n_sent, self.n_lang = token_ids.shape

    def __len__(self) -> int:
        return self.n_sent

    def __getitem__(self, idx: int):
        # Return all language versions; collate picks src/tgt
        return {
            "sentence_id": idx,
            "ids_per_lang": [self.token_ids[idx, j] for j in range(self.n_lang)],
            "lens_per_lang": self.lengths[idx],
        }


class Opus100Dataset(IterableDataset):
    """Streaming dataset over sharded opus-100 npz files (packed-int32 format).

    Each shard contains:
      src_tokens   (Tsrc,) int32     concatenated token IDs
      src_offsets  (N+1,)  int32     pair i is src_tokens[offsets[i]:offsets[i+1]]
      tgt_tokens   (Ttgt,) int32     same for English side
      tgt_offsets  (N+1,)  int32

    The arrays are loaded with mmap_mode='r' so random row access is O(1) and
    memory-cheap (kernel page cache handles repeated reads).

    Each yield is a single (src, tgt) pair plus the (src_lang, tgt_lang) pair
    indices. Batching happens in `make_streaming_collate`.

    Random sampling: pick a shard uniformly, pick a row uniformly, then 50/50
    swap directions (X→en vs en→X). The shard's lang_code identifies X.
    """

    def __init__(self, root_dir: str, short_codes: list[str],
                 seed: int = 42, langs: list[str] | None = None):
        self.root_dir = root_dir
        self.short_codes = short_codes
        self.seed = seed

        shards = sorted(glob.glob(os.path.join(root_dir, "*_en.npz")))
        if langs is not None:
            shards = [s for s in shards
                      if os.path.basename(s).split("_en.npz")[0] in langs]
        if not shards:
            raise FileNotFoundError(f"No *_en.npz shards found in {root_dir}")

        self.shards = shards
        self.shard_meta = []  # list of (lang_short_code, n_pairs, path)
        for path in shards:
            with np.load(path, allow_pickle=False) as d:
                if "src_offsets" not in d:
                    raise ValueError(
                        f"{path} is in old object-array format. Re-run "
                        f"build_opus100_corpus.py to regenerate as packed int32.")
                lang = str(d["lang_code"])
                n = int(d["n_pairs"])
            self.shard_meta.append((lang, n, path))

        self.lang_to_idx = {c: i for i, c in enumerate(short_codes)}
        if "en" not in self.lang_to_idx:
            raise ValueError("short_codes must include 'en'")
        self.en_idx = self.lang_to_idx["en"]

        for lang, _, _ in self.shard_meta:
            if lang not in self.lang_to_idx:
                raise ValueError(
                    f"shard lang '{lang}' not in short_codes {short_codes}")

        # Per-shard mmap'd views, lazy-built per worker
        self._mmaps: dict[str, dict] = {}

    def _load_shard(self, path: str) -> dict:
        if path not in self._mmaps:
            # np.load with mmap_mode='r' still requires memory-mapping each
            # array, but for plain int32 arrays this is a true mmap (kernel
            # page cache). With allow_pickle=False, no unpickling happens.
            self._mmaps[path] = dict(np.load(path, mmap_mode="r", allow_pickle=False))
        return self._mmaps[path]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # Add a per-rank offset so DDP ranks see different sample streams
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except RuntimeError:
            rank = 0
        rng = np.random.default_rng(self.seed + rank * 100000 + worker_id * 1000)

        n_shards = len(self.shard_meta)
        while True:
            shard_idx = int(rng.integers(0, n_shards))
            lang, n_pairs, path = self.shard_meta[shard_idx]
            d = self._load_shard(path)
            row = int(rng.integers(0, n_pairs))

            so = d["src_offsets"]
            to = d["tgt_offsets"]
            x_ids = np.array(d["src_tokens"][so[row]:so[row+1]])  # copy out of mmap
            en_ids = np.array(d["tgt_tokens"][to[row]:to[row+1]])

            if rng.random() < 0.5:
                src_ids, tgt_ids = x_ids, en_ids
                src_lang, tgt_lang = self.lang_to_idx[lang], self.en_idx
            else:
                src_ids, tgt_ids = en_ids, x_ids
                src_lang, tgt_lang = self.en_idx, self.lang_to_idx[lang]

            yield {
                "src_ids": src_ids,
                "tgt_ids": tgt_ids,
                "src_lang_id": src_lang,
                "tgt_lang_id": tgt_lang,
            }


def make_streaming_collate(pad_token_id: int):
    """Collate fn for items yielded by Opus100Dataset (each item already has
    src/tgt picked, so we just pad+stack)."""
    def collate(batch: list[dict]) -> dict:
        B = len(batch)
        T_src = max(len(b["src_ids"]) for b in batch)
        T_tgt = max(len(b["tgt_ids"]) for b in batch)

        src_arr = np.full((B, T_src), pad_token_id, dtype=np.int64)
        tgt_arr = np.full((B, T_tgt), pad_token_id, dtype=np.int64)
        src_lang = np.empty(B, dtype=np.int64)
        tgt_lang = np.empty(B, dtype=np.int64)
        for i, b in enumerate(batch):
            src_arr[i, : len(b["src_ids"])] = b["src_ids"]
            tgt_arr[i, : len(b["tgt_ids"])] = b["tgt_ids"]
            src_lang[i] = b["src_lang_id"]
            tgt_lang[i] = b["tgt_lang_id"]

        return {
            "src_ids": torch.from_numpy(src_arr),
            "tgt_ids": torch.from_numpy(tgt_arr),
            "src_lang_id": torch.from_numpy(src_lang),
            "tgt_lang_id": torch.from_numpy(tgt_lang),
        }
    return collate


def make_collate(
    pad_token_id: int,
    n_lang: int,
    src_langs: list[int] | None = None,
    tgt_langs: list[int] | None = None,
    rng: np.random.Generator | None = None,
):
    """Returns a collate fn that picks (src_lang, tgt_lang) per example.

    If src_langs / tgt_langs is None, sample uniformly over all n_lang.
    Same lang allowed (same-language reconstruction).
    """
    if rng is None:
        rng = np.random.default_rng()
    src_pool = src_langs if src_langs is not None else list(range(n_lang))
    tgt_pool = tgt_langs if tgt_langs is not None else list(range(n_lang))

    def collate(batch: list[dict]) -> dict:
        B = len(batch)
        src_lang = int(rng.choice(src_pool))
        tgt_lang = int(rng.choice(tgt_pool))

        src_seqs = [b["ids_per_lang"][src_lang] for b in batch]
        tgt_seqs = [b["ids_per_lang"][tgt_lang] for b in batch]

        T_src = max(len(s) for s in src_seqs)
        T_tgt = max(len(s) for s in tgt_seqs)

        src_arr = np.full((B, T_src), pad_token_id, dtype=np.int64)
        tgt_arr = np.full((B, T_tgt), pad_token_id, dtype=np.int64)
        for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
            src_arr[i, : len(s)] = s
            tgt_arr[i, : len(t)] = t

        return {
            "src_ids": torch.from_numpy(src_arr),
            "tgt_ids": torch.from_numpy(tgt_arr),
            "tgt_lang_id": torch.full((B,), tgt_lang, dtype=torch.int64),
            "src_lang_id": torch.full((B,), src_lang, dtype=torch.int64),
        }

    return collate
