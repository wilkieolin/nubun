#!/usr/bin/env bash
# Unpack opus100 shards for the CS-3 input workers (run once, on the user node).
#   bash cerebras/unpack_opus.sh <data/opus100 with *_en.npz [+ *_en.sem.npz]> <out dir>
# An .npz is a zip of STORED .npy members; unpacked, numpy memory-maps them
# (np.load ignores mmap_mode for .npz, so the packed form costs ~1.5 GB RAM per
# semantic shard per worker). Output layout, read by cerebras/opus_input.py:
#   <out>/<lang>_en/{src_tokens,src_offsets,tgt_tokens,tgt_offsets,n_pairs,lang_code}.npy
#   <out>/<lang>_en.sem/{src_sem,tgt_sem}.npy
set -euo pipefail
src="${1:?src dir}"; out="${2:?out dir}"; mkdir -p "$out"
for f in "$src"/*_en.npz "$src"/*_en.sem.npz; do
  [[ -f "$f" ]] || continue
  b="$(basename "$f" .npz)"; mkdir -p "$out/$b"; unzip -o -q "$f" -d "$out/$b"; echo "unpacked $b"
done
du -sh "$out"
