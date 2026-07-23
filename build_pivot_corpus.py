"""Phase 8 data lever — pivot-mine non-English-centric X-Y pairs from opus100.

opus100 is all X-en. But when the SAME English sentence appears in two shards
(X-en and Y-en), its X and Y translations are both renderings of that meaning,
so (X, Y) is a valid non-en-centric parallel pair. This mines all such pairs
across the 9 non-en shards -> data/pivot_corpus.npz.

Stores undirected pairs (a-side / b-side + their lang ids); the data loader picks
the direction at train time (either side can be source). Packed int32, opus-style
offsets. Run: python build_pivot_corpus.py
"""
import numpy as np

LANGS = ['ar', 'de', 'es', 'fr', 'hi', 'ja', 'pt', 'ru', 'zh']
# lang -> global short_codes index (must match training meta.short_codes order)
# meta.short_codes = en,zh,es,fr,ar,ru,hi,ja,pt,de (from parallel_corpus)
SHORT = ['en', 'zh', 'es', 'fr', 'ar', 'ru', 'hi', 'ja', 'pt', 'de']
LANG_ID = {lg: SHORT.index(lg) for lg in LANGS}


def main():
    # Pass 1: en_hash -> {lang: (src_start, src_end)} first occurrence per lang.
    shards = {}
    en2rows = {}
    for lg in LANGS:
        d = np.load(f'data/opus100/{lg}_en.npz', allow_pickle=True)
        shards[lg] = (d['src_tokens'], d['src_offsets'], d['tgt_tokens'], d['tgt_offsets'])
        st, so, tt, to = shards[lg]
        n = len(to) - 1
        for i in range(n):
            h = hash(tt[to[i]:to[i + 1]].tobytes())
            slot = en2rows.get(h)
            if slot is None:
                en2rows[h] = {lg: (so[i], so[i + 1])}
            elif lg not in slot:
                slot[lg] = (so[i], so[i + 1])
        print(f'  scanned {lg}: {n} rows', flush=True)

    # Pass 2: emit all C(k,2) pairs for en shared by >=2 langs.
    a_tok, a_off, a_lang = [], [0], []
    b_tok, b_off, b_lang = [], [0], []
    n_pairs = 0
    for slot in en2rows.values():
        if len(slot) < 2:
            continue
        items = list(slot.items())  # [(lang, (s,e)), ...]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (la, (sa, ea)), (lb, (sb, eb)) = items[i], items[j]
                xa = shards[la][0][sa:ea]
                xb = shards[lb][0][sb:eb]
                a_tok.append(xa); a_off.append(a_off[-1] + len(xa)); a_lang.append(LANG_ID[la])
                b_tok.append(xb); b_off.append(b_off[-1] + len(xb)); b_lang.append(LANG_ID[lb])
                n_pairs += 1
    print(f'  emitted {n_pairs} pivot pairs')

    np.savez('data/pivot_corpus.npz',
             a_tokens=np.concatenate(a_tok).astype(np.int32),
             a_offsets=np.array(a_off, dtype=np.int64),
             a_lang=np.array(a_lang, dtype=np.int16),
             b_tokens=np.concatenate(b_tok).astype(np.int32),
             b_offsets=np.array(b_off, dtype=np.int64),
             b_lang=np.array(b_lang, dtype=np.int16),
             n_pairs=n_pairs, short_codes=np.array(SHORT))
    print('  wrote data/pivot_corpus.npz')


if __name__ == '__main__':
    main()
