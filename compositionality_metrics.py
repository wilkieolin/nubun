"""compositionality_metrics.py — PHASE9 A3 + A6: is the code compositional?

Encoder-only. Encodes every FLORES sentence (dev + devtest, 2009 x 10 languages;
all held out from the opus100-trained models) and scores the RVQ code stacks
against cheap meaning attributes labelled on the English side with spaCy and
keyword lists, then copied to the other nine languages (FLORES is parallel, so
attributes are properties of the meaning, not of one language).

Messages: a sentence's code is C[slot, level] (64 x 8). Two symbol views per
depth d: the level-(d-1) code alone ("level"), and the stack prefix of levels
0..d-1 as one symbol ("prefix", the character up to depth d).

Metrics (all with an attribute-shuffle or code-permutation control):
  topsim   Spearman(code distance, MiniLM meaning distance) over sentence pairs;
           bag-of-symbols Jaccard distance and positional Hamming.
  posdis   Chaabouni et al. 2020: per slot, (I1 - I2) / H(slot) over attributes.
  bosdis   same, per symbol, on its count in the message (order-free).
  MIG      per attribute, (I1 - I2) / H(attr) over slots (positional) and over
           symbol counts (bag).
  TRE      Andreas 2019, additive composition: held-out ridge fit of the pooled
           quantised vector (cumulative to depth d) from one-hot attributes;
           reported as residual / variance (1 - R^2). Control: rows permuted.
  nPMI     level-0 code x attribute value, per language; cross-language
           stability = mean pairwise Pearson of the nPMI tables.
  stack    A6: cross-lingual Jaccard of distinct stack prefixes (depth 1..8),
           chance = sentence i vs i + n/2 (FLORES is article-ordered, so i+1 is
           usually the same article), kappa = (real - chance) / (1 - chance);
           plus positional slot agreement.

Gate (PHASE9 A3): posdis >> bosdis -> keep RVQ (positions matter); bosdis
dominant -> order-free code, consider additive quantisation (B4).

Usage:
    python compositionality_metrics.py --checkpoint <ckpt> [--name cs3_500k]
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
import torch

import vqvae_select
vqvae_select.select_from_argv()  # pre-port checkpoints need pre-port vqvae/
from vqvae.data import load_corpus
from code_usage_audit import build_model, encode_all_levels

FIELDS = {
    "health": "disease virus patient hospital doctor medical health cancer infection vaccine "
              "medicine symptom illness drug",
    "sport": "game team player match season championship cup score race coach olympic "
             "tournament league athlete",
    "politics": "government president minister election party parliament vote law policy "
                "political senator congress governor",
    "nature": "animal species plant bird fish forest tree river sea ocean island mountain "
              "insect wildlife",
    "travel": "travel tourist hotel visa flight airport passport trip visitor destination "
              "tour journey",
    "science": "scientist research study university theory physics chemical planet star "
               "energy experiment",
    "disaster": "storm hurricane earthquake flood rain wind fire weather tornado volcano",
}
FIELD_OF = {w: f for f, ws in FIELDS.items() for w in ws.split()}
NEG_LEMMAS = {"no", "never", "nothing", "none", "nobody", "neither", "nor", "without"}
NER_GROUPS = {"person": {"PERSON"}, "place": {"GPE", "LOC", "FAC"}, "org": {"ORG", "NORP"},
              "time": {"DATE", "TIME"}, "number": {"CARDINAL", "QUANTITY", "MONEY", "PERCENT"}}


# ---------------------------------------------------------------- attributes
ATTR_NAMES = ["neg", "tense", "subj_person", "subj_number", "field",
              *[f"ner_{g}" for g in NER_GROUPS]]


def doc_attributes(doc):
    """One spaCy doc -> {attr: string value} (also used by augment/minpairs.py)."""
    a = {}
    root = next((t for t in doc if t.dep_ == "ROOT"), doc[0])
    a["neg"] = str(int(any(t.dep_ == "neg" or t.lemma_.lower() in NEG_LEMMAS for t in doc)))
    aux = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    tense = "none"
    if any(x.lower_ in ("will", "shall", "'ll") for x in aux):
        tense = "fut"
    else:
        for t in [*aux, root]:  # first finite tense: "was taken" -> past, "has gone" -> pres
            ts = t.morph.get("Tense")
            if ts:
                tense = ts[0].lower()
                break
    a["tense"] = tense
    subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    if subj is None:
        person, number = "none", "none"
    else:
        p = subj.morph.get("Person")
        # 1st and 2nd person are each <2% of FLORES: merge as speech participants
        person = ("12" if p[0] in ("1", "2") else p[0]) if (subj.pos_ == "PRON" and p) else "3n"
        n = subj.morph.get("Number")
        number = n[0].lower() if n else "none"
    a["subj_person"] = person
    a["subj_number"] = number
    hits = {}
    for t in doc:
        f = FIELD_OF.get(t.lemma_.lower())
        if f:
            hits[f] = hits.get(f, 0) + 1
    best = sorted(hits.items(), key=lambda kv: -kv[1])
    a["field"] = (best[0][0] if best and (len(best) == 1 or best[0][1] > best[1][1])
                  else "none")
    labels = {e.label_ for e in doc.ents}
    for g, s in NER_GROUPS.items():
        a[f"ner_{g}"] = str(int(bool(labels & s)))
    return a


def label_attributes(texts):
    """English sentences -> {attr: array of string values}."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    cols = {k: [] for k in ATTR_NAMES}
    for doc in nlp.pipe(texts, batch_size=64):
        for k, v in doc_attributes(doc).items():
            cols[k].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def encode_attrs(attrs, min_frac):
    """Drop degenerate attributes; map values to ints (rare values pooled to 'other')."""
    out, names = [], []
    for k, v in attrs.items():
        vals, cnt = np.unique(v, return_counts=True)
        keep = set(vals[cnt >= min_frac * len(v)])
        v = np.array([x if x in keep else "other" for x in v])
        vals, cnt = np.unique(v, return_counts=True)
        if len(vals) < 2 or cnt.min() < min_frac * len(v):
            continue
        out.append(np.searchsorted(vals, v))
        names.append((k, list(vals)))
    return np.stack(out, 1), names


# ---------------------------------------------------------------- info theory
def entropy(x):
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info(x, y):
    """Plug-in I(X;Y) in bits for integer arrays."""
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def disentanglement(sym, A):
    """sym: (N, F) integer factors (slots, or symbol counts); A: (N, K) attributes.
    Returns (dis, mig): Chaabouni's per-factor gap / H(factor) and per-attribute
    MIG gap / H(attr)."""
    F_, K = sym.shape[1], A.shape[1]
    I = np.zeros((F_, K))
    H = np.zeros(F_)
    for f in range(F_):
        H[f] = entropy(sym[:, f])
        if H[f] > 1e-9:
            for k in range(K):
                I[f, k] = mutual_info(sym[:, f], A[:, k])
    live = H > 1e-9
    if not live.any() or K < 2:
        return 0.0, 0.0
    s = np.sort(I[live], 1)[:, ::-1]
    dis = float(np.mean((s[:, 0] - s[:, 1]) / H[live]))
    HA = np.array([entropy(A[:, k]) for k in range(K)])
    t = np.sort(I[live], 0)[::-1]
    mig = float(np.mean((t[0] - t[1]) / HA)) if t.shape[0] > 1 else float(np.mean(t[0] / HA))
    return dis, mig


def symbol_counts(sym, max_symbols):
    """(N, M) symbols -> (N, S) counts of the S most frequent symbols."""
    vals, cnt = np.unique(sym, return_counts=True)
    top = vals[np.argsort(-cnt)[:max_symbols]]
    return np.stack([(sym == v).sum(1) for v in top], 1)


def prefix_symbols(C, depth):
    """(N, M, L) codes -> (N, M) one int per slot for the stack prefix 0..depth-1."""
    K = int(C.max()) + 1
    out = np.zeros(C.shape[:2], dtype=np.int64)
    for l in range(depth):
        out = out * K + C[:, :, l]
    return out


# ---------------------------------------------------------------- metrics
def topsim(sym, meaning, n_pairs, rng, positional):
    from scipy.stats import spearmanr
    N = sym.shape[0]
    i, j = rng.integers(0, N, n_pairs), rng.integers(0, N, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    if positional:
        dc = (sym[i] != sym[j]).mean(1)
    else:
        sets = [set(r.tolist()) for r in sym]
        dc = np.array([1 - len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in zip(i, j)])
    dm = 1 - (meaning[i] * meaning[j]).sum(1)
    return float(spearmanr(dc, dm).statistic)


def tre(X, A, names, sent_of, rng, ridge=1e-2):
    """Held-out 1 - R^2 of X from one-hot attributes; split by sentence."""
    onehot = np.concatenate([np.eye(len(v))[A[:, k]] for k, (_, v) in enumerate(names)], 1)
    D = np.concatenate([onehot, np.ones((len(X), 1))], 1)
    sents = np.unique(sent_of)
    test = np.isin(sent_of, rng.choice(sents, len(sents) // 5, replace=False))

    def fit_eval(Y):
        W = np.linalg.solve(D[~test].T @ D[~test] + ridge * np.eye(D.shape[1]), D[~test].T @ Y[~test])
        r = Y[test] - D[test] @ W
        return float((r ** 2).sum() / ((Y[test] - Y[~test].mean(0)) ** 2).sum())

    return fit_eval(X), fit_eval(X[rng.permutation(len(X))])


def npmi_stability(C0, A, names, lang_of, n_langs, K):
    """Per-language nPMI(level-0 code present, attribute value); mean pairwise Pearson."""
    pres = np.zeros((len(C0), K), bool)
    pres[np.arange(len(C0))[:, None], C0] = True
    tables = []
    for l in range(n_langs):
        m = lang_of == l
        P, cols = pres[m], []
        for k, (_, vals) in enumerate(names):
            for v in range(len(vals)):
                a = A[m, k] == v
                pxy = (P & a[:, None]).mean(0)
                px, py = P.mean(0), a.mean()
                with np.errstate(divide="ignore", invalid="ignore"):
                    npmi = np.log(pxy / (px * py)) / -np.log(pxy)
                npmi[(pxy == 0)] = -1.0
                npmi[px == 0] = np.nan
                cols.append(npmi)
        tables.append(np.stack(cols, 1))  # (K, n_attr_values)
    T = np.stack(tables)                    # (langs, K, V)
    ok = ~np.isnan(T).any(0)
    rs = [np.corrcoef(T[a][ok], T[b][ok])[0, 1] for a, b in combinations(range(n_langs), 2)]
    return float(np.mean(rs)), np.nanmean(T, 0)


def stack_interlingua(C, sent_idx, lang_of, n_langs, n_sent, depth):
    """A6: cross-lingual Jaccard of distinct stack prefixes, real vs i/i+n/2 chance."""
    sym = prefix_symbols(C, depth)
    by = {}
    for r, (s, l) in enumerate(zip(sent_idx, lang_of)):
        by[(s, l)] = r
    sets = [set(r.tolist()) for r in sym]

    def jac(a, b):
        return len(a & b) / len(a | b)

    real, chance, pos_real, pos_chance = [], [], [], []
    for a, b in combinations(range(n_langs), 2):
        for s in range(n_sent):
            ra, rb, rc = by[(s, a)], by[(s, b)], by[((s + n_sent // 2) % n_sent, b)]
            real.append(jac(sets[ra], sets[rb]))
            chance.append(jac(sets[ra], sets[rc]))
            pos_real.append((sym[ra] == sym[rb]).mean())
            pos_chance.append((sym[ra] == sym[rc]).mean())
    r, c = float(np.mean(real)), float(np.mean(chance))
    pr, pc = float(np.mean(pos_real)), float(np.mean(pos_chance))
    return dict(real=r, chance=c, lift=r - c, kappa=(r - c) / (1 - c),
                pos_real=pr, pos_chance=pc, pos_kappa=(pr - pc) / (1 - pc))


# ---------------------------------------------------------------- main
@torch.no_grad()
def meaning_embeddings(ids_en, pad, device, name, batch=64):
    from transformers import AutoModel
    st = AutoModel.from_pretrained(name).eval().to(device)
    out = []
    for s in range(0, len(ids_en), batch):
        seqs = ids_en[s:s + batch]
        T = max(len(x) for x in seqs)
        bt = torch.full((len(seqs), T), pad, dtype=torch.long)
        for i, x in enumerate(seqs):
            bt[i, :len(x)] = torch.as_tensor(x)
        bt = bt.to(device)
        attn = (bt != pad).long()
        h = st(input_ids=bt, attention_mask=attn).last_hidden_state
        m = attn.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    del st
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", default=None, help="label for output files")
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--tokenizer", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--attr-cache", default="data/flores_en_attributes.json")
    p.add_argument("--min-frac", type=float, default=0.02, help="min share of an attribute value")
    p.add_argument("--max-symbols", type=int, default=256, help="bosdis/bag-MIG: top-N symbols")
    p.add_argument("--n-pairs", type=int, default=100000)
    p.add_argument("--n-shuffles", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/phase9")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.set_grad_enabled(False)  # analysis only; codebooks are trainable parameters
    rng = np.random.default_rng(a.seed)
    name = a.name or os.path.splitext(os.path.basename(a.checkpoint))[0]
    os.makedirs(a.outdir, exist_ok=True)

    ck = vqvae_select.load_checkpoint(a.checkpoint)
    dev_ids, devtest_ids, _, _, meta = load_corpus(a.corpus)
    ids = np.concatenate([dev_ids, devtest_ids], 0)        # (2009, 10) held out from opus100
    n_sent, n_langs = ids.shape
    pad = meta.pad_token_id

    # English attributes (cached; independent of the checkpoint)
    if os.path.exists(a.attr_cache):
        attrs = {k: np.array(v) for k, v in json.load(open(a.attr_cache)).items()}
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        texts = [tok.decode(ids[i, 0], skip_special_tokens=True) for i in range(n_sent)]
        attrs = label_attributes(texts)
        json.dump({k: v.tolist() for k, v in attrs.items()}, open(a.attr_cache, "w"))
    A_sent, names = encode_attrs(attrs, a.min_frac)

    # Encode every sentence in every language
    emb = torch.load(a.embedding_table, map_location="cpu")
    model = build_model(ck["args"], emb, meta).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    model = model.to(a.device)
    q = model.quantizer
    codes, zq, sent_of, lang_of = [], [], [], []
    for l in range(n_langs):
        for s in range(0, n_sent, a.batch_size):
            seqs = [ids[i, l] for i in range(s, min(s + a.batch_size, n_sent))]
            T = max(len(x) for x in seqs)
            bt = torch.full((len(seqs), T), pad, dtype=torch.long)
            for i, x in enumerate(seqs):
                bt[i, :len(x)] = torch.as_tensor(x)
            c = encode_all_levels(model, bt.to(a.device))          # (B, M, L)
            # cumulative quantised vector per level, pooled over slots: (B, L, d_code)
            vecs = torch.stack([cb[c[..., j]] for j, cb in enumerate(q.codebooks)], 2)
            zq.append(vecs.cumsum(2).mean(1).float().cpu())
            codes.append(c.cpu())
            sent_of += list(range(s, s + len(seqs)))
            lang_of += [l] * len(seqs)
    C = torch.cat(codes).numpy()
    ZQ = torch.cat(zq).numpy()
    sent_of, lang_of = np.array(sent_of), np.array(lang_of)
    A = A_sent[sent_of]
    N, M, L = C.shape
    K = q.k
    meaning = meaning_embeddings([ids[i, 0] for i in range(n_sent)], pad, a.device,
                                 a.semantic_model)[sent_of]
    del model
    if a.device == "cuda":
        torch.cuda.empty_cache()

    def shuffled_A():
        return A_sent[rng.permutation(n_sent)][sent_of]   # same shuffle for all languages

    res = {"checkpoint": a.checkpoint, "step": ck.get("step"), "N": int(N),
           "attributes": [[k, v] for k, v in names], "depth": {}}
    lines = [f"Compositionality: {a.checkpoint} ({ck.get('step')} steps)",
             f"{N} encodings = {n_sent} FLORES sentences x {n_langs} langs; {M} slots x {L} levels",
             "attributes (English spaCy/keywords, copied across languages):"]
    for k, v in names:
        share = np.bincount(A_sent[:, [n for n, _ in names].index(k)]) / n_sent
        lines.append(f"  {k:12s} " + ", ".join(f"{x}:{s:.2f}" for x, s in zip(v, share)))

    hdr = (f"\n{'view':7s} {'d':>2s} | {'posdis':>6s} {'(shuf)':>7s} | {'bosdis':>6s} {'(shuf)':>7s} | "
           f"{'MIGpos':>6s} {'MIGbag':>6s} {'(shuf)':>13s} | {'topsim-bag':>10s} {'topsim-pos':>10s} "
           f"{'(perm)':>7s} | {'TRE':>5s} {'(perm)':>6s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for view in ("level", "prefix"):
        for d in range(1, L + 1):
            sym = C[:, :, d - 1] if view == "level" else prefix_symbols(C, d)
            cnt = symbol_counts(sym, a.max_symbols)
            posdis, migpos = disentanglement(sym, A)
            bosdis, migbag = disentanglement(cnt, A)
            sh = []
            for _ in range(a.n_shuffles):
                As = shuffled_A()
                (pd_, mp_), (bd_, mb_) = disentanglement(sym, As), disentanglement(cnt, As)
                sh.append((pd_, bd_, mp_, mb_))
            sh = np.mean(sh, 0)
            ts_bag = topsim(sym, meaning, a.n_pairs, rng, positional=False)
            ts_pos = topsim(sym, meaning, a.n_pairs, rng, positional=True)
            ts_perm = topsim(sym, meaning[rng.permutation(N)], a.n_pairs, rng, positional=False)
            if view == "prefix":
                tr, tr_perm = tre(ZQ[:, d - 1], A, names, sent_of, rng)
            else:
                tr = tr_perm = float("nan")
            r = dict(posdis=posdis, posdis_shuf=sh[0], bosdis=bosdis, bosdis_shuf=sh[1],
                     mig_pos=migpos, mig_bag=migbag, mig_pos_shuf=sh[2], mig_bag_shuf=sh[3],
                     topsim_bag=ts_bag, topsim_pos=ts_pos, topsim_perm=ts_perm,
                     tre=tr, tre_perm=tr_perm)
            res["depth"][f"{view}{d}"] = {k: float(v) for k, v in r.items()}
            lines.append(f"{view:7s} {d:2d} | {posdis:6.4f} {sh[0]:7.4f} | {bosdis:6.4f} {sh[1]:7.4f} | "
                         f"{migpos:6.4f} {migbag:6.4f} {sh[2]:6.4f}/{sh[3]:6.4f} | {ts_bag:10.3f} "
                         f"{ts_pos:10.3f} {ts_perm:7.3f} | {tr:5.3f} {tr_perm:6.3f}")
            print(lines[-1], flush=True)

    # nPMI stability across languages (level 0)
    stab, npmi = npmi_stability(C[:, :, 0], A, names, lang_of, n_langs, K)
    stab_sh, _ = npmi_stability(C[:, :, 0], shuffled_A(), names, lang_of, n_langs, K)
    res["npmi_stability"], res["npmi_stability_shuf"] = stab, stab_sh
    lines.append(f"\nnPMI(level-0 code, attribute value) cross-language stability: "
                 f"mean pairwise r = {stab:.3f} (attribute-shuffled {stab_sh:.3f})")
    lines.append("top level-0 codes per attribute value (nPMI, averaged over languages):")
    col = 0
    for k, vals in names:
        for v in vals:
            top = np.argsort(-np.nan_to_num(npmi[:, col], nan=-9))[:5]
            lines.append(f"  {k}={v:8s} " + ", ".join(f"{c}:{npmi[c, col]:+.2f}" for c in top))
            col += 1

    # A6: stack-level interlingua
    lines.append("\nA6 stack-level interlingua (distinct stack prefixes, all 45 language pairs):")
    lines.append(f"{'depth':>5s} | {'real':>6s} {'chance':>6s} {'lift':>6s} {'kappa':>6s} | "
                 f"{'slot-agree':>10s} {'chance':>6s} {'kappa':>6s}")
    res["stack_interlingua"] = {}
    for d in range(1, L + 1):
        s = stack_interlingua(C, sent_of, lang_of, n_langs, n_sent, d)
        res["stack_interlingua"][d] = s
        lines.append(f"{d:5d} | {s['real']:6.3f} {s['chance']:6.3f} {s['lift']:+6.3f} {s['kappa']:6.3f} | "
                     f"{s['pos_real']:10.3f} {s['pos_chance']:6.3f} {s['pos_kappa']:6.3f}")

    txt = "\n".join(lines) + "\n"
    print(txt)
    with open(os.path.join(a.outdir, f"compositionality_{name}.txt"), "w") as f:
        f.write(txt)
    with open(os.path.join(a.outdir, f"compositionality_{name}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
