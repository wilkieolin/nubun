"""substitution_test.py — Phase 9 C2: does editing level-0 codes change what the
decoder says, and only that?

Two modes, both on the minimal-pair shards (consecutive rows A, B differ in one
factor, negation or tense; `augment/README.md`):

  transplant  Encode A and B. In A's codes, set the level-0 slots where A and B
              differ to B's code, and decode. If the factor lives in level 0,
              the output should take B's polarity / tense and keep the rest of A.
              Branch 2's success criterion. Variants:
                level0  B's level-0 code, A's deeper levels kept
                stack   B's whole stack at those slots
                control the same slots, random level-0 codes from the marginal
  radical     --radical C: insert level-0 code C into sentences without it, at
              the slots C usually holds; remove it from sentences with it (each
              C becomes the sentence's most frequent other level-0 code).
              Control: a random active code at the same slots. The A1 probe's
              named radicals (seed 2: 5 negation, 102 today) are the targets.

  enrich      No decoding: which level-0 codes mark negated sentences (English-side
              label), by presence in negated vs affirmative members. Use it to pick
              --radical per model; radicals found at one padding need not hold at another.

Scoring:
  - Negation, lexical: a negator list per language. Each member's polarity comes
    from its English side; pairs where exactly one member is negated are kept.
    The detector's accuracy on the reference texts is printed per language, so
    a weak detector (zh, ja substrings) is visible.
  - Any factor, MiniLM: progress = (1 + (cos(g, gB) - cos(g, gA)) / (1 - cos(gA, gB))) / 2,
    against the model's own reconstructions gA, gB: 0 = still A, 1 = became B,
    0.5 = as close to one as the other. ">0.5" is the share closer to B.
    Pairs whose reconstructions are near-identical (1 - cos < 0.02) are skipped.

Rows are padded as the checkpoint trained (code_usage_audit.train_pad_len).

  python substitution_test.py --checkpoint ckpt.eval.pt --minpairs-dir data/minpairs_npy
  python substitution_test.py --checkpoint ckpt.eval.pt --minpairs-dir data/minpairs_npy \
      --mode radical --radical 5
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

import vqvae_select
from code_usage_audit import build_model, encode_all_levels, pad_batch, train_pad_len
from roundtrip_eval import embed, generate
from vqvae.data import load_corpus

FACTORS = ("negation", "tense")

# Whole words, lowercased; "n't" and French "n'" are handled in negated().
NEGATORS = {
    "en": {"not", "no", "never", "nobody", "nothing", "none", "neither", "nor",
           "cannot", "nowhere"},
    "de": {"nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines", "nie",
           "niemals", "niemand", "nichts", "weder"},
    "fr": {"ne", "pas", "jamais", "rien", "aucun", "aucune", "ni"},
    "es": {"no", "nunca", "jamás", "nada", "nadie", "ningún", "ninguno", "ninguna",
           "tampoco", "ni"},
    "pt": {"não", "nunca", "jamais", "nada", "ninguém", "nenhum", "nenhuma", "nem"},
    "ru": {"не", "нет", "никогда", "ничего", "никто", "ни"},
    "ar": {"لا", "لم", "لن", "ليس", "ليست", "غير"},
    "hi": {"नहीं", "न", "मत"},
}
# Substrings, for languages without spaces. Noisy (不同 "different", 少ない "few").
NEG_SUBSTR = {"zh": ("不", "没", "沒", "未", "别", "無", "无"),
              "ja": ("ない", "ません", "なかった", "ず", "ぬ")}


def negated(text, lang):
    t = text.lower()
    if lang in NEG_SUBSTR:
        return any(s in t for s in NEG_SUBSTR[lang])
    if lang == "en" and re.search(r"n\s?['’´\u0301]+\s?t\b", t):     # n't, and the decoder's "don ́t"
        return True
    if lang == "fr" and re.search(r"\bn['’]", t):
        return True
    return any(w in NEGATORS.get(lang, ()) for w in re.findall(r"\w+", t))


def load_shards(minpairs_dir, langs):
    """{lang: dict of arrays}, packed <lang>_en.npz or unpacked <lang>_en/."""
    out = {}
    for lang in langs:
        npz, d = os.path.join(minpairs_dir, f"{lang}_en.npz"), os.path.join(minpairs_dir, f"{lang}_en")
        if os.path.exists(npz):
            out[lang] = dict(np.load(npz))
        elif os.path.isdir(d):
            out[lang] = {k[:-4]: np.load(os.path.join(d, k)) for k in os.listdir(d) if k.endswith(".npy")}
    return out


class Model:
    """Encode to RVQ codes, rebuild z_q from codes, decode to ids and text."""

    def __init__(self, a, meta):
        ck = vqvae_select.load_checkpoint(a.checkpoint)
        emb = torch.load(a.embedding_table, map_location="cpu")
        self.net = build_model(ck["args"], emb, meta).eval()
        self.net.load_state_dict(ck["model_state"], strict=True)
        self.net = self.net.to(a.device)
        if not hasattr(self.net.quantizer, "codebooks"):
            raise SystemExit("substitution_test needs an RVQ checkpoint")
        if ck["args"].get("use_stop_mask") or ck["args"].get("compression_ratio"):
            raise SystemExit("stop masks / length caps are not handled; wafer models use neither")
        self.cbs = list(self.net.quantizer.codebooks)
        self.meta, self.dev, self.a = meta, a.device, a
        self.T = train_pad_len(ck["args"], a.pad_to)
        self.step = ck.get("step")

    def batch(self, seqs):
        return pad_batch(seqs, self.meta.pad_token_id, self.meta.eos_token_id, self.T).to(self.dev)

    @torch.no_grad()
    def codes(self, seqs):
        bs = self.a.batch_size
        return torch.cat([encode_all_levels(self.net, self.batch(seqs[s:s + bs]))
                          for s in range(0, len(seqs), bs)]).cpu()       # (N, M, L)

    def z_q(self, codes):
        codes = codes.to(self.dev)
        return sum(cb[codes[..., l]] for l, cb in enumerate(self.cbs))

    @torch.no_grad()
    def check(self, seqs):
        """z_q rebuilt from codes must match the quantizer's own output."""
        src = self.batch(seqs[:8])
        zq = self.net.quantizer(self.net.encoder(src))[0]
        err = (zq - self.z_q(encode_all_levels(self.net, src))).abs().max().item()
        if err > 1e-3:
            raise SystemExit(f"z_q rebuilt from codes differs from the quantizer by {err:.2e}")

    @torch.no_grad()
    def decode(self, codes, lang):
        """codes (N, M, L) -> list of id tensors (eos-trimmed), in lang."""
        m = self.meta
        li = m.short_codes.index(lang)
        out = []
        for s in range(0, len(codes), self.a.batch_size):
            zq = self.z_q(codes[s:s + self.a.batch_size])
            mask = torch.ones(zq.shape[:2], dtype=torch.bool, device=self.dev)
            lt = torch.full((zq.size(0),), li, dtype=torch.long, device=self.dev)
            out.append(generate(self.net, zq, mask, lt, m.bos_token_id, m.eos_token_id,
                                m.pad_token_id, self.a.max_gen_len, self.dev).cpu())
        rows = []
        for ids in out:
            rows += [r for r in ids]
        return rows


class Scorer:
    def __init__(self, a, meta):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.semantic_model)
        self.st = AutoModel.from_pretrained(a.semantic_model).eval().to(a.device)
        self.pad, self.dev, self.bs = meta.pad_token_id, a.device, a.batch_size

    def text(self, ids):
        return [self.tok.decode(r.tolist(), skip_special_tokens=True).strip() for r in ids]

    @torch.no_grad()
    def emb(self, ids):
        out = []
        for s in range(0, len(ids), self.bs):
            ch = ids[s:s + self.bs]
            bt = torch.full((len(ch), max(len(r) for r in ch)), self.pad, dtype=torch.long)
            for i, r in enumerate(ch):
                bt[i, :len(r)] = r
            out.append(F.normalize(embed(self.st, bt.to(self.dev), self.pad).float(), dim=-1).cpu())
        return torch.cat(out)


def progress(e, eA, eB, min_gap=0.02):
    """0 at A's reconstruction, 1 at B's; NaN where A and B decode alike."""
    gap = 1 - (eA * eB).sum(-1)
    p = (1 + ((e * eB).sum(-1) - (e * eA).sum(-1)) / gap.clamp(min=1e-6)) / 2
    return torch.where(gap > min_gap, p, torch.full_like(p, float("nan"))).numpy()


def summarize(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return (float(np.median(v)), float(np.mean(v > 0.5)), len(v)) if len(v) else (float("nan"),) * 2 + (0,)


def select_pairs(d, factor_id, T, n, rng):
    """First rows of pairs of this factor whose members both fit T on both sides."""
    mem, fac = d["member"], d["factor"]
    first = np.flatnonzero((mem == 0) & (fac == factor_id))
    if T:
        ln = np.maximum(np.diff(d["src_offsets"]), np.diff(d["tgt_offsets"]))
        first = first[(ln[first] <= T) & (ln[first + 1] <= T)]
    return np.sort(rng.choice(first, min(n, len(first)), replace=False))


def rows(d, side, idx):
    tok, off = (d["src_tokens"], d["src_offsets"]) if side == "x" else (d["tgt_tokens"], d["tgt_offsets"])
    return [tok[off[i]:off[i + 1]].astype(np.int64) for i in idx]


def run_transplant(a, model, sc, shards, rng, report):
    marg = None
    for lang, d in shards.items():
        side_lang = lang if a.side == "x" else "en"
        tgts = [side_lang if t == "src" else t for t in a.tgt_langs.split(",")]
        for fi, f in enumerate(FACTORS):
            first = select_pairs(d, fi, model.T, a.n_pairs, rng)
            if not len(first):
                continue
            cA, cB = model.codes(rows(d, a.side, first)), model.codes(rows(d, a.side, first + 1))
            if marg is None:
                vals = torch.cat([cA, cB])[..., 0].flatten().numpy()
                marg = np.bincount(vals, minlength=model.cbs[0].shape[0]).astype(float)
                marg /= marg.sum()
            # English-side polarity labels (negation only)
            en = {m: [sc.tok.decode(r.tolist(), skip_special_tokens=True)
                      for r in rows(d, "en", first + m)] for m in (0, 1)}
            lab = {m: np.array([negated(t, "en") for t in en[m]]) for m in (0, 1)}
            clean = lab[0] != lab[1]
            recon = {}                                                  # (member, tgt) -> ids, emb, text

            def rec(m, tl):
                if (m, tl) not in recon:
                    g = model.decode((cA, cB)[m], tl)
                    recon[m, tl] = (g, sc.emb(g), sc.text(g))
                return recon[m, tl]
            for direction, (src, dst) in {"A->B": (cA, cB), "B->A": (cB, cA)}.items():
                diff = src[..., 0] != dst[..., 0]                       # (N, M)
                conds = {}
                c = src.clone(); c[..., 0] = torch.where(diff, dst[..., 0], src[..., 0]); conds["level0"] = c
                c = src.clone(); c[diff] = dst[diff]; conds["stack"] = c
                rnd = torch.from_numpy(rng.choice(len(marg), size=diff.shape, p=marg))
                c = src.clone(); c[..., 0] = torch.where(diff, rnd, src[..., 0]); conds["control"] = c
                n_diff = diff.sum(1).numpy()
                for tl in tgts:
                    key = (lang, f, direction, tl)
                    (_, eA, tA), (_, eB, tB) = rec(direction != "A->B", tl), rec(direction == "A->B", tl)
                    r = {"n": int(len(first)), "slots_changed_median": float(np.median(n_diff)),
                         "identical_level0": float(np.mean(n_diff == 0))}
                    if f == "negation":
                        la, lb = (lab[0], lab[1]) if direction == "A->B" else (lab[1], lab[0])
                        ok = clean & (np.array([negated(t, tl) for t in tA]) == la) \
                                   & (np.array([negated(t, tl) for t in tB]) == lb)
                        r["neg_clean"], r["neg_decodable"] = int(clean.sum()), int(ok.sum())
                    for cn, cc in conds.items():
                        g = model.decode(cc, tl)
                        e = sc.emb(g)
                        r[f"{cn}_progress"] = summarize(progress(e, eA, eB))
                        r[f"{cn}_keep_A"] = float((e * eA).sum(-1).mean())
                        if f == "negation":
                            hit = np.array([negated(t, tl) for t in sc.text(g)]) == lb
                            r[f"{cn}_flip"] = float(hit[ok].mean()) if ok.any() else float("nan")
                        if cn == "level0":
                            r["examples"] = [{"A": x, "B": y, "level0": z} for x, y, z in
                                             list(zip(tA, tB, sc.text(g)))[:a.n_examples]]
                    report.append({"lang": lang, "factor": f, "direction": direction, "tgt": tl, **r})
                    print_row(report[-1])


def run_radical(a, model, sc, shards, rng, report):
    C = a.radical
    for lang, d in shards.items():
        side_lang = lang if a.side == "x" else "en"
        tgts = [side_lang if t == "src" else t for t in a.tgt_langs.split(",")]
        idx = np.concatenate([select_pairs(d, fi, model.T, a.n_pairs, rng) for fi in range(len(FACTORS))])
        idx = np.concatenate([idx, idx + 1])
        codes = model.codes(rows(d, a.side, idx))
        l0 = codes[..., 0]
        has = (l0 == C).any(1)
        if not has.any():
            print(f"{lang}: code {C} never appears at level 0; skipped")
            continue
        slot_rate = (l0[has] == C).float().mean(0)                          # (M,)
        slots = slot_rate >= a.slot_frac
        slots[slot_rate.argmax()] = True
        active = np.flatnonzero(np.bincount(l0.flatten().numpy(), minlength=model.cbs[0].shape[0]))
        active = active[active != C]
        edits = {}
        ins = codes[~has].clone(); ins[:, slots, 0] = C; edits["insert"] = (~has, ins)
        ctl = codes[~has].clone()
        ctl[:, slots, 0] = torch.from_numpy(rng.choice(active, size=(len(ctl), 1))).expand(-1, int(slots.sum()))
        edits["insert_control"] = (~has, ctl)
        rem = codes[has].clone()
        for i in range(len(rem)):
            other = rem[i, :, 0][rem[i, :, 0] != C]
            fill = int(torch.mode(other).values) if len(other) else int(active[0])
            rem[i, :, 0][rem[i, :, 0] == C] = fill
        edits["remove"] = (has, rem)
        for tl in tgts:
            for en_name, (sel, cc) in edits.items():
                if not sel.any():
                    continue
                base = model.decode(codes[sel], tl)
                g = model.decode(cc, tl)
                tb, tg = sc.text(base), sc.text(g)
                nb = np.array([negated(t, tl) for t in tb])
                ng = np.array([negated(t, tl) for t in tg])
                # insert: of outputs affirmative before, share negated after; remove: the reverse
                base_ok = ~nb if en_name.startswith("insert") else nb
                r = {"lang": lang, "tgt": tl, "edit": en_name, "code": C, "n": int(sel.sum()),
                     "slots": np.flatnonzero(slots.numpy()).tolist(),
                     "neg_before": float(nb.mean()), "neg_after": float(ng.mean()),
                     "n_base": int(base_ok.sum()),
                     "changed": float((ng != nb)[base_ok].mean()) if base_ok.any() else float("nan"),
                     "keep": float((sc.emb(g) * sc.emb(base)).sum(-1).mean()),
                     "unchanged_text": float(np.mean([x == y for x, y in zip(tb, tg)])),
                     "examples": [{"before": x, "after": y} for x, y in list(zip(tb, tg))[:a.n_examples]]}
                report.append(r)
                verb = "negated" if en_name.startswith("insert") else "un-negated"
                print(f"{lang:3s} ->{tl:3s} {en_name:15s} n={r['n']:4d} slots={len(r['slots']):2d}  "
                      f"{verb} {r['changed']:4.0%} of {r['n_base']:3d}  (negated {r['neg_before']:.2f} -> "
                      f"{r['neg_after']:.2f})  keep {r['keep']:.3f}  text unchanged {r['unchanged_text']:.0%}")


def run_enrich(a, model, sc, shards, rng, report):
    """Level-0 codes by presence in negated vs affirmative members (clean pairs)."""
    K = model.cbs[0].shape[0]
    neg_p, aff_p = [], []
    for lang, d in shards.items():
        first = select_pairs(d, 0, model.T, a.n_pairs, rng)
        lab = {m: np.array([negated(sc.tok.decode(r.tolist(), skip_special_tokens=True), "en")
                            for r in rows(d, "en", first + m)]) for m in (0, 1)}
        keep = lab[0] != lab[1]
        for m in (0, 1):
            c = model.codes(rows(d, a.side, first[keep] + m))[..., 0]
            pres = torch.zeros(len(c), K, dtype=torch.bool).scatter_(1, c, True).numpy()
            neg_p.append(pres[lab[m][keep]]); aff_p.append(pres[~lab[m][keep]])
    N, A = np.concatenate(neg_p), np.concatenate(aff_p)
    pn, pa = N.mean(0), A.mean(0)
    lift = (pn + 1e-3) / (pa + 1e-3)
    order = np.argsort(-lift)[:a.n_examples * 2]
    cover = N[:, order].cumsum(1) > 0
    print(f"negation radicals: {len(N)} negated, {len(A)} affirmative sentences")
    for i, j in enumerate(order):
        print(f"  code {j:3d}: in {pn[j]:5.1%} of negated, {pa[j]:5.1%} of affirmative, "
              f"lift {lift[j]:5.1f}; top {i + 1} cover {cover[:, i].mean():.0%} of negated")
        report.append({"code": int(j), "neg": float(pn[j]), "aff": float(pa[j]), "lift": float(lift[j]),
                       "cum_cover": float(cover[:, i].mean())})


def print_row(r):
    s = (f"{r['lang']:3s} {r['factor']:8s} {r['direction']} ->{r['tgt']:3s} n={r['n']:4d} "
         f"slots {r['slots_changed_median']:4.0f} (=0 {r['identical_level0']:4.0%}) | progress median/>0.5:")
    for cn in ("level0", "stack", "control"):
        m, frac, _ = r[f"{cn}_progress"]
        s += f" {cn} {m:+.2f}/{frac:3.0%}"
    if r["factor"] == "negation":
        s += f" | flip ({r['neg_decodable']}/{r['neg_clean']} decodable):"
        for cn in ("level0", "stack", "control"):
            s += f" {cn} {r[cn + '_flip']:.0%}"
    print(s)


def calibrate(sc, shards):
    """Negation detector vs English-side labels, on reference text per language."""
    print("negation detector on references (accuracy vs English-side polarity, clean pairs):")
    for lang, d in shards.items():
        nf = np.flatnonzero((d["member"] == 0) & (d["factor"] == 0))[:300]
        if not len(nf):
            continue
        idx = np.concatenate([nf, nf + 1])
        en = [negated(sc.tok.decode(r.tolist(), skip_special_tokens=True), "en") for r in rows(d, "en", idx)]
        en = np.array(en)
        clean = np.concatenate([en[:len(nf)] != en[len(nf):]] * 2)
        x = np.array([negated(sc.tok.decode(r.tolist(), skip_special_tokens=True), lang)
                      for r in rows(d, "x", idx)])
        print(f"  {lang}: {np.mean(x[clean] == en[clean]):.1%} on {clean.sum()} rows "
              f"({np.mean(clean[:len(nf)]):.0%} of pairs clean on the English side)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--minpairs-dir", required=True)
    p.add_argument("--mode", choices=["transplant", "radical", "enrich"], default="transplant")
    p.add_argument("--radical", type=int, default=None, help="level-0 code for --mode radical")
    p.add_argument("--slot-frac", type=float, default=0.5,
                   help="radical: insert at slots holding the code in >= this share of its sentences")
    p.add_argument("--langs", default="de,fr,es,ru,ja", help="minimal-pair shards to read")
    p.add_argument("--side", choices=["x", "en"], default="x", help="which side to encode")
    p.add_argument("--tgt-langs", default="en,src", help="decode into these; src = the encoded side's language")
    p.add_argument("--n-pairs", type=int, default=100, help="per factor per shard")
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--max-gen-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pad-to", type=int, default=None,
                   help="row length; default the training format (64 for CS-3, batch max for GPU); 0 = batch max")
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON report")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.mode == "radical" and a.radical is None:
        p.error("--mode radical needs --radical")

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    _, _, _, _, meta = load_corpus(a.corpus)
    model = Model(a, meta)
    sc = Scorer(a, meta)
    shards = load_shards(a.minpairs_dir, a.langs.split(","))
    if not shards:
        raise SystemExit(f"no shards for {a.langs} in {a.minpairs_dir}")
    first = next(iter(shards.values()))
    model.check(rows(first, a.side, np.arange(8)))
    print(f"{a.checkpoint} (step {model.step})  mode={a.mode}  side={a.side}  "
          f"pad_to={model.T or 'batch max'}  shards={','.join(shards)}")
    report = []
    if a.mode != "enrich":
        calibrate(sc, shards)
    {"transplant": run_transplant, "radical": run_radical, "enrich": run_enrich}[a.mode](
        a, model, sc, shards, rng, report)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"checkpoint": a.checkpoint, "args": vars(a), "rows": report}, fh, indent=1,
                      ensure_ascii=False)


if __name__ == "__main__":
    main()
