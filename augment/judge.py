"""judge.py — LLM-judge quorum and planted-defect calibration for minimal pairs (augment/README.md §10).

The automatic checks of minpairs.py (spaCy flip/keep, negation regexes, back-translation) have
unknown recall and no faithfulness check. This adds:

  build    item pool in <work>/judge/items.jsonl: every real English edit and translation with
           its automatic verdict, plus planted defects made from auto-accepted pairs (an LLM
           corrupts them: no_flip, extra_change, subject_change, ungrammatical, mistranslation;
           mismatch swaps in another pair's translation). Planted translations are
           back-translated so the automatic checks can be scored on them too.
  run      one judge model over every item (blind to origin, items shuffled, 10 per request)
  score    per judge and for the quorum (>= 2 of 3 fail): catch rate on planted defects by
           defect and language, pass rates and agreement with the automatic checks on real
           items; writes judge/judged.jsonl and judge/score.json
  sample   ~50 real items for the human check of the quorum -> judge/sample.json

The planting model should be the generator (its own mistakes are what we want to catch); the
judges should not be (a model approves its own output).

  P="python augment/judge.py --work augment/work/muse_e2e"
  $P build --edit-model muse-spark-bf16 --translate-model muse-spark-bf16 \\
           --backtrans-model meta-llama/Meta-Llama-3.1-70B-Instruct --plant-model muse-spark-bf16
  $P run --model openai/gpt-oss-120b --cluster sophia
  $P run --model google/gemma-4-31B-it --cluster sophia
  $P run --model inkling-bf16 --cluster minerva
  $P score --judges openai/gpt-oss-120b,google/gemma-4-31B-it,inkling-bf16
  $P sample --judges ...
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minpairs as mp  # noqa: E402

DEFECTS_EN = ["no_flip", "extra_change", "subject_change", "ungrammatical"]
DEFECTS_TR = DEFECTS_EN + ["mistranslation", "mismatch"]
JUDGE_CHUNK = 10
PLANT_CHUNK = 10

FACTOR_TEXT = {
    "negation": ("polarity", "one member is negated, the other affirmative",
                 ", and the same tense",
                 "do-support (go -> do not go), contractions, any/anyone/anything -> "
                 "some/someone/something"),
    "tense": ("tense", "one member is present tense, the other past tense",
              ", and the same polarity (negated or not)",
              "verb agreement, and sequence of tense in dependent clauses "
              "(I promise I won't -> I promised I wouldn't)"),
}

PLANT_SYSTEM = """You help test a quality checker for minimal-pair data. Each item gives a pair A, B in {language} that should differ only in {fname} ({fdesc}; "relation" says which way), and a defect to introduce. Return a corrupted version of the pair that contains exactly that defect and is otherwise as close as possible to the original. Keep it realistic: the kind of mistake a careless annotator or a weak translation model makes, not obvious garbling.

Defects:
- no_flip: rewrite B so it has the same {fname} as A (undo the flip), but make one small other change (a synonym, a different determiner) so B is not identical to A. Keep A unchanged.
- extra_change: keep the flip, but also change one content word in B (a noun, name, number or adjective) to a word with a different meaning. Keep A unchanged.
- subject_change: keep the flip, but also change B's grammatical subject in person or number (I -> we, he -> they), adjusting agreement. Keep A unchanged.
- ungrammatical: keep the flip, but introduce one grammatical error in B that a fluent speaker would notice (agreement, verb form, particle or word order). Keep A unchanged.
- mistranslation: change the meaning of both A and B in the same way (a different verb or object), keeping the flip between them, so they no longer translate the English pair EN_A, EN_B.

Output only the JSON object {{"items": [{{"id": <id>, "A": <sentence>, "B": <sentence>}}, ...]}}, one item per input id. No commentary."""

JUDGE_EN = """You audit English minimal pairs for a dataset. In each item, B should be A with exactly one change in {fname} ({fdesc}); "relation" says which way.
For each item decide:
- "flipped": B differs from A in {fname} exactly as the relation states, in the main clause.
- "minimal": apart from that flip, B means the same as A: same subject (person and number), same content words, names, numbers and places{other}. Changes the flip forces are fine: {allowed}.
- "grammatical": B is grammatical, natural English, at least as well formed as A.
Be strict: these pairs teach a model what {fname} is, so a pair with any other difference is useless.

Output only the JSON object {{"items": [{{"id": <id>, "flipped": true|false, "minimal": true|false, "grammatical": true|false, "problem": <a few words, or null>}}, ...]}}, one item per input id, in input order. No commentary."""

JUDGE_TR = """You audit {language} translations of English minimal pairs for a dataset. Each item gives an English pair EN_A, EN_B that differ only in {fname} ({fdesc}; "relation" says which way), and a {language} pair A, B.
For each item decide:
- "faithful": A translates EN_A and B translates EN_B (meaning, not word for word).
- "flipped": B differs from A in {fname} as the relation states, with the language's normal grammatical device{device}.
- "minimal": apart from that flip, A and B are worded identically, except for changes the flip forces in {language} (agreement, particles, word order).
- "grammatical": A and B are both grammatical, natural {language}.
Be strict: these pairs teach a model what {fname} is, so a pair with any other difference is useless.

Output only the JSON object {{"items": [{{"id": <id>, "faithful": true|false, "flipped": true|false, "minimal": true|false, "grammatical": true|false, "problem": <a few words, or null>}}, ...]}}, one item per input id, in input order. No commentary."""

DEVICE_NOTE = {("tense", "ja"): " (the past form of the main predicate: た, ました, でした, かった)"}
CRITERIA_EN = ("flipped", "minimal", "grammatical")
CRITERIA_TR = ("faithful",) + CRITERIA_EN


def lang_name(lang):
    return "English" if lang == "en" else mp.LANG_NAMES[lang]


def jdir(work, *parts):
    os.makedirs(work.p("judge"), exist_ok=True)
    return work.p("judge", *parts)


def digest(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def chunked(xs, n):
    return [xs[k:k + n] for k in range(0, len(xs), n)]


def send(work, name, items, model, args, max_tokens):
    """items: [(custom_id, body)] -> run interactively, return {custom_id: content}."""
    paths = mp.write_input(work, name, items, "openai", {"stage": name.split(".")[0], "model": model})
    mp.run_interactive(work, name, paths, model, args.cluster, args.workers, args.timeout)
    return mp.load_results(work, name)


def parsed_items(content):
    obj = mp.parse_json_answer(content)
    items = obj.get("items") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
    out = {}
    for it in items or []:
        if isinstance(it, dict) and "id" in it:
            try:
                out[int(it["id"])] = it
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------- build
def real_items(args, work):
    items = []
    for factor in mp.FACTORS:
        for r in mp.read_jsonl(work.p("verified", f"edit.{factor}.{mp.slug(args.edit_model)}.jsonl")):
            if not (isinstance(r.get("b"), str) and r["b"].strip()):
                continue                                   # skips: nothing to judge
            items.append({"iid": f"R:en:{factor}:{r['pair_id']}", "origin": "real", "defect": None,
                          "factor": factor, "direction": r["direction"], "lang": "en",
                          "pair_id": r["pair_id"], "en_a": r["a"], "en_b": r["b"],
                          "a": r["a"], "b": r["b"],
                          "auto": {"accepted": r["accepted"], "reason": r["reason"]}})
        bt_path = work.p("verified", mp.stage_name("bt", factor, args.backtrans_model,
                                                   args.translate_model) + ".jsonl")
        bt = {(r["pair_id"], r["lang"]): r for r in mp.read_jsonl(bt_path)}
        for r in mp.accepted_translations(work, factor, args.translate_model):
            for lang, v in r["langs"].items():
                if not (isinstance(v.get("a"), str) and isinstance(v.get("b"), str)):
                    continue
                b = bt.get((r["pair_id"], lang))
                if not v["accepted"]:
                    auto = {"accepted": False, "reason": v["reason"]}
                elif b is None:
                    auto = {"accepted": True, "reason": None, "bt": None}
                else:
                    auto = {"accepted": b["accepted"], "reason": b["reason"] and "bt:" + b["reason"],
                            "bt": {"a": b["a"], "b": b["b"]}}
                items.append({"iid": f"R:{lang}:{factor}:{r['pair_id']}", "origin": "real",
                              "defect": None, "factor": factor, "direction": r["direction"],
                              "lang": lang, "pair_id": r["pair_id"],
                              "en_a": r["en"]["a"], "en_b": r["en"]["b"],
                              "a": v["a"], "b": v["b"], "auto": auto})
    return items


def auto_translation(factor, direction, lang, a, b, min_ratio=0.6):
    """The translate-stage checks of minpairs.verify_tr for one language."""
    ok, reason, _ = mp.verify_translation(factor, direction, a, b, min_ratio)
    if ok and factor == "negation":
        neg, aff = (b, a) if direction == "add" else (a, b)
        if mp.NEG_COUNTERS[lang](neg) < mp.NEG_COUNTERS[lang](aff) + 1:
            ok, reason = False, "neg_marker"
    return ok, reason


def cmd_build(args, work):
    rng = random.Random(args.seed)
    real = real_items(args, work)
    slots = defaultdict(list)
    for it in real:
        if it["auto"]["accepted"]:
            slots[(it["factor"], it["lang"])].append(it)
    # planted: distinct source pairs per (factor, lang); each defect gets args.per_defect
    plan = []
    for (factor, lang), pool in sorted(slots.items()):
        defects = DEFECTS_EN if lang == "en" else DEFECTS_TR
        src = rng.sample(pool, min(len(pool), args.per_defect * len(defects)))
        for k, it in enumerate(src):
            plan.append((defects[k % len(defects)], it))
    planted, to_llm = [], []
    for defect, it in plan:
        p = {**{k: it[k] for k in ("factor", "direction", "lang", "pair_id", "en_a", "en_b")},
             "iid": f"P:{defect}:{it['lang']}:{it['factor']}:{it['pair_id']}", "origin": "planted",
             "defect": defect, "a": it["a"], "b": it["b"], "orig": {"a": it["a"], "b": it["b"]}}
        if defect == "mismatch":
            other = rng.choice([o for o in slots[(it["factor"], it["lang"])]
                                if o["pair_id"] != it["pair_id"]])
            p["a"], p["b"] = other["a"], other["b"]
            planted.append(p)
        else:
            to_llm.append(p)
    # LLM corruption, grouped per (factor, lang) so the system prompt names the language
    groups = defaultdict(list)
    for p in to_llm:
        groups[(p["factor"], p["lang"])].append(p)
    reqs, slot_of = [], {}
    for (factor, lang), ps in sorted(groups.items()):
        fname, fdesc, _, _ = FACTOR_TEXT[factor]
        system = PLANT_SYSTEM.format(language=lang_name(lang), fname=fname, fdesc=fdesc)
        for k, ch in enumerate(chunked(ps, PLANT_CHUNK)):
            cid = f"plant:{factor}:{lang}:{k}"
            user = [{"id": j, "defect": p["defect"],
                     "relation": mp.FACTOR_GLOSS[(factor, p["direction"])], "A": p["a"], "B": p["b"],
                     **({"EN_A": p["en_a"], "EN_B": p["en_b"]} if lang != "en" else {})}
                    for j, p in enumerate(ch)]
            reqs.append((cid, mp.chat_body(args.plant_model, system,
                                           json.dumps(user, ensure_ascii=False), 3000, False)))
            slot_of[cid] = ch
    res = send(work, f"plant.all.{mp.slug(args.plant_model)}", reqs, args.plant_model,
               argparse.Namespace(cluster=args.plant_cluster, workers=args.workers,
                                  timeout=args.timeout), 3000)
    dropped = Counter()
    for cid, ch in slot_of.items():
        got = parsed_items(res.get(cid))
        for j, p in enumerate(ch):
            it = got.get(j)
            a, b = (it or {}).get("A"), (it or {}).get("B")
            if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()):
                dropped["no_answer"] += 1
                continue
            a, b = a.strip(), b.strip()
            if p["defect"] == "mistranslation" and a == p["a"]:
                dropped["unchanged"] += 1
                continue
            if p["defect"] != "mistranslation" and (b == p["b"] or b == a):
                dropped["unchanged"] += 1
                continue
            planted.append({**p, "a": a, "b": b})
    print(f"planted {len(planted)} ({len(plan)} planned; dropped {dict(dropped)})")
    # automatic checks on planted items: English verifier; translate checks + back-translation
    en = [p for p in planted if p["lang"] == "en"]
    for factor in mp.FACTORS:
        rs = [p for p in en if p["factor"] == factor]
        mp.run_english_verifier(factor, rs, args.min_cos)
        for p in rs:
            p["auto"] = {"accepted": p.pop("accepted"), "reason": p.pop("reason")}
            for k in ("edit_dist", "edit_budget", "cosine", "attrs_a", "attrs_b"):
                p.pop(k, None)
    tr = [p for p in planted if p["lang"] != "en"]
    bt_reqs = []
    for p in tr:
        ok, reason = auto_translation(p["factor"], p["direction"], p["lang"], p["a"], p["b"])
        p["auto"] = {"accepted": ok, "reason": reason}
        if ok:
            bt_reqs.append((f"pbt:{p['iid']}", mp.chat_body(
                args.backtrans_model, mp.BT_SYSTEM,
                json.dumps({"A": p["a"], "B": p["b"]}, ensure_ascii=False), mp.MAX_TOKENS["bt"], False)))
    res = send(work, f"pbt.all.{mp.slug(args.backtrans_model)}", bt_reqs, args.backtrans_model,
               argparse.Namespace(cluster="sophia", workers=args.workers, timeout=args.timeout), 300)
    for factor in mp.FACTORS:
        chk = []
        for p in tr:
            if p["factor"] != factor or not p["auto"]["accepted"]:
                continue
            obj = mp.parse_json_answer(res.get(f"pbt:{p['iid']}"))
            if not (isinstance(obj, dict) and isinstance(obj.get("A"), str) and isinstance(obj.get("B"), str)):
                p["auto"] = {"accepted": False, "reason": "bt:unparsed"}
                continue
            chk.append({"direction": p["direction"], "a": obj["A"].strip(), "b": obj["B"].strip(), "p": p})
        mp.run_english_verifier(factor, chk, args.min_cos, main_clause=False)
        for c in chk:
            c["p"]["auto"] = {"accepted": c["accepted"], "reason": c["reason"] and "bt:" + c["reason"],
                              "bt": {"a": c["a"], "b": c["b"]}}
    items = real + planted
    mp.write_jsonl(jdir(work, "items.jsonl"), items)
    caught = Counter((p["defect"], not p["auto"]["accepted"]) for p in planted)
    print(f"wrote {jdir(work, 'items.jsonl')}: {len(real)} real, {len(planted)} planted")
    for d in DEFECTS_TR:
        n = caught[(d, True)] + caught[(d, False)]
        if n:
            print(f"  automatic checks catch {d:<15s} {caught[(d, True)]:>4d}/{n}")


# ---------------------------------------------------------------- run
def judge_requests(items, model, seed):
    """Deterministic: (factor, lang) groups, shuffled so planted items mix with real ones."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[(it["factor"], it["lang"])].append(it)
    tag = digest([it["iid"] for it in items] + [seed])
    reqs, mapping = [], {}
    for (factor, lang), its in sorted(groups.items()):
        its = sorted(its, key=lambda it: it["iid"])
        rng.shuffle(its)
        fname, fdesc, other, allowed = FACTOR_TEXT[factor]
        if lang == "en":
            system = JUDGE_EN.format(fname=fname, fdesc=fdesc, other=other, allowed=allowed)
        else:
            system = JUDGE_TR.format(language=lang_name(lang), fname=fname, fdesc=fdesc,
                                     device=DEVICE_NOTE.get((factor, lang), ""))
        for k, ch in enumerate(chunked(its, JUDGE_CHUNK)):
            cid = f"judge:{tag}:{factor}:{lang}:{k}"
            user = [{"id": j, "relation": mp.FACTOR_GLOSS[(factor, it["direction"])],
                     **({"EN_A": it["en_a"], "EN_B": it["en_b"]} if lang != "en" else {}),
                     "A": it["a"], "B": it["b"]} for j, it in enumerate(ch)]
            reqs.append((cid, mp.chat_body(model, system, json.dumps(user, ensure_ascii=False),
                                           1500, False)))
            mapping[cid] = [it["iid"] for it in ch]
    return reqs, mapping


def load_items(work):
    items = mp.read_jsonl(jdir(work, "items.jsonl"))
    if not items:
        raise SystemExit(f"{jdir(work, 'items.jsonl')} missing: run build")
    return items


def cmd_run(args, work):
    items = load_items(work)
    reqs, mapping = judge_requests(items, args.model, args.seed)
    name = f"judge.all.{mp.slug(args.model)}"
    with open(jdir(work, name + ".map.json"), "w") as f:
        json.dump(mapping, f)
    send(work, name, reqs, args.model, args, 1500)


def verdicts(work, model, items_by_iid):
    """{iid: {"ok": bool|None, criteria..., "problem"}} for one judge; None = no answer."""
    name = f"judge.all.{mp.slug(model)}"
    mapping = json.load(open(jdir(work, name + ".map.json")))
    res = mp.load_results(work, name)
    out = {}
    for cid, iids in mapping.items():
        got = parsed_items(res.get(cid))
        for j, iid in enumerate(iids):
            it = got.get(j)
            crit = CRITERIA_EN if items_by_iid[iid]["lang"] == "en" else CRITERIA_TR
            if not it or not all(isinstance(it.get(c), bool) for c in crit):
                out[iid] = {"ok": None}
                continue
            out[iid] = {**{c: it[c] for c in crit}, "ok": all(it[c] for c in crit),
                        "problem": it.get("problem")}
    return out


# ---------------------------------------------------------------- score
def quorum(vs, need):
    """vs: judge verdicts for one item -> True (pass) / False (fail) / None (too few answers)."""
    oks = [v["ok"] for v in vs if v["ok"] is not None]
    if len(oks) < need:
        return None
    return sum(oks) >= need


def judged(work, judges):
    items = load_items(work)
    by_iid = {it["iid"]: it for it in items}
    per = {m: verdicts(work, m, by_iid) for m in judges}
    need = len(judges) // 2 + 1
    for it in items:
        it["judges"] = {m: per[m].get(it["iid"], {"ok": None}) for m in judges}
        it["quorum"] = quorum(list(it["judges"].values()), need)
    return items


def pct(a, n):
    return f"{a:>4d}/{n:<4d} {100 * a / n:5.1f}%" if n else f"{'-':>15s}"


def cmd_score(args, work):
    judges = args.judges.split(",")
    short = [m.split("/")[-1][:14] for m in judges]
    items = judged(work, judges)
    mp.write_jsonl(jdir(work, "judged.jsonl"), items)
    planted = [it for it in items if it["origin"] == "planted"]
    real = [it for it in items if it["origin"] == "real"]
    score = {"judges": judges, "planted": {}, "real": {}}

    def caught(it, who):
        if who == "auto":
            return not it["auto"]["accepted"]
        if who == "quorum":
            return it["quorum"] is False
        if who == "auto|quorum":
            return not it["auto"]["accepted"] or it["quorum"] is False
        return it["judges"][who]["ok"] is False

    cols = ["auto"] + judges + ["quorum", "auto|quorum"]
    names = ["auto"] + short + ["quorum", "auto|quorum"]
    print("\nCatch rate on planted defects (higher is better)")
    print(f"{'defect':<16s}" + "".join(f"{n:>17s}" for n in names))
    for d in DEFECTS_TR + ["all"]:
        ps = [p for p in planted if d == "all" or p["defect"] == d]
        row = {c: sum(caught(p, c) for p in ps) for c in cols}
        score["planted"][d] = {"n": len(ps), **row}
        print(f"{d:<16s}" + "".join(f"{pct(row[c], len(ps)):>17s}" for c in cols))
    print("\nCatch rate on planted defects by language: auto / quorum / auto|quorum")
    for lang in ["en"] + mp.TARGET_LANGS:
        ps = [p for p in planted if p["lang"] == lang]
        if ps:
            print(f"  {lang:<4s} n={len(ps):<4d}" + "".join(
                f"{c}: {pct(sum(caught(p, c) for p in ps), len(ps))}   "
                for c in ("auto", "quorum", "auto|quorum")))
    print("\nReal items: pass rates (auto, quorum) and agreement")
    print(f"{'slot':<14s}{'n':>6s}{'auto pass':>17s}{'quorum pass':>17s}{'both pass':>17s}"
          f"{'auto only':>17s}{'quorum only':>17s}{'no quorum':>10s}")
    for (factor, lang) in sorted({(it["factor"], it["lang"]) for it in real},
                                 key=lambda x: (x[0], x[1] != "en", x[1])):
        rs = [it for it in real if it["factor"] == factor and it["lang"] == lang]
        n = len(rs)
        a = sum(it["auto"]["accepted"] for it in rs)
        q = sum(it["quorum"] is True for it in rs)
        both = sum(it["auto"]["accepted"] and it["quorum"] is True for it in rs)
        ao = sum(it["auto"]["accepted"] and it["quorum"] is False for it in rs)
        qo = sum(not it["auto"]["accepted"] and it["quorum"] is True for it in rs)
        nq = sum(it["quorum"] is None for it in rs)
        score["real"][f"{factor}/{lang}"] = {"n": n, "auto": a, "quorum": q, "both": both,
                                             "auto_only": ao, "quorum_only": qo, "no_quorum": nq}
        print(f"{factor[:3] + '/' + lang:<14s}{n:>6d}{pct(a, n):>17s}{pct(q, n):>17s}"
              f"{pct(both, n):>17s}{pct(ao, n):>17s}{pct(qo, n):>17s}{nq:>10d}")
    print("\nPer judge on real items: pass rate, unanswered; pairwise agreement")
    for m, s in zip(judges, short):
        oks = [it["judges"][m]["ok"] for it in real]
        print(f"  {s:<16s} pass {pct(sum(o is True for o in oks), len(oks))}  "
              f"unanswered {sum(o is None for o in oks)}")
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            both = [(it["judges"][judges[i]]["ok"], it["judges"][judges[j]]["ok"]) for it in items]
            both = [(x, y) for x, y in both if x is not None and y is not None]
            agree = sum(x == y for x, y in both)
            pa = sum(x for x, _ in both) / max(1, len(both))
            pb = sum(y for _, y in both) / max(1, len(both))
            pe = pa * pb + (1 - pa) * (1 - pb)
            kappa = (agree / max(1, len(both)) - pe) / max(1e-9, 1 - pe)
            print(f"  {short[i]} vs {short[j]}: agree {pct(agree, len(both))}  kappa {kappa:.2f}")
    print("\nWhy the quorum fails real items it fails (criteria failed by >= 2 judges)")
    for lang_kind in ("en", "tr"):
        cnt = Counter()
        for it in real:
            if it["quorum"] is not False or (it["lang"] == "en") != (lang_kind == "en"):
                continue
            crit = CRITERIA_EN if it["lang"] == "en" else CRITERIA_TR
            for c in crit:
                if sum(v.get(c) is False for v in it["judges"].values()) >= 2:
                    cnt[c] += 1
        print(f"  {lang_kind}: {dict(cnt.most_common())}")
    with open(jdir(work, "score.json"), "w") as f:
        json.dump(score, f, indent=1)
    print(f"\nwrote {jdir(work, 'judged.jsonl')}, {jdir(work, 'score.json')}")


# ---------------------------------------------------------------- sample for the human check
def cmd_sample(args, work):
    """~50 real items: English edits and Japanese tense weighted to disagreements, plus
    items every judge and the automatic checks passed (does unanimous mean right?)."""
    rng = random.Random(args.seed)
    items = [it for it in judged(work, args.judges.split(",")) if it["origin"] == "real"]

    def split(it):
        oks = {v["ok"] for v in it["judges"].values()}
        return len(oks - {None}) > 1 or (it["quorum"] is not None and it["quorum"] != it["auto"]["accepted"])

    def unanimous(it):
        return it["auto"]["accepted"] and all(v["ok"] is True for v in it["judges"].values())

    picked, seen = [], set()

    def take(pool, n, why):
        pool = [it for it in pool if it["iid"] not in seen]
        for it in rng.sample(pool, min(n, len(pool))):
            seen.add(it["iid"])
            picked.append({**it, "why": why})

    en = [it for it in items if it["lang"] == "en"]
    ja_t = [it for it in items if it["lang"] == "ja" and it["factor"] == "tense"]
    take([it for it in en if split(it)], args.n_en - args.n_en // 3, "disagreement")
    take(en, args.n_en - len(picked), "random")
    k = len(picked)
    take([it for it in ja_t if split(it)], args.n_ja - args.n_ja // 3, "disagreement")
    take(ja_t, args.n_ja - (len(picked) - k), "random")
    take([it for it in items if it["lang"] in ("en", "de", "fr", "ja") and unanimous(it)],
         args.n_unanimous, "unanimous_pass")
    rng.shuffle(picked)
    keep = ("iid", "factor", "direction", "lang", "en_a", "en_b", "a", "b", "why", "auto", "quorum")
    out = [{**{k: it[k] for k in keep},
            "judges": {m.split("/")[-1]: {"ok": v["ok"], "problem": v.get("problem")}
                       for m, v in it["judges"].items()}} for it in picked]
    with open(args.out or jdir(work, "sample.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out or jdir(work, 'sample.json')}: {len(out)} items "
          f"{dict(Counter((it['lang'], it['why']) for it in out))}")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.path.join(mp.REPO, "augment", "work"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build")
    s.add_argument("--edit-model", required=True)
    s.add_argument("--translate-model", required=True)
    s.add_argument("--backtrans-model", required=True)
    s.add_argument("--plant-model", required=True)
    s.add_argument("--plant-cluster", choices=sorted(mp.CHAT_URL), default="minerva")
    s.add_argument("--per-defect", type=int, default=6, help="planted items per defect and (factor, lang)")
    s.add_argument("--min-cos", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--timeout", type=float, default=600)

    s = sub.add_parser("run")
    s.add_argument("--model", required=True)
    s.add_argument("--seed", type=int, default=0)
    mp.add_run_args(s)

    for name in ("score", "sample"):
        s = sub.add_parser(name)
        s.add_argument("--judges", required=True, help="comma-separated judge models")
        if name == "sample":
            s.add_argument("--n-en", type=int, default=25)
            s.add_argument("--n-ja", type=int, default=15)
            s.add_argument("--n-unanimous", type=int, default=10)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--out")

    args = p.parse_args()
    work = mp.Work(args.work)
    {"build": cmd_build, "run": cmd_run, "score": cmd_score, "sample": cmd_sample}[args.cmd](args, work)


if __name__ == "__main__":
    main()
