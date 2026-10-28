"""minpairs.py — minimal-pair augmentation pilot (PHASE9 §8 branch 2; spec: augment/README.md).

Drives the ALCF inference batch service (Sophia, vLLM) through the stages of
README §3, with a local LLM-free verifier after every LLM stage:

  candidates   opus100 English sides -> filtered candidates          (spaCy, tokenizer)
  smoke        3-line batch to settle the request/result format       (README §4)
  prepare      write a stage's batch input for one model              (edit | translate | backtrans)
  submit       create a batch for a prepared input
  status       poll every submission (--wait: until none are pending)
  collect      fetch result metadata, copy results files into the work dir
  inspect      print raw result lines of one batch and how they parse
  verify       parse a stage's results and apply the local verifier   (spaCy, MiniLM)
  report       per stage x model x factor (x language) counts and rejection reasons
  package      accepted pairs -> opus100-layout shards + sidecar JSONL

Work directory (--work, default augment/work):
  candidates.jsonl                one line per candidate, pair_id = candidate id
  inputs/<name>.jsonl             batch input (+ .ids order sidecar, .meta.json upstream models)
  submissions.jsonl               one line per created batch (latest line per batch_id wins)
  results/<batch_id>.results.jsonl, <batch_id>.result.json
  verified/<name>.jsonl           per-item verdicts with rejection reasons
  report.json

<name> is <stage>.<factor>.<model slug>, stage in {edit, tr, bt}.

The batch service reads inputs and writes outputs on /eagle. --remote-root is
the work directory as Sophia sees it (default: the work directory's absolute
path, right when this runs on a Polaris/Sophia login node with the work
directory on /eagle). If the work directory is elsewhere, rsync inputs/ to
<remote-root>/inputs before `submit`, and rsync <remote-root>/outputs back into
<work>/outputs before `collect`; collect maps remote result paths onto the
work directory.

Typical pilot (README §5):
  python augment/minpairs.py candidates --n 300
  python augment/minpairs.py smoke --model meta-llama/Meta-Llama-3.1-8B-Instruct
  python augment/minpairs.py status --wait && python augment/minpairs.py collect
  python augment/minpairs.py inspect --batch <id>
  for m in <models>; do
    python augment/minpairs.py prepare edit --factor negation --model $m --submit
    python augment/minpairs.py prepare edit --factor tense --model $m --submit
  done
  python augment/minpairs.py status --wait && python augment/minpairs.py collect
  python augment/minpairs.py verify edit --factor negation --model <m>     (each model, factor)
  python augment/minpairs.py prepare translate --factor negation --edit-model <best> --model <m> --submit
  ...  verify translate, prepare/verify backtrans, report, package

Only the ALCF stages (smoke, submit, status, collect, inspect) run with the
standard library plus requests; candidates/verify/package need numpy, spaCy
(en_core_web_sm), transformers and torch.
"""
import argparse
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

API = "https://inference-api.alcf.anl.gov/resource_server"
CREATE_URL = API + "/sophia/vllm/v1/batches"
BATCH_URL = API + "/v1/batches"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TARGET_LANGS = ["zh", "es", "fr", "ar", "ru", "hi", "ja", "pt", "de"]
SHARD_LANGS = ["ar", "de", "es", "fr", "hi", "ja", "pt", "ru", "zh"]  # data/opus100/{x}_en.npz
LANG_NAMES = {"zh": "Chinese (Simplified)", "es": "Spanish", "fr": "French", "ar": "Arabic",
              "ru": "Russian", "hi": "Hindi", "ja": "Japanese", "pt": "Portuguese",
              "de": "German"}
FACTORS = ["negation", "tense"]
FACTOR_ID = {"negation": 0, "tense": 1}
DIRECTIONS = {"negation": ("add", "remove"), "tense": ("to_past", "to_present")}
MAX_LINES = 150_000          # per batch input file (README §4)
MAX_TOKENS = {"edit": 1200, "tr": 1500, "bt": 300}
EDIT_CHUNK = 20
REASONING_MODELS = ("gpt-oss", "muse-spark", "nemotron", "beam-", "inkling")   # think before answering
TERMINAL = {"completed", "failed", "cancelled", "expired"}

TIME_ADVERBS = {"yesterday", "tomorrow", "ago", "last", "next", "now", "currently", "soon"}
QUOTE_RE = re.compile(r"[\"“”„«»「」『』‘]|(^|\s)'|'(\s|$)")
URL_RE = re.compile(r"https?://|www\.|\.(com|org|net|html?)\b", re.I)
# subtitle register that the edit models mishandle: OCR "l" for "I", got to/gotta, ain't, ...
COLLOQ_RE = re.compile(r"\bl\b|\b(gotta|gonna|wanna|ain't|got to|y'all|kinda|sorta)\b", re.I)


# ---------------------------------------------------------------- small utils
def slug(model):
    return model.replace("/", "--")


def stage_name(stage, factor, model, translate_model=None):
    """<stage>.<factor>.<model slug>; back-translation also names whose translations
    it checks (<bt model>~<translate model>), so two translators never share files."""
    name = f"{stage}.{factor}.{slug(model)}"
    return name + f"~{slug(translate_model)}" if stage == "bt" else name


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_answer(text):
    """Model answer -> object, tolerant of code fences and prose around the JSON."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def token_edit_distance(a, b):
    """Whitespace-token edits (difflib opcodes; a replace counts its longer side)."""
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def char_ratio(a, b):
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def sampled(pair_id, frac, salt="bt"):
    h = hashlib.sha1(f"{salt}:{pair_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2 ** 32 < frac


class Work:
    def __init__(self, root, remote_root=None):
        self.root = os.path.abspath(root)
        self.remote = (remote_root or self.root).rstrip("/")
        for d in ("inputs", "results", "verified", "outputs"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def to_remote(self, local):
        return self.remote + "/" + os.path.relpath(local, self.root)

    def to_local(self, remote):
        if os.path.exists(remote):
            return remote
        if remote.startswith(self.remote + "/"):
            return self.p(remote[len(self.remote) + 1:])
        return None

    @property
    def candidates(self):
        return self.p("candidates.jsonl")

    @property
    def submissions(self):
        return self.p("submissions.jsonl")

    def latest_submissions(self):
        latest = {}
        for r in read_jsonl(self.submissions):
            latest[r["batch_id"]] = {**latest.get(r["batch_id"], {}), **r}
        return latest


# ---------------------------------------------------------------- ALCF client
AUTH_HELPER = os.environ.get("ALCF_AUTH_HELPER", os.path.expanduser("~/inference_auth_token.py"))
_token = {"value": None, "at": 0.0}


def _helper_token():
    """Token from ALCF's inference_auth_token.py (refreshes itself; needs globus_sdk)."""
    if not os.path.exists(AUTH_HELPER):
        return None
    pys = [os.environ.get("ALCF_AUTH_PYTHON"), sys.executable,
           os.path.expanduser("~/miniconda3/envs/globus_env/bin/python")]
    for py in filter(None, pys):
        if not os.path.exists(py):
            continue
        r = subprocess.run([py, AUTH_HELPER, "get_access_token"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[-1].strip()
    return None


def access_token(refresh=False):
    """Cached 10 min; the helper is preferred over env vars so 48 h expiry is handled."""
    if _token["value"] and not refresh and time.time() - _token["at"] < 600:
        return _token["value"]
    tok = _helper_token() or os.environ.get("ALCF_ACCESS_TOKEN") or os.environ.get("ALCF_BEARER_TOKEN")
    if tok:
        _token.update(value=tok.strip(), at=time.time())
        return _token["value"]
    return _legacy_token()


def _legacy_token():
    if shutil.which("alcf-tokens"):
        out = subprocess.run(["alcf-tokens", "get-token", "inference"], capture_output=True,
                             text=True, check=True).stdout.strip()
        if out:
            return out.splitlines()[-1].strip()
    try:
        from alcf_tokens.auth import get_access_token
        try:
            return get_access_token("inference")
        except TypeError:
            return get_access_token()
    except ImportError:
        pass
    raise SystemExit(f"no ALCF token: {AUTH_HELPER} (python inference_auth_token.py authenticate), "
                     "ALCF_ACCESS_TOKEN / ALCF_BEARER_TOKEN, or alcf-tokens")


def api(method, url, body=None):
    import requests
    r = requests.request(method, url, json=body, timeout=120,
                         headers={"Authorization": f"Bearer {access_token()}"})
    if r.status_code >= 400:
        raise SystemExit(f"{method} {url} -> HTTP {r.status_code}: {r.text[:2000]}")
    try:
        return r.json()
    except ValueError:
        return {"_text": r.text}


def create_batch(work, model, input_path, meta):
    out_dir = work.to_remote(work.p("outputs"))
    body = {"model": model, "input_file": work.to_remote(input_path),
            "output_folder_path": out_dir}
    resp = api("POST", CREATE_URL, body)
    bid = resp.get("batch_id") or resp.get("id") or (resp.get("batch") or {}).get("batch_id")
    if not bid:
        raise SystemExit(f"create response carries no batch id (README §4 unknown): {resp}")
    row = {"batch_id": bid, "model": model, "input_file": input_path,
           "remote_input": body["input_file"], "output_folder": out_dir, "status": "pending",
           "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "create_response": resp, **meta}
    append_jsonl(work.submissions, row)
    print(f"submitted {os.path.basename(input_path)} on {model}: batch {bid}")
    return bid


# ---------------------------------------------------------------- request lines
def chat_body(model, system, user, max_tokens, json_mode):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if any(k in model for k in REASONING_MODELS):
        # reasoning counts against max_tokens. On a 20-item edit, default effort spends
        # 12-14k (muse, beam, inkling, nemotron); "low" cuts that to 1.6-3k, except
        # nemotron (~9k) and gpt-oss (1-3k). Headroom is free: only used tokens are billed.
        body["reasoning_effort"] = "low"
        body["max_tokens"] = max_tokens + 20000
    return body


def request_line(custom_id, body, line_format):
    if line_format == "body":
        return body
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
            "body": body}


def write_input(work, name, items, line_format, meta):
    """items: [(custom_id, body)] -> one or more input files (split at MAX_LINES)."""
    for old in os.listdir(work.p("inputs")):
        if old.startswith(name + ".") and (old[len(name) + 1:].startswith("part")
                                           or old[len(name) + 1:] in ("jsonl", "ids", "meta.json")):
            os.remove(work.p("inputs", old))
    nparts = max(1, math.ceil(len(items) / MAX_LINES))
    paths = []
    for k in range(nparts):
        chunk = items[k * MAX_LINES:(k + 1) * MAX_LINES]
        stem = name if nparts == 1 else f"{name}.part{k}"
        path = work.p("inputs", stem + ".jsonl")
        write_jsonl(path, [request_line(cid, body, line_format) for cid, body in chunk])
        with open(work.p("inputs", stem + ".ids"), "w") as f:
            f.write("\n".join(cid for cid, _ in chunk) + "\n")
        paths.append(path)
    with open(work.p("inputs", name + ".meta.json"), "w") as f:
        json.dump({**meta, "name": name, "n_requests": len(items), "line_format": line_format,
                   "parts": [os.path.basename(p) for p in paths]}, f, indent=1)
    size = sum(os.path.getsize(p) for p in paths) / 1e6
    print(f"wrote {name}: {len(items)} requests in {len(paths)} file(s), {size:.1f} MB")
    return paths


# ---------------------------------------------------------------- prompts
EDIT_SYSTEM = {
    "negation": """You edit English sentences to create minimal pairs that differ ONLY in the polarity of the main clause.

For each input sentence, toggle the polarity of the MAIN clause's finite verb:
- affirmative -> negated: op "add"; negated -> affirmative: op "remove".
- Use do-support where English needs it ("She left." -> "She did not leave.").
- Match the source's contraction register: if the source uses contractions (don't, isn't), the variant does too; if it spells out (do not, is not), so does the variant.
- Change other words ONLY where grammar forces it: some/any, already/yet, too/either. Keep every other word, word order and punctuation identical.
- Never add or remove adverbs such as never, anymore, at all, still.
- Use op "skip" (variant null) for: fragments, headlines, quotations, sentences whose only negation is in a subordinate clause, and idioms whose meaning the toggle would change (can't help, not only, no doubt, nothing but).

Input: a JSON array of {"id", "text"}.
Output only the JSON object {"items": [{"id": <id>, "op": "add" | "remove" | "skip", "variant": <edited sentence or null>}, ...]}, one item per input id, in input order. No commentary.

Examples:
{"id": 1, "text": "The committee has approved the budget."} -> {"id": 1, "op": "add", "variant": "The committee has not approved the budget."}
{"id": 2, "text": "She doesn't like the new rules."} -> {"id": 2, "op": "remove", "variant": "She likes the new rules."}
{"id": 3, "text": "I saw some birds in the garden."} -> {"id": 3, "op": "add", "variant": "I did not see any birds in the garden."}
{"id": 4, "text": "Not only did he win, he also set a record."} -> {"id": 4, "op": "skip", "variant": null}""",
    "tense": """You edit English sentences to create minimal pairs that differ ONLY in the tense of the main clause.

For each input sentence, switch the MAIN clause between present and past:
- present -> past: op "to_past"; past -> present: op "to_present".
- Carry sequence-of-tense agreement into clauses that depend on the main verb ("He says he knows." <-> "He said he knew.").
- Keep aspect: progressive stays progressive (is going <-> was going), perfect stays perfect (has gone <-> had gone).
- Change nothing else: same words, word order, punctuation, person and number.
- Use op "skip" (variant null) for: timeless generalities (scientific facts, definitions, proverbs), sentences with a time adverbial that conflicts with the new tense, futures, modals (can, must, should, may, might, would, could), headlines, quotations and fragments.

Input: a JSON array of {"id", "text"}.
Output only the JSON object {"items": [{"id": <id>, "op": "to_past" | "to_present" | "skip", "variant": <edited sentence or null>}, ...]}, one item per input id, in input order. No commentary.

Examples:
{"id": 1, "text": "The children are playing in the park."} -> {"id": 1, "op": "to_past", "variant": "The children were playing in the park."}
{"id": 2, "text": "He said that he knew the answer."} -> {"id": 2, "op": "to_present", "variant": "He says that he knows the answer."}
{"id": 3, "text": "The company has opened three new offices."} -> {"id": 3, "op": "to_past", "variant": "The company had opened three new offices."}
{"id": 4, "text": "Water boils at 100 degrees."} -> {"id": 4, "op": "skip", "variant": null}""",
}

FACTOR_GLOSS = {
    ("negation", "add"): "A is affirmative; B is the same sentence with the main clause negated.",
    ("negation", "remove"): "A is negated; B is the same sentence with the main-clause negation removed.",
    ("tense", "to_past"): "A is in the present tense; B is the same sentence in the past tense.",
    ("tense", "to_present"): "A is in the past tense; B is the same sentence in the present tense.",
}

TR_SYSTEM = """You translate English minimal pairs into nine languages: """ + ", ".join(
    f"{LANG_NAMES[l]} ({l})" for l in TARGET_LANGS) + """.

A and B are the same English sentence and differ ONLY in one factor ({factor_desc}).
For each language:
1. Translate A naturally and faithfully.
2. Derive B from YOUR translation of A by the minimal change that flips the factor. Never translate B independently.
3. The two translations must be word-for-word identical except for the factor.
4. Realise the factor with the language's smallest conventional device: {device}. Never use a time adverb or a different word choice to express it.
5. If a minimal pair is impossible in a language, return null for that language.

Output only the JSON object {{"zh": {{"a": ..., "b": ...}}, "es": {{...}}, "fr": ..., "ar": ..., "ru": ..., "hi": ..., "ja": ..., "pt": ..., "de": ...}} (each value an object or null). No commentary."""

TR_FACTOR = {
    "negation": ("polarity: one member is negated, the other affirmative",
                 "a negation particle or a negative verb form"),
    "tense": ("tense: one member is present, the other past",
              "the past-tense verb form; in Chinese the aspect marker 了 or 过, in Japanese the "
              "た-form (or ました/でした)"),
}

BT_SYSTEM = """You translate a pair of sentences into English as literally as possible.
The two sentences A and B are a minimal pair: translate each one literally, keeping the two English translations word-for-word identical wherever the sources are identical. Do not smooth over or normalise any grammatical difference between them (tense, negation, number, person).
Output only the JSON object {"A": <English translation of A>, "B": <English translation of B>}. No commentary."""


# ---------------------------------------------------------------- NLP helpers
_nlp = _st = _tok = None


def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tok


_doc_attributes = None


def doc_attributes(doc):
    global _doc_attributes
    if _doc_attributes is None:
        sys.path.insert(0, REPO)
        from compositionality_metrics import doc_attributes as f  # same rules as A3 labels
        _doc_attributes = f
    return _doc_attributes(doc)


def cosines(pairs, batch=64):
    """Mean-pooled MiniLM cosine for each (a, b) text pair."""
    global _st
    import torch
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _st is None:
        _st = AutoModel.from_pretrained(MODEL_NAME).eval().to(dev)
    texts = [t for ab in pairs for t in ab]
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tokenizer()(texts[s:s + batch], padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(dev)
            h = _st(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
            out.append(e.cpu())
    if not out:
        return []
    e = torch.cat(out)
    return (e[0::2] * e[1::2]).sum(-1).tolist()


def root_of(doc):
    return next((t for t in doc if t.dep_ == "ROOT"), doc[0])


def n_finite(doc):
    return sum(1 for t in doc if "Fin" in t.morph.get("VerbForm"))


# ---------------------------------------------------------------- candidates
def text_prefilter(t):
    if not t or QUOTE_RE.search(t) or URL_RE.search(t):
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 10 or sum(c.isupper() for c in letters) > 0.6 * len(letters):
        return False                                    # all-caps headline
    if COLLOQ_RE.search(t):
        return False
    n = len(t.split())
    return (5 <= n <= 30 and t[0].isupper() and t[-1] in ".!"   # no questions: rhetorical/tag
            and ".." not in t and "--" not in t and "…" not in t)   # ellipsis/dash: fragments


def classify(doc):
    """-> (attrs, {factor: direction}) for every factor this sentence can serve."""
    if len(list(doc.sents)) != 1:
        return None, {}
    words = [t for t in doc if not (t.is_punct or t.is_space)]
    if not 6 <= len(words) <= 25:
        return None, {}
    root = root_of(doc)
    aux = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children):
        return None, {}
    if not any("Fin" in t.morph.get("VerbForm") for t in [root, *aux]):
        return None, {}
    if any("Imp" in t.morph.get("Mood") for t in [root, *aux]) or any(t.dep_ == "intj" for t in doc):
        return None, {}                                  # imperatives; "Look, ...", "Okay, uh"
    a = doc_attributes(doc)
    if a["tense"] == "none":
        return None, {}
    ok = {}
    if a["neg"] == "1":
        if any(c.dep_ == "neg" for c in root.children):   # negation on the main clause
            ok["negation"] = "remove"
    else:
        ok["negation"] = "add"
    modal = any(x.tag_ == "MD" for x in aux) or root.tag_ == "MD"
    if (a["tense"] in ("pres", "past") and not modal
            and not any(t.lower_ in TIME_ADVERBS for t in doc)):
        ok["tense"] = "to_past" if a["tense"] == "pres" else "to_present"
    return a, ok


def cmd_candidates(args, work):
    import numpy as np
    rng = np.random.default_rng(args.seed)
    tok = tokenizer()
    shards = [l for l in SHARD_LANGS if os.path.exists(os.path.join(args.data, f"{l}_en.npz"))]
    if not shards:
        raise SystemExit(f"no opus100 shards in {args.data}")
    # quota per (shard, factor, direction): balanced across shards and directions
    quota = {(s, f, d): math.ceil(args.n / 2 / len(shards))
             for s in shards for f in FACTORS for d in DIRECTIONS[f]}
    rows, seen = [], set()
    for s in shards:
        z = np.load(os.path.join(args.data, f"{s}_en.npz"))
        toks, offs = z["tgt_tokens"], z["tgt_offsets"]
        order = rng.permutation(len(offs) - 1)
        drawn = 0
        for start in range(0, len(order), args.chunk):
            if all(quota[(s, f, d)] <= 0 for f in FACTORS for d in DIRECTIONS[f]):
                break
            if drawn >= args.max_draw:
                break
            idx = order[start:start + args.chunk]
            drawn += len(idx)
            texts = [(int(i), tok.decode(toks[offs[i]:offs[i + 1]].tolist(), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True).strip()) for i in idx]
            texts = [(i, t) for i, t in texts if t not in seen and text_prefilter(t)]
            for (i, t), doc in zip(texts, nlp().pipe([t for _, t in texts], batch_size=256)):
                attrs, ok = classify(doc)
                need = [(f, d) for f, d in ok.items() if quota[(s, f, d)] > 0]
                if not need:
                    continue
                f, d = need[rng.integers(len(need))]
                quota[(s, f, d)] -= 1
                seen.add(t)
                rows.append({"factor": f, "direction": d, "source": {"shard": f"{s}_en", "row": i},
                             "text": t, "attrs": attrs})
        left = {f"{f}/{d}": quota[(s, f, d)] for f in FACTORS for d in DIRECTIONS[f]}
        print(f"  [{s}_en] drew {drawn} rows; unfilled quota {left}")
    # trim to n per factor (ceil quotas overshoot), keep directions balanced
    out = []
    for f in FACTORS:
        for d in DIRECTIONS[f]:
            fd = [r for r in rows if r["factor"] == f and r["direction"] == d]
            rng.shuffle(fd)
            out += fd[:args.n // 2 if d == DIRECTIONS[f][0] else args.n - args.n // 2]
    for pid, r in enumerate(out):
        r["pair_id"] = pid
    out = [{"pair_id": r.pop("pair_id"), **r} for r in out]
    write_jsonl(work.candidates, out)
    c = Counter((r["factor"], r["direction"]) for r in out)
    print(f"wrote {len(out)} candidates to {work.candidates}: {dict(c)}")


# ---------------------------------------------------------------- prepare
def accepted_edits(work, factor, edit_model):
    path = work.p("verified", f"edit.{factor}.{slug(edit_model)}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run verify edit --factor {factor} --model {edit_model}")
    return [r for r in read_jsonl(path) if r["accepted"]]


def accepted_translations(work, factor, tr_model):
    path = work.p("verified", f"tr.{factor}.{slug(tr_model)}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run verify translate --factor {factor} --model {tr_model}")
    return read_jsonl(path)


def cmd_prepare(args, work):
    m = args.model
    meta = {"stage": {"edit": "edit", "translate": "tr", "backtrans": "bt"}[args.stage],
            "factor": args.factor, "model": m, "json_mode": args.json_mode}
    items = []
    if args.stage == "edit":
        cands = [c for c in read_jsonl(work.candidates) if c["factor"] == args.factor]
        if not cands:
            raise SystemExit(f"no {args.factor} candidates in {work.candidates}")
        for k in range(0, len(cands), EDIT_CHUNK):
            chunk = cands[k:k + EDIT_CHUNK]
            user = json.dumps([{"id": c["pair_id"], "text": c["text"]} for c in chunk],
                              ensure_ascii=False)
            items.append((f"edit:{args.factor}:c{k // EDIT_CHUNK:05d}",
                          chat_body(m, EDIT_SYSTEM[args.factor], user, MAX_TOKENS["edit"],
                                    args.json_mode)))
    elif args.stage == "translate":
        if not args.edit_model:
            raise SystemExit("prepare translate needs --edit-model")
        meta["edit_model"] = args.edit_model
        desc, device = TR_FACTOR[args.factor]
        system = TR_SYSTEM.format(factor_desc=desc, device=device)
        for r in accepted_edits(work, args.factor, args.edit_model):
            user = json.dumps({"relation": FACTOR_GLOSS[(args.factor, r["direction"])],
                               "A": r["a"], "B": r["b"]}, ensure_ascii=False)
            items.append((f"tr:{args.factor}:{r['pair_id']}",
                          chat_body(m, system, user, MAX_TOKENS["tr"], args.json_mode)))
    else:
        if not args.translate_model:
            raise SystemExit("prepare backtrans needs --translate-model")
        meta["translate_model"] = args.translate_model
        meta["sample_frac"] = args.sample_frac
        for r in accepted_translations(work, args.factor, args.translate_model):
            if args.factor == "negation" and not sampled(r["pair_id"], args.sample_frac):
                continue
            for lang, v in r["langs"].items():
                if not v.get("accepted"):
                    continue
                user = json.dumps({"A": v["a"], "B": v["b"]}, ensure_ascii=False)
                items.append((f"bt:{args.factor}:{r['pair_id']}:{lang}",
                              chat_body(m, BT_SYSTEM, user, MAX_TOKENS["bt"], args.json_mode)))
    if not items:
        raise SystemExit("nothing to submit")
    name = stage_name(meta["stage"], args.factor, m, args.translate_model)
    paths = write_input(work, name, items, args.line_format, meta)
    if args.run:
        run_interactive(work, name, paths, m, args.cluster, args.workers, args.timeout, args.limit)
    elif args.submit:
        for p in paths:
            create_batch(work, m, p, {"name": name, "stage": meta["stage"], "factor": args.factor})
    elif work.remote != work.root:
        print(f"stage the inputs: rsync -av {work.p('inputs')}/ <host>:{work.to_remote(work.p('inputs'))}/")


# ---------------------------------------------------------------- smoke, submit, status, collect
SMOKE_SENTENCES = [{"id": 0, "text": "The minister has signed the agreement."},
                   {"id": 1, "text": "We don't know where the documents are."}]


def cmd_smoke(args, work):
    m = args.model
    items = [
        ("smoke:plain:0", chat_body(m, "You are a terse assistant.",
                                    "Reply with the single word OK.", 10, False)),
        ("smoke:json:1", chat_body(m, "You answer in JSON.",
                                   'Return {"ok": true, "n": 3}.', 30, True)),
        ("smoke:edit:2", chat_body(m, EDIT_SYSTEM["negation"], json.dumps(SMOKE_SENTENCES),
                                   300, False)),
    ]
    name = f"smoke.{args.line_format}.{slug(m)}"
    paths = write_input(work, name, items, args.line_format,
                        {"stage": "smoke", "factor": None, "model": m})
    if work.remote != work.root and not args.yes_staged:
        print(f"inputs must be visible to Sophia at {work.to_remote(paths[0])}; rsync, then:\n"
              f"  python augment/minpairs.py submit --input {paths[0]} --model {m}")
        return
    create_batch(work, m, paths[0], {"name": name, "stage": "smoke", "factor": None})


def cmd_submit(args, work):
    path = os.path.abspath(args.input)
    meta_path = re.sub(r"(\.part\d+)?\.jsonl$", ".meta.json", path)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    model = args.model or meta.get("model")
    if not model:
        raise SystemExit("--model needed (no .meta.json next to the input)")
    create_batch(work, model, path, {"name": meta.get("name", os.path.basename(path)[:-6]),
                                     "stage": meta.get("stage"), "factor": meta.get("factor")})


def refresh_status(work):
    rows = []
    for bid, s in work.latest_submissions().items():
        if s.get("status") not in TERMINAL and s.get("mode") != "interactive":
            r = api("GET", f"{BATCH_URL}/{bid}")
            st = str(r.get("status", "unknown")).lower()
            if st != s.get("status"):
                append_jsonl(work.submissions, {"batch_id": bid, "status": st,
                                                "status_response": r})
            s["status"] = st
        rows.append(s)
    return rows


def cmd_status(args, work):
    while True:
        rows = refresh_status(work)
        for s in sorted(rows, key=lambda s: s.get("submitted_at", "")):
            print(f"  {s['batch_id']:<40s} {s.get('status', '?'):<10s} {s.get('name', '?')}")
        pending = [s for s in rows if s.get("status") not in TERMINAL]
        if not args.wait or not pending:
            break
        print(f"{len(pending)} pending; next poll in {args.interval}s")
        time.sleep(args.interval)


def cmd_collect(args, work):
    for bid, s in work.latest_submissions().items():
        if s.get("status") != "completed":
            continue
        dst = work.p("results", f"{bid}.results.jsonl")
        if os.path.exists(dst) and not args.force:
            continue
        meta = api("GET", f"{BATCH_URL}/{bid}/result")
        with open(work.p("results", f"{bid}.result.json"), "w") as f:
            json.dump(meta, f, indent=1)
        remote = meta.get("results_file") or meta.get("result_file")
        if not remote:
            print(f"  {bid}: result response has no results_file: {str(meta)[:300]}")
            continue
        local = work.to_local(remote)
        if not local or not os.path.exists(local):
            print(f"  {bid}: {remote} not visible here; rsync <remote-root>/outputs into "
                  f"{work.p('outputs')} and re-run collect")
            continue
        shutil.copyfile(local, dst)
        print(f"  {bid}: {sum(1 for _ in open(dst))} result lines -> {dst}")


# ---------------------------------------------------------------- interactive runner
CHAT_URL = {"sophia": API + "/sophia/vllm/v1/chat/completions",
            "minerva": API + "/minerva/api/v1/chat/completions",
            "metis": API + "/metis/api/v1/chat/completions"}
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def chat_once(url, body, timeout, retries=6):
    """POST one chat body -> (http status, response json or {"error"}). Retries 429/5xx/timeouts."""
    import requests
    delay, err = 5.0, None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {access_token(refresh=attempt > 0 and err == 401)}"})
            if r.status_code == 200:
                return 200, r.json()
            err = r.status_code
            if r.status_code not in RETRY_STATUS | {401} or (r.status_code == 401 and attempt > 1):
                return r.status_code, {"error": r.text[:2000]}
            msg = r.text[:300]
        except (requests.Timeout, requests.ConnectionError) as e:
            err, msg = "timeout", str(e)[:300]
        except ValueError as e:
            err, msg = "bad_json", str(e)[:300]
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return err, {"error": msg}


def input_digest(paths):
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:8]


def run_interactive(work, name, paths, model, cluster, workers, timeout, limit=None):
    """Send every request line of `paths` to the chat endpoint; resumable.

    Results go to results/<run_id>.results.jsonl in the OpenAI batch result shape
    ({"custom_id", "response": {"status_code", "body"}, "error"}), registered in
    submissions.jsonl as a completed batch, so verify/report read them unchanged.
    run_id changes with the input content, so an edited prompt starts a fresh run.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    run_id = f"local-{name}-{input_digest(paths)}"
    dst = work.p("results", f"{run_id}.results.jsonl")
    done = {r["custom_id"] for r in read_jsonl(dst)
            if (r.get("response") or {}).get("status_code") == 200}
    lines = []
    for p in paths:
        for rec in read_jsonl(p):
            cid, body = (rec["custom_id"], rec["body"]) if "body" in rec else (None, rec)
            if cid is None:
                raise SystemExit(f"{p}: body-only lines carry no custom_id; prepare with "
                                 "--line-format openai to run interactively")
            if cid not in done:
                lines.append((cid, {**body, "model": model}))
    if limit:
        lines = lines[:limit]
    subs = work.latest_submissions()
    if run_id not in subs:
        append_jsonl(work.submissions, {
            "batch_id": run_id, "model": model, "input_file": paths[0], "name": name,
            "mode": "interactive", "cluster": cluster, "status": "running",
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"{run_id}: {len(done)} done, {len(lines)} to send on {model} ({cluster}, {workers} workers)")
    url = CHAT_URL[cluster]
    t0, n_ok, n_err, toks, lat = time.time(), 0, 0, 0, []
    with open(dst, "a") as out, ThreadPoolExecutor(workers) as pool:
        futs = {}
        for cid, body in lines:
            futs[pool.submit(lambda b=body: (time.time(), chat_once(url, b, timeout)))] = cid
        for k, fut in enumerate(as_completed(futs), 1):
            cid = futs[fut]
            t_start, (status, resp) = fut.result()
            lat.append(time.time() - t_start)
            ok = status == 200
            n_ok += ok
            n_err += not ok
            toks += ((resp.get("usage") or {}).get("completion_tokens") or 0) if ok else 0
            out.write(json.dumps({"custom_id": cid,
                                  "response": {"status_code": status, "body": resp if ok else None},
                                  "error": None if ok else resp.get("error")},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if k % max(1, len(lines) // 20) == 0 or k == len(lines):
                el = time.time() - t0
                print(f"  {k}/{len(lines)}  ok {n_ok} err {n_err}  {el:.0f}s  "
                      f"{toks / max(el, 1e-9):.0f} out tok/s  median latency "
                      f"{sorted(lat)[len(lat) // 2]:.1f}s", flush=True)
    el = time.time() - t0
    metrics = {"mode": "interactive", "model": model, "cluster": cluster, "workers": workers,
               "sent": len(lines), "ok": n_ok, "errors": n_err, "wall_s": round(el, 1),
               "completion_tokens": toks, "out_tok_per_s": round(toks / max(el, 1e-9), 1),
               "median_latency_s": round(sorted(lat)[len(lat) // 2], 2) if lat else None}
    with open(work.p("results", f"{run_id}.result.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    remaining = len(lines) - n_ok
    append_jsonl(work.submissions, {"batch_id": run_id,
                                    "status": "completed" if remaining == 0 else "partial",
                                    "metrics": metrics})
    if n_err:
        errs = Counter(str(r.get("response", {}).get("status_code")) for r in read_jsonl(dst)
                       if r.get("response", {}).get("status_code") != 200)
        print(f"  {n_err} failed this pass ({dict(errs)} in file); re-run the same command to retry them")
    return run_id


def cmd_run(args, work):
    path = os.path.abspath(args.input)
    meta_path = re.sub(r"(\.part\d+)?\.jsonl$", ".meta.json", path)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    model = args.model or meta.get("model")
    if not model:
        raise SystemExit("--model needed (no .meta.json next to the input)")
    parts = [work.p("inputs", p) for p in meta.get("parts", [])] or [path]
    run_interactive(work, meta.get("name", os.path.basename(path)[:-6]), parts, model,
                    args.cluster, args.workers, args.timeout, args.limit)


# ---------------------------------------------------------------- results parsing
def find_choices(rec):
    if isinstance(rec, dict):
        if isinstance(rec.get("choices"), list):
            return rec["choices"]
        for k in ("response", "body", "result", "output"):
            c = find_choices(rec.get(k))
            if c is not None:
                return c
    return None


def result_content(rec):
    """Results line -> (custom_id or None, content or None, error or None)."""
    cid = rec.get("custom_id")
    ch = find_choices(rec)
    content = None
    if ch:
        msg = ch[0].get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if content is None:
            content = ch[0].get("text")
    err = rec.get("error")
    if not err and isinstance(rec.get("response"), dict):
        err = rec["response"].get("error")
    return cid, content, err


def load_results(work, name):
    """All completed batches of an input name -> {custom_id: content}; latest batch wins."""
    out = {}
    subs = sorted((s for s in work.latest_submissions().values() if s.get("name") == name),
                  key=lambda s: s.get("submitted_at", ""))
    if not subs:
        raise SystemExit(f"no submissions for {name}")
    for s in subs:
        path = work.p("results", f"{s['batch_id']}.results.jsonl")
        if not os.path.exists(path):
            print(f"  note: {s['batch_id']} ({s.get('status')}) not collected; skipped")
            continue
        ids_path = re.sub(r"\.jsonl$", ".ids", s["input_file"])
        order = open(ids_path).read().split() if os.path.exists(ids_path) else []
        for k, rec in enumerate(read_jsonl(path)):
            cid, content, _ = result_content(rec)
            if cid is None and k < len(order):
                cid = order[k]           # body-only line format: results in input order
            if cid is not None:
                out[cid] = content
    return out


def cmd_inspect(args, work):
    path = work.p("results", f"{args.batch}.results.jsonl")
    rows = read_jsonl(path)
    print(f"{path}: {len(rows)} lines")
    for rec in rows[:args.n]:
        print("--- keys:", list(rec.keys()))
        print(json.dumps(rec, ensure_ascii=False)[:1500])
        cid, content, err = result_content(rec)
        print(f">>> custom_id={cid!r} error={err!r}\n>>> content={content!r}\n"
              f">>> parsed={parse_json_answer(content)!r}")
    meta = work.p("results", f"{args.batch}.result.json")
    if os.path.exists(meta):
        print("--- result endpoint:", open(meta).read()[:1500])


# ---------------------------------------------------------------- English verifier (3.3)
KEEP = {"negation": ["tense", "subj_person", "subj_number", "ner_person", "ner_place",
                     "ner_org", "ner_time", "ner_number"],
        "tense": ["neg", "subj_person", "subj_number", "ner_person", "ner_place", "ner_org",
                  "ner_time", "ner_number"]}
FLIP = {("negation", "add"): ("neg", "0", "1"), ("negation", "remove"): ("neg", "1", "0"),
        ("tense", "to_past"): ("tense", "pres", "past"),
        ("tense", "to_present"): ("tense", "past", "pres")}


# negative-polarity items: licensed by the negation, ungrammatical or meaning-changing once it is
# removed ("did even have", "exists anymore", "want anyone"); "even" only counts when it followed
# the negation in the negated member
NPI = {"any", "anyone", "anybody", "anything", "anywhere", "anymore", "ever", "either"}
NEG_NPI_RE = re.compile(r"\b(not|n't|never|no)\s+(even|yet)\b|\bat all\b|\bany more\b", re.I)


def stranded_npi(neg, aff):
    """NPIs carried over from the negated member into the affirmative one (direction "remove":
    the model deleted the negation but kept what it licensed)."""
    wn = set(re.findall(r"[a-z']+", neg.lower()))
    wa = set(re.findall(r"[a-z']+", aff.lower()))
    hit = sorted(NPI & wn & wa)
    for m in NEG_NPI_RE.finditer(neg):
        tail = m.group(2) or m.group(0)
        if tail.lower() in aff.lower():
            hit.append(tail.lower())
    return hit


def verify_english(factor, direction, pairs, min_cos):
    """pairs: [(a, b)] -> [dict(accepted, reason, ...)] (README §3.3)."""
    docs = list(nlp().pipe([t for ab in pairs for t in ab], batch_size=128))
    cos = cosines(pairs)
    out = []
    attr, va, vb = FLIP[(factor, direction)]
    for k, (a, b) in enumerate(pairs):
        da, db = docs[2 * k], docs[2 * k + 1]
        aa, ab = doc_attributes(da), doc_attributes(db)
        ed = token_edit_distance(a, b)
        budget = 3 if factor == "negation" else n_finite(da) + 1
        changed = [x for x in KEEP[factor] if aa[x] != ab[x]]
        neg_doc = db if direction == "add" else da
        if a.strip() == b.strip():
            reason = "identical"
        elif (aa[attr], ab[attr]) != (va, vb):
            reason = f"{attr}_no_flip({aa[attr]}->{ab[attr]})"
        elif factor == "negation" and not any(c.dep_ == "neg" for c in root_of(neg_doc).children):
            reason = "neg_not_main_clause"     # doc-level neg flipped, but not on the ROOT
        elif factor == "negation" and direction == "remove" and stranded_npi(a, b):
            reason = "npi_stranded"             # e.g. "didn't even" -> "did even"
        elif changed:
            reason = "changed_" + "+".join(changed)
        elif ed > budget:
            reason = "edit_distance"
        elif cos[k] < min_cos:
            reason = "cosine"
        else:
            reason = None
        out.append({"accepted": reason is None, "reason": reason, "edit_dist": ed,
                    "edit_budget": budget, "cosine": round(cos[k], 4),
                    "attrs_a": aa, "attrs_b": ab})
    return out


def run_english_verifier(factor, records, min_cos):
    """records with 'direction', 'a', 'b' -> same records + verdict fields, grouped by direction."""
    for d in DIRECTIONS[factor]:
        idx = [k for k, r in enumerate(records) if r["direction"] == d]
        if idx:
            vs = verify_english(factor, d, [(records[k]["a"], records[k]["b"]) for k in idx],
                                min_cos)
            for k, v in zip(idx, vs):
                records[k].update(v)


# ---------------------------------------------------------------- translation verifier (3.5)
def _ws_tokens(t):
    return re.findall(r"[^\s.,!?;:।،؟\"'()\[\]«»]+", t)


NEG_COUNTERS = {
    "zh": lambda t: len(re.findall(r"不|没|无|未", t)),
    "ja": lambda t: len(re.findall(r"ませんでした|なかった|ません|ない|ず", t)),
    "de": lambda t: len(re.findall(r"\b(nicht|kein|keine|keinen|keinem|keiner|keines)\b", t, re.I)),
    "ru": lambda t: len(re.findall(r"\b(не|нет|ни)\b", t, re.I)),
    "es": lambda t: len(re.findall(r"\b(no|não|nunca|nada|nem|ni)\b", t, re.I)),
    "pt": lambda t: len(re.findall(r"\b(no|não|nunca|nada|nem)\b", t, re.I)),
    "fr": lambda t: len(re.findall(r"\bne\b|\bn['’]|\bpas\b|\bjamais\b|\baucun\w*", t, re.I)),
    # Arabic/Devanagari: whitespace tokens (\b is unreliable with combining marks)
    "ar": lambda t: sum(bool(re.fullmatch(r"[وف]?(لا|لم|لن|ليس\w*|لست\w*|ما|غير)", w))
                        for w in _ws_tokens(t)),
    "hi": lambda t: sum(w in ("नहीं", "न", "मत") for w in _ws_tokens(t)),
}
TENSE_MARKERS = {"zh": r"了|过", "ja": r"でした|ました|かった|た|だ"}


def verify_translation(factor, direction, a, b, min_ratio):
    """One language's {a, b} -> (accepted, reason, extra)."""
    if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()):
        return False, "null", {}
    a, b = a.strip(), b.strip()
    if a == b:
        return False, "identical", {}
    ratio = char_ratio(a, b)
    extra = {"char_ratio": round(ratio, 4)}
    if ratio < min_ratio:
        return False, "char_ratio", extra
    return True, None, extra


def cmd_verify(args, work):
    stage = {"edit": "edit", "translate": "tr", "backtrans": "bt"}[args.stage]
    if stage == "bt" and not args.translate_model:
        raise SystemExit("verify backtrans needs --translate-model")
    name = stage_name(stage, args.factor, args.model, args.translate_model)
    meta_path = work.p("inputs", name + ".meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    res = load_results(work, name)
    out = work.p("verified", name + ".jsonl")
    if stage == "edit":
        recs = verify_edit(args, work, res)
    elif stage == "tr":
        recs = verify_tr(args, work, res, meta)
    else:
        recs = verify_bt(args, work, res, meta)
    write_jsonl(out, recs)
    print(f"wrote {out}")


def verify_edit(args, work, res):
    cands = {c["pair_id"]: c for c in read_jsonl(work.candidates) if c["factor"] == args.factor}
    by_id, parsed_chunks = {}, set()
    for cid, content in res.items():
        obj = parse_json_answer(content)
        items = obj.get("items") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
        if not isinstance(items, list):
            continue
        parsed_chunks.add(cid)
        for it in items:
            if isinstance(it, dict) and "id" in it:
                try:
                    by_id[int(it["id"])] = it
                except (TypeError, ValueError):
                    pass
    recs, to_check = [], []
    for k, (pid, c) in enumerate(cands.items()):   # same order and chunking as prepare
        chunk = f"edit:{args.factor}:c{k // EDIT_CHUNK:05d}"
        r = {"pair_id": pid, "factor": args.factor, "direction": c["direction"],
             "source": c["source"], "a": c["text"], "b": None, "op": None,
             "returned": chunk in res and res[chunk] is not None,
             "parsed": chunk in parsed_chunks, "accepted": False, "reason": None}
        it = by_id.get(pid)
        if not r["returned"]:
            r["reason"] = "not_returned"
        elif not r["parsed"]:
            r["reason"] = "unparsed"
        elif it is None:
            r["reason"] = "missing_item"
        else:
            r["op"], r["b"] = it.get("op"), it.get("variant")
            # the op label is advisory (models mislabel correct edits); the verifier decides
            r["op_label_ok"] = r["op"] == c["direction"]
            if r["op"] == "skip" or (r["op"] not in DIRECTIONS[args.factor] and not r["b"]):
                r["reason"] = "skip"
            elif not isinstance(r["b"], str) or not r["b"].strip():
                r["reason"] = "no_variant"
            else:
                r["b"] = r["b"].strip()
                to_check.append(r)
        recs.append(r)
    run_english_verifier(args.factor, to_check, args.min_cos)
    summarize(f"edit {args.factor} {args.model}", recs)
    return recs


def verify_tr(args, work, res, meta):
    edits = {r["pair_id"]: r for r in accepted_edits(work, args.factor, meta["edit_model"])}
    recs = []
    for pid, e in edits.items():
        cid = f"tr:{args.factor}:{pid}"
        content = res.get(cid)
        obj = parse_json_answer(content) if content is not None else None
        r = {"pair_id": pid, "factor": args.factor, "direction": e["direction"],
             "en": {"a": e["a"], "b": e["b"]}, "returned": content is not None,
             "parsed": isinstance(obj, dict), "langs": {}}
        for lang in TARGET_LANGS:
            v = obj.get(lang) if isinstance(obj, dict) else None
            a = v.get("a") if isinstance(v, dict) else None
            b = v.get("b") if isinstance(v, dict) else None
            if not r["returned"]:
                ok, reason, extra = False, "not_returned", {}
            elif not r["parsed"]:
                ok, reason, extra = False, "unparsed", {}
            else:
                ok, reason, extra = verify_translation(args.factor, e["direction"], a, b,
                                                       args.min_ratio)
            if ok and args.factor == "negation":
                neg_member, aff_member = (b, a) if e["direction"] == "add" else (a, b)
                cn, ca = NEG_COUNTERS[lang](neg_member), NEG_COUNTERS[lang](aff_member)
                extra.update(neg_count_neg=cn, neg_count_aff=ca)
                if cn < ca + 1:
                    ok, reason = False, "neg_marker"
            if ok and args.factor == "tense" and lang in TENSE_MARKERS:
                past, pres = (b, a) if e["direction"] == "to_past" else (a, b)
                cp = len(re.findall(TENSE_MARKERS[lang], past))
                cq = len(re.findall(TENSE_MARKERS[lang], pres))
                extra["aspect_marker_ok"] = cp > cq          # soft: logged, not decisive
            r["langs"][lang] = {"a": a, "b": b, "accepted": ok, "reason": reason, **extra}
        recs.append(r)
    flat = [{"accepted": v["accepted"], "reason": v["reason"], "lang": l}
            for r in recs for l, v in r["langs"].items()]
    summarize(f"translate {args.factor} {args.model}", flat, by="lang")
    if args.factor == "tense":
        for lang in TENSE_MARKERS:
            vs = [r["langs"][lang] for r in recs if r["langs"][lang]["accepted"]]
            if vs:
                print(f"  {lang} aspect marker in past member only: "
                      f"{sum(v['aspect_marker_ok'] for v in vs)}/{len(vs)} (soft)")
    return recs


def verify_bt(args, work, res, meta):
    trs = {r["pair_id"]: r for r in accepted_translations(work, args.factor,
                                                          meta["translate_model"])}
    recs, to_check = [], []
    for cid, content in res.items():
        _, _, pid, lang = cid.split(":")
        pid = int(pid)
        t = trs.get(pid)
        if t is None:
            continue
        obj = parse_json_answer(content)
        r = {"pair_id": pid, "lang": lang, "factor": args.factor, "direction": t["direction"],
             "returned": content is not None, "parsed": isinstance(obj, dict),
             "accepted": False, "reason": None, "a": None, "b": None}
        if not r["returned"]:
            r["reason"] = "not_returned"
        elif not r["parsed"]:
            r["reason"] = "unparsed"
        elif not (isinstance(obj.get("A"), str) and isinstance(obj.get("B"), str)):
            r["reason"] = "missing_member"
        else:
            r["a"], r["b"] = obj["A"].strip(), obj["B"].strip()
            to_check.append(r)
        recs.append(r)
    run_english_verifier(args.factor, to_check, args.min_cos)
    summarize(f"backtrans {args.factor} {args.model}", recs, by="lang")
    if args.factor == "negation":   # calibration of the regex check (it already passed)
        n = len(recs)
        agree = sum(r["accepted"] for r in recs)
        print(f"  regex/back-translation agreement on the negation sample: {agree}/{n}")
    return recs


def summarize(title, recs, by=None):
    print(f"{title}: {sum(r['accepted'] for r in recs)}/{len(recs)} accepted")
    groups = defaultdict(list)
    for r in recs:
        groups[r.get(by) if by else "all"].append(r)
    for g, rs in sorted(groups.items()):
        reasons = Counter(r["reason"] for r in rs if not r["accepted"])
        acc = sum(r["accepted"] for r in rs)
        print(f"  {g:<4s} {acc:>6d}/{len(rs):<6d} {100 * acc / max(1, len(rs)):5.1f}%  "
              + ", ".join(f"{k}:{v}" for k, v in reasons.most_common(6)))


# ---------------------------------------------------------------- report
def cmd_report(args, work):
    subs = work.latest_submissions().values()
    metrics = {}
    for s in subs:
        p = work.p("results", f"{s['batch_id']}.result.json")
        if os.path.exists(p):
            m = json.load(open(p))
            metrics.setdefault(s.get("name"), []).append(
                {k: v for k, v in m.items() if k not in ("results_file", "progress_file")})
    report = []
    for fn in sorted(os.listdir(work.p("verified"))):
        if not fn.endswith(".jsonl"):
            continue
        name = fn[:-6]
        stage, factor, model = name.split(".", 2)
        recs = read_jsonl(work.p("verified", fn))
        meta_path = work.p("inputs", name + ".meta.json")
        n_req = json.load(open(meta_path)).get("n_requests") if os.path.exists(meta_path) else None
        if stage == "tr":
            rows = [{"lang": l, "returned": r["returned"], "parsed": r["parsed"]
                     and v.get("reason") != "null", **v} for r in recs for l, v in r["langs"].items()]
        else:
            rows = recs
        groups = defaultdict(list)
        for r in rows:
            groups[r.get("lang", "en")].append(r)
        for g, rs in sorted(groups.items()):
            report.append({
                "stage": stage, "factor": factor, "model": model.replace("--", "/"), "lang": g,
                "requests": n_req, "submitted": len(rs),
                "returned": sum(bool(r.get("returned")) for r in rs),
                "parsed": sum(bool(r.get("parsed")) for r in rs),
                "accepted": sum(bool(r["accepted"]) for r in rs),
                "reasons": dict(Counter(r["reason"] for r in rs if not r["accepted"])),
                "batch_metrics": metrics.get(name)})
    with open(work.p("report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"{'stage':<5s} {'factor':<9s} {'model':<42s} {'lang':<4s} {'subm':>6s} {'ret':>6s} "
          f"{'pars':>6s} {'acc':>6s} {'acc%':>6s}  top rejections")
    for r in report:
        top = ", ".join(f"{k}:{v}" for k, v in Counter(r["reasons"]).most_common(3))
        print(f"{r['stage']:<5s} {r['factor']:<9s} {r['model']:<42s} {r['lang']:<4s} "
              f"{r['submitted']:>6d} {r['returned']:>6d} {r['parsed']:>6d} {r['accepted']:>6d} "
              f"{100 * r['accepted'] / max(1, r['submitted']):>5.1f}%  {top}")
    print(f"wrote {work.p('report.json')}")


# ---------------------------------------------------------------- package (3.7)
def cmd_package(args, work):
    import numpy as np
    tok = tokenizer()
    pairs = []          # (pair_id, factor, direction, {lang: (a, b)}, verdicts)
    for factor in args.factors.split(","):
        edits = {r["pair_id"]: r for r in accepted_edits(work, factor, args.edit_model)}
        trs = {r["pair_id"]: r for r in accepted_translations(work, factor, args.translate_model)}
        bt_ok = None
        if factor == "tense" or args.require_bt_negation:
            if not args.backtrans_model:
                raise SystemExit("--backtrans-model needed for tense")
            bt_path = work.p("verified", stage_name("bt", factor, args.backtrans_model,
                                                    args.translate_model) + ".jsonl")
            bt_ok = {(r["pair_id"], r["lang"]) for r in read_jsonl(bt_path) if r["accepted"]}
        kept = 0
        for pid, t in trs.items():
            if pid not in edits:
                continue
            langs = {"en": (edits[pid]["a"], edits[pid]["b"])}
            for lang, v in t["langs"].items():
                if v["accepted"] and (bt_ok is None or (pid, lang) in bt_ok):
                    langs[lang] = (v["a"], v["b"])
            if len(langs) - 1 >= args.min_langs:
                pairs.append((pid, factor, t["direction"], langs, t["langs"]))
                kept += 1
        print(f"{factor}: {kept}/{len(trs)} pairs with >= {args.min_langs} accepted target languages")
    os.makedirs(args.out, exist_ok=True)

    def enc(t):
        return tok.encode(t, add_special_tokens=True, truncation=True, max_length=args.max_len + 1)

    # one shard per target language in the opus100 layout: src = X member, tgt = English member;
    # the two members of a pair are consecutive rows with the same pair_id
    for lang in TARGET_LANGS:
        src, tgt, pid_arr, fac_arr, mem_arr = [], [], [], [], []
        for pid, factor, _, langs, _ in pairs:
            if lang not in langs:
                continue
            ids = [(enc(langs[lang][m]), enc(langs["en"][m])) for m in (0, 1)]
            if any(len(x) > args.max_len for s_t in ids for x in s_t):
                continue
            for m, (s, t) in enumerate(ids):
                src.append(s); tgt.append(t)
                pid_arr.append(pid); fac_arr.append(FACTOR_ID[factor]); mem_arr.append(m)
        so = np.zeros(len(src) + 1, np.int32); np.cumsum([len(s) for s in src], out=so[1:])
        to = np.zeros(len(tgt) + 1, np.int32); np.cumsum([len(t) for t in tgt], out=to[1:])
        path = os.path.join(args.out, f"{lang}_en.npz")
        np.savez(path,
                 src_tokens=np.array([x for s in src for x in s], np.int32), src_offsets=so,
                 tgt_tokens=np.array([x for t in tgt for x in t], np.int32), tgt_offsets=to,
                 n_pairs=np.array(len(src), np.int32), lang_code=np.array(lang),
                 pair_id=np.array(pid_arr, np.int32), factor=np.array(fac_arr, np.uint8),
                 member=np.array(mem_arr, np.uint8))
        print(f"  [{lang}] {len(src) // 2} pairs ({len(src)} rows) -> {path}")
    side = os.path.join(args.out, "minpairs.jsonl")
    write_jsonl(side, [{"pair_id": pid, "factor": f, "direction": d,
                        "texts": {l: {"a": a, "b": b} for l, (a, b) in langs.items()},
                        "verdicts": verdicts} for pid, f, d, langs, verdicts in pairs])
    print(f"sidecar {side}")


# ---------------------------------------------------------------- main
def add_run_args(s):
    s.add_argument("--cluster", choices=sorted(CHAT_URL), default="sophia")
    s.add_argument("--workers", type=int, default=8, help="concurrent requests")
    s.add_argument("--timeout", type=float, default=600, help="per-request seconds")
    s.add_argument("--limit", type=int, help="send at most this many requests (testing)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.path.join(REPO, "augment", "work"))
    p.add_argument("--remote-root", default=os.environ.get("NUBUN_AUGMENT_REMOTE"),
                   help="work dir as Sophia sees it (on /eagle); default: --work itself")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("candidates")
    s.add_argument("--n", type=int, default=300, help="candidates per factor")
    s.add_argument("--data", default=os.path.join(REPO, "data", "opus100"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--chunk", type=int, default=4000, help="rows decoded per spaCy pass")
    s.add_argument("--max-draw", type=int, default=400_000, help="row cap per shard")

    s = sub.add_parser("smoke")
    s.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    s.add_argument("--line-format", choices=["openai", "body"], default="openai")
    s.add_argument("--yes-staged", action="store_true",
                   help="inputs are already visible to Sophia at --remote-root")

    s = sub.add_parser("prepare")
    s.add_argument("stage", choices=["edit", "translate", "backtrans"])
    s.add_argument("--factor", choices=FACTORS, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--edit-model", help="translate: whose accepted English pairs")
    s.add_argument("--translate-model", help="backtrans: whose accepted translations")
    s.add_argument("--sample-frac", type=float, default=0.1,
                   help="backtrans negation: fraction of pairs (tense: all)")
    s.add_argument("--json-mode", action="store_true", help="add response_format json_object")
    s.add_argument("--line-format", choices=["openai", "body"], default="openai")
    s.add_argument("--submit", action="store_true", help="create an ALCF batch (inputs on /eagle)")
    s.add_argument("--run", action="store_true", help="send interactively (no /eagle needed)")
    add_run_args(s)

    s = sub.add_parser("run", help="send a prepared input to the interactive endpoint (resumable)")
    s.add_argument("--input", required=True)
    s.add_argument("--model")
    add_run_args(s)

    s = sub.add_parser("submit")
    s.add_argument("--input", required=True)
    s.add_argument("--model")

    s = sub.add_parser("status")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--interval", type=int, default=300)

    s = sub.add_parser("collect")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("inspect")
    s.add_argument("--batch", required=True)
    s.add_argument("--n", type=int, default=3)

    s = sub.add_parser("verify")
    s.add_argument("stage", choices=["edit", "translate", "backtrans"])
    s.add_argument("--factor", choices=FACTORS, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--min-cos", type=float, default=0.3,
                   help="MiniLM floor; negation lowers paraphrase cosine by design (README 3.3)")
    s.add_argument("--min-ratio", type=float, default=0.6)
    s.add_argument("--translate-model", help="backtrans: whose translations were back-translated")

    sub.add_parser("report")

    s = sub.add_parser("package")
    s.add_argument("--factors", default="negation,tense")
    s.add_argument("--edit-model", required=True)
    s.add_argument("--translate-model", required=True)
    s.add_argument("--backtrans-model")
    s.add_argument("--require-bt-negation", action="store_true",
                   help="also require back-translation for negation (only sampled pairs survive)")
    s.add_argument("--min-langs", type=int, default=9, help="accepted target languages per pair")
    s.add_argument("--max-len", type=int, default=128)
    s.add_argument("--out", default=os.path.join(REPO, "data", "minpairs"))

    args = p.parse_args()
    work = Work(args.work, args.remote_root)
    {"candidates": cmd_candidates, "smoke": cmd_smoke, "prepare": cmd_prepare,
     "run": cmd_run, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect,
     "inspect": cmd_inspect, "verify": cmd_verify, "report": cmd_report,
     "package": cmd_package}[args.cmd](args, work)


if __name__ == "__main__":
    main()
