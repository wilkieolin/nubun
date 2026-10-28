# Minimal-pair augmentation: specification (Phase 9, branch 2)

Status: implemented as `minpairs.py`; pilot run interactively on 2026-10-28
(results in §9). The script drives the ALCF inference service through the
stages below (`python augment/minpairs.py --help` for the subcommands); this
document is its target. **Interactive mode (`run`, `prepare ... --run`) is the
default path**: it needs only the token, not `/eagle`; the batch path (§4) is
kept but unused.

## 1. Why

`PHASE9.md` §7 A4 and §8 finding 7: parallel translations identify the meaning
of a sentence as a whole (the interlingua is seed-robust) but not its
factorisation into radicals (95% of level 0 is seed-specific). Translations
share every meaning factor and differ only in language, so they cannot
identify structure inside meaning. Pairs that share all meaning factors but
one, with the changed factor known, do identify it (Locatello et al. 2020,
weakly supervised disentanglement; Shu et al. ICLR 2020, consistency and
restrictiveness; Klindt et al. ICLR 2021 and Ahuja et al. 2022, sparse
transitions). This directory produces such pairs for the two factors that are
categorical, frequent in every language and expressible as a sentence edit:
**polarity of the main clause (negation)** and **tense of the main clause
(present vs past)**.

Two kinds of pair are needed (Shu et al.):

| pair type | members | what it identifies | source |
|---|---|---|---|
| change pair (minimal pair) | same sentence, factor flipped | restrictiveness: the factor's effect is confined to one code | generated here |
| share pair | two unrelated sentences with the same factor value | consistency: that code depends only on the factor | free, from the attribute labels `compositionality_metrics.py` already computes |

Only change pairs need generation.

## 2. Deliverable

A corpus of minimal pairs, each pair present in all ten languages, with:

- `pair_id`, `factor` (`negation` | `tense`), `direction` (`add`/`remove` or
  `to_past`/`to_present`), `source` (opus100 shard and row),
- for each language: `a` and `b` text, their XLM-R token ids (the
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` tokenizer, as
  the rest of the corpus), and the verification verdicts,
- packaged in the opus100 shard layout (`src_tokens`/`src_offsets`, see
  `build_opus100_corpus.py`) plus `pair_id` and `factor` arrays, so the trainer
  can co-locate the two members of a pair in one batch.

Target size: tens of thousands of pairs (20k per factor to start), mixed into
the 2M-pair opus100 corpus, not replacing it.

## 3. Pipeline

```
opus100 English sides ──filter──> candidates ──batch LLM edit──> English pairs
   ──local verify──> accepted English pairs ──batch LLM joint translation──> 10-language pairs
   ──local verify (+ batch back-translation for tense)──> accepted pairs ──package──> shards
```

Every LLM stage is followed by a local, LLM-free verifier. The verifier is
what makes the corpus trustworthy; the prompts only make the yield high.

### 3.1 Candidates

Source: the English side (`tgt_*` arrays) of the opus100 shards in
`data/opus100/`, detokenised with the MiniLM tokenizer, or the raw
`Helsinki-NLP/opus-100` text when it is cached. Seeded sample, balanced across
the nine shards. Filters, using spaCy `en_core_web_sm` with the same rules as
`label_attributes()` in `compositionality_metrics.py`:

- 6 to 25 tokens, exactly one sentence, no quotation marks, no URL, no all-caps
  headline;
- a ROOT with a finite tense and a subject;
- negation: half the candidates currently negated (`neg=1`, direction
  `remove`), half not (`add`);
- tense: `tense` in {pres, past}, no future auxiliary, no modal on the root, no
  time adverbial from a stop list (yesterday, tomorrow, ago, last, next, now,
  currently, soon);
- 300 per factor for the pilot, 50k per factor for production (expect ~40%
  loss by the end).

### 3.2 LLM edit (English)

One request per 20 candidates, JSON answer with one item per input id:
`op` (`add` | `remove` | `skip`, or `to_past` | `to_present` | `skip`) and
`variant`. The prompt for each factor must state:

- negation: toggle polarity on the main clause's finite verb only, with
  do-support where English needs it; match the source's contraction register;
  allow only forced agreement changes (some/any, already/yet, too/either);
  never add or remove adverbs (never, anymore, at all); skip fragments,
  headlines, quotations, sentences whose only negation is in a subordinate
  clause, and idioms whose meaning the toggle changes (can't help, not only).
- tense: switch the main clause between simple present and simple past,
  carrying sequence-of-tense agreement, keeping aspect (progressive stays
  progressive, perfect stays perfect); change nothing else; skip timeless
  generalities, sentences with a conflicting time adverbial, futures, modals,
  headlines, quotations, fragments.
- both: three or four worked examples including one `skip`; "output only the
  JSON object"; temperature 0; for reasoning models, the lowest reasoning
  effort.

### 3.3 Local verification of English pairs

Accept a pair only if all hold:

| check | negation | tense |
|---|---|---|
| spaCy attribute diff | `neg` flips; `tense`, `subj_person`, `subj_number`, NER flags unchanged | `tense` flips pres<->past; `neg`, `subj_*`, NER flags unchanged |
| whitespace-token edit distance (difflib) | <= 3 (do-support + particle + one agreement word) | <= number of finite verbs + 1 |
| MiniLM cosine between members | >= 0.6 | >= 0.6 |
| identity | members differ | members differ |

Record the reason for every rejection. Expected yield 85 to 95%.

### 3.4 LLM joint translation

One request per accepted English pair returning all nine target languages
(`zh, es, fr, ar, ru, hi, ja, pt, de`), each as `{"a", "b"}`. The prompt must
state: A and B differ only in the named factor; translate A, then derive B
from the translation of A by the minimal change (never translate B
independently); the two translations must be word-for-word identical except
for the factor; realise the factor with the language's smallest conventional
device (negation particle or negative verb form; past-tense form, or in Chinese
and Japanese the aspect/past marker 了 / 过 / た-form), never with a time
adverb or a different word choice; return null for a language where this is
impossible.

### 3.5 Local verification of translations

Per language, accept only if:

- the two members differ and their character-level similarity ratio
  (difflib) is >= 0.6, so a re-translated B is rejected;
- negation: a per-language negation-marker pattern matches exactly one member
  (zh 不|没|无|未; ja ない|ません|ず|なかった|ませんでした; de nicht|kein|keine|keinen;
  ru не|нет|ни; es/pt no|não|nunca|nada|nem; fr ne|n'|pas|jamais|aucun; ar لا|لم|لن|ليس|ما|غير;
  hi नहीं|न|मत), with the richer member's count higher by at least one;
- tense in zh/ja: the aspect/past marker is present in exactly one member
  (soft check, logged, not decisive);
- tense in all languages: decided by back-translation (3.6).

### 3.6 Back-translation (tense, and a sample of negation)

One request per (pair, language) asking for literal English translations of
both members. The English verifier of 3.3 is then applied to the two
back-translations: accept the language if the factor flips and nothing else
changes. Also run on a 10% sample of negation pairs as a calibration of the
regex check.

### 3.7 Packaging

Accepted pairs tokenised with the MiniLM tokenizer; written as shards in the
opus100 layout with two extra arrays, `pair_id` (int32, same id on the two
members) and `factor` (uint8: 0 negation, 1 tense); a sidecar JSONL with the
text and verdicts for inspection. The trainer change (pair co-location in a
batch, a penalty on the number of slots whose code stack differs between the
members beyond a budget of one or two, and a consistency term on share pairs)
is specified in `PHASE9.md` §8 branch 2 and is not part of this directory.

## 4a. ALCF interactive mode (used)

`prepare <stage> ... --run` (or `run --input inputs/<name>.jsonl`) POSTs each
request line's body to `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1/chat/completions`
with `--workers` concurrent requests, retrying 429/5xx/timeouts. Results are
written as `results/local-<name>-<input sha>.results.jsonl` in the OpenAI batch
result shape and registered in `submissions.jsonl`, so `verify`/`report` are
unchanged. Re-running the same command resumes (skips custom_ids already
answered); editing a prompt changes the input digest and starts a fresh run.

- Token: `~/inference_auth_token.py get_access_token` (run with a Python that has
  `globus_sdk`, e.g. the `globus_env` conda env; override with `ALCF_AUTH_PYTHON`),
  cached 10 min; falls back to `ALCF_ACCESS_TOKEN` / `ALCF_BEARER_TOKEN`. A 401
  "high-assurance timeout" means the Globus session must be redone
  (logout at app.globus.org, then `inference_auth_token.py authenticate --force`
  with an anl.gov / alcf.anl.gov identity); a refresh alone does not clear it.
- Minerva (`/resource_server/minerva/api/v1`, `--cluster minerva`) on 2026-10-28:
  muse-spark-bf16, nemotron-3-ultra, beam-1.0-bf16, inkling-bf16, gpt-oss-120b;
  Metis: Mistral-Large-3-675B-Instruct-2512, gemma-4-31B-it, gpt-oss-120b.
- Hot on Sophia on 2026-10-28 (`GET .../resource_server/sophia/jobs`): Llama 3.1
  8B/70B, gpt-oss-20b/120b (2 nodes), gemma-4-31B-it (2 nodes), gemma-3-27b-it.
  405B was not running.
- gpt-oss: `reasoning_effort: low` still spends 1-3k reasoning tokens on a
  20-item edit, which count against `max_tokens`; `chat_body` adds 4000.
- Measured: gpt-oss-120b 8-16 workers 800-1300 out tok/s; Llama 3.1 70B ~27
  tok/s per stream, ~190 tok/s at 8-16 workers. Pilot stages take 1-7 min each.

## 4. ALCF batch service (not used)

From docs.alcf.anl.gov/services/inference-endpoints/batch/ and /models/:

- Batches run only on **Sophia** (A100, vLLM) and only for models flagged B:
  `meta-llama/Meta-Llama-3.1-8B-Instruct`, `-70B-Instruct`, `-405B-Instruct`,
  `meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`,
  `google/gemma-3-27b-it`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`.
  Minerva's `muse-spark-bf16` and the other Minerva/Metis/Tara models are
  interactive only (chat completions), no batch endpoint.
- Auth: bearer token with the `inference` scope
  (`alcf-tokens get-token inference`; Python `alcf_tokens.auth.get_access_token`).
- Create: `POST https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1/batches`
  with `{"model", "input_file", "output_folder_path"}`; input must be JSONL on
  `/eagle/argonne_tpc/...` or a world-readable path.
- Status: `GET .../resource_server/v1/batches/{batch_id}` (pending | running |
  failed | completed). Result: `GET .../{batch_id}/result` returns
  `results_file` (`<input>_<model>_<batch>/*.results.jsonl`), `progress_file`
  and metrics. List: `GET .../resource_server/v1/batches`. Cancel is not
  self-service.
- Limits: 150k requests per file, 10 MB per request; cold start 10-15 min for
  models that are not always hot (405B, Llama 3.3 70B, Scout, Gemma 3 are not
  hot; 3.1 8B/70B and gpt-oss are); dynamically loaded models unload after 2 h
  idle. No published turnaround or concurrency limits.
- **Unknowns to settle with a 3-line smoke batch before anything else:** the
  per-line request format (assumed vLLM/OpenAI batch format
  `{"custom_id","method":"POST","url":"/v1/chat/completions","body":{...}}`;
  fallback: the chat body alone), the results line shape, whether
  `response_format` JSON mode / guided decoding is honoured, and the body of
  the create response (assumed to carry `batch_id`).
- `collect` must run where `/eagle` is mounted (Sophia/Polaris login node), or
  the results files are rsynced to the GB10 first; verification and packaging
  can run anywhere with spaCy and the tokenizer.

Request line conventions: `custom_id = <stage>:<factor>:<pair_id>[:<lang>]`;
`temperature 0`; `max_tokens` 1200 for edits (20 items), 1500 for a nine-language
translation, 300 for a back-translation; the system prompt identical across a
stage so any server-side prefix caching applies.

## 5. Pilot protocol

Goal: choose the production model(s) by measured minimality, per stage.

1. Smoke batch on `Meta-Llama-3.1-8B-Instruct` (hot) to fix the line format.
2. 300 candidates per factor. Submit the same edit input to at least:
   `Meta-Llama-3.1-405B-Instruct`, `google/gemma-3-27b-it`, `openai/gpt-oss-120b`
   (and `Llama-3.3-70B-Instruct` if cheap). Verify. Record acceptance per
   model and rejection reasons.
3. Take the accepted pairs of the best edit model; submit the same translation
   input to the same model set. Verify. Record acceptance per model and per
   language; expect Chinese and Japanese tense to be lowest.
4. Back-translate with a hot 70B model; apply the English verifier.
5. Optionally score the same 300 pairs through `muse-spark-bf16`
   interactively (chat completions on Minerva, packing 20 sentences or 5 pairs
   per request) to see the ceiling the batch models give up.
6. Human read of 30 accepted and 30 rejected translations per language for
   zh, ja, ar, hi, to calibrate the regex and similarity thresholds.

Decision rule: per stage, the cheapest model whose per-language acceptance is
within 5 points of the best model's. A split (a smaller model for English
edits, the largest for translation) is the expected outcome.

Report (`report` stage): for each stage x model x factor (x language):
submitted, returned, parsed, accepted, rejection-reason histogram, and the
model's batch metrics (throughput, response time) from the result endpoint.

## 6. Success criteria for the corpus

- >= 85% of English edits accepted for both factors on the chosen model.
- >= 70% of translations accepted in every language (>= 60% for zh/ja tense).
- Human calibration agrees with the automatic verdict on >= 90% of the
  sample.
- Production run: 20k accepted pairs per factor in all ten languages.

Downstream success (not this directory): on two seeds of the new recipe, the
cross-seed recurring fraction of level-0 radicals rises above the current
three-seed core (3-4 content radicals, 4-5%), negation and tense enter the
core, and the substitution test (swap the identified code into real encodings)
flips the decode as the factor predicts.

## 7. Cost and throughput

40k accepted pairs from 100k candidates: ~5k edit requests (20 items each,
~1k tokens in, ~600 out) and ~40k translation requests (~500 in, ~700 out),
plus ~360k short back-translation requests if run for every language (or
~40k if run for tense only in a 10% sample plus all of zh/ja). All within the
150k-per-file limit when split by stage and language. ALCF batch is
allocation-billed; the only external cost is time. For reference, the same
volume on a commercial API at frontier-model rates is in the low hundreds of
dollars with batch discounts; on a small hot model it is tens.

## 8. Open questions

- Whether `response_format` is honoured on the batch path; if not, strict
  "JSON only" instructions plus a tolerant parser (first `{` to last `}`).
- Whether the detokenised English from the shards is faithful enough, or the
  raw opus-100 text should be re-downloaded on the GB10 for candidates.
- Chinese and Japanese tense: whether the aspect-marker realisation counts as
  the same factor for the trainer's purposes, or whether tense pairs should be
  restricted to the eight languages with inflectional tense.
- How many share pairs the consistency term needs, and whether they should be
  drawn from opus100 or from the generated pairs' members.

## 9. Pilot results (2026-10-28, 300 candidates per factor)

Candidate filter additions after a first look (opus100 English is mostly
subtitles): no questions, no `..`/`--`/ellipsis, no imperatives (spaCy
Mood=Imp), no interjections, no OCR "l"/gotta/gonna/ain't/got to. The model's
`op` label is advisory (Llama mislabels correct edits); the verifier decides.
`--min-cos` lowered 0.6 -> 0.3: negation lowers the paraphrase cosine by
design (0.6 rejected 42/300 correct Llama pairs).

English edit acceptance (§3.3 verifier):

| model | negation | tense |
|---|---|---|
| gpt-oss-120b | 82.0% | 79.7% |
| gemma-4-31B-it | 79.7% | 74.0% |
| Llama-3.1-70B | 69.3% | 45.0% |
| gemma-3-27b-it | 69.0% | 36.3% |
| muse-spark-bf16 (Minerva) | 82.7% | 71.0% |
| inkling-bf16 (Minerva) | 84.7% | 69.3% |
| beam-1.0-bf16 (Minerva) | 83.3% | 69.0% |
| nemotron-3-ultra (Minerva) | 81.7% | 62.0% |
| Mistral-Large-3-675B (Metis) | 61.7% | 48.0% |

The Minerva models are reasoning models: at default effort a 20-item edit
spends 12-14k reasoning tokens and ran out of a 7.2k budget on half the
requests; `reasoning_effort: low` (now set for all reasoning models, with
+20k max_tokens headroom) cuts it to 1.6-3k (nemotron ~9k), 15 requests in
~1 min at 8 workers. Acceptance overstates gpt-oss on tense: of 14 sampled
items gpt-oss edited and muse skipped, ~10 gpt-oss edits are wrong in ways the
verifier cannot see (event dated in the past made present, "Someone once gets",
half-flipped sequence of tense "says ... would"). Where both accept, the
variants are identical 190/203 (tense), 209/237 (negation); on the negation
differences muse repairs negative-polarity items (anyone -> someone) that
gpt-oss leaves stranded. Acceptance rate is not precision: the model choice
needs the §5.6 human read.

Translation of gpt-oss's accepted edits, per-language acceptance (§3.5):
gpt-oss-120b negation 91-99%, tense 81-94%; gemma-4-31B returns null for most
Arabic (2-6%) and many zh/hi tense items, and its zh negation edits are
sometimes ungrammatical. gpt-oss-120b chosen for both stages.

Back-translation (Llama 3.1 70B, independent family) of gpt-oss translations:
tense 74-90% in eight languages, **zh 17%**: sentence-final 了 on statives
marks change of state ("doesn't like Russians anymore"), not past, so the zh
tense pairs are wrong and the soft marker check (196/202 "ok") is
uninformative. Negation 10% sample: back-translation agrees with the regex on
203/233 (87%), disagreements mostly `neg_not_main_clause` (to inspect).

Known leaks in accepted English pairs: stranded negative-polarity items on
`remove` ("didn't even" -> "did even"), negating discourse frames ("I mean"),
tense flips against explicit dates ("from 1995 to 2009").

### Translation: muse and inkling vs gpt-oss (same source: gpt-oss's accepted edits)

Negation = translation accepted (regex) / 246 source pairs; tense = translation
accepted AND Llama 3.1 70B back-translation accepted / 239 source pairs.

| | ar | de | es | fr | hi | ja | pt | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| neg gpt-oss-120b | 95.9 | 97.6 | 99.2 | 91.1 | 98.8 | 95.1 | 99.2 | 97.2 | 98.8 | 97.0 |
| neg muse-spark | 98.8 | 100 | 100 | 100 | 100 | 97.6 | 100 | 99.6 | 99.2 | 99.5 |
| neg inkling | 98.4 | 99.2 | 100 | 100 | 100 | 96.7 | 100 | 99.6 | 99.6 | 99.3 |
| tense gpt-oss-120b | 59.8 | 78.2 | 82.4 | 72.4 | 76.6 | 77.0 | 83.7 | 75.3 | 14.6 | 75.7 (no zh) |
| tense muse-spark | 75.3 | 91.6 | 92.1 | 88.7 | 86.6 | 83.7 | 89.5 | 85.4 | 18.8 | 86.6 (no zh) |
| tense inkling | 74.5 | 83.3 | 90.0 | 78.7 | 84.1 | 73.2 | 84.9 | 84.1 | 21.3 | 81.6 (no zh) |

Read side by side, muse's translations are also more accurate (gpt-oss: ja
記事12 for "article 12", de "Du wirst nicht im Voraus gesagt"). But muse
translates broken English B members faithfully where gpt-oss returned null
("His tomb exists anymore" -> fr "Son tombeau existe plus", which means the
opposite), so gpt-oss's nulls were a hidden filter. Fixed upstream: the English
verifier now rejects `remove` edits that keep a negative-polarity item licensed
by the deleted negation (`npi_stranded`: any/anyone/anything/anymore/ever/either,
"not even", "not yet", "at all", "any more"); 8/300 gpt-oss and 5/300 muse
negation edits, ~2 of each false positives (free-choice "any other", "if ...
any"). zh tense fails for every model (15-21%): drop zh from tense pairs.

Back-translation files are now named `bt.<factor>.<bt model>~<translate model>`
(`verify backtrans` and `package` take `--translate-model`), so two translators'
back-translations no longer share inputs or results.
