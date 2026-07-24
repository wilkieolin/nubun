# Nubun — Top Radicals Discovered (Phase 8 winner, C1)

Model: `phase8_C1_d512_cos130k_step130000.pt` — Perceiver encoder → **RVQ (8×128)**
bottleneck → unfrozen-embedding AR decoder, **d_model 512** (widened past the XLM-R
384 embedding via input/output projections), tied embeddings, trained on opus100
(130k steps, gentle cosine schedule, semantic + weighted-CE losses).
**Round-trip cross-lingual meaning = 0.5045** (58.6% of the gold-translation
ceiling of 0.861) — the best of the Phase 8 scale-up (0.435 → 0.5045 via wider
network + gentle schedule; see `PHASE8.md`).

**Method (`analyze_codebook.py`, output-side discovery).** For each codebook entry we
decode many latent sequences containing that code vs. matched controls without it, in
all 10 languages, and score every output token by log-odds (test vs. control). A
code's "meaning" is the set of tokens it reliably induces. Reported here are **level-0
codes** (the coarsest RVQ residual = the primary radical); levels 1–7 add finer,
progressively less interpretable corrections. Scores in parentheses are cross-lingual
aggregate log-odds. Full inventory: `results/C1_codebook_L0.txt`.

The higher-capacity winner resolves a **richer, more differentiated** set of concepts
than the earlier d384 models — finer gender/number/temporal distinctions and new
function-word and entity classes.

## Pronouns (emergent)
- **HE** — Char 41: 他 / Er / него / he / He
- **YOU** — Char 59: Vous / êtes / You / أنت / тебя
- **WE** — Char 28: we / nosotros / 他们 / रहे

## Function words (new at this scale)
- **CAN (modal)** — Char 32: puedo / peux / pode / puede / peut / могу / 不能
- **SAY / said** — Char 38: gesagt / disse / dijo / sagen / dit / कहा / says
- **NEGATION** — Char 60: No / Нет; Char 39: not / ليس / ذلك
- **connective / affirmation** — Char 44: Por / Y / و / और / हाँ; Char 26: OK (ठीक / حسن / Да / Well)

## Number & measure
- **TWO / count** — Char 4: 2 / 3 / dos / 以上 / bis
- **legal ARTICLE** — Char 33: article / articles / artículo / المادة / Artikel / 第

## Content concepts
- **WOMAN** — Char 18: женщин / Frauen / mujeres / 妇女 / mujer / femme / femmes / 女
- **MAN** — Char 45: man / hommes / homens / Mann
- **NAME** — Char 14: nombre / اسم / nom / имя
- **PHONE / call** — Char 21: phone / 電話 / call / llama / llamada / फोन
- **HOUSE / room** — Char 53: Haus / Raum / доме / quarto
- **LIFE / world** — Char 61: vida / जीवन / 世界 / Welt / life
- **NIGHT** — Char 8: رात / Nacht (+ homem)
- **POLICE / victim** — Char 40: 警 / polícia / victim
- **URL / web** — Char 50: www (+11.81, strongest single code) / http / W
- **development / system** — Char 16: sistema / деятельности / 研究 / вопросы; Char 12: Entwicklung / desarrollo

Each fires on the same meaning across up to 10 languages and 5 scripts.

## Interlingua (bag-of-codes)
`crosslingual_consistency.py` on C1 (all 10 langs, 45 pairs): mean real Jaccard
**0.433**, chance 0.171, **lift +0.262** — every language pair positive. Parallel
translations share ~43% of their primary radicals vs ~17% at chance. The interlingua
is **not data-limited**: adding direct non-English-centric pairs (Phase 8 data lever)
did not raise the lift (+0.268, tied) — the codes are already language-neutral from
English-pivoted training + the semantic loss.
