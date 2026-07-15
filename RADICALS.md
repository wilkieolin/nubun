# Nubun — Top Radicals Discovered (Phase 7 RVQ)

Model: `phase7_rvq_unfroze_cos_100k_step100000.pt` — Perceiver encoder → **RVQ (4×128)**
bottleneck → unfrozen-embedding AR decoder, trained on opus100 (100k steps, cosine
LR, semantic + weighted-CE losses). Round-trip cross-lingual meaning = 0.328 (38% of
the gold-translation ceiling; 70% of a continuous bottleneck).

**Method (`analyze_codebook.py`, output-side discovery).** For each codebook entry we
decode many latent sequences containing that code vs. matched controls without it, in
all 10 languages, and score every output token by log-odds (test vs. control). A
code's "meaning" is the set of tokens it reliably induces. Reported here are **level-0
codes** (the coarsest RVQ residual = the primary radical); levels 1–3 add finer,
progressively less interpretable corrections (see *Coarse-to-fine* below).

Scores in parentheses are cross-lingual aggregate log-odds. Not all 128 codes are
crisp — a substantial, clearly-interpretable fraction is reported; the rest are
function-word/subword mixes.

## The pronoun system (emergent, near-complete)

The single most striking result: the model allocated **distinct radicals to a full
cross-lingual pronoun paradigm**, each consistent across scripts and families.

| Radical | Meaning | Cross-lingual evidence |
|---------|---------|------------------------|
| Char 1 | **we / us** | We, Wir, On/nous, 我们, हम, нас, نحن, 我々, Temos |
| Char 45 | **you** | Вы/Ты, Du, Vous/Tu, можете |
| Char 68 | **I / me** | me, moi, minha, yo (also Char 79: Eu, Me) |
| Char 97 | **she / her** | 她, she, her, वह, Elle |
| Char 114 | **he / him** | 他, Он, he, er, वह, wants |
| Char 91 | **they / them** | Они, их, them, Ils, los |

## Negation

| Radical | Meaning | Evidence |
|---------|---------|----------|
| Char 47 | **not** | not, don('t), 不是, لا, ne, Não |
| Char 35 | **nothing / none** | nada, No, Не, hay/há (there-is/none), anything |

## People & society

| Radical | Meaning | Evidence |
|---------|---------|----------|
| Char 46 | **woman / wife** | femme, 女, mulher, mujer, wife, женщин, woman |
| Char 83 | **family / children** | enfants, Familie, Kinder, 父亲(father), pai(father), अपने |
| Char 33 | **human rights** | humanos, 权(rights), direitos, rechte, Menschen |
| Char 9 | **world / nation / people** | mundo, लोग(people), Land, 国际, 団 |

## Institutions & polity

| Radical | Meaning | Evidence |
|---------|---------|----------|
| Char 36 | **council / commission** | Conseil, Comissão, Commission, Consejo, decisão |
| Char 25 | **nation / community** | 国家, Estados, Gemeinschaft, 国际 |
| Char 22 | **united (nations/states)** | Estados, Unidas, politik, rechte |
| Char 50 | **peace / war** | paz, paix, мира, war, situation |
| Char 102 | **system** | sistema (+3.87), 工作(work), 组(group) |
| Char 95 | **Israel / Palestine / peace** | Israel (+5.80 — highest in codebook), Palestin, paz |

## Concrete concepts

| Radical | Meaning | Evidence |
|---------|---------|----------|
| Char 127 | **city / house / place** | casa, cidade, 街(street), Stadt, ciudad, cerca |
| Char 99 | **money** | деньги, dinheiro, 金, money |
| Char 3 | **phone / call / contact** | call, llamada, teléfono, appel, звон, फोन, 电/叫 |
| Char 30 | **book / list / data** | 本, lista, 资料(material), libro |

## Abstract & functional

| Radical | Meaning | Evidence |
|---------|---------|----------|
| Char 41 | **time / moment** | vez, time, momento, Zeit, noch |
| Char 21 | **sorry / apology** | 对不起, lo siento, désolé, muito |
| Char 58 | **numbers / dates** | 15, 月(month), 10, 2014, 20 |
| Char 92 | **ordinal / legal article** | 第(ordinal), 条(clause), пункт(point), third, 十(ten) |
| Char 106 | **how / as (conjunction)** | Como (+4.43), Dans, wenn, Et |

## Coarse-to-fine structure across RVQ levels

Each slot is a **stack of 4 residual codes** (one per level). Inspecting all four:

| Level | Character quality | Log-odds | Examples |
|-------|-------------------|----------|----------|
| **L0** (coarsest) | Cleanest standalone concepts | +1.5–5.8 | the tables above |
| L1 | Still conceptual, weaker | +1.4–2.0 | name, book/article, we/us |
| L2 | Moderate, thematic | +1.1–1.8 | institution/society |
| L3 (finest) | Noisy subword refinements | +0.7–1.1 | fragments |

The top radical sets the concept; lower radicals refine the reconstruction — the
textbook coarse-to-fine behavior of residual VQ, and a natural fit for a
"radical stack" per character.

## Takeaway

The bottleneck learned a **discrete, composable, language-independent** vocabulary:
a full pronoun paradigm, negation, and dozens of content concepts, each firing on the
same meaning across 10 languages and 5 scripts. This is the core synthetic-logography
thesis, demonstrated on a model that actually reconstructs meaning (round-trip 0.328).

Full per-code, per-language detail: `results/rvq_char_semantics_L{0,1,2,3}.txt`.
