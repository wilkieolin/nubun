# Literature review — where Nubun sits, and what the field already knows

Compiled 2026-10-12 from three surveys (machine learning; linguistics and writing
systems; mathematics and information theory). Each entry: citation, what it
shows, why it matters here. Verification status is marked where it is not a
canonical reference: **[verified]** = title/venue/year confirmed by search this
session; **[standard]** = canonical, not re-searched; **[preprint]** = 2025-26
arXiv, read from the abstract; **UNVERIFIED** = could not be confirmed, check
before citing. Nubun-specific conclusions are collected in §0 and in
`PHASE9.md`.

## 0. Synthesis: what the literature says about our plan

1. **The gap Nubun occupies is real.** No verified 2025-26 work offers a
   learned, discrete, multilingual, human-readable sentence-level vocabulary.
   The nearest neighbours each lack one property: Meta's Large Concept Model
   (continuous SONAR space, not readable), Token Assorted and the 2026
   abstract-chain-of-thought preprints (VQ codes as a reasoning vocabulary, but
   monolingual and flat), SAE feature dictionaries (interpretable but
   continuous, not sequences), emergent-language work (discrete and
   compositional but not grounded in natural language).
2. **Why Phases 1-2 failed and the VQ-VAE worked is now explainable.**
   Dictionary recovery from static embeddings is non-identifiable when atoms
   are coherent (Gribonval & Schnass 2010; Spielman et al. 2012), embedding
   geometry is seed-unstable (Wendlandt et al. 2018; Antoniak & Mimno 2018),
   and unsupervised disentanglement is impossible without an inductive bias
   (Locatello et al. 2019). Parallel sentences in 3+ languages are conditionally
   independent views of one latent meaning, which is exactly the Kruskal
   condition for generic identifiability of a discrete latent (Allman, Matias &
   Rhodes 2009). Multilingual end-to-end training is the supervision signal the
   decomposition approach lacked.
3. **Codebook collapse is a solved-enough problem.** FSQ (Mentzer et al. 2024)
   and LFQ (Yu et al. 2024) remove it structurally; SimVQ (Zhu et al. 2025),
   CVQ-VAE (Zheng & Vedaldi 2023), affine re-parameterisation (Huh et al. 2023)
   and encoder freezing (NSVQ 2026) are drop-ins for a learned codebook.
4. **Hierarchy is justified by interpretability and description length, not by
   reconstruction.** Reyhanian & Wiskott (2026) show a matched single-level VQ
   reconstructs as well as a hierarchical one. The case for 8 RVQ levels must
   rest on compositional structure and MDL, which we have not yet measured.
5. **A reconstruction-only objective minimises the information in the codes**
   (Kharitonov et al. 2020) and effective codes need not be compositional
   (Kottur et al. 2017). If level-0 is to be a semantic radical by design rather
   than by accident, it needs an explicit pressure: a cross-lingual agreement
   or information-bottleneck term on level-0 itself.
6. **Measure compositionality with existing tools.** Topographic similarity
   (Brighton & Kirby 2006), positional and bag-of-symbols disentanglement
   (Chaabouni et al. 2020), tree reconstruction error with additive composition
   (Andreas 2019), MIG (Chen et al. 2018), normalised PMI morpheme induction
   (Boldt & Mortensen 2025/26), and a COGS-style held-out-combination split
   (Kim & Linzen 2020). None has been run on Nubun codes.
7. **The human-readable layer has hard numbers to design against.** About 3
   strokes per glyph with 50% redundancy across 100+ scripts (Changizi &
   Shimojo 2005); perimetric complexity predicts letter-identification
   efficiency and a new alphabet is learned in ~3,000 trials (Pelli et al.
   2006); at most 4 chunks in the focus of attention (Cowan 2001); 5-8 bits per
   perceptual unit and ~39 bits/s regardless of language (Coupé et al. 2019);
   200-600 semantic components index 7k-47k characters in every real
   logographic system (Kangxi 214, Shuowen 540, GF 0014-2009 514 components).
8. **Every pure-semantic script in history failed for the same reasons**
   (Eco 1995): an a priori taxonomy that broke on real vocabulary (Wilkins
   1668, Leibniz), unmemorable abstract glyphs with low transparency
   (Blissymbolics; Hetzroni et al. 2002), and no phonological route (DeFrancis
   1984/1989; Unger 2004). Nubun's codes are learned bottom-up, which answers
   the first objection; the glyph layer must be designed to answer the other
   two, and every glyph needs a pronounceable reading.
9. **Frequent concepts should be the shortest symbols** (Zipf 1949; Huffman
   1952; Piantadosi et al. 2011 on contextual surprisal). The quantizer's
   placement of "not", "we", "water" at level 0 is consistent with a
   Huffman-like allocation; variable depth per slot would make "deep stack =
   rare, specific concept" a readable regularity.
10. **Keep redundancy on purpose.** A minimum-entropy code is maximally fragile
    (Shannon 1948; Plotkin & Nowak 2000). Human scripts are ~50% redundant;
    a readable Nubun should constrain which level-1 codes may follow a level-0
    radical, the way phonetic components are constrained in Chinese.

## 1. Machine learning

### 1.1 Discrete latents and vector quantization for text

- **van den Oord, Vinyals, Kavukcuoglu. Neural Discrete Representation Learning. NeurIPS 2017.** [standard] VQ-VAE: nearest-codebook lookup, straight-through gradient, commitment loss. Our bottleneck; its dead-code failure mode is the Phase 8 E3 collapse.
- **Kaiser et al. Fast Decoding in Sequence Models Using Discrete Latent Variables. ICML 2018.** [verified] Autoencodes a sentence into a shorter discrete latent sequence, then decodes in parallel; introduces decomposed VQ. The 2018 ancestor of our design.
- **Roy, Vaswani, Neelakantan, Parmar. Theory and Experiments on Vector Quantized Autoencoders. arXiv 2018.** [standard] EM / soft-assignment codebook updates stabilise VQ for NMT latents.
- **Zeghidour et al. SoundStream. IEEE/ACM TASLP 2021.** [verified] Popularised residual VQ with quantizer dropout for variable bitrate. Quantizer dropout would give our levels a coarse-to-fine reading by construction and enable variable depth per slot.
- **Lee et al. Autoregressive Image Generation using Residual Quantization (RQ-VAE). CVPR 2022.** [verified] Depth helps reconstruction more than codebook size; huge codebooks collapse. Supports "many levels, small codebook". Uses one shared codebook across depths; we use one per level, which is better for interpretability.
- **Mentzer, Minnen, Agustsson, Tschannen. Finite Scalar Quantization: VQ-VAE Made Simple. ICLR 2024.** [verified] Round each of a few dimensions to a small grid; no learned codebook, no collapse. Collapse-free and intrinsically factorised, but codes are coordinates, not concepts. Best used as a control: if FSQ at matched bits shows the same cross-lingual sharing, the radical phenomenon is a property of the data, not of RVQ.
- **Yu et al. Language Model Beats Diffusion: Tokenizer is Key (MAGVIT-v2, LFQ). ICLR 2024.** [verified] Lookup-free binary codes with an entropy penalty. The most jamo-like factorisation available.
- **Huh, Cheung, Agrawal, Isola. Straightening Out the Straight-Through Estimator. ICML 2023.** [verified] Affine codebook re-parameterisation and alternating optimisation fix index collapse.
- **Zheng, Vedaldi. Online Clustered Codebook (CVQ-VAE). ICCV 2023.** [verified] Revives dead codes toward encoder features online.
- **Zhu et al. Addressing Representation Collapse in VQ Models with One Linear Layer (SimVQ). ICCV 2025.** [verified] Codes = frozen basis x one learnable linear map.
- **Lu et al. Beyond Stationarity: Rethinking Codebook Collapse. arXiv 2602.18896; NSVQ arXiv 2606.11363.** [preprint] Collapse as encoder drift; stage-wise encoder freezing.
- **Huijben et al. Residual Quantization with Implicit Neural Codebooks (QINCo). ICML 2024.** [verified] Each level's codebook is a function of the partial reconstruction. Models context-dependent component meaning (a Chinese component means different things in different characters) at the cost of level-k codes not being readable in isolation.
- **Rajput et al. Recommender Systems with Generative Retrieval (TIGER). NeurIPS 2023.** [verified] RQ-VAE "Semantic IDs" as a generative vocabulary at scale. Our sentence codes are Semantic IDs for sentences.
- **Singh et al. Better Generalization with Semantic IDs. RecSys 2024.** [verified] SentencePiece over code strings beat hand-built n-grams. Suggests a second tokenisation layer over our 8x64 code strings: frequent code bigrams become compound glyphs.
- **Zhao, Lee, Eskenazi. Unsupervised Discrete Sentence Representation Learning (DI-VAE). ACL 2018.** [verified] Interpretable discrete sentence latents; batch-prior regulariser against posterior collapse.
- **Tucker et al. Emergent Discrete Communication in Semantic Spaces. NeurIPS 2021.** [verified] VQ-style tokens as an emergent language.
- **Li et al. Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey. IEEE TPAMI 2026.** [verified] Single reference for the quantiser design space (VQ, RVQ, PQ, FSQ, LFQ).
- **Yu et al. Vector Quantized Latent Concepts. arXiv 2602.02726.** [preprint] VQ codebook over LM token representations read as a concept vocabulary.

### 1.2 Latent and compressed reasoning in LLMs

- **Hao et al. Training LLMs to Reason in a Continuous Latent Space (Coconut). COLM 2025.** [verified] Feeds the hidden state back as the next input. Our codes are a discrete, readable alternative to opaque continuous thoughts.
- **Shen et al. CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation. arXiv 2502.21074.** [verified] Self-distillation route to a compressed reasoning space; a training recipe for teaching an LLM to emit Nubun codes.
- **Su et al. Token Assorted: Mixing Latent and Text Tokens for Improved LM Reasoning. ICML 2025.** [verified] Compresses CoT with a VQ-VAE (codebook 1024, 16 tokens to 1 code) and mixes codes with text via randomised replacement. The most direct precedent for "VQ codes as a reasoning vocabulary"; the minimal Nubun demo is this recipe with our codes.
- **Zhang et al. Soft Thinking. NeurIPS 2025.** [verified] Probability-weighted mixture of embeddings as a concept token; evidence that concept-level steps help.
- **Goyal et al. Think Before You Speak: Pause Tokens. ICLR 2024.** [verified] Baseline for any "code tokens in an LLM" experiment.
- **Pagnoni et al. Byte Latent Transformer. ACL 2025.** [verified] Entropy-segmented byte patches replace a fixed tokenizer; sentence codes are the far end of the granularity axis.
- **LCM team (Meta FAIR). Large Concept Models. arXiv 2412.08821.** [verified] Autoregression over sentence-level SONAR embeddings at 1.6-7B; a quantised-SONAR variant underperformed diffusion. The closest industrial analogue; it does not pursue readability.
- **Hwang, Wang, Gu. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net). ICLR 2026.** [verified] Learned content-dependent segmentation. Our fixed 64-slot bottleneck could become learned, variable-length chunking.
- **Tack et al. LLM Pretraining with Continuous Concepts (CoCoMix). ICLR 2026.** [verified] SAE-derived concepts predicted and mixed into the hidden state; more sample-efficient than next-token prediction. Nubun codes could serve as the concept targets.
- **2026 preprints on discrete latent reasoning** [preprint]: Thinking Without Words (arXiv 2604.22709; 64 abstract tokens bottlenecked from verbal CoT); Interpretable Discrete Latent Reasoning via Rendered Compression (arXiv 2606.29712; up to 20x); Do Latent Tokens Think? (arXiv 2512.21711); Concept Tokens (Findings of ACL 2026). The field has moved toward our thesis, monolingually and without an orthography.

### 1.3 Interlingua and language-agnostic sentence representations

- **Lu, Keung, Ladhak, Bhardwaj, Zhang, Sun. A Neural Interlingua for Multilingual MT. WMT 2018.** [verified] Explicit attentional interlingua layer. Continuous version of our bottleneck.
- **Vázquez, Raganato, Tiedemann, Creutz. Multilingual NMT with a Language-Independent Attention Bridge. RepL4NLP 2019.** [verified] Fixed number of attention heads gives a fixed-size language-agnostic representation; our 64 Perceiver latents are an attention bridge plus quantisation.
- **Artetxe, Schwenk. LASER. TACL 2019.** [verified] **Feng et al. LaBSE. ACL 2022.** [verified] **Duquenne, Schwenk, Sagot. SONAR. arXiv 2308.11466.** [verified] Continuous baselines; evaluate our code space with the same xsim retrieval metrics, and consider SONAR as an initialisation or quantisation ablation.
- **Liu et al. Composable Text Controls in Latent Space with ODEs (LatentOps). EMNLP 2023.** [verified] Composable editing in a sentence latent; discrete code substitution should permit the same.

### 1.4 Emergent communication and compositionality metrics

- **Lazaridou, Peysakhovich, Baroni. ICLR 2017.** [verified] Origin of the modern emergent-language program.
- **Kottur, Moura, Lee, Batra. Natural Language Does Not Emerge 'Naturally'. EMNLP 2017.** [verified] Emergent codes are effective but not compositional unless constrained. Cross-lingual agreement does not imply compositionality.
- **Chaabouni, Kharitonov, Bouchacourt, Dupoux, Baroni. Compositionality and Generalization in Emergent Languages. ACL 2020.** [verified] Positional (posdis) and bag-of-symbols (bosdis) disentanglement. posdis vs bosdis on our codes decides whether levels are an ordered hierarchy (RVQ) or an unordered bag (additive quantisation).
- **Ren, Guo, Labeau, Cohen, Kirby. Compositional Languages Emerge in a Neural Iterated Learning Model. ICLR 2020.** [verified] Periodic re-learning by fresh agents amplifies compositional structure; suggests periodic decoder re-initialisation.
- **Andreas. Measuring Compositionality in Representation Learning. ICLR 2019.** [verified] Tree reconstruction error; with addition as the composition function it directly tests "codes sum to meaning".
- **Brighton, Kirby. Artificial Life 12(2), 2006.** [standard] Topographic similarity: correlation of meaning distances with message distances.
- **Kharitonov, Chaabouni, Bouchacourt, Baroni. Entropy Minimization in Emergent Languages. ACL 2020.** [verified] Emergent codes spontaneously minimise information about the input. A reconstruction-only VQ-VAE will do the same.
- **Boldt, Mortensen. Morpheme Induction for Emergent Language. EMNLP 2025; Communicating in Emergent Language with an Induced Morphological Phrasebook. ACL 2026.** [verified] Normalised PMI between forms and meanings beats topsim as a compositionality measure; the direct method for finding radicals in a code corpus.
- **Lake, Baroni. SCAN. ICML 2018; Kim, Linzen. COGS. EMNLP 2020.** [standard] Held-out-combination benchmarks; a COGS-style split over code combinations tests "composable" rather than "interpretable".

### 1.5 Interpretability of discrete codes and SAE features

- **Bricken et al. Towards Monosemanticity. Transformer Circuits, 2023.** [standard] **Templeton et al. Scaling Monosemanticity. 2024.** [verified] **Cunningham et al. Sparse Autoencoders Find Highly Interpretable Features. ICLR 2024.** [standard] Dictionary learning yields monosemantic, multilingual, steerable features. A feature firing for one concept in any language is the SAE analogue of our "water" code; SAE features are a sparse continuous dictionary, ours is a dense discrete one. No paper treats SAE features as an orthography.

### 1.6 Multilingual LLM concept space

- **Wendler, Veselovsky, Monea, West. Do Llamas Work in English? ACL 2024.** [verified] A mid-layer concept space closer to English than to the input language.
- **Schut, Gal, Farquhar. Do Multilingual LLMs Think in English? arXiv 2502.15603.** [verified] English-pivot bias a neutral interlingua would avoid.
- **Wu, Yu, Yogatama, Lu, Kim. The Semantic Hub Hypothesis. ICLR 2025.** [verified] Semantically equivalent inputs across languages and modalities converge in intermediate layers; interventions transfer. Probing whether Nubun codes are linearly decodable from an LLM's middle layers would show they are a readout of an existing hub.
- **Tang et al. Language-Specific Neurons. ACL 2024; Kojima et al. NAACL 2024.** [verified] Language-specific neurons concentrate in first/last layers and overlap <5%. Shared structure lives in the middle, where our bottleneck sits.
- **Mondal et al. Language-Specific Neurons Do Not Facilitate Cross-Lingual Transfer. NAACL 2025 workshop.** [verified] Shared structure must be learned explicitly.
- **Zhao et al. How Do LLMs Handle Multilingualism? NeurIPS 2024.** UNVERIFIED this session.

## 2. Linguistics, writing systems, constructed scripts, cognition

### 2.1 Writing-system typology and Chinese character structure

- **Gelb. A Study of Writing. 1952.** [standard] Full writing encodes language, not meaning; semasiography is a forerunner. The orthodox view Nubun's semantic layer must answer.
- **Sampson. Writing Systems. 1985**, ch. 8. [verified] Coins "featural": Hangul letters encode articulatory features by stroke addition. The direct precedent for our Hangul-style component layer.
- **DeFrancis. Visible Speech. 1989.** [verified] Rejects featural classification and denies semasiography is writing: readers need a phonological route.
- **Hunminjeongeum Haerye (1446).** [verified] Five basic consonants depict the articulators, the rest derived by stroke addition; vowels from three primitives; stated learnability target "a morning to ten days". A designed script with ~8 primitives and systematic derivation.
- **DeFrancis. The Chinese Language: Fact and Fantasy. 1984.** [verified] The "ideographic myth": characters are morphosyllabic, 80-90% phono-semantic. **Unger. Ideogram. 2004.** [verified] Extends to kanji. A script of pure meaning has never worked at scale.
- **Shuowen Jiezi (c. 100 CE, 540 bushou); Zihui (1615, 214); Kangxi Dictionary (1716, 214 radicals for 47,035 characters).** [verified] Liushu: ~81% phono-semantic, ~13% semantic compounds (huiyi), ~4% pictographs. Semantic composition alone did not scale; phonetics was recruited.
- **Component standards:** GF 3001-1997 (560 primitives for 20,902 characters); GF 0014-2009 (514 for common characters); GF 0011-2009 (201 principal + 100 associated indexing components); Chinese Character Information Dictionary (623 for 7,785 characters). Character coverage: 1,000 characters ~90% of running text, 2,500 ~98%, 3,500 ~99.5%. A component-level coverage curve was not found (compute it from CHISE/cjkvi-ids). Real logographies converge on **500-600 primitive components**, Zipf-distributed.
- **Unicode Ideographic Description Sequences (U+2FF0-2FFB); CHISE IDS; cjkvi-ids; Wikimedia CJK decomposition data.** [verified] A formal grammar of composition operators plus components, machine-readable, with the caveat that one structure has several encodings and needs canonicalisation. A ready-made formalism for Nubun compound glyphs.
- **Myers. The Grammar of Chinese Characters. Routledge 2019.** [verified] Character structure is grammar-like and productive. Supports a generative glyph design.
- **Shu, Chen, Anderson, Wu, Xuan. Properties of School Chinese. Child Development 2003.** [verified] High-frequency characters are opaque and simple; low-frequency ones complex and transparent. Design rule: opaque short head, transparent compositional tail.
- **Han, Kelly, Winters, Kemp. Simplification Is Not Dominant in the Evolution of Chinese Characters. Open Mind 2022.** [verified] Complexity rises to keep glyphs discriminable as the inventory grows.
- **Mullaney. The Chinese Typewriter. MIT Press 2017.** [verified] Large character sets were made inputtable by component-based retrieval; a readable Nubun needs an input method, and composition from radicals is the proven route.

### 2.2 Constructed semantic and ideographic scripts

- **Wilkins. An Essay towards a Real Character and a Philosophical Language. 1668.** [verified] 40 genera subdivided into differences and species, encoded positionally; brittle to new knowledge, symbols unmemorable. **Dalgarno. Ars Signorum. 1661.** **Leibniz. De arte combinatoria (1666) and the characteristica universalis.** [standard] An alphabet of human thought whose primitives were never actually discovered; Nubun's VQ-VAE is the empirical attempt at that discovery. **Maat. Philosophical Languages in the Seventeenth Century. Kluwer 2004.** **Eco. The Search for the Perfect Language. 1995.** [verified] Best secondary source on why ~83 projects failed: arbitrary taxonomy, unlearnable glyphs, no community of use.
- **Bliss. Semantography (Blissymbolics). 1949/1965.** [verified] ~100 basic shapes compose several hundred symbols; >5,000 authorised words; adopted for AAC in 1971. Studies (Hetzroni, Quist & Lloyd 2002 and later) find it the least transparent major AAC symbol set. The closest existing analogue to Nubun and its clearest lesson: abstract-geometric compositionality costs guessability; plan for learned glyphs and a 100-150 element starter set.
- **Neurath. International Picture Language (Isotype). 1936.** [verified] Successful restricted-domain semasiography.
- **Ota. LoCoS. Icographic 1973.** [verified] ~8 primitives with spatial syntax; no user base.
- **Weilgart. aUI: The Language of Space. 1979.** [verified; 1962 date UNVERIFIED] 31 sememe primitives produce unmemorably long words: a lower bound on inventory size.
- **Brown. Loglan. Scientific American 1960; Cowan. The Complete Lojban Language. 1997.** [verified] ~1,350 root words plus combining forms: a plausible upper target for radical + compound inventory.
- **Lang. Toki Pona. 2014; Toki Pona Dictionary 2021.** [verified] ~120-137 words with heavy context and compounding, and the largest living minimalist-conlang community: a lower bound for a usable semantic basis.
- **Quijada. Ithkuil. 2004/2011; Foer, The New Yorker 2012.** [verified] Maximal precision per symbol defeats learnability: do not optimise information density alone.
- **Uchida, Zhu, Della Senta. Universal Networking Language. UNU/IAS 1996-.** [verified] Concept nodes anchored to English headwords; coverage never scaled, stayed parasitic on English. Nubun's learned, language-neutral codes are the fix UNL needed.

### 2.3 Semantic primitives and concept inventories

- **Wierzbicka. Semantic Primitives. 1972; Goddard & Wierzbicka. Words and Meanings. OUP 2014.** [verified] Natural Semantic Metalanguage: ~65 primes (I, YOU, NOT, WANT, KNOW, SAY, DO, GOOD, BIG, ...) tested in ~30 languages. Our level-0 inventory already contains several (we, you, I, not, say, can); NSM is the natural seed/validation set.
- **Swadesh. IJAL 1955.** [verified] 100/200-item core vocabulary; **Concepticon (List et al. 2016)** links concept lists to one ID space.
- **Ogden. Basic English. 1930.** [verified] 850 words; Bliss aligned to it.
- **Longman Defining Vocabulary (~2,000 words); Xu. Lexikos 2012.** [verified] The practical "define everything" basis; polysemy inflates nominal counts.
- **Schank. Conceptual Dependency. Cognitive Psychology 1972.** [venue UNVERIFIED] ~11 primitive ACTs for events.
- **Jackendoff. Semantic Structures. MIT Press 1990.** [verified] ~8 ontological categories (THING, PLACE, PATH, EVENT, ...) and functions (GO, CAUSE, BE, ...). Suggests a glyph grammar should encode a few type categories structurally, like Hangul's initial/medial/final slots.
- **Miller. WordNet. CACM 1995** (117,659 synsets in 3.0); **Navigli & Ponzetto. BabelNet. AIJ 2012.** [verified] Upper estimate of distinct lexicalised concepts (~1e5) and a cross-lingual inventory to align codes against.
- **Dixon. A Semantic Approach to English Grammar. OUP 2005.** [verified] Dozens of verb semantic types: a candidate mid-level "category radical" tier.

### 2.4 Compositionality, morphology, reading compound characters

- **Frege's principle; Szabó, Compositionality, SEP.** [standard] Natural systems are only partially compositional. **Fodor & Pylyshyn 1988.** [standard] Systematicity and productivity arguments. **Hupkes, Dankers, Mul, Bruni. Compositionality Decomposed. JAIR 2020.** [verified] Five operational tests (systematicity, productivity, substitutivity, localism, overgeneralisation) usable on our codes.
- **Hockett. The Origin of Speech. Scientific American 1960.** [verified] Duality of patterning: meaningless units compose into meaningful ones. Nubun's two tiers (strokes to radicals to compounds) mirror it.
- **Packard. The Morphology of Chinese. CUP 2000.** [verified] ~80% of Mandarin vocabulary is disyllabic compounds. **Semantic transparency studies** (Wang & Peng 2000; Tsang, Zou & Tse 2022; Wei et al. 2023; Wan et al. 2024). [verified] Readers decompose compounds regardless of transparency; heads are more transparent than modifiers. Design compounds head-marked, 2 units.
- **Shibatani. The Languages of Japan. CUP 1990; NINJAL surveys.** [verified] Two-kanji compounds dominate the learned lexicon.
- **Korean mixed script and Hanja decline** (Open Korean Historical Corpus, arXiv 2510.24541; Hanja in headlines, JAPC 2025). [verified] A featural phonographic script displaced a semantic-radical script within a century despite ~57% Sino-Korean vocabulary, at the cost of homophone ambiguity. A historical A/B of the two halves of our hybrid: learnability won; the semantic layer fills the ambiguity gap.
- **Taft & Zhu. Submorphemic Processing in Reading Chinese. JEP:LMC 1997.** [verified] Readers represent radicals with position. Fix radical slots in the glyph grammar.
- **Perfetti, Liu, Tan. The Lexical Constituency Model. Psychological Review 2005.** [verified] Phonology is activated even in a "semantic" script: every glyph needs a reading.
- **Shen & Ke. Radical Awareness and Word Acquisition. Modern Language Journal 2007.** [verified] Explicit radical instruction transfers to vocabulary.

### 2.5 Perceptual and cognitive constraints

- **Pelli, Burns, Farell, Moore-Page. Feature Detection and Letter Identification. Vision Research 2006.** [verified] Efficiency inversely proportional to perimetric complexity; ~7 features per letter; fluent efficiency on a novel alphabet in ~3,000 trials. **Pelli, Farell, Moore. Nature 2003.** [verified] Words are read letter by letter: keep compounds to 2-3 radicals.
- **Changizi & Shimojo. Proc. R. Soc. B 2005.** [verified] ~3 strokes per character across >100 scripts regardless of inventory size; ~50% redundancy. **Changizi, Zhang, Ye, Shimojo. American Naturalist 2006.** [verified] ~36 topological junction types match natural-scene statistics; build primitives from the frequent ones.
- **Morin. Spontaneous Emergence of Legibility. Cognitive Science 2018.** [verified] Cardinal strokes dominate from the start; legibility is designed in, not evolved. **Miton & Morin. Graphic Complexity in Writing Systems. Cognition 2021.** [findings UNVERIFIED]
- **Miller 1956; Cowan. The Magical Number 4. BBS 2001.** [verified] ~4 chunks in focus of attention: at most 4 radicals per block, 4 blocks per clause chunk.
- **Coupé, Oh, Dediu, Pellegrino. Science Advances 2019.** [verified] 5-8 bits per syllable, ~39 bits/s across 17 languages: design for 5-8 bits (32-256 alternatives) per perceptual unit.

### 2.6 Information-theoretic linguistics

- **Zipf 1935/1949.** [standard] Frequent words are short. **Piantadosi, Tily, Gibson. PNAS 2011.** [verified] Word length tracks contextual surprisal, not frequency: allocate glyph complexity by surprisal.
- **Kemp & Regier. Kinship Categories. Science 2012.** [verified] Attested systems sit near the simplicity-informativeness frontier. **Zaslavsky, Kemp, Regier, Tishby. Efficient Compression in Color Naming. PNAS 2018.** [verified] Lexical categories are near-optimal information-bottleneck compressions and evolve by phase transitions. The number of radicals is a trajectory, not a constant. **Kemp, Xu, Regier. Annual Review of Linguistics 2018.** [verified] **Gibson et al. How Efficiency Shapes Human Language. TiCS 2019.** [verified] Tolerate ambiguity where context resolves it; keep dependency distances short.

## 3. Mathematics, information theory, signal processing

### 3.1 Sparse coding and why Phases 1-2 could not work

- **Olshausen & Field. Nature 1996.** [standard] Sparse overcomplete bases give interpretable atoms for images, where the generative structure is sparse-linear; word embeddings are not generated that way.
- **Aharon, Elad, Bruckstein. K-SVD. IEEE TSP 2006.** [standard] k-means + OMP is a degenerate K-SVD; atom duplication, dead atoms and init-dependence are its known failures.
- **Pati et al. OMP. 1993; Chen, Donoho, Saunders. Basis Pursuit. SIAM 1998.** [standard] Recovery requires incoherent atoms; semantic directions (water/liquid/river) are nearly collinear.
- **Gribonval & Schnass. Dictionary Identification. IEEE TIT 2010.** [verified] **Spielman, Wang, Wright. Exact Recovery of Sparsely-Used Dictionaries. COLT 2012.** [verified] Identifiability needs incoherence and random sparse coefficients; neither holds for semantics, so there is no "true" radical set to recover by decomposition.
- **Murphy, Talukdar, Mitchell. NNSE. COLING 2012.** [standard] **Faruqui et al. Sparse Overcomplete Word Vectors. ACL 2015.** [verified] **Subramanian et al. SPINE. AAAI 2018.** [verified] **Arora et al. Linear Algebraic Structure of Word Senses. TACL 2018.** [standard] Interpretable sparse features over embeddings exist, but they are rotations or discourse atoms, not a stable cross-lingual compositional inventory, and K-SVD runs are unstable.
- **Wendlandt, Kummerfeld, Mihalcea. NAACL 2018; Antoniak & Mimno. TACL 2018.** [verified] Embedding neighbourhoods are unstable across seeds and samples; any dictionary on top inherits the noise.

### 3.2 Vector quantization theory

- **Lloyd 1982; Linde, Buzo, Gray 1980.** [verified] Codebook EMA is online LBG; collapse is the empty-cell problem.
- **Shannon 1959; Berger 1971; Cover & Thomas 2006 ch. 10.** [standard] Rate-distortion: "is 8x128 the right split" is a rate-allocation question answerable by distortion at equal bits.
- **Gray & Neuhoff. Quantization. IEEE TIT 1998.** [verified] RVQ and PQ are constrained VQ families that give up optimality for codebook size.
- **Juang & Gray. Multiple Stage VQ. ICASSP 1982.** [verified] Our quantizer: 1,024 stored vectors give an effective 128^8 codebook; greedy stage-wise encoding is suboptimal.
- **Barnes, Rizvi, Nasrabadi. Advances in Residual VQ: A Review. IEEE TIP 1996.** [verified] Joint codebook re-optimisation and beam-search encoding over stages lower distortion at fixed rate; many small stages generalise better than few large ones.
- **Jégou, Douze, Schmid. Product Quantization. TPAMI 2011.** [standard] Right model if radicals occupy independent subspaces; RVQ if they are ordered refinements. Our level-0 behaviour indicates refinement.
- **Babenko & Lempitsky. Additive Quantization. CVPR 2014.** [standard] Codes sum with no residual ordering, encoded jointly. The mathematically correct model of an order-free radical stack; RVQ is AQ with greedy encoding.
- **Conway & Sloane 1999; Zamir 2014.** [standard] Lattice quantizers as the no-codebook baseline.
- **Fang et al. A Unified Rate-Distortion Perspective on Vector, Product and Scalar Quantization. arXiv 2609.02107.** [preprint] Hold total bits fixed and compare (slots, levels, codebook) configurations.
- **Reyhanian & Wiskott. Is Hierarchical Quantization Essential for Optimal Reconstruction? arXiv 2601.22244.** [preprint] At matched budget with collapse mitigation, one level matches two for reconstruction. Our case for 8 levels must rest on structure and description length.

### 3.3 Information bottleneck, MDL, compression

- **Tishby, Pereira, Bialek. The Information Bottleneck Method. 1999.** [standard] Complexity-accuracy curve with phase transitions where categories split. **Strouse & Schwab. Deterministic IB. UAI 2016.** [verified] H(code) - beta I(code; meaning) is the right objective to monitor for a hard discrete codebook.
- **Rissanen 1978; Grünwald 2007.** [standard] Two-part MDL (codebook description + corpus code length under a prior) is the principled rule for choosing codes per level and number of levels.
- **Li & Vitányi 2008; Hutter Prize.** [standard] "Compression = understanding" is the frame but the right analogue for a lossy code is rate-distortion.
- **Delétang et al. Language Modeling Is Compression. ICLR 2024.** [verified] Suggests the canonical evaluation of codes as a semantic space: train a small autoregressive prior over code sequences and report bits per sentence against a byte-level LM at equal reconstruction fidelity.

### 3.4 Compositionality formalisms and identifiability

- **Smolensky. Tensor Product Variable Binding. AIJ 1990.** [standard] slot x code is a role-filler binding; exactly decodable if roles are orthogonal.
- **Plate. Holographic Reduced Representations. IEEE TNN 1995; Kanerva 1988/2009; Gayler 2003; Kleyko et al. Survey on Hyperdimensional Computing. ACM CSUR 2022/23.** [verified: survey] A fixed-width vector can be a compositional symbol with readable parts; the residual sum of 8 codes is a bundle, recoverable only if level codebooks are near-orthogonal (measured in `PHASE9.md` §2: adjacent-level subspace overlap 0.16-0.51).
- **Higgins et al. beta-VAE. ICLR 2017.** [standard] **Locatello et al. ICML 2019.** [standard] Unsupervised disentanglement is impossible without inductive bias; multilingual parallel data is ours. Test cross-seed code stability.
- **Allman, Matias, Rhodes. Identifiability of Parameters in Latent Structure Models. Annals of Statistics 2009.** [verified] Kruskal's theorem: 3+ conditionally independent views identify a discrete latent. The theoretical reason multilingual training produces stable radicals.
- **Kivva, Rajendran, Ravikumar, Aragam. Identifiability of Deep Generative Models without Auxiliary Information. NeurIPS 2022.** [verified] Mixture-prior deep generative models are identifiable up to affine maps.

### 3.5 Combinatorics and coding theory for a character set

- **Counting.** R radicals in k ordered slots give R^k characters (128^8 ~ 7e16 per slot); k unordered distinct radicals give C(R,k) (C(128,8) ~ 1.4e12). Making order irrelevant costs a factor of k!, so order-free readable stacks need either ordered levels or a larger R.
- **Shannon 1948/1951.** [standard] Redundancy buys error correction; English is ~75% redundant. **Plotkin & Nowak. Language Evolution and Information Theory. J. Theor. Biol. 2000.** [verified] Combinable signals break the error limit of a flat inventory: small R, moderate k.
- **Huffman 1952; Zipf 1949; Mandelbrot 1953; Ferrer-i-Cancho & Solé. PNAS 2003.** [verified: F-i-C] Zipf scaling appears at the phase transition between useless and referential systems; the code-frequency exponent is a cheap diagnostic (measured: a log-log slope of -1.88 over the active level-0 codes of the 500k model, steeper than Zipf because of the long tail; see `PHASE9.md`).
- **Piantadosi, Tily, Gibson. PNAS 2011.** [verified] Allocate depth by contextual surprisal: variable-depth RVQ.

### 3.6 Measuring compositionality of discrete codes

- **Brighton & Kirby 2006** (topsim); **Chaabouni et al. 2020** (posdis, bosdis); **Andreas 2019** (TRE); **Chen, Li, Grosse, Duvenaud. NeurIPS 2018** (MIG) [standard]; **Boldt & Mortensen 2025/26** (nPMI morpheme induction); **Lake & Baroni 2018; Kim & Linzen 2020** (held-out combinations). All computable on existing checkpoints; see `PHASE9.md` §4 workstream A.

## 4. Open items from the surveys

- Compute a component-level coverage curve for Chinese from CHISE/cjkvi-ids (how many components cover 90% of characters); the literature gives inventory sizes but not the curve.
- UNVERIFIED items to check before citing: Zhao et al. NeurIPS 2024; Schank 1972 venue; aUI 1962 date; Miton & Morin 2021 findings; Chan, Gupta & Gersho 1992 (joint RVQ codebook design; Barnes et al. 1996 covers the same material and is verified).
