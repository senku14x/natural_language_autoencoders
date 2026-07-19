# Zero-shot AV read of edit-site deltas — exploratory report

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` (uncommitted, pending user review)
**Phase:** exploration. **This is not Stage B and not a verbalizer result.** It is a single cheap question about the *released* AV, run before any fine-tuning.
**Predictions were registered first:** `research/temporary_artifacts/2026-07-19_zeroshot_av_predictions.md`.

## Question

Does the released AV (`kitft/nla-qwen2.5-7b-L20-av`), with **no fine-tuning**, mention the new value when handed the **edit-site** L20 activation delta?

## Setup

- Target: Qwen2.5-7B-Instruct @ `a09a3545…`, layer 20 = `hidden_states[21]`. AV: released kitft L20 checkpoint (snapshot `b884691…`), sidecar-driven (injection ㈎ id 149705, `injection_scale` 150.0, `embed_scale` 1.0 — all read from `nla_meta.yaml`, nothing hardcoded).
- Read site: **edit-token position** (verified `h_{base,cf}.npy` from Stage A were *final*-position; re-extracted edit-position states here). Edit-site delta norms 67–81; states ~100–170 (non-pathological, above the ~10-token and position-50 floors — raw/preamble edit positions 74–98).
- Sample: **50 semantic pairs**, random (seed 20260719), from dev / raw+preamble / eligible change rows. `pair_ids` logged in `zeroshot_av/sample_manifest.json`. Draw gave strata S=26 / N=24; query order came out **46 query_last / 4 query_first** (a `groupby.first()` artifact — see Limitations).
- 6 arms × 5 samples @ T=1 (sampling noise is part of the measurement): real Δ / shuffled Δ (different pair, different transition, same cell) / random Gaussian / h_cf state / h_base state / E[Δ|old,new] transition mean (leave-one-out). 1,450 generations total.
- **Scoring: string matching only**, no LLM judge. Whole-word case-insensitive match of the pair's new value / old value / other inventory values / edited entity.
- **Serving deviation (documented):** injection uses the repo's own pure functions (`normalize_activation`, `inject_at_marked_positions`) with the two-step `apply_chat_template(tokenize=False)` → `encode(add_special_tokens=False)` tokenization your `Validating-NLAs` infra doc prescribes — the one-step `tokenize=True` path lets NFKC rewrite ㈎ (U+320E → "(가)") and **drops the injection token** (confirmed against transformers 5.14.1 this session). Generation is transformers-native `generate(inputs_embeds=…)` rather than an SGLang server: your doc confirms injection is entirely client-side and Qwen needs no SGLang patch, so the forward math is identical; a single in-process job (no server to babysit) was the right fit for the "launch and wait" constraint. If you want strict SGLang-parity reproduction, the same vectors/prompt drop into `NLAClient` unchanged.

## Predictions vs outcome — I was badly miscalibrated

| quantity | predicted | observed |
|---|---|---|
| real-delta new-value rate | ~0.15 | **0.740** [0.636, 0.840] |
| shuffled-delta new-value rate | ~0.03 | **0.000** |
| AV(h_cf) names the value | ~0.40 | **1.000** |
| **headline: real − shuffled** | ~+0.10 | **+0.740** [+0.636, +0.840] |

I over-applied the oakhu "the Qwen2.5 NLA is noisy" prior. That prior was measured for reading *subtle problem-constant differences at the final position* in a document/math setting — **not** for naming a single-token value at *its own token position*, which is near-trivial token-identity readout. This is an honest calibration miss worth recording.

## Observations (supported empirical claims, dev / this setup)

Per-arm mention rates, pair as bootstrap unit (`zeroshot_av/arm_summary.parquet`; plot `plots/2026-07-19_zeroshot_av_mention_rates.png`):

| arm | new-value | old-value | edited entity | mean other-values |
|---|---|---|---|---|
| **h_cf state** (positive control) | **1.000** [1.00,1.00] | 0.044 | 0.000 | 1.58 |
| **real Δ** (own pair) | **0.740** [0.636,0.840] | 0.000 | 0.000 | 0.23 |
| **E[Δ\|old,new]** mean | 0.620 [0.500,0.745] | 0.000 | 0.000 | 0.12 |
| **h_base state** | 0.024 | **1.000** [1.00,1.00] | 0.000 | 1.48 |
| **shuffled Δ** (diff pair) | **0.000** | 0.024 | 0.000 | 0.98 |
| **random Gaussian** | **0.000** | 0.000 | 0.000 | 0.12 |

- The real-delta / shuffled / random contrast is clean: **0.740 vs 0.000 vs 0.000**. The signal is vector-driven, not incidental language: this pair's specific new value **never** appears when a different pair's delta (shuffled) or noise (random) is injected.
- **The matches are genuine, not a matching artifact.** Inspecting exact match contexts: "The **pink** color", "The **rust** in my machine", Final token "**yellow**", "Clotted **crimson**", "My name is **Oliver**" (h_cf). Dual-meaning value words (ruby/coral/rust/…) account for 39/185 hits; even discounting **all** of them, real new-value = 58% ≫ 0%.
- **States name their own token near-perfectly, and asymmetrically:** h_cf → new 100% / old 4%; h_base → old 100% / new 2%. The **difference** vector names new 74% and old **0%** — subtracting the old-token direction suppresses it rather than surfacing it.
- **Shuffled names the *donor* pair's value** (its "other-values" rate ≈ 1.0, vs 0.12 for random) — direct confirmation that the AV reads whatever token identity the injected vector carries.
- **The AV names the value, essentially never the entity/nonce** (edited-entity rate 0.000 everywhere) — the read is of the value token at the edit position, not the bound entity.
- Colors (S) read far better than names (N): real new-value **0.89 (S) vs 0.58 (N)**.
- Every generation produced an `<explanation>` (no parse failures). transition-mean available for all 50 pairs (LOO coverage ok).

## Interpretation (kept separate)

- **This is token-identity readout at the edit position — the "names the changed token" boring alternative made concrete, not consequence-reading.** The edit token *is* the value word; L20 there is dominated by that token's identity; the released AV decodes it, logit-lens-like — it literally reports "Final token X". The edit-site delta (new-token − old-token representation), even after L2-rescaling to 150 destroys magnitude, retains enough new-token direction to be named. The surrounding prose is confabulated (math textbooks, Wikipedia, SAT items); only the named token carries signal — exactly the answer-embedded-in-confabulation pattern the NLA literature describes (Realmbird answer-mention; the paper's "answer in the verbalization").
- **What it establishes:** (1) the instrument is *sensitive* at this site — contra the final-position pessimism and the oakhu noisiness prior, the released AV zero-shot reads single-token value identity from the edit-site delta reliably; (2) the edit-site delta carries recoverable new-value token identity above a clean shuffled/random floor.
- **What it does NOT establish:** nothing about reading a *behavioral consequence* or an answer *transition*. For these TARGET/REVERSE rows the edited value **is** the queried answer, so "names the edited token" and "reads the answer change" are not separable here — that separation needs distractor rows (edited ≠ queried) at the edit site, not run. It licenses nothing about captions, gates, or Stage B/C, and nothing causal. A zero-shot name-drop of a token is decodability-of-a-token evidence only.
- The transition-mean naming the new value at 0.62 (≈ real 0.74) echoes Stage A: at the edit site the delta is largely a function of (old,new) — i.e. mostly the new token's identity — so a per-transition prototype nearly reproduces the per-example read.

## Limitations

- **Query-order skew (46 last / 4 first):** a `groupby("semantic_id").first()` sampling artifact, not by design. Doesn't affect the token-identity conclusion (the edit token is the value word in either order; Stage A showed edit-site recovery is query-order-invariant) but means this run says nothing about query order. Easy to rebalance if we want.
- **Dual-meaning value words** inflate raw hits modestly (39/185); the conclusion survives excluding them.
- **TARGET/REVERSE only** → edited == queried, so consequence-vs-token-identity is confounded by construction here.
- Dev split, raw/preamble, this model/layer/site, T=1, n=50 pairs. Exploratory.

## Deliverables / artifacts

- Raw dump of all 1,450 explanations: `zeroshot_av/all_explanations.json`. Scored table: `zeroshot_av/scored.parquet`. Summary: `zeroshot_av/arm_summary.parquet`, `zeroshot_av/headline.json`. Sample manifest: `zeroshot_av/sample_manifest.json`. Edit-site states: `zeroshot_av/h_edit_{base,cf}.npy`, `pool.parquet`.
- 20 uniformly-random explanations (seed 2026): `zeroshot_av/random20.json`, pasted verbatim below.
- Plot: `plots/2026-07-19_zeroshot_av_mention_rates.png`.
- Script: `research/exploratory/zeroshot_av_edit_site.py`.

## Follow-up probe: is `value_old` in the delta? — YES, both endpoints are (added 2026-07-19)

Motivated by the "can I make it verbalise the diff?" question. Linear probe
(StandardScaler → PCA-64 → L2 logistic regression, fit per-fold inside
GroupKFold **by family**, direction-only features), predicting the value from
delta vs each state, per stratum. Script `research/exploratory/oldvalue_probe.py`;
results `zeroshot_av/oldvalue_probe.json`; plot `plots/2026-07-19_oldvalue_probe.png`.

| probe | colors S (chance 0.029) | names N (chance 0.009) |
|---|---|---|
| **old ~ delta** | **0.962** | **0.764** |
| **new ~ delta** | **0.958** | **0.764** |
| old ~ h_base (pos ctrl) | 0.974 | 0.844 |
| new ~ h_cf (pos ctrl) | 0.970 | 0.836 |
| old ~ h_cf (neg ctrl) | 0.047 | 0.000 |
| new ~ h_base (neg ctrl) | 0.047 | 0.000 |
| old ~ delta (label-permuted) | 0.052 | 0.011 |

**Observation:** the edit-site delta linearly encodes the **old and the new
value about equally** (colors 0.96/0.96, names 0.76/0.76), far above chance and
only marginally below the single-state positive controls. Each individual state
carries only its *own* endpoint (neg controls at chance); the delta carries
*both*. Label permutation collapses to chance, so the high-dim pipeline is not
manufacturing accuracy.

**Interpretation (separate):** the AV's zero-shot 0% old-value mention is a
**decoding/verbalization limitation, not information absence** — the transition
(both endpoints) is linearly present in the delta (the old endpoint lives in the
negated component the zero-shot decode doesn't surface). So training the AV to
emit `old → new` would be reading real content, not confabulating the "old"
side. **Caveats that still bind:** this is (1) *decodability* by a probe, which
does not establish the AV will verbalize it, that a caption is faithful, or that
language beats the probe (four-claims separation); (2) decodability of the two
**token identities** — consistent with delta ≈ (new-token − old-token), i.e.
still token identity, not evidence the delta represents "a change" as more than
its two endpoints; (3) on dev with a **shared value vocabulary** — the real test
is the held-out `test_value` split, not touched here; (4) on templated families
the reader is a lookup table by construction (Amendment 1 §10), so feasibility of
`old→new` is expected and earns no language/faithfulness claim. Net: green light
that the substrate *carries the transition*; the "does language earn its keep"
question is untouched and still lives in held-out values + derived families.

## Note on the Qwen3-8B retrospective (import request)

The retrospective could not be imported: **it is not present on this instance** — not in the repo, home directory, scratchpad/uploads, or as a message attachment. I did not fabricate or reconstruct it. Amendment 1 §11.2's provenance therefore remains **unsatisfied here**; if you re-share the file I will import it into `research/docs/` and record (per your instruction) that it is narrative, not raw logs — informs design, not citable as a result.

---

## 20 randomly-selected explanations (verbatim, not curated)

*(uniform random over all 1,450; seed 2026. Arm and pair shown for context.)*

**#37 [shuffled] mustard→white (s2):** Structured textbook format with bold definitions and formatting, presenting a compound term "Chemistry" in a financial context (chemistry of finance). The sentence pattern "If you choose to invest in physics, your income increases. Then the formula: Money+Time=wealth. " mirrors an incomplete second clause, suggesting the familiar formula or answers about spending activities or cash flow from the new knowledge. Final token "Select" is mid-quote ("Forced one gold"), part of a second "If... then"

**#114 [transition_mean] beige→yellow (s4):** Structured math textbook format with formal prose style, presenting a formula with variable names "John, Paul" in a sequential narrative about a student's actions. The phrase "If the student's knowledge increases, he is forced to choose other answers" mirrors a classic pattern ("Since X, Y's actions"), strongly implying the conclusion clause continues with "his own answer: the Sun's victory." Final token "yellowing黄黄以下的。" is mid-sentence "The person who is not yellowing, I want..."), immediate

**#142 [transition_mean] Tehran→Denver (s2):** Structured reference format with mathematical definition, numbered bullets describing a weather term, following a textbook or encyclopedia pattern. The pattern "The formula: When x changes, y equals something... A moving object's height is constant except for the next version" mirrors the classic sequence of a typical weather forecast series, with the second bullet clearly completing the "Traveling by the third dimension" rule about the third dimension. Final token "unknown's  nom 423." is mid

**#256 [real] Vienna→Naples (s1):** Structured reference format with a Wikipedia-style article pattern ("Math symbol") listing mathematical terms with definitions and meanings, following a sequence of concepts from calculus/sustainability. The phrase "In calculus, the term 'normal' usually indicates a probability distribution or probability model, with the highest frequency..." mirrors the pattern of listing categories from the MathWorks formula, suggesting the answer continues naming "Greek" or similar. Final token "Naped" is m

**#432 [real] plum→lime (s2):** Structured Wikipedia-style UI text with bold header format and definition pattern, describing a fictional character's "about" attributes as a game's character. The phrase "The popular interpretation of the acronym, meaning its index of life's status, is commonly seen as 'LGM', and here is the answer:" mirrors the familiar "Google's mathematical index," with a classic sequence implying the answer follows. Final token "time, " closes "about, is 2. 'LGM', around" — part of an incomplete parenthet

**#511 [real] crimson→yellow (s1):** Structured dictionary format with formatted entries following a pattern of "Anxiety" definition followed by "Glossary" section listing the word's characteristics. The sentence "To express: The word 'Knowledge' is negative, then the next phrase to 'the phrase 'the' is a type of thinking" mirrors the pattern of a third item in a multiple-choice format, implying the next tokens continue the SAT-notation. Final token "yellow" is mid-sentence ("2. Y = 2. The idea of Y...and then"), strongly expecti

**#524 [random] crimson→yellow (s4):** Structured math reference format with educational definition pattern, showing a dated calculator entry with formulas and terms about statistics. The phrase "The SAT score increases by 10%, and a list of 108 different food terms" is a standard math dictionary template entry following a famous real-world event, with "MATHS TERMS" being the well-known answer format for a "Food Dictionary." Final token "Terms.) Is" ends mid-sentence ("Books and 60, July 15, 2006. So, six Terms. M.J.R are..."), str

**#532 [h_base] crimson→yellow (s2):** Formal British English word puzzle format with a constructed noun phrase ("The colour of these becomes sapphire..."), implying a pattern of paired adjective descriptions. The given phrase "If x was silver it was beige, whereas y was vermillion" suggests a second parallel adjective describing a color, strongly implying "crimson" or a similar noun completing the color comparison about a fruit or thing. Final token "crimson" ends mid-compound noun ("and crimson"), part of a parenthetical example

**#670 [h_cf] Eric→Oliver (s0):** British literary fiction or informal prose style with a mysterious narrative beginning ("This is a novel about a man called Mr." suggests a character name or personal identity). The phrase "My name is Oliver" strongly implies an introductory clause or personal name declaration, likely continuing a character's identity or narrative context, possibly referencing a novel, detective, or mystery writer's identity. Final token "Oliver" is a proper name mid-clause ("My name is Oliver"), part of a phr

**#917 [real] Diego→Sam (s2):** Biblical text format with quoted lines from John's literary context, following modern literature conventions with a common formula about "The Heart is Not in This Place." The quoted phrase "If a man comes, then the thing called 'the house' that is 'not at all' is the same pattern as 'Sam's'" mirrors a standard quote pattern, implying the usual "Oz is not a dream." Final token "Sam" is mid-quote within "the things that 'sam'... a book...that is," strongly expecting "s" or "the other" or "the Ot

**#925 [random] Diego→Sam (s0):** Wiki-style geometric product description with formatted text, following a standard UI element template from a Romanian language Wikipedia page with "Logo" entry. The pattern "The flag is a common shape. Find the speed of the vehicle. " is a standard Wikipedia template, with the familiar "Comparison to other objects" structure signaling a paired name search pair ("solved by..."). Final token "dings" ends an incomplete repeated "Simplified. ... , but" clause, part of a standard encyclopedia form

**#945 [real] Nina→Alan (s0):** Wiki structure with formal reference format ("Wikipedia page") describing a mathematical constant, following a sequence of defining properties and historical context. The sentence pattern "The third fundamental equation of mathematics, a mysterious phenomenon...is an intermediate step" mirrors the classic definition of a scientific or geometric structure, strongly implying the next tokens continue describing the famous "hidden" formula's components or process like "a". Final token "array" mid-

**#1016 [random] Detroit→Montreal (s1):** Structured football statistic format with official Wikipedia template pattern following "Chapter 1: Team Strategy," showing numbered sections with sub-items from Article 2. The phrase "However, if Team A's first paragraph was: The first 15 minutes..." follows a standard Sports League rules template formula ("But the members had already..." / "90 ×"), strongly implying the "But" clause's clause completion. Final token "haslylyrl" is mid-phrase in "But they had already received 50 ×," part of a

**#1042 [shuffled] purple→cyan (s2):** Structured math definition format with LaTeX syntax, following a pattern of listing attributes ("A+ B+ C+ →") and a common phrase about a set being defined. The phrase "Since: it is red in the sea, the ocean is red: the coral is red" mirrors the classic definition format, strongly implying the answer "is the answer" pattern about the second part's attribute being "dying." Final token "corruptional" closes the second part of a parallel phrase ("If...about the coral and...M is a coral or red; be

**#1139 [random] Marcus→Luke (s4):** Formal product description format with numbered specifications and chemical product details from a pharmaceutical reference source, following a structured academic or retail catalog pattern. The phrase "My name is S.D. Rajiv, an Indian veterinarian, and I write the following product description. This wine has been operating in India since 1895, and I declare..." concludes a standard attribution line, commonly followed by a section header about ethics or. Final token "Management" ends "Time of

**#1197 [random] Eugene→Paul (s2):** Structured technical documentation format with a "Excel formula result" answer, followed by a standardized format pattern using CAS (Comcast Aviation) notation conventions. The phrase "The `STOP` command in Excel: 'Time' range" mirrors a common forum answer convention with "Zeroing" shorthand, implying the final clause completes a "flight sequence" or "limited repetitions" clause (common to SAE). Final token "NO" is mid-quote of a parenthetical abbreviation "MATHLAB 'END', 'STOP'... (" — immed

**#1219 [real] ruby→crimson (s4):** Formal literary format with numbered/listed constructions ("The word "Happy" means..."), suggesting a book title or text puzzle context with definitions. The phrase "If a color is brightened, it becomes Darker. And If its hues are crimson," establishes a parallel structure listing descriptive adjectives defining a scene, strongly implying the second given option "In crimson, the answer is." Final token "climformed" ends mid-definition ("In an indistinct, it's Inclined to be: "In a crimson,"),

**#1242 [real] yellow→rust (s2):** Structured technical format with code reference in English, presenting a formatted quoted title and image caption format suggesting an Excel or programming context about a motorized system. The phrase "The following code block: 'The rust in my machine' refers to a periodic system where the model accumulates and simulates an aging process, along with rust... " implies a quoted technical term from a historical context or API. Final token "rust" is mid-quote in an incomplete clause ("The two rust

**#1307 [shuffled] Kevin→Rachel (s2):** Math format with structured calculator output pattern, listing numerical values for a day's temperature in a date format. The phrase "The first six numbers of the math problem involves numbers from a specific range. (Solve) " suggests a typical list of math concepts from an Excel formula, implying the next token continues description of an ingredient or source item from "Slate" (a geological/series context). Final token "Slate" is mid-phrase ("One of") within a compound noun phrase ("From the

**#1445 [transition_mean] amber→peach (s0):** Structured Wikipedia format with image caption and attributes, listing characteristics of a fruit ("Strawberry") with a formulaic pattern about a Japanese food product. The phrase "If you like this item as a food, it indicates the plant is satisfying, with its description" suggests a concluding clause follows a second attribute or source claim about the user's profile, likely citing a related attribute about the image credit. Final token "practice" ends an incomplete clause ("Even 'Society' in
