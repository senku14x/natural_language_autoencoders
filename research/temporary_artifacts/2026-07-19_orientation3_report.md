# Orientation report #3 — Counterfactual Difference NLA (fresh instance)

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` · **Phase:** orientation only — no GPU work, no weights downloaded
**Mode:** absorb + spot-check + extend the two prior same-day orientations (`ARTIFACTS.md`, `temporary_artifacts/2026-07-19_orientation_report.md`), not re-derive them.

---

## 1. The project in my own words

- **North Star.** Turn activation-to-language interfaces from plausible summarizers into instruments whose *individual claims about a specific internal change* are exactly gradable and causally replayable — i.e., make the behaviorally relevant activation **difference** the object of the language bottleneck, with calibrated failure modes.
- **Proxy task.** Qwen2.5-7B-Instruct, layer 20 (block-20 output = HF `hidden_states[21]`), final pre-answer position, frozen two-slot binding dataset (58,320 rows): verbalize `Δ_final = h_cf − h_base` into a deterministic caption (Amendment-1 2×2 renders), and have a separately trained frozen reader convert the parsed, canonically re-rendered caption into a patch that moves the answer distribution in the stated direction. Gates: v1 §9 as amended by Amendment 1 §6.
- **Validity argument.** A behaviorally certified single-fact delta is the easiest possible case. Failure there bounds all richer difference-verbalization claims; success yields a calibrated instrument plus a reusable controls factory (availability, distributional change, derived relations are gated extensions, not v1).
- **Divergence risk.** Every gate can pass through boring channels — answer-token promotion at the final site, transition-prototype lookup, template/metadata leakage — and with deterministic captions the frozen reader is a lookup table **by construction** (Amendment 1 §10). The language claim lives only in held-out values/names/entities and later derived-property families. v1 §10's outcome table prices this in; the discipline is keeping it priced in at write-up time.

## 2. Environment fingerprint (this instance — differs from the prior report)

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, capability (9,0)/sm_90, 81,080 MiB free, driver 580.105.08 |
| CUDA / nvcc | 12.8 / 12.8.93 · torch **2.7.0+cu128 preinstalled** |
| Python | 3.12.3 (Ubuntu 24.04.4) |
| **transformers** | **absent** — prior report's "5.14.1 matches generation env" described the *previous* instance; must be installed (pin 5.14.1) before Stage A |
| Installed this session (user-level) | pyarrow 25.0.0, pandas 2.1.4, pyyaml, pytest; gh 2.96.0 + hf CLI 1.24.0 (both authenticated) |
| Persistent volume | `/home/ubuntu/counterfactualnlas` (virtiofs mount of `/lambda/nfs/counterfactualnlas`) — **not** the prior report's `/lambda/nfs/cot-oracle`, which does not exist here. Repo cloned to `<volume>/natural_language_autoencoders`. Everything outside the volume (incl. `~/.cache`, `~/.local`) is ephemeral root disk (2.7 TB free). |
| `HF_HOME` | unset — set to a path under the persistent volume before any download |
| Egress | open (HF, GitHub, LW, transformer-circuits, arXiv all reachable; pip + gh work) |
| pytest | system 9.1.1; `research/data/tests` with `-p no:libtmux` → **65 passed, 1 skipped** (integration test needs the pinned tokenizer — gitignored, absent from clone) |

## 3. What I verified from prior sessions (independently, this session)

All from the frozen parquet / code / primary sources, not from the manifest or the prior entries:

- **Dataset counts, exact:** 58,320 rows / 7,290 `semantic_id` pairs / 1,240 families / **2,514** unique ordered *edit* transitions; cells 19,360 × {TARGET_EDIT, DISTRACTOR_EDIT, REVERSE} + 240 NULL_AA (all `plumbing_only`); perfect 29,160/29,160 balance on query_order, prompt_format, preamble; distractor flavors 9,680 + 9,680; splits train 36,240 / dev 4,800 / test_context 4,800 / test_transition 4,800 / test_value 2,880 / test_entity 2,880 / test_name 1,920. (My first pass got 2,726 "transitions" by including DISTRACTOR answer-columns — wrong quantity; the 2,514 figure is correct for edit transitions and equals the TARGET+REVERSE answer-transition count.)
- **Held-out axes, exact:** test_value \ train = {charcoal, cream, green, lavender}, stratum S only; test_name \ train = 10 names, N only; test_entity \ train = 5 nonces; zero leakage.
- **Caption audit re-run dataset-wide with the repo's own renderer/parser:** all 58,320 rows × (4-way round-trip re-render + cross-format canonical equality + all 4 sha256s) → **0 failures**.
- **Link integrity:** 38,720 `crossed_with` + 19,360 `reverse_of` row-level links (= 4,840/2,420 semantic × 8 variants — consistent with the prior entry's semantic-level counts), 0 dangling, 0 crossing a split boundary.
- **Position floor, exact per cell:** raw/no-pre 18–30 (100% < 50), chat/no-pre 47–59 (29.3% < 50), both preamble cells ≥ 90 (0%); total 18,848 rows (32.3%) below the released AV's `_MIN_POSITION = 50`.
- **Injection contract in code:** embedding-row replacement at ㈎ with neighbor check (`nla/injection.py`; stale ㊗ docstring confirmed); `config.py:182–183` — absent `injection_scale` → None → hard assert, only `mse_scale` defaults to √d (confirming the `schema.py:82–84` docstring is wrong for `injection_scale`); `normalize_activation` clamps only exact-zero (1e-12) — the epsilon-guard gap is real; `_MIN_POSITION = 50`; hook = `layers[K]` forward output; AR = truncated 21-block backbone, final-LN→Identity, `Linear(d,d)` value head, `tokens[-1]` anchor; `resolve_embed_scale` (inference only) = 1.0 for Qwen.
- **NEW — released sidecar verified from the HF release itself** (`kitft/nla-qwen2.5-7b-L20-av/raw/main/nla_meta.yaml`, no weights downloaded): `injection_scale: 150.0`, `mse_scale: 59.8665… = √3584` exactly, ㈎ id 149705, neighbors 29/522, `extraction_layer_index: 20`, and the exact AV prompt template (`<concept>㈎</concept>`, "2-3 text snippets") and AR template (`Summary of the following text: <text>…</text> <summary>`). Prior sessions had this from code paths only; the released artifact itself now independently confirms every number.
- **Values audit:** colors 42/52, names 91/124, cities 80/85, nonces 24/24 survive all four surface forms — exact match.
- **Qwen3-8B prior patching artifacts: still absent** (no logs/model/layer/site/patch formula anywhere in `research/`; only doc mentions). **Fifth consecutive flag.** The Amendment 1 §11.2 citation embargo remains in force; any claim resting on that result stays unsupported until the logs are imported.
- **Both open decisions still open** (`research/data/README.md` decisions 4 & 5): `NOT_IDENTIFIABLE_FROM_THIS_STATE` (parser-valid, never emitted — awaiting user call) and preamble text (PROVISIONAL — awaiting sign-off). Also still pending: name-survivor list review, slot2 R4/R6 retention.

Literature spot-checks (targeted verbatim extraction from the primary sources, sample chosen for numeric density):

| Source | Claim checked | Result |
|---|---|---|
| Zhang & Turner (turntrout) | FVE 0.68 implausible vs 0.70 plausible; plausibility 0.7% vs 7.6%; RL decreases 21%→7.6%; 0.70 round-trip at 20k docs vs Anthropic 0.75 at 100k; last-paragraph-only FVE 0.67 | **exact** |
| oakhu & ryan_greenblatt | "rock" (mean-direction) beats Qwen NLA dataset-wide at trained layer and within-variant for all three NLAs; Qwen NLA misses problem-constant changes, Gemma-27B detects them; steering 11/65 self-correct ≈ 10/65 random ≈ 17% targeted | **exact** (see §4 for two registry-row phrasings I could not re-verify) |
| Anurin | 0.748 vs paper 0.752; shuffled < −0.6; snippets→empty −0.091, →REDACTED −0.759 (below mean baseline); quote-ban 0.774→0.757; transfer ~0.51–0.68; diff-of-means qualitative | **exact** |
| Prabhu | 81.4% explanation flip (from 26.5% baseline error), 99.6% answer-label preservation, per-sample prefix-tuning-style vectors, Qwen2.5-7B L20 | **exact** |
| loops | λ=0.001 −28% tokens +0.007 FVE; λ=0.002 ~−40% −0.015; λ=0.03 degenerates to tail repetition; Karvonen's random-K-prefix reader proposal in comments | **exact** |
| Chalnev | probes beat decoder on every task; AG News ~3pp … Language ID ~49pp; Alice/apples→Sarah/marbles verbatim; cycle consistency ≠ faithfulness stated explicitly | **exact**, one precision note (§4) |
| Realmbird | kitft L20 AV/AR checkpoints; answer-mention rises toward final answer; higher counterfactual importance ↔ lower reconstruction loss; GSM8K-Aug, SGLang | **exact** |
| nanoNLA README | maintainer "I used to be confident that this was significantly better, I no longer am" re additive vs replacement | **verbatim** |

## 4. What I could not verify, and what I think needs correcting

1. **Could not re-verify (page-size limit, not evidence of error):** the NLA paper's auditing-case decoupling figure (reconstruction reward smooth, surfacing erratic with an "unexplained spike") — the transformer-circuits page exceeds what my fetch pass could cover. Orientation #2 verified it directly with the correct calibration ("suggestive, not established" is the paper-accurate reading); I carry it on trust from that session.
2. **Registry-row phrasings I could not find in the source (flag, low stakes):** `research/CLAUDE.md`'s oakhu row says "FVU > 1 on OOD short prompts" and "poor cross-layer generalization." Two targeted extractions found the *substance* (rock beats NLA ⇒ FVU > 1 w.r.t. dataset variance is a fair gloss) but no literal FVU>1 statement and no cross-layer-generalization result in the post body. Either appendix content my extractor missed, or glosses that should be reworded to what the post shows. The design consequence (Stage B.1 parity is load-bearing; nulls need positive controls) rests on the confirmed rock result either way.
3. **Corrected for precision (edited in `research/CLAUDE.md` this session):** the Chalnev row's "25–33pp on gender/number" is the **mean-over-tokens** condition (25.4pp gender, 33.3pp singular/plural); last-token gaps are 19.7pp and 7.7pp. Range claim (~3–49pp) unchanged.
4. **ARTIFACTS.md lost another entry header:** the first orientation entry (commit 5d1637d) had no `## date — title` line and read as a continuation of Orientation #2 — the *same* failure mode Orientation #2 fixed for the Amendment-1 entry. Restored this session. Two occurrences in two days says the newest-first append workflow is error-prone; suggestion: always paste the full `##` header template before writing an entry.
5. **`research/CLAUDE.md` item 5 was stale:** `AO_NLA_JSPACE_METHODS_LEARNINGS.md` is now **provided** (commit 639300a, user upload, after Orientation #2 ended). Updated the line. I read the doc in full; its five sources are all published (arXiv 2512.15674, the Jakkli workshop paper, arXiv 2606.02609, the two Transformer Circuits papers) — I found nothing in it that names or reproduces unpublished material. Its content is consistent with the context doc and adds the AO-line synthesis (four confounds, mandatory baseline set, solvability + text-inversion-hardness design principle, probe-ceiling observation) that the project's controls already encode.
6. **Prior report's environment table is stale for this instance** (§2 above): different persistent volume path, transformers absent. Anything scripted against `/lambda/nfs/cot-oracle` must be repointed.

Nothing I checked invalidates any frozen artifact or any prior conclusion. The two prior orientations were accurate on every number I re-derived.

## 5. Where the literature bounds our claims — and the ten working conclusions

The ten working conclusions in the session brief all survive my spot-checks; none is overstated. Per-conclusion status: (1) exact; (2) carried with its own "suggestive, not established" caveat — which prior verification showed is precisely the paper-accurate reading; (3) exact, two independent sources; (4) exact, incl. the below-mean-baseline collapse and the comments-thread provenance of the truncated-reader fix (backlog-only is right); (5) exact in substance — and its implication is the single most important operational fact for us: **a null from the released checkpoint on our deltas is uninterpretable without a positive control**; (6) exact incl. the verbatim entity-substitution example; grade exactly, expect AV ≪ probe; (7) exact, on our exact checkpoints — final-position captions are at risk of being answer-transition readouts, which v1 §10 already accepts as a capped-claim outcome; (8) exact — scope everything to naturally-arising deltas, no monitor claims; (9) exact (0.748/0.752 replication, shuffled < −0.6, 0.70@20k) — these are the Stage B.1 parity targets, now doubly grounded; (10) our own norm, and every strong source we checked follows it.

Boundary summary: reconstruction ≠ faithfulness (1,2,4); locality and answer-leakage pressures (3,7); off-policy ablation is not semantic deletion (4); our instrument is noisy relative to our effect sizes (5,9); expect under-extraction vs probes (6); no adversarial/monitor claims (8). The v1 gates + Amendment 1 + the context doc's §4/§5 already encode all of these; the project's exposure is execution discipline, not design gaps.

## 6. Frozen vs. open — and the single next decision

**Frozen (and now triply verified):** v1 scope, claim ladder L0–L6, stages A–D; caption schema at renderer `amendment1-r2` with the four deterministic caption columns; the dataset, splits, and links; tokenizer SHA `a09a3545…`; the injection contract (code + released sidecar).

**Open:** Stage A itself; format & site freeze; entity-content decision (gated on Stage B mean hierarchy + queried-entity probe); gate freezing after dev inspection; four user-input items — preamble sign-off (PROVISIONAL, baked into half the rows), name-survivor review, slot2 R4/R6 retention, Qwen3-8B log import (embargo in force) — plus three pre-registrations flagged by Orientation #2 that I endorse: exclude same-transition pairs from the "unrelated delta" control; score AO/probe baselines on margins/AUC only; pre-register expected Stage B.1 parity per format×preamble cell.

**The single next research decision:** does the frozen layer-20 final-position delta pass the causal-substrate gate on dev families — **median normalized counterfactual-margin recovery ≥ 0.50**, correct direction ≥ 80% of held-out families, shuffled/unrelated/matched-norm controls < 0.10, reverse deltas reversing. Nothing downstream is interpretable until that number exists. (The single next *user* decision is preamble sign-off, because a swap forces regeneration and would invalidate Stage A work on preamble rows — it should precede GPU spend.)

## 7. Disagreements with the plan

No structural disagreement — after three independent audits the plan remains unusually well pre-registered. Three additions, stated as recommendations:

1. **Pre-register the format-freeze trade-off rule before Stage A.** The cell most in-distribution for the released AV (preamble, final_pos ≥ 90) and the cell most likely to give the cleanest causal substrate (short raw prompts) may disagree. Stage A folds the format decision into the behavioral screen; decide *now* how substrate quality trades against AV in-distribution-ness (e.g., "freeze the cell with the best margin recovery subject to ≥1 cell above the B.1 parity floor"), so the choice can't drift post hoc.
2. **Make the epsilon guard a Stage A deliverable with a measured threshold, not a TODO.** Concrete proposal: during Level-0 plumbing, measure the numerical noise floor of extraction (repeated forward passes on identical NULL_AA prompts; their deltas are exactly zero in exact arithmetic, so any nonzero norm is floor), then set ε = k × that floor (k ~ 10) and treat ‖Δ‖ < ε as zero at the AV interface. Costs minutes on the same forward passes Stage A already runs.
3. **Pin `transformers==5.14.1` for Stage A** (the generation environment's version) and re-run the tokenizer integration test + bit-identical regeneration check as the very first Stage A action, before any activation is extracted. Chat-template behavior is version-sensitive and the frozen `input_ids` are the ground truth for everything downstream.

## 8. Stage A readiness

**Stage A needs:** Qwen2.5-7B-Instruct at a pinned revision (~15 GB; no NLA checkpoints), the pinned tokenizer files at SHA `a09a3545…`, transformers/accelerate/safetensors/matplotlib installed, `HF_HOME` on the persistent volume. All egress is open; nothing else is missing.

**Recommended order:**

1. `HF_HOME=/home/ubuntu/counterfactualnlas/hf_home` (persisted in `~/.profile`).
2. `pip install --user transformers==5.14.1 tokenizers accelerate safetensors matplotlib` (+ record the lock in the ARTIFACTS entry).
3. Re-fetch the pinned tokenizer at SHA `a09a3545…` into `research/data/tokenizer/` → integration test (66/66) → bit-identical regeneration check against the frozen manifest (`config_hash 0d85e5a6…`).
4. Download Qwen2.5-7B-Instruct, pin and record the model revision.
5. Level-0 plumbing: frozen `input_ids` re-tokenization match; hook parity (`layers[20]` forward output ≡ `hidden_states[21]`); alpha-zero identity patch; batch-size-1 vs batched parity; extraction noise floor → freeze ε (see §7.2).
6. Patch-control matrix on dev families, per format×preamble cell: real / reverse / null / shuffled / unrelated (same-transition excluded) / matched-norm random / output-token direction / mean hierarchy; margin recovery + JS + top-k; then site/format freeze and the §6 gate check.

**Before GPU spend, wanted from the user:** preamble sign-off, name-survivor list, slot2 R4/R6 call, Qwen3-8B logs (or explicit "keep embargoed"), and sign-off on the three pre-registrations in §6.
