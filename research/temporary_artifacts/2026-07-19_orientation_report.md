# Orientation report — Counterfactual Difference NLA

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` · **Phase:** orientation only, no GPU work, nothing downloaded
**Companion log entries:** `research/ARTIFACTS.md` (both 2026-07-19 entries) — this file is the standalone summary of the same session.

---

## 1. What the project claims (in the assistant's own words)

- **North Star.** Move activation-to-language interfaces from "plausible summarizer" to "instrument whose individual claims about a specific internal *change* are causally checkable" — i.e., make a behaviorally relevant activation **difference**, not a full state, the object of the language bottleneck, with claims that can be graded exactly and replayed causally.
- **Proxy task.** On Qwen2.5-7B-Instruct, layer 20 (block-20 output = HF `hidden_states[21]`), final pre-answer position, controlled two-slot binding task: verbalize `Δ_final = h_cf − h_base` into a rigid caption (`old -> new` / `NO_CHANGE`; four deterministic renders per Amendment 1), and have a separately trained frozen reader turn the parsed caption back into a patch that shifts the answer distribution in the stated direction. Gates: v1 doc §9 as amended by Amendment 1 §6.
- **Validity argument.** If a maximally clean, behaviorally certified, single-fact delta cannot be verbalized above the probe/prototype/no-activation floor and causally replayed, richer difference-verbalization claims are unfounded. If it can, the result is a calibrated instrument plus a reusable controls factory for the harder questions (query availability, distributional changes, derived relations).
- **Divergence risk.** The gates can pass via boring mechanisms — answer-token promotion at the final site, transition-prototype lookup, template/metadata leakage — and with deterministic captions the reader is a lookup table **by construction** (Amendment 1 §10). The language claim lives only in held-out values and later derived-property families. The v1 doc's §10 outcome table already prices most of this in.

## 2. Environment fingerprint (recorded; nothing downloaded)

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, capability sm_90 (9,0), 78.7/79.2 GiB free |
| Driver / CUDA | 580.105.08 / CUDA 12.8, nvcc 12.8.93 |
| torch | 2.7.0+cu128 |
| transformers | 5.14.1 (matches the dataset-generation environment) |
| `HF_HOME` | **unset** — set to a path under `/lambda/nfs/cot-oracle/` before any model download (root disk is ephemeral) |
| Persistent volume | `/lambda/nfs/cot-oracle` (NFS v4.1) — repo lives here |
| Egress | **open** on this instance (HF, GitHub, arXiv, LW all reachable) — unlike the generation session's proxy block |
| pytest | 9.1.1 aborts at collection via the *system* libtmux plugin; run `python3 -m pytest research/data/tests -q -p no:libtmux` → **65 passed, 1 skipped** |
| Pinned tokenizer | gitignored, absent from clone → integration test skips; bit-identical regeneration not currently executable (re-fetch at SHA `a09a3545…` is cheap now that egress is open) |

## 3. Frozen-dataset verification (independent, from the parquet — not the manifest)

- **Counts:** 58,320 rows / 7,290 semantic pairs / 1,240 families / 2,514 unique ordered transitions. Cells 19,360 × {TARGET, DISTRACTOR, REVERSE} + 240 NULL_AA; variant axes perfectly balanced (29,160 per level of query_order / prompt_format / preamble); distractor flavors 9,680 crossed_query + 9,680 other_slot. Splits: train 36,240 / dev 4,800 / test_context 4,800 / test_transition 4,800 / test_value 2,880 / test_entity 2,880 / test_name 1,920.
- **Caption audit, all 58,320 rows, five checks** (round-trip re-render, cross-format canonical equality, sha256, caption-vs-answer semantics, edited-entity token identity): **0 failures**. The last two checks are new relative to the prior session's audit.
- **Link integrity** (new): 4,840 `crossed_with` + 2,420 `reverse_of` links; 0 dangling; all crossed pairs share the edit and differ in query; all reverse pairs swap values; no link crosses a split boundary.
- **Held-out-value splits are real:** test_value = {charcoal, lavender, cream, green} (S-only), test_name = 10 complete names (N-only), test_entity = 5 nonces; zero leakage into train. Value survivors from the tokenizer audit: colors 42/52, names 91/124, cities 80/85, nonces 24/24.
- **NULL_AA:** 240 rows, identical prompts (delta exactly zero), all `plumbing_only`.
- **Position floor (sharpened):** by format×preamble cell (14,580 rows each): raw/no-preamble final_pos 18–30 → **100% below** the released AV's `_MIN_POSITION = 50`; chat/no-preamble 47–59 → **29.3% below**; both preamble cells ≥ 90 → 0%. Total below floor: 18,848 rows (32.3%). **Stage B.1 parity must be reported per format×preamble cell (4 cells), not per preamble (2).**
- **Two multiplicity statistics** over the same 2,514 transitions: answer-transitions (TARGET+REVERSE only — the prior plot's quantity, verified correct): city max 10 / median 5, names 92% singletons. **Edit**-transitions (including DISTRACTOR cells — what feeds the `E[Δ|old,new]` prototype baseline): city max 15 (Tokyo↔Singapore), colors median 3, names median 2. Consequence: the §8 "unrelated real delta" control must exclude same-transition pairs or it is not a null.
- **Plots:** `research/plots/2026-07-19_v1_dataset_composition.png`, `…_v1_transition_distribution.png` (prior session, numbers re-verified), `…_v1_final_pos_vs_training_floor.png` (new).

## 4. Injection contract as verified in code (constrains Stage A/B)

- Extraction: hook on `layers[K]` output → `layer_index=20` = block-20 output = HF `hidden_states[21]` (`nla/datagen/extractors.py`); raw vectors, `norm="none"`; original NLA training used positions ≥ 50 only.
- AV: embedding-row **replacement** at the ㈎ marker (id 149705), neighbor-checked scan inside the hook (`nla/injection.py`); vector rescaled to L2 = `injection_scale` = **150.0** from the released sidecar (paper heuristic: α ≈ 75th-percentile activation norm; ambient √d ≈ 59.9). Magnitude is destroyed at the interface — Amendment 1's magnitude ban is structurally correct. Only exactly-zero vectors stay zero (`clamp_min(1e-12)`); the epsilon-guard policy for tiny-but-nonzero deltas exists nowhere in code yet (GPU-stage TODO).
- AR: truncated 21-block backbone, final-LN → Identity, `Linear(d,d)` value head, suffix-anchored extraction at `tokens[-1]`, direction-only score MSE = 2(1−cos) with `mse_scale` = √d.
- `resolve_embed_scale` (inference file only) is the architecture embedding multiplier (1.0 for Qwen) — distinct from the training-side `resolve_target_scale` (`nla/schema.py`).
- Community forks (EasyNLA, nanoNLA) train with **additive norm-matched injection at the layer-1 output** (`h'_p = h_p + ‖h_p‖·v/‖v‖`) — a different interface; their results do not transfer to our checkpoints without re-verification, and the additive-vs-replacement question is **unsettled** (no formal ablation exists; the nanoNLA maintainer has walked back confidence).

## 5. Frozen vs. open — and the single next decision

**Frozen and doubly verified:** v1 scope + claim ladder L0–L6 + stages A–D; caption schema at renderer `amendment1-r2`; the dataset and its splits; tokenizer SHA; the injection-contract facts above.

**Open:** Stage A itself; format freeze (raw vs chat); site freeze; entity-content decision (gated on Stage B mean hierarchy + queried-entity probe); gate freezing after dev inspection; and four user-input items: preamble sign-off (PROVISIONAL), name-survivor list review, slot2 R4/R6 retention, Qwen3-8B patching-log import (citation embargo in force — logs absent from `research/` at the fourth check).

**The single next decision:** does the frozen layer-20 final-position delta pass the causal-substrate gate on dev families? **The number: median normalized counterfactual-margin recovery ≥ 0.50** (with ≥ 80% correct-direction rate and shuffled/unrelated/matched-norm controls < 0.10). Nothing downstream is interpretable until that number exists.

## 6. Contradictions and gaps

Re-confirmed independently from the prior session's audit: `research/PLAN.md` dangling reference (file does not exist); companion methods-review doc still not provided; docs/design.md §2 claims absent `injection_scale` ⇒ `sqrt_d_model` while code resolves absent ⇒ None ⇒ hard assert (code wins); stale ㊗ docstring in `nla/injection.py` (actual marker is ㈎); stale "blocked on tokenizer access" STATUS in `research/data/README.md`; Stage C-before-D vs context-doc §5.8 wording tension (materially dissolved by Amendment 1 §1.2 for transition-only content, one reconciliation sentence still wanted); `resolve_embed_scale` naming; missing epsilon guard.

New this session:

1. `nla/schema.py:82–84` — `resolve_target_scale`'s docstring claims config.py supplies the sqrt_d default for absent keys; true only for `mse_scale`, false for `injection_scale`. Second instance of the design.md §2 error, in the shared module both readers import.
2. ARTIFACTS.md had lost the `##` header of the Amendment-1 §11 entry (now restored).
3. Position floor is a 4-cell phenomenon, not 2 (see §3) — parity reporting requirement tightened.
4. The "unrelated real delta" control in v1 §8 is underspecified given edit-transition multiplicity (see §3).
5. One PDF referenced in the session instructions is absent from the repo and has no public copy — must not be registered, cited, or committed until provided and cleared; ask the user.
6. v1 doc §14 sources the prior Qwen3-8B patching evidence to two files on the user's local machine; until imported, the Amendment 1 §11.2 embargo stands.

## 7. Literature: all ten working conclusions verified against primary sources

Every source was read directly (papers, posts, and fork source code). All ten conclusions in the session brief survive; exact figures and per-conclusion notes are in the ARTIFACTS entry. Highlights:

- **Verified exactly:** implausible-init NLA FVE 0.68 vs 0.70 control with 99.3% implausible claims; RL *decreases* plausible-init plausibility 21% → 7.6% (Zhang & Turner). Post-hoc snippet replacement collapses FVE to −0.09 … −0.76, below the mean baseline; on-policy constrained decoding barely hurts (Anurin). Qwen2.5-7B round-trip FVE 0.748 replicated vs 0.752 (paper appendix) on held-out UltraFineWeb; shuffled-verbalization null < −0.6 (Anurin) — these are the Stage B.1 parity targets. For Qwen, a mean-direction "rock" beats the NLA at reconstruction even dataset-wide at the trained layer; its NLA does not track problem-constant changes (oakhu & ryan_greenblatt) — hence a null on our deltas is uninterpretable without a positive control. 81.4% explanation flip with 99.6% behavior preservation under per-sample optimization (Prabhu) — no monitor claims; scope to natural deltas. Answer-mention rate rises toward the final answer and correlates with lower reconstruction loss, on our exact checkpoints (Realmbird) — final-position captions are at risk of being answer-transition readouts. Probes beat decoder extraction on every tested task, by ~3–49pp depending on task (Chalnev) — expect "probe succeeds, AV fails" as the default.
- **Calibration notes:** the auditing-task decoupling (reconstruction reward smooth, surfacing erratic) is per the paper "generally increase … with an unexplained spike", possibly string-matching noise — the brief's "suggestive, not established" caveat is the paper-accurate reading. The Jakkli near-chance AO sycophancy AUC is disputed by Bauer et al. §A.1 as a calibration artifact (Yes/No logit margin gives 0.83) — **rule adopted: score AO/probe baselines on logit margins or AUC, never sampled strings.** Bauer's "every run with NLA-style injection performed significantly worse" is explicitly not a formal ablation.
- **Controls-provenance map:** activation-dependence controls ← Zhang & Turner; Stage B.1 parity gate ← oakhu FVU > 1; crossed pairs ← AObench's "Activation Sensitivity" task (same tokens, different upstream context); matched-norm random patch control ← oakhu's steering null (11/65 targeted ≈ 10/65 random vectors); shuffle nulls ← Anurin; matched-length in-distribution ablation rule ← Anurin; truncated/masked-caption reader training (backlog, not v1) ← public comments on the length-penalty post.
- Useful precedents registered: zero-shot diff-of-means verbalization shows qualitative signal (anger/pirate/language visible; SAE feature directions mostly not) — closest public precedent for Stage B.3, bounding expectations at "qualitative, unreliable"; the AR value head is near-identity (+0.007 FVE, Gemma-27B); registry rows for Chalnev/Jakkli/Bauer/EasyNLA/nanoNLA corrected accordingly in `research/CLAUDE.md`.

## 8. Disagreements / spec gaps (small, pre-GPU)

1. Define "unrelated delta" to exclude same-transition pairs (else contaminated by the shared prototype).
2. Pre-register the expected Stage B.1 parity outcome per format×preamble cell now, so a raw/no-preamble parity failure is spent as confirmation, not news.
3. The preamble is PROVISIONAL but baked into half the frozen rows — sign-off (or swap + regeneration) belongs **before** Stage A GPU spend.

No structural disagreement with the plan; it is in an unusually well-pre-registered state.

## 9. Recommended next action

**Run Stage A.** It needs only the target model at the pinned revision — no NLA checkpoints — and its margin-recovery number gates the entire project: set `HF_HOME` to the NFS volume → re-fetch the pinned tokenizer (restores the integration test and freeze verification) → download Qwen2.5-7B-Instruct → Level-0 plumbing parity (frozen `input_ids` match, hook parity, alpha-zero identity, batch-size-1 parity) → patch-control matrix on dev families with margin + JS + top-k, reported per format×preamble cell, folding the format-freeze decision into the same screen. Every evaluation post read this session says the released checkpoint is too noisy to interpret without exactly this kind of certified substrate.
