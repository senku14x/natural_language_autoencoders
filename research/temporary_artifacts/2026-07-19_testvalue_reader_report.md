# test_value reader-level eval — report (the registered single spend)

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2`
**Phase:** validation of the reader-generalization question (cruxes §C3, probe item 4).
**Predictions and protocol registered first:** `2026-07-19_testvalue_predictions.md`.
**Split usage:** second and final touch of `test_value` (first: EXP-2 probe-level 1-NN). Run once with user sign-off; no iteration.

## Question

Is the SFT'd reader (work item 2 adapter) a general value decoder, or a closed
set over its ~100 train caption values? EXP-2 established unseen values are
*represented* consistently (1-NN 0.89); this tests whether a trained reader can
*name* them, using the open-vocabulary decoder EXP-2's method lesson demands.

## Setup (as registered)

480 test_value raw/preamble change rows (60 fresh families, stratum S only;
held-out colors {charcoal, cream, green, lavender}, verified absent from all
eligible train captions). Screened with the identical rule/machinery as the
train pass; Δ at edit_pos, canonical-shape forwards. Arms: SFT adapter × real Δ
(greedy, strict fullmatch parse); SFT adapter × permuted Δ; zero-shot released
AV × real Δ (5 @ T=1, string match). Endpoint classes scored separately.

## Predictions vs outcome

| quantity | predicted | observed |
|---|---|---|
| eligibility | ~55% | **44.2%** (212/480) — miss; held-value rows are less eligible (0.35 vs 0.54 seen→seen; held→held 0/8) |
| seen→seen pair-exact (in-split reference) | ~0.99 | **1.000** (n=130) |
| held-side exact naming | ~0.45 [0.05, 0.90]; P(≥0.7)=35% | **0.939** strict [0.87, 0.99] / **0.963** case-insensitive |
| zero-shot reference on held values | ≈ seen (0.85–0.90) | held-new **0.941** vs seen-new 0.800 |
| permuted-Δ own-label | ≈ 0 | **0.000** (donor-label 0.976) |
| miss structure | nearest-seen color substitutions | **wrong — surface-form garbles of the correct word instead** |

## Observations (test_value, this setup; family-bootstrap 95% CI)

1. **SFT × real Δ, pair-exact by endpoint class:** seen→seen **1.000** (130) ·
   seen→held **1.000** (41) · held→seen **0.878** [0.74, 0.98] (41). Held→held
   had no eligible rows (0/8).
2. **The adapter names values it never emitted in training.** Held-side field
   accuracy **0.939** strict / **0.963** case-insensitive (82 fields), vs
   seen-side 0.994. All five held-side misses are surface-form garbles of the
   **correct** word — "Charcoal -> Navy" (×2, capitalization only), "charlie",
   "crem", "cremation" — and **zero** are nearest-seen-color substitutions
   (no gray-for-charcoal, ivory-for-cream, etc.).
3. **The asymmetry is on the old (sign-negative) side:** seen→held (held value
   in the *new* field) is perfect; held→seen (held value in the *old* field)
   carries all the misses — consistent with 1a's finding that the
   negative-direction read is the less native one.
4. **Zero-shot reference:** the released AV mentions held-out new values at
   0.941 (vs 0.800 for seen new values on these rows) — as predicted, it has
   no train/held distinction; SFT did not need to *add* held-value readability,
   and did not remove it.
5. **Permuted-Δ:** own-label 0.000, donor-label 0.976 — activation dependence
   total on this split too. The Δ-ignoring floor for held values is exactly 0
   (they are unreachable from the train caption distribution).
6. **Eligibility miss (owned):** 44.2% vs predicted 55%; rows with held-value
   answers screen at 0.35 vs 0.54 for seen→seen. Plausible contributors: the
   target model answers rarer color words less reliably, and charcoal/lavender
   have only one audited single-token surface form (fewer argmax forms count as
   correct). Not diagnosed further; scoped to the screen, not the reader.
7. Plot: `plots/2026-07-19_testvalue_reader.png`. Data:
   `data/artifacts/v1/testvalue_reader/` (rows, states, three eval parquets,
   summary.json).

## Interpretation (kept separate)

- **Registered rule outcome: "reader generalizes."** Combined with EXP-2
  (representation-level 1-NN 0.89), the value code is now general at **both**
  levels under these conditions: the delta encodes unseen values consistently,
  and a LoRA-SFT'd reader maps them to the correct token strings despite never
  having produced them in training. The feared SFT-induced closed-vocabulary
  collapse did not occur; the residual held-side deficit is surface-form
  (casing/tokenization of never-emitted words), not semantic.
- **Claim, scoped:** on this model/layer/site, raw/preamble colors stratum,
  the edit-site delta → `old -> new` channel is a **general token-identity
  code with a general reader** — not a per-value lookup at either level. The
  Amendment 1 §10 cap now binds at its final line: with values generalizing,
  the remaining "language earns its keep" question lives **entirely in derived
  families** (and paraphrase/edit semantics), exactly as the backlog ordered.
- Still NOT established: anything about consequence/NO_CHANGE (site is
  behaviorally blind), names/cities strata (test_value is S-only), language
  beyond a two-slot token code, or robustness beyond greedy decode on this
  template.
- Speculation (flagged): seen→held at 1.000 vs held→seen at 0.878 suggests the
  positive-direction (new) read is the AV's native operation and the negative
  side is the SFT-taught one, with less headroom for rare surface forms.

## Limitations

- n=41 per held class; held→held empty (screen); one seed/adapter; strict
  grading counts case-only errors ("Charcoal") as misses — both strict and
  case-insensitive reported. The split is now spent for reader-level questions.
