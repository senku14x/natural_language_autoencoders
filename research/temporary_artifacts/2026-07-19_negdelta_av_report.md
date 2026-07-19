# −Δ arm through the zero-shot AV — report (work item 1a)

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` (uncommitted, pending user review)
**Phase:** exploration. One added arm to the existing zero-shot experiment; released AV, no fine-tuning.
**Predictions registered first:** `2026-07-19_negdelta_logitlens_predictions.md`.

## Question

The released AV names the new value at 0.74 and the old value at 0.00 from the
same delta that a linear probe decodes both endpoints from at ~0.96/0.96. Is the
old side AV-inaccessible, or is the asymmetry a sign convention — the AV naming
whichever token identity sits in the positive direction?

## Setup

- Reused artifacts, nothing re-extracted: `zeroshot_av/h_edit_{base,cf}.npy`
  (594 eligible dev raw/preamble change rows), the same 50-pair sample
  (`sample_manifest.json`, seed 20260719), the same whole-word case-insensitive
  string rule, same AV checkpoint (snapshot `b884691…`), sidecar-driven injection
  (㈎ 149705, scale 150, two-step render→encode), 5 samples @ T=1 per pair,
  `max_new_tokens=220`. Script `research/exploratory/negdelta_av_arm.py`.
- **Environment (this instance differs from the prior session):** fresh H100 80GB,
  driver 580.126.20, torch **2.12.0+cu130** (prior: 2.7.0+cu128), transformers
  5.14.1 (pinned), Python 3.12.13. `/workspace` is **not volume-backed**
  (`workspace_is_volume: false`) — nothing on this instance survives
  recycle/destroy. Only the AV generation pass ran on this hardware; all target-model
  activations are the prior session's saved arrays. pytest 65 passed / 1 skipped;
  pinned tokenizer re-fetched.

## Pre-registered fact that sharpened the prediction

Every REVERSE pair is the **exact prompt swap** of its TARGET partner
(200/200 raw/preamble rows verified: `rev.base_input_ids == tgt.cf_input_ids` and
vice versa; `reverse_of` resolves via semantic_id). Under the canonical-shape
deterministic forward this makes **−Δ(row) bitwise equal to the real Δ of its
partner row** — a vector class the prior run already read at 0.77 (its REVERSE-cell
rows). Outcome 1 ("sign convention") was therefore heavily favored a priori
(predictions file, P1–P3); the run verifies the antisymmetry end-to-end through
the AV rather than discovering something new.

## Predictions vs outcome — hit

| quantity | predicted | observed |
|---|---|---|
| −Δ old-value mention | ~0.74 [0.60, 0.85] | **0.696** [0.580, 0.804] |
| −Δ new-value mention | 0.00–0.04 | **0.000** |
| per stratum (S / N) | ~0.89 / ~0.58 | **0.854 / 0.525** |

## Observations (dev, this setup; 50 pairs × 5 samples; pair-bootstrap 95% CI)

- **−Δ names the old value 0.696 [0.580, 0.804] and the new value 0.000.** The
  exact mirror of the real arm (new 0.740 / old 0.000).
- Paired per-pair comparison: (−Δ old-rate) − (real-Δ new-rate) = **−0.044
  [−0.176, +0.092]** — statistically indistinguishable from the real arm.
- Stratum gap mirrors the real arm: colors 0.854 vs names 0.525 (real: 0.892/0.575).
- Edited-entity mention 0.000, as in every prior arm. Explanations show the same
  pattern as the real arm: the old-value word embedded in confabulated prose
  (random inspection: "'mustard' or 'mustard'", "'Crimson'", "olive" in a list
  context); no parse failures.
- Full dump: `zeroshot_av/negdelta_explanations.json`; scored rows
  `negdelta_scored.parquet`; summary `negdelta_summary.json`.
- Plot: `plots/2026-07-19_negdelta_av_mention_rates.png` (all seven arms on one
  axes, shuffled/random floor drawn).

## Interpretation (kept separate)

- **The 0.74/0.00 asymmetry is a sign convention, as hypothesized.** The AV
  treats its input as a state and names the token identity in the positive
  direction; a negated component reads as absence, not as "points away from."
  Both endpoints are AV-accessible — one per sign. This was near-implied by the
  prompt-swap identity plus the prior REVERSE-row readings; the run's marginal
  contribution is verifying the vector-level antisymmetry end-to-end through the
  AV interface with no deviation beyond sampling noise (and doing so on this new
  instance/torch version, using the saved activations).
- **Consequence for the SFT (work item 2):** training the AV to emit
  `old -> new` teaches a *bidirectional read of a code the model already decodes
  one direction at a time* — surfacing the negative-sign component alongside the
  positive one, not creating a new decodable quantity. A large SFT gain on the
  old field should therefore not be sold as the AV "learning the transition";
  the information and (sign-conditional) decodability were both already present.
- Scope cap unchanged: this is **token-identity readout** of the transition code
  (Amendment 1 §10; NEW_WARMSTART_CRUXES §C3). Nothing here is a consequence,
  caption, or language result.

## Limitations

- Same 50 pairs / dev / raw+preamble / this model/layer/site as the parent run;
  T=1 sampling; generation ran on different hardware+torch than the prior arms
  (the injected vectors are identical; cross-arm comparisons assume the AV's
  sampling distribution is not materially torch-version-dependent — the h_cf/
  h_base/real-arm rates were not re-measured this session).
- TARGET/REVERSE rows only: edited == queried remains confounded by construction.
