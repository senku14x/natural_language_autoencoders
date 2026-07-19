# AV SFT on edit-site deltas — report (work item 2)

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` (uncommitted, pending user review)
**Phase:** execution/feasibility (Stage-C-shaped, but **not** the registered Stage C — see scope note).
**Predictions and frozen hyperparameters registered first:** `2026-07-19_sft_predictions.md`.

## Question

Can a LoRA fine-tune of the released kitft L20 AV emit the full transition
`old -> new` (`caption_arrow_transition`, strict fullmatch grading) from an
edit-site L20 delta?

## Setup (as registered; nothing tuned post hoc)

- **Train:** 4,348 eligible train raw/preamble change rows (72.5% of 6,000;
  eligibility = Stage A screen rule computed on this instance; stratum split
  N 99.6% / S 54.4% — matches the frozen dev pattern N 100% / S 57.1%, a data
  property: the model answers name/city bindings more reliably than colors).
  Δ at `edit_pos`, canonical-shape forwards; delta norms q25/50/75 = 67.5/74.6/82.0
  (matches prior sessions). Captions: bare `"old -> new"` + EOS after the
  unmodified sidecar AV prompt (injection contract unchanged: ㈎ replacement,
  L2→150).
- **Dev eval:** 594 eligible raw/preamble change rows (frozen Stage A labels;
  this-instance screen agreement 99.25%, 6/800 flips — the torch 2.12 vs 2.7
  drift is marginal). Dev activations re-extracted this instance for train/eval
  consistency. **Families fully disjoint from train (overlap 0). All dev values
  train-seen; 46.5% of dev rows (276/594) have a train-unseen ordered
  transition.** `test_value` untouched.
- **Runs:** two, identical (LoRA r=64 α=128 dropout 0.05 on q/k/v/o/gate/up/down;
  AdamW lr 2e-5 cosine, 5% warmup, 3 epochs, batch 32, bf16, seed 20260719;
  161M trainable = 2.08%), differing only in Δ assignment: real vs permuted
  across train examples (transition-coincidence rate of the permutation 0.000).
- **Eval arms:** real-SFT × real Δ; shuffled-SFT × real Δ; real-SFT × permuted
  dev Δ (also scored against the donor's labels). Greedy, max 16 new tokens,
  strict parse (`parse_arrow_transition`); unparseable = wrong.
- Script `research/exploratory/sft_transition_av.py` (+
  `sft_extract_train_deltas.py`); artifacts `data/artifacts/v1/sft_transition/`
  (adapters, eval parquets, loss curves, `sft_summary.json`).

## Predictions vs outcome — direction right, magnitude under again

| quantity | predicted | observed |
|---|---|---|
| real × real, pair-exact | 0.82 [0.65, 0.92] | **0.997** [0.992, 1.000] |
| … per stratum S / N | 0.90 / 0.70 | **1.000 / 0.994** |
| shuffled × real, pair-exact | ≤ 0.05 | **0.000** |
| real × permuted, own-label | ~0.00–0.03 | **0.002** |
| real × permuted, donor-label | mechanism predicted | **0.997** |
| parse failures | < 2% | 0.0% (all arms) |
| train eligibility | ~74% | 72.5% |

Calibration note (recurring): this is the second consecutive under-prediction of
delta token-identity readability (zero-shot: predicted 0.15, got 0.74; here
0.82 → 0.997), while I over-predicted the trivial decoder (0.45 → 0.04). The
probe numbers (0.945/0.870) were not a ceiling — a PCA-64 logistic probe on dev
pools is a weaker reader than a 7B SFT on 4.3k rows.

## Observations (dev, this setup; family-bootstrap 95% CI)

1. **real-SFT × real Δ: ordered-pair exact 0.997 [0.992, 1.000]** — old-field
   0.997, new-field 0.998; S 1.000 [1,1], N 0.994 [0.981,1]. The zero-shot
   stratum gap (0.89 vs 0.58) closed. The only errors: 2 generations of one
   semantic pair (Hans → Helen: "Jsans -> Helen", "Rs -> R").
2. **Unseen ordered transitions of seen values: 0.993** (276 rows; S 1.000 /
   N 0.987) — indistinguishable from seen-transition rows. The mapping
   composes per-endpoint; it is not a per-transition lookup.
3. **shuffled-SFT × real Δ: 0.000** pair-exact (old 0.003 / new 0.007), at the
   analytic Δ-ignoring floor (0.003; mode transition Singapore→Tokyo at
   frequency 0.003, mode field values 0.012). Greedy decode collapsed to
   essentially one caption ("Hassan -> Ralph", 583/594). Its train loss is
   pinned at ~1.95 nats (≈ caption-prior entropy) while the real run reaches
   ~0.001 (curve in the plot).
4. **real-SFT × permuted Δ: own-label 0.002, donor-label 0.997** — the model
   emits the injected vector's transition, not its row's. Activation dependence
   at eval is total; the caption channel carries no row-metadata leak (the AV
   sees only the vector — this arm demonstrates it end-to-end).
5. Registered gate analog: real − shuffled = **+0.997** ≫ 40pp. Amendment 1 §6
   spirit-gates (transition-slot ≥0.90; real ≥ prior+40pp; shuffled collapses
   to prior) all clear on this one format/init.
6. Plot: `plots/2026-07-19_sft_transition_accuracy.png` (accuracy by arm ×
   stratum with the analytic floor, trivial-decoder 0.04, and zero-shot 0.74
   drawn on the same axes; loss curves real vs shuffled).

## Interpretation (kept separate)

- **Feasibility is settled: the released AV, LoRA-adapted, emits the exact
  bidirectional transition from the edit-site delta at ceiling on seen-value
  dev data, against a clean empirical Δ-ignoring floor.** The old endpoint —
  zero-shot inaccessible (0.00) but present in the negative sign direction
  (1a) — is fully surfaced by SFT (0.997): consistent with the SFT teaching a
  read of the sign-negative component, per 1a's sign-convention account.
- **The transition-composition result (0.993 on unseen pairs) moves the lookup
  bound from the (old,new) pair to the individual value** — the learned reader
  is a per-value token-identity dictionary applied twice with a sign
  convention, not a memorized pair table. This is the strongest form the
  seen-value result can take, and it is exactly what the position-sweep/1-NN
  geometry predicted (delta ≈ new-token − old-token in a general value basis).
- **Hard caps that stand (Amendment 1 §10; cruxes §C3):** dev shares the value
  vocabulary with train, so this certifies a **transition token code over seen
  values** — not language (deterministic template ⇒ reader-is-lookup at the
  value level), not held-out-value generalization (`test_value` untouched, the
  one split with remaining discriminating power), not consequence/NO_CHANGE
  (deliberately excluded from training because the site is behaviorally
  blind). Per 1b, the code the SFT reads is at least non-trivially readable
  (unembedding/input-embedding baselines fail) — but "reads a non-trivial
  token-identity code" is the whole claim.
- **Scope note vs the registered plan:** this is one format (arrow) × one init
  (released), no vanilla-init arm, no sentence format, no per-format
  no-activation baseline (the shuffled-SFT arm empirically bounds the
  Δ-ignoring strategy, which subsumes the no-activation floor for this
  question). It satisfies the Stage-C *feasibility* question but is **not**
  the Amendment 1 §5 2×2, and Stage C should not be marked passed.
- Speculation (flagged as such): near-zero train loss with 0.997 disjoint-family
  dev accuracy suggests the LoRA capacity is spent on the value dictionary +
  format, not memorization; nothing here tests whether the same recipe scales
  to non-templated captions.

## Limitations

- Dev only; seen values; raw/preamble cell; one model/layer/site; one seed; one
  format/init cell of the registered 2×2; greedy decode. Train eligibility
  computed under torch 2.12 forwards (dev agreement check bounds the drift).
- The S-stratum trains on only 54% of its rows (screen eligibility), so the S
  result is scoped to model-answerable color bindings.
- The 2-error pair (Hans → Helen) was not diagnosed beyond inspection.

## Recommended next (not run; user decision)

Held-out values (`test_value`) are now the single most informative cheap eval:
the extraction + eval infrastructure is in place, the adapter is saved, and the
crux ("general code vs per-value dictionary" — cruxes §C3/probe item 4) is
exactly what it discriminates. But it is the last unspent split — one shot —
so the eval protocol should be registered and user-approved first.
