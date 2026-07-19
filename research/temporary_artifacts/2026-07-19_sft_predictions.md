# Registered predictions — AV SFT on edit-site deltas (work item 2)

**Logged BEFORE the train-split screen/extraction and before any training.**
Timestamp: 2026-07-19, after work items 1a/1b were written up.

## Question

Can a LoRA fine-tune of the released kitft L20 AV emit the full transition
`old -> new` (caption `caption_arrow_transition`, strict `parse_arrow_transition`
grading) from an edit-site L20 delta?

## Design frozen before running

- **Data:** eligible **train** rows, raw/preamble, change cells only
  (TARGET_EDIT + REVERSE; no NO_CHANGE / distractor rows — the edit-site delta is
  behaviorally blind [AUC ≈ 0.5], so a consequence field would train
  confabulation). Eligibility = the Stage A screen rule (argmax correct on both
  prompts in any audited surface form + margin toward cf), computed on this
  instance for train. Δ at `edit_pos`, canonical-shape forwards (B=64, L=131,
  left-pad, explicit position_ids).
- **Eval:** dev eligible raw/preamble change rows (594 per the frozen Stage A
  screen; dev activations re-extracted on this instance in the same pass for
  train/eval consistency — frozen dev eligibility labels used for selection,
  with this-instance screen agreement reported as a robustness datum).
  **test_value untouched.**
- **Target:** the exact caption string + EOS after the unmodified sidecar AV
  prompt (injection contract unchanged: ㈎ replacement, normalize to L2=150).
  Greedy decode at eval, max_new_tokens 16, strict fullmatch parse; unparseable
  counts as wrong on all fields (parse-failure rate reported separately).
- **Arms (three, all evaluated on the same dev rows):**
  1. real-SFT model × real dev Δ (the result);
  2. shuffled-SFT model × real dev Δ (**non-negotiable control** — identical
     training except Δ permuted across train examples, fixed seed; transition
     frequencies are skewed, so a Δ-ignoring model scores well above zero);
  3. real-SFT model × permuted dev Δ (activation-dependence at eval).
- **Hyperparameters (frozen now, no post-hoc tuning):** LoRA r=64, α=128,
  dropout 0.05, targets q/k/v/o/gate/up/down projections; AdamW lr 2e-5, cosine
  decay, 5% warmup, 3 epochs, batch 32, bf16, seed 20260719; identical for both
  runs (same data order; only the Δ assignment differs). Per-epoch dev eval of
  arm 1 (learning-curve insurance; the reported number is end-of-training).
- **Metrics:** exact old-field, exact new-field, exact ordered pair; per stratum
  (S/N); family-bootstrap 95% CIs; the analytic Δ-ignoring floor drawn on the
  plot (dev frequency of the train-mode transition / mode field values); the 1b
  trivial-decoder number (0.04) on the same axes.

## Predictions

1. **Train screen eligibility:** ~74% of the 6,000 train raw/pre change rows
   (dev was 74.3%) → ~4,400 eligible. Dev screen agreement with the frozen
   labels ≥ 98% (torch 2.12 vs 2.7 forwards; margins are decisive, median
   32–53 logits).
2. **Real-SFT × real Δ (dev):** new-field ~0.90 (S ~0.94 / N ~0.82); old-field
   ~0.87 (both endpoints are in the delta and sign-accessible per 1a);
   **ordered-pair exact ~0.82** [0.65, 0.92] (S ~0.90 / N ~0.70). Basis: the
   zero-shot free-prose read is 0.74 (0.89/0.58); supervised extraction with an
   exact format should exceed it; the probe ceiling is 0.945/0.870 per endpoint.
3. **Shuffled-SFT × real Δ:** pair-exact ≤ 0.05; single fields ≤ 0.15 (greedy
   likely collapses to one or a few high-frequency captions). This IS the
   Δ-ignoring floor made empirical.
4. **Real-SFT × permuted Δ:** emits the donor pair's transition (mechanism
   already demonstrated in the zero-shot shuffled arm), so own-label accuracy
   ≈ coincidence rate ≈ 0.00–0.03.
5. Parse-failure rate after SFT < 2%.
6. **Registered gate analog (Amendment 1 §6 spirit):** real − shuffled ≥ 40pp
   on pair-exact for the run to count as "the SFT reads the delta."

## Registered interpretation rules

- Real ≫ shuffled with per-stratum gaps ≈ predicted → "a LoRA-adapted released
  AV emits the full ordered transition from the edit-site delta on held-in-value
  dev data." Capped, by construction, at a **transition token code** on a
  templated family (Amendment 1 §10; cruxes §C3): the reader is a lookup table
  over (old, new) here; this earns **no** language, faithfulness, or consequence
  claim, and says nothing about held-out values.
- Real ≈ shuffled → the SFT failed to use Δ. Diagnose before concluding
  anything: the information is present (probe 0.95/0.87) and AV-accessible
  zero-shot (0.74), so the failure would be optimization/interface (learning
  rate, injection in the training path, masking bug), not information absence.
- Old-field ≫ or ≪ new-field asymmetries are informative either way: 1a showed
  the old endpoint is only sign-inaccessible, so a large residual old-field
  deficit after SFT would suggest the positive-direction read is privileged in
  the AV's decoding pathway, not just in its output convention.
