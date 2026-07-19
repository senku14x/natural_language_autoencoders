# Re-check of the behavioral-blindness null under a nonlinear reader — resolution

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2`
**Trigger:** user-supplied re-analysis claiming the target-vs-distractor null is
a property of the linear probe: GBM AUC ~0.63 under query_first with query_last
at chance, presented as a causal dissociation a confound could not produce.
**Phase:** validation (adversarial check of a load-bearing null).

## Question

Is the "edit-site delta is behaviorally blind" null (target-vs-distractor
AUC ≈ 0.5, the basis for "consequence/NO_CHANGE not groundable" and the
derived-relation pivot) an artifact of the linear probe used to measure it —
i.e., does a nonlinear reader find a real relevance signal the linear null
missed? Why it matters: the downstream reader (the AV) is a full LM, so a
linear-only null under-instruments the claim; if a real query_first consequence
signal existed at the edit site, the pivot premise would change.

## Bottom line

The user's critique was **half right, and the half that was right improves the
project**: the original null was linear-only and its positive control did not
certify sensitivity in the relevant regime — correct, and now fixed. But the
**positive finding is an artifact**: the query_first GBM signal is `edit_pos`
leakage, and it vanishes under exact position matching. **The blindness null
stands — now properly earned against a nonlinear reader with a position
control, in both query orders.** The load-bearing inference (consequence /
NO_CHANGE not groundable from this edit-site delta; pivot to derived families)
is unchanged, on strengthened footing.

## What was run (all on cached `position_sweep` artifacts, no GPU)

1. **Reproduction of the user's script verbatim**
   (`exploratory/recheck_blindness_nonlinear.py`): confirmed, slightly
   stronger than reported — raw/pre value-matched, StandardScaler → PCA-128
   (inside fold) → HistGBM, GroupKFold by family, 5 seeds:
   query_first **0.658** [0.641–0.692] vs query_last **0.505** [0.497–0.517];
   linear 0.42/0.48; family-permuted null 0.52/0.51.
2. **Surface-metadata diagnostics** on the same subsets:
   - `edit_pos` ALONE classifies target-vs-distractor at **0.868** under
     query_first (0.537 under query_last). A metadata-only reader
     (edit_pos + stratum + value identity, no activations) reaches **0.867 /
     0.535 — exceeding the GBM-on-delta and reproducing its "causal
     dissociation" exactly.**
   - Mechanism: under query_first the query text precedes the edit token, so
     which entity is queried (target vs crossed/other-slot distractor) shifts
     the edit token's **absolute position** via token-length differences
     (target edit_pos mean 89.0 vs distractor 87.6; disjoint enough to
     classify). Under query_last the query is downstream and positions are
     class-uncorrelated (0.537). Absolute-position information is present in
     L20 states, so the delta carries the template's fingerprint.
3. **The decisive control — exact position matching** (per-edit_pos class
   subsampling to 50/50, 5 seeds, same GBM pipeline):
   - query_first **0.511** [0.453–0.576] (n≈164)
   - query_last **0.489** [0.469–0.521] (n≈296)
   The nonlinear signal is entirely accounted for by position.

## Why the "a confound wouldn't respect causal masking" argument failed

Any **query-correlated surface feature** is also only available under
query_first — the confound (query text upstream) exists exactly when the causal
path exists. Causal masking dissociates *query-upstream information* from
*query-downstream information*; it cannot dissociate "the state encodes
relevance" from "the state encodes its own absolute position, which the
template makes class-correlated." Only feature-level controls (position
matching) can.

Note also: the user's metadata check tested value identity (0.492 — correct,
no leak there) but not position, which was the actual leaky channel; and the
family-permuted null assigns one label per family while 92/97 families contain
both classes, so it is structurally weaker than a row-level artifact test.

## What survives of the critique (adopted)

1. **Linear-only nulls are inadequate instrumentation for "not groundable"
   claims when the downstream reader is an LM.** The corrected null — GBM,
   PCA inside fold, position-matched, both query orders, chance across seeds —
   is what the blindness claim now rests on.
2. **Variance-ranked PCA does not certify sensitivity for low-variance
   signals.** The 0.95 value-decode positive control only certified the
   high-variance regime. Habit adopted: for any null intended to carry weight,
   run a planted-direction sensitivity check at the relevant scale and avoid
   unsupervised variance bottlenecks, or verify them against the planted floor.
3. The pooled numbers in the original report mixed cells unevenly; within-cell
   reporting is the right default (the corrected numbers here are within
   raw/pre).
4. The user's own transductive-PCA warning (0.88–1.00 when fit before CV) is
   the correct trap to document.

## Caveats on the correction itself

- Position matching cuts query_first to n≈164; the check reliably detects
  AUC ≳ 0.65 but cannot exclude a very weak (<0.6) residual signal. The claim
  is "no detected signal with the artifact removed," not proof of absence.
- Even if a position-robust signal existed, edit_pos-driven separability would
  not constitute a consequence *representation* — it is template geometry, and
  it would not survive any template change.
- Scope: dev split, raw/pre, L20 edit site, this model. EXP-1's layer-sweep
  null remains linear-only; re-running it nonlinearly with position matching is
  the cheap outstanding item (the same artifact risk applies there — final
  position is shared, but layer-sweep deltas at the final position have no
  class-correlated position shift, so the exposure is smaller).

## Status of load-bearing claims

- `research/CLAUDE.md` / `NEW_WARMSTART_CRUXES.md` "behaviorally blind" →
  **stands**, with the basis upgraded from "linear probe at chance" to
  "linear AND nonlinear readers at chance once the query_first position
  artifact is controlled; metadata baseline explains the naive nonlinear
  signal."
- The derived-relation pivot premise is unchanged.
- New standing rule for any future discriminator on this dataset: **position-
  match (or covariate-control edit_pos) under query_first; report the
  metadata-only baseline alongside.**
