# Registered predictions — test_value reader-level eval (single spend)

**Logged BEFORE screening, extracting, or evaluating anything on test_value.**
Timestamp: 2026-07-19, after user sign-off on spending the split. **One eval,
no iteration afterward**; results reported whatever they are. This is the
split's second touch (first: EXP-2 probe-level 1-NN, user-approved).

## Question

Can the SFT'd reader (work item 2 adapter) *name* held-out values — i.e. is the
learned read a general value decoder, or a closed set over the ~100 train
caption values? EXP-2 established the representation side (unseen-value deltas
internally consistent, 1-NN 0.89); this is the reader side, with an
open-vocabulary decoder (the AV itself), which EXP-2's method lesson demands.

## Protocol (frozen)

- Rows: test_value raw/preamble change cells (480 rows, 60 families, all
  stratum S). Screen with the identical eligibility rule/machinery as the train
  pass; extract Δ at edit_pos under canonical-shape forwards.
- Endpoint classes scored separately: seen→seen (240 rows pre-screen; the
  in-split reference), seen→held (116), held→seen (116), held→held (8).
  Held-out colors {charcoal, cream, green, lavender}; none appear in eligible
  train captions (verified); all four have single-token surface forms.
- Arms, all in this single spend:
  1. **real-SFT adapter × real Δ** (greedy, strict fullmatch parse) — headline;
  2. **real-SFT adapter × permuted Δ** (within-split permutation) — floor;
  3. **zero-shot released AV × real Δ** (5 samples @ T=1, whole-word string
     match, same rule as the original zero-shot run) — reference that separates
     "SFT vocabulary collapse" from "representation failure": the released AV
     has no train/held distinction, so it should read held-out colors exactly
     as well as seen ones.
- Metrics: exact accuracy per endpoint side (seen-side vs held-side) and per
  endpoint class; substitution table for held-side misses (is a miss the
  nearest seen color?); family bootstrap; parse rate. Analytic floor: held-out
  values are unreachable by any Δ-ignoring strategy (never in train captions),
  so any exact held-side hit is signal; ~0 floor.

## Predictions

1. Eligibility ≈ 55% (dev S 57.1%, train S 54.4%) → ~260 eligible rows.
2. Seen→seen rows, SFT arm: ≈ 0.99 (dev S was 1.000; new families).
3. Zero-shot arm: held-side new-value string match ≈ seen-side ≈ 0.85–0.90
   (the released AV has no reason to distinguish them).
4. **The crux — SFT arm, held-side exact naming: genuinely uncertain, ~0.45
   with a wide interval [0.05, 0.90].** Competing forces: the delta code is
   general (1-NN 0.89) and LoRA does not delete vocabulary; but 3 epochs to
   ~0.001 train loss over ~100 output values is strong prior-narrowing, and the
   shuffled run showed this model class collapses hard when it can.
   - P(≥ 0.70, "reader generalizes") ≈ 35%
   - P(≤ 0.10 with systematic nearest-seen substitutions, "vocabulary
     collapse") ≈ 30%
   - P(intermediate) ≈ 35%
5. Permuted-Δ arm own-label ≈ 0; emits donor transitions.
6. Misses on the held side will be dominated by near-synonym seen colors
   (charcoal→gray/slate, cream→ivory/beige, lavender→violet/purple,
   green→lime/teal) rather than random values.

## Registered interpretation rules

- SFT held-side high (≳0.7) with zero-shot reference normal → reader-level
  generalization to unseen values: the general-value-code claim upgrades from
  representation to reader. **Still token identity** — no language/consequence
  upgrade (Amendment 1 §10 cap unchanged).
- SFT held-side ≈ 0 with nearest-seen substitutions while zero-shot reads
  held-out colors fine → **SFT-induced closed-vocabulary collapse**: the code
  is general, the fine-tuned reader is not. Concrete warm-start design lesson
  (value-diverse captions needed), not a representation failure.
- Both SFT and zero-shot fail on held-side while seen-side is normal → tension
  with EXP-2's 1-NN 0.89; diagnose (extraction/screen bug first) before any
  science interpretation.
- Intermediate → report the number and the substitution structure; no forced
  binary.
