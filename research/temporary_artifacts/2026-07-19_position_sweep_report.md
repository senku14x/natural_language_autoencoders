# Position sweep + scaled probe + target/distractor discriminator

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` (uncommitted) · **Phase:** exploration
Follow-on to the zero-shot AV read and old-value probe. Probe-only; no new prompts generated; **`test_value` (held-out values) untouched**. Predictions registered in the script header before running.

## Setup

Edit-position L20 states extracted for **all 1,656 eligible dev change + distractor rows** across every cell (raw/nopre edit-pos 3–26 · raw/pre 74–98 · chat/pre 98–122). Linear probe = StandardScaler → PCA-64 → L2 logistic regression, GroupKFold **by family**, direction-only features. Data `data/artifacts/v1/position_sweep/`; script `research/exploratory/probe_position_sweep.py`.

## Observations

**(A) Scaled probe, tighter CIs — both endpoints, ~equal, robust.**

| | value_old | value_new | chance |
|---|---|---|---|
| colors S (n=637) | 0.945 [0.908,0.974] | 0.945 [0.912,0.974] | 0.028 |
| names N (n=378) | 0.870 [0.834,0.948] | 0.870 [0.834,0.952] | 0.009 |

Names rose from 0.76 (raw/pre-only, n=320) to 0.87 (all cells) with the extra data; CIs tightened. old ≈ new throughout — both value-token identities are linearly present in the delta.

**(B) The value basis is position-invariant (your hypothesis — confirmed).** Cross-position transfer for colors (train one edit-position bucket, test another), new value:

- delta: early→late 0.99, late→early 0.99, all off-diagonals 0.92–1.00 (worst 0.85 later→later).
- h_cf state: same picture, 0.91–1.00.
- Clean within-raw/pre (fixed format): hi→lo 0.985 (lo→hi 0.806 on n=68).

Transfer is near-ceiling from position 3 to 122, for both the delta and the raw state. My prediction that early positions (3–26) would degrade from underinformativeness was **wrong** — a single value token is cleanly linearly decodable even at position 3 (the "~10-token poor-decode" caveat is about the AV's document-activation generation quality, not linear decodability of one value token). Plot `plots/2026-07-19_position_transfer.png`.

**(C) The edit-site delta is behaviorally BLIND — the decisive result.** Target-vs-distractor discriminability at the edit site (can the delta tell a behaviorally-relevant edit from an irrelevant one?), ROC AUC, GroupKFold by family:

| query order | AUC | n (target/distractor) |
|---|---|---|
| query_last | 0.508 ± 0.066 | 458 / 295 |
| query_first | 0.473 ± 0.023 | 558 / 345 |
| both | 0.495 ± 0.040 | 1016 / 640 |

**All at chance, in both query orders.** Meanwhile the edited value is fully decodable from the *distractor* deltas (colors 0.946, names 0.930) — so *which* value changed is present at ~95%, but *whether that change matters to the answer* is not linearly present at all. Plot `plots/2026-07-19_target_distractor_auc.png`.

## Interpretation (kept separate)

- **The basis-invariance result confirms the boring reading, not the exciting one.** A value's L20 representation is a stable, position-general linear direction — i.e. token identity. That it transfers across positions is exactly what "the delta is (new-token − old-token)" predicts. Robust, but token identity.
- **(C) is the sobering finding: the edit-site delta encodes *what changed*, not *whether it matters*.** A verbalizer reading this delta would name the changed value identically for a consequential edit and an inconsequential one — it cannot ground a behavioral consequence or a NO_CHANGE claim. This is the "names the changed token, not the consequence" boring alternative (v1 §5.4), now directly demonstrated, and it holds **even under query-first** (the relevance computation is not at the edit token's L20 state, even when the query precedes it).
- **Clean dissociation across Stage A + this experiment:**
  - *Edit position* carries the value-token identity robustly (95%, position-invariant) and is causally sufficient for the answer (Stage A 0.98) — but only because for TARGET rows the edited token *is* the answer; it is blind to behavioral relevance.
  - *Final position* carries the behavioral consequence (the answer margin) but the L20 state there is ~1.4% causally sufficient — the consequence is computed downstream (layers 21–27) from the query attending back to the value.
  - Neither single site gives "read the behavioral difference." Edit site = *what changed*; final site = *that the output differs*; they are dissociated, exactly as the causal-attention argument (v1 §4) predicts.
- **What this means for "can I make it verbalise the diff?":** yes for the *transition slot* — `old → new` (both token identities) is robustly, position-invariantly, held-out-family readable, so a warm-start AV would very likely learn to emit it. **No** for a behavioral-difference / consequence / NO_CHANGE claim from the edit-site delta — that information is not there. The v1 caption is transition-only, so this is on-plan; but it caps the claim firmly at "reads an answer-transition token code," and the "language earns its keep" question still lives entirely in held-out **values** (generalization beyond learned per-value directions) and derived-relation families.

## Caveats

- Position buckets co-vary with format/preamble (raw/nopre = early, raw/pre = late, chat/pre = later); the within-raw/pre split (fixed format) is the clean-but-narrow check and agrees.
- (C) is a clean test: target and distractor rows share the value inventory and, within a cell, format/position — so any above-chance AUC would have to be relevance (or a value confound that would only *inflate* AUC); AUC at 0.5 means no relevance signal, a fortiori.
- All on dev with shared value vocabulary. Held-out values not tested (deliberately).

## Bottom line

The transition (both value tokens) is decodable robustly and position-invariantly — genuinely strong for v1's `old→new` caption. But the edit-site delta is behaviorally blind (target vs distractor at chance), so this is a robust *token-transition reader*, not evidence of reading a behavioral difference. The exciting version of the claim needs held-out values and/or a site/representation that carries the consequence — not the edit-site delta.
