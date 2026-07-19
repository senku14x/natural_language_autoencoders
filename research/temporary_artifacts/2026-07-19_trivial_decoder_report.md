# Trivial-decoder (unembedding / logit-lens) baseline — report (work item 1b)

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` (uncommitted, pending user review)
**Phase:** exploration → baseline for interpretation of the zero-shot AV result.
**Predictions and the decision rule registered first:** `2026-07-19_negdelta_logitlens_predictions.md`.

## Question

The zero-shot report called the AV's edit-site read "logit-lens-like" four times
and never measured it. Does a linear readout of the unembedding — `W_U·Δ` and
`W_U·RMSNorm(Δ)` — recover the new value from the same deltas at anything near
the AV's 0.74?

## Setup

Same 50 pairs, same saved `h_edit_{base,cf}.npy`, same seed. Readout = top-1/top-5
of the 152k-vocab logits; hit = any top-k token id is an audited single-token
surface form of the value (`values_audit.json`; the token-level translation of the
string rule — a looser "decoded string contains the value" check gives identical
conclusions). Shuffled floor uses the **exact donor pairs** from the prior run
(`scored.parquet.shuffled_from`). Arms: +Δ→new (primary), −Δ→old, cross terms,
shuffled donor, and **h_cf→new / h_base→old state positive controls** (instrument
sensitivity). Both conventions: raw `W_U·Δ` and `W_U·RMSNorm_final(Δ)` (γ from
`model.norm.weight`; the scalar rescale cannot change top-k order, so the
conventions differ only by the elementwise γ weighting). No forwards — only
`lm_head.weight` + `model.norm.weight` loaded (untied, verified). Plumbing check:
a W_U row ranks itself top-1. Script `research/exploratory/trivial_decoder_baseline.py`.

## Predictions vs outcome — miss (in the informative direction)

Predicted top-5(new|+Δ) ~0.45 [0.10, 0.80], state control ~0.5; assigned only ~20%
to "AV ≫ trivial readout." Observed: **the trivial decoder is at the floor
everywhere, including the state controls.**

## Observations (dev, this setup; 50 pairs, bootstrap 95% CI)

| readout | target | top-1 | top-5 | median rank (of 152,064) |
|---|---|---|---|---|
| `W_U·RMSNorm(Δ)` | new | 0.02 [0, .06] | **0.04** [0, .10] | 11,900 |
| `W_U·Δ` (raw) | new | 0.00 | 0.00 | 12,822 |
| `W_U·RMSNorm(−Δ)` | old | 0.00 | 0.00 | 13,237 |
| `W_U·RMSNorm(h_cf)` state (pos. ctrl) | new | 0.00 | 0.00 | 15,022 |
| `W_U·RMSNorm(h_base)` state (pos. ctrl) | old | 0.00 | 0.00 | 24,059 |
| shuffled-donor Δ | own new | 0.00 | 0.00 | 68,386 |
| **released AV (string match), reference** | new | — | **0.740** [0.64, 0.84] | — |
| **released AV, −Δ (1a), reference** | old | — | **0.696** [0.58, 0.80] | — |
| linear probe (prior), reference | new / old | 0.96 S / 0.76 N | 1.00 | — |

- Top-5 decodes are mid-layer noise — rare/multilingual tokens and code fragments
  ("ysqli", "SharedPtr", "℠", "都是非常"), never value-adjacent words
  (`trivial_decoder_top5.json`).
- The value's rank is far above chance (median ~12k of 152k ≈ top 8% for Δ, vs
  ~68k for the shuffled donor) — a weak alignment exists — but four orders of
  magnitude away from *naming*.
- Both conventions agree; γ-weighting moves the state ranks (88k→15k for h_cf)
  but no arm approaches top-5.
- Plot: `plots/2026-07-19_trivial_decoder_vs_av.png` (AV, lens, floors, probe
  ceiling on the same axes, both directions).

## Decision-rule outcome (registered before looking)

**The second branch fires: AV ≫ logit lens** (0.740 vs 0.04 best-convention
top-5; CIs nowhere near overlap). Under the registered rule: the AV is decoding
something the trivial unembedding readout is not — the first result in this arc
that a trivial baseline does not reproduce.

## Interpretation (kept separate)

- **The "logit-lens-like" gloss in the zero-shot report was wrong and should be
  retired.** The value identity at the edit-site L20 is not in the unembedding
  basis (states fail too, so this is a property of the site/layer representation,
  not of differencing). It is also not in the input-embedding basis (EXP-2's
  retrieval null, prior session). It is a "middle" code: linearly decodable by a
  trained probe (0.96/0.87) and decodable by the trained AV (0.74), invisible to
  both weight-derived readouts.
- **What this does and does not establish.** It establishes the AV's zero-shot
  read is not reproducible by a trivial linear readout of the model's own
  weights — the AV's 21 layers + NLA training perform a basis translation the
  unembedding does not provide. It does **not** establish the AV beats a *trained
  probe* (it doesn't: 0.74 < 0.87–0.96, consistent with the literature-wide
  "readers match, never exceed, probes" ceiling), and it does not upgrade the
  content of what is read — still token identity (Amendment 1 §10 cap).
- **Instrument note.** The state-control failure was anticipated as a possible
  outcome: W_U at 71% depth at a copied input token's own position reads
  next-token dispositions, not current-token identity. This is why the
  registered rule compared the AV to the readout on the *same vectors* rather
  than assuming the lens is a sensitive instrument here.
- **Consequence for the SFT writeup:** the SFT is not merely interface
  groundwork over a trivially-readable code; but the correct claim remains "the
  AV decodes a non-trivially-readable token-identity code," not "the AV reads
  the transition." Follow-up that would earn more (not this session): where
  between the injection layer and the output the AV rotates the injected
  direction into nameable form, and whether the probe direction and the AV's
  effective readout direction coincide.

## Limitations

- 50 pairs, dev, raw/preamble, this model/layer/site. Top-5 budget vs the AV's
  ~220-token generation is conservative in the AV's favor by design (registered);
  the median-rank column shows the conclusion is not a top-k artifact — the AV
  reference would need the value inside ~top-5 of 152k to be matched, vs observed
  ~12k.
- The probe reference comes from the prior session's pipeline (PCA→logreg,
  GroupKFold), not re-fit on exactly these 50 pairs.
