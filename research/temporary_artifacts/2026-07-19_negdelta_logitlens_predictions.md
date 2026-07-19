# Registered predictions — (1a) −Δ arm through the zero-shot AV, (1b) trivial-decoder (logit-lens) baseline

**Logged BEFORE running either experiment.** Timestamp: 2026-07-19, after environment
setup, before any AV inference or unembedding readout in this session.
Both experiments reuse `research/data/artifacts/v1/zeroshot_av/h_edit_{base,cf}.npy`
(594 eligible dev raw/preamble change rows), the same 50-pair sample
(`sample_manifest.json`, seed 20260719), the same whole-word case-insensitive
string rule, and (for the shuffled floor) the exact donor pairs recorded in
`scored.parquet.shuffled_from`.

## Pre-registration inputs (facts checked before predicting; no new outcome data)

1. **A REVERSE pair is the exact prompt swap of its TARGET partner.** Verified on
   200/200 random raw/preamble REVERSE rows: `rev.base_input_ids == tgt.cf_input_ids`
   and `rev.cf_input_ids == tgt.base_input_ids` (link resolved via `reverse_of` →
   semantic_id + matching variant coordinates). Under the canonical-shape
   deterministic forward this entails **−Δ(row) is bitwise equal to the real Δ of its
   partner row** (same prompts → same states → sign flip), with the same edit position.
2. The prior run's real arm *already included* REVERSE-cell rows and read them at
   new-value 0.766 (n=145 generations) vs TARGET 0.705 (n=105) — i.e. the AV reads
   "reverse-direction" transition vectors at the same rate as forward ones.
3. Prior per-stratum real rates: colors S 0.892, names N 0.575. Prior floors:
   shuffled 0.000, random 0.000. Prior old-value mention on real Δ: 0.000.
4. Old/new are ~equally linearly decodable from the same deltas (probe 0.96/0.96
   colors, 0.76/0.76 names on this pool).

## 1a — −Δ through the AV (5 samples × 50 pairs, T=1)

Because of input (1), the −Δ arm is very close to a *logical consequence* of the
prior real arm: each −Δ vector IS a real transition delta (the partner row's), whose
"new value" is this pair's old value. The honestly-new information is (a) verifying
the vector-level antisymmetry end-to-end through the AV, and (b) the small
possibility that the AV's read depends on something beyond the injected vector
(it cannot — the vector is identical — so any deviation beyond sampling noise
would indicate a bug, not psychology).

**P1 (−Δ names old):** old-value mention on −Δ ≈ **0.74** (anything in 0.60–0.85
unsurprising; per stratum ≈ 0.89 S / 0.58 N, mirroring the real arm).
**P2 (−Δ suppresses new):** new-value mention on −Δ ≈ **0.00–0.04**.
**P3 (probability of the "names neither" outcome):** ≤ 5–10%, and if observed with
the same vectors I will first suspect an implementation difference (injection,
tokenization, scaling) before a science interpretation — the identical-vector
argument leaves no room for the AV to treat −Δ differently from a partner real Δ.

**Registered interpretation rule (from the session brief, unchanged):**
- −Δ names old ≈ 0.74 / new ≈ 0 → both endpoints are AV-accessible; the 74/0
  asymmetry of the real arm is a **sign convention** (the AV names whatever token
  identity sits in the positive direction). The SFT is then teaching a
  bidirectional read of a code the model already decodes one way at a time.
- −Δ names neither → the probe-vs-AV gap is not about sign; report it as the more
  interesting outcome (it changes what the SFT is being asked to learn) — but per
  P3, first rule out implementation drift vs the prior run.

Note the scope cap that binds regardless of outcome: this remains **token-identity
readout** of a transition code (Amendment 1 §10, NEW_WARMSTART_CRUXES §C3);
a positive is not a language or consequence result.

## 1b — trivial-decoder baseline: W_U · Δ and W_U · RMSNorm(Δ)

The zero-shot report asserted "logit-lens-like" without measuring it; this measures
it. Score = does the value's audited single-token surface form (any of the 4 forms
from `values_audit.json`) appear in the top-1 / top-5 of the readout — the exact
token-level translation of the string rule. Same 50 pairs. Arms: +Δ→new (primary),
−Δ→old, +Δ→old and −Δ→new (should be ≈0), shuffled-donor Δ→own new (floor, exact
donors from `scored.parquet`), and h_cf→new / h_base→old (**instrument positive
control** — if the unembedding cannot read a value from a raw *state* at this site,
a Δ null would say nothing about "difference-ness"). Both norm conventions reported:
raw `W_U·Δ` and `W_U·RMSNorm_final(Δ)` (Qwen's final norm is RMSNorm; the scalar
rescale cannot change top-k order, so the conventions differ exactly by the
elementwise γ weighting).

**Predictions (wide — genuinely uncertain):**
- top-1(new | +Δ): ~**0.2** [0.02, 0.6]; top-5: ~**0.45** [0.10, 0.80].
- RMSNorm convention within ±0.10 of raw; no strong prior on which is higher.
- −Δ→old ≈ +Δ→new (the readout is linear, so this symmetry is near-mechanical;
  deviations only from which tokens sit near the negated direction).
- Cross terms (+Δ→old, −Δ→new) ≤ 0.05. Shuffled floor ≤ 0.05.
- State controls (h_cf→new): ~0.5 [0.15, 0.95] — mid-depth states at a token's own
  position are only partially rotated into the output basis; W_U may read the
  *next*-token prediction (e.g. punctuation after the value word) rather than the
  current token.
- P(top-5 best-convention ≥ 0.74, i.e. logit lens matches/beats the AV): **~40%**.
- P(top-5 best-convention ≤ 0.10, i.e. AV ≫ trivial readout): **~20%**.

**Registered decision rule (from the session brief, fixed before looking):**
- **logit lens ≥ 0.74** (best convention, top-5, new-value rate) → the AV adds
  nothing over a linear readout of the unembedding on this task; say so plainly.
  The SFT then becomes interface groundwork, and the writeup must not claim the
  AV "reads" the transition.
- **AV ≫ logit lens** (operationalized: AV 0.74 exceeds the best logit-lens top-5
  by ≥ 20pp with non-overlapping pair-bootstrap CIs) → the AV is decoding something
  the trivial readout is not; that is the first non-trivial result in the arc and
  deserves its own follow-up.
- In between → partial; report both numbers with CIs and do not collapse the
  distinction.

Top-1 is also reported but top-5 is the primary comparison — deliberately generous
to the trivial baseline (do not sandbag the boring alternative; the AV gets ~220
tokens of output to land one string match, so a 5-token budget for the lens is
conservative in the AV's favor if anything).

## Environment note (fingerprint for both runs)

Fresh H100 80GB instance (driver 580.126.20), torch **2.12.0+cu130** (prior
sessions: 2.7.0+cu128 — cross-instance bitwise reproducibility of forwards is not
expected; both 1a and 1b consume the *saved* activations, so this affects only the
AV generation pass), transformers 5.14.1 (pinned, matches generation env),
Python 3.12.13, `/venv/main`. `/workspace` is NOT volume-backed on this instance
(`workspace_is_volume: false`) — nothing here survives recycle/destroy; HF_HOME =
`/workspace/.hf_home`. Base model @ `a09a3545…`, AV snapshot `b884691…` (same as
prior session). pytest: 65 passed, 1 skipped. Pinned tokenizer re-fetched at the
dataset SHA.
