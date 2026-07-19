# Stage A report — exact-model causal check (dev split)

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` (report committed by user request; code, plots, data artifacts, and the ARTIFACTS.md entry remain uncommitted pending user approval)
**Setup:** Qwen2.5-7B-Instruct @ `a09a3545…` (bf16, sdpa), torch 2.7.0+cu128, transformers 5.14.1 (generation-env match), H100 80GB. Layer 20 = block-20 output = HF `hidden_states[21]`. Seed 20260719. Full manifest: `research/data/artifacts/v1/stage_a/stage_a_env.json`.
**Scope:** dev split only (4,800 rows + 240 NULL_AA). No test split touched. No NLA checkpoints used.

## Headline

1. **The pre-registered primary site fails the causal-substrate gate decisively.** Replacing the *final-position* L20 state of the base prompt with the counterfactual's state (which the additive `h_base + Δ_final` patch does exactly) recovers a **median 1.4%** of the counterfactual answer margin — 0 of 1,016 eligible rows reach the 0.50 gate; p99 = 0.17.
2. **The substrate exists — one position to the left of where v1 pointed.** A single-vector patch at the **edit-token position** recovers **median 0.983** of the margin (100% of rows > 0.5, 100% direction-correct, all three live cells, both strata, both query orders), against clean controls. An all-positions patch recovers **exactly 1.000** (positive control; certifies the machinery end-to-end).
3. **The edit-site effect is selective, not disruptive.** Distractor-edit deltas at the same site with the *same norm* (~76) leave the queried answer unchanged (99.2% retention, JS-to-base median 0.0006).

## Level-0 plumbing (all pass — `stage_a/level0.json`)

- Frozen `input_ids` re-encode exactly (500-row raw spot check); `final_pos == len(input_ids)−1` for all dev rows; dataset regeneration was verified **bit-identical** to the frozen artifact earlier this session (config_hash `0d85e5a6…`, 66/66 tests).
- Hook on `layers[20]` output ≡ `hidden_states[21]`: max abs diff **0.0**.
- Zero patch is an exact identity: max abs logit diff **0.0**.
- **Measurement-policy finding:** forward results are a deterministic function of (input, batch shape). Identical shapes are bitwise reproducible; different batch shapes (padding length / batch size) differ by up to ~0.6 logit and ~2.65 h20-norm units — the same order as 15% of the median final-position delta (17.4). **Policy adopted: every Stage A forward runs at one canonical shape (B=64, L=131, left-padded, explicit position_ids).** Under it, repeats and batch recompositions are bitwise identical and all 240 NULL_AA deltas are exactly zero.
- Epsilon guard (provisional): ε = 3.97 = 1.5 × max observed cross-shape noise; within the canonical policy, true-null deltas are exactly 0. Revisit if Stage B ever mixes shapes.

## Behavioral screen (`stage_a/screen.parquet`, Fig `2026-07-19_stage_a_screen_eligibility.png`)

Eligibility = full-vocab argmax is a surface form of the correct answer on *both* prompts (+ margin moves toward cf for change cells). Margin convention frozen: raw → leading-space ids (dataset columns); chat → dominant audited form per stratum measured on dev (N → `bare_cap`, S → `bare`).

| cell | base correct | eligible (change rows) |
|---|---|---|
| raw / preamble | 83.3% | **74.3%** (594/800) |
| raw / no preamble | 43.9% | 32.8% (262/800) — stratum S only (2 N rows) |
| chat / preamble | 46.1% | 20.0% (160/800) |
| chat / no preamble | 0.4% | **0.0%** (0/800) |

chat/no-preamble is behaviorally dead — the chat model opens with "The/Your/Based", never the bare answer token. The PROVISIONAL preamble is what makes the task work; distractor eligibility 40%, NULL_AA plumbing rows 100% zero-delta. Eligible change rows: 1,016 across 100 families (~8–12 rows each). Margin gaps are decisive: median 32–53 logits.

## Patch-control matrix, final position (`stage_a/patch_metrics.parquet`, Fig `…_margin_recovery_matrix.png`)

Median normalized margin recovery (change cells pooled; per-cell numbers in the parquet):

| condition | median recovery | direction-correct |
|---|---|---|
| real Δ (own pair) | **0.014** | 75.7% |
| W_U[new]−W_U[old], norm-matched | 0.100 | 100% |
| E[Δ\|old→new] LOFO / same-transition other pair | 0.021 / 0.024 | 90% / 87% (n=78) |
| unrelated Δ / matched-norm random | 0.002 / 0.002 | 56% / 53% |
| zero patch | 0.000 (exact) | — |
| −Δ (reverse) | −0.005 | 34.5% |
| global mean Δ LOFO | 0.000 | — (≈zero vector → identity) |

JS-to-cf barely moves (0.982 → 0.981 of a near-maximal 1.0 gap). The real delta beats unspecific controls ~7× — a *specific but tiny* causal signal. Since `h_base + Δ_final = h_cf` bitwise at that position, this is a mechanistic statement: **the final-position L20 state carries ~1.4% of the answer; the rest is fetched by layers 21–27 attention from earlier positions.** Distractor deltas at the final position are inert (99.4% answer retention).

**Gate verdict (final position): FAIL** — median 0.014 ≪ 0.50; direction 75.7% < 80%.

## Site decomposition (`stage_a/site_decomposition.parquet`, Fig `…_site_decomposition.png`)

Pre-edit deltas are exactly zero (asserted — prompt identity). Median recovery by patch scope:

| scope | median | rows > 0.5 | direction |
|---|---|---|---|
| ALL positions (positive control) | **1.000** | 100% | 100% |
| edit token + final | 0.998 | 100% | 100% |
| **edit token only** | **0.983** | **100%** | **100%** |
| edit site, same-transition other pair | 0.955 | 100% (n=78) | 100% |
| edit site, unrelated Δ (diff. transition) | 0.169 | — | 98.8% |
| edit site, matched-norm random | 0.019 | — | 79.6% |
| edit site, −Δ (reverse) | −0.025 | — | 37.0% |
| post-edit positions (excl. edit) | 0.017 | 0% | 78.1% |
| final position only (v1 primary site) | 0.014 | 0% | 75.7% |

Per-cell, family-bootstrap 95% CIs for edit-only real: raw/pre 0.984 [0.982, 0.987] (94 families), raw/nopre 0.974 [0.968, 0.978] (60), chat/pre 0.987 [0.980, 0.992] (40); 100% of families ≥ 80% direction-correct in every cell. JS recovery 0.995; top-10 overlap 0.5 → 0.9. Strata: N 0.988 / S 0.976. Query orders identical (0.982 both). Edit-site delta norms 68–83 (token-identity difference).

**Distractor arm at the edit site** (`stage_a/distractor_edit_site.parquet`): same-norm distractor deltas (median 76) → 99.2% answer retention, JS-to-base 0.0006 (crossed_query 100.0%, other_slot 98.4%). The site is value-binding-selective, not norm-sensitive.

**Gate verdict (edit site): PASS with one flagged exception** — median 0.974–0.987 ≥ 0.50 ✓; direction 100% ≥ 80% ✓; random/reverse controls < 0.10 ✓; reverse pushes negative ✓ (small magnitude — margin_base is near its floor, so −Δ has little room; direction split 28–51% is the informative read). **Exception: the unrelated-real-delta control is 0.16–0.19 > 0.10.** An unrelated value implanted at the edit site partially breaks the old binding and lifts the (new−old) margin nonspecifically. The specific:nonspecific ratio is ~6:1 (0.98 vs 0.17). The §9 control bound was written for the final site; per §9's own rule, adjusting it requires a one-time written justification before test-split work — **user decision, not taken here.**

## Interpretation (kept separate from observations)

- **This is the gating system doing its job.** v1's premise — "Δ_final is causally sufficient for the answer change" — is false on this task/model/layer. Had we trained the AV on final-position deltas first, every downstream null would have been uninterpretable. Stage A cost one GPU-day and re-pointed the project at a certified substrate.
- **The certified substrate is the edit-site delta**, which the plan's own §4 (causal-attention constraint) anticipated as the site where a query-last delta cannot encode the query — the dataset's `query_order` axis and `NOT_IDENTIFIABLE` label exist precisely for this. Consequences for caption grounding: the transition slot (`old -> new`) is groundable from an edit-site delta in any query order; the *behavioral* claim ("answer unaffected" / NO_CHANGE-as-consequence) is groundable only in query-first cells at this site. This reshapes Stage B/C cell usage but requires **no dataset regeneration** (edit_pos is recorded; raw/pre edit positions 74–98 are above the released AV's position-50 floor).
- **The same-transition prototype (0.955) nearly ties the own-pair delta (0.983)** — at the edit site the delta is largely a function of (old value, new value), i.e., v1 §10's "value-transition mean ties the method" outcome is likely. This is priced in: acceptable for the boring pass, caps the claim at reading a transition code; the per-example-vs-prototype question belongs to Stage B's mean hierarchy on the larger train split (dev coverage is only 78 rows).
- **Boring alternatives not yet excluded:** at the edit site the patch implants what is largely the *token identity* of the new value; "the delta encodes the transition" and "the delta is the new token's representation minus the old's" are not yet distinguished (Stage B geometry + probes; the entity slot decision). Also the ~0.17 nonspecific disruption component must be subtracted in any later effect accounting.
- Scope: dev split, one model/revision, L20, this binding task, canonical-shape bf16 forwards. Nothing here says anything about the AV's ability to *read* these deltas — that is Stage B/C.

## Proposed next actions (require user sign-off)

1. **Freeze the substrate as: edit-token position, layer 20, raw/preamble primary cell** (594/800 eligible, both strata, positions above the AV floor), keeping raw/nopre + chat/pre as robustness cells. Formal site-freeze is a plan decision → user.
2. Decide the §9 unrelated-control bound (keep 0.10 and report the exception, or re-justify once in writing per §9).
3. Stage B no-training diagnostics on the *edit-site* deltas (parity B.1 per cell, mean hierarchy on train-scale coverage, probes, geometry, §11.1 queried-entity probe on crossed pairs — final-position deltas remain available for it).
4. The preamble sign-off is now empirically loaded: it is the difference between a 74%-eligible and a 33%-eligible primary cell.

## Artifacts

- `research/data/artifacts/v1/stage_a/`: `level0.json`, `screen.parquet`, `screen_meta.json`, `h_base.npy`, `h_cf.npy`, `logits_{base,cf}_fp16.npy`, `patch_metrics.parquet`, `site_decomposition.parquet`, `distractor_edit_site.parquet`, `stage_a_env.json`
- `research/stage_a/`: `stage_a_lib.py`, `run_level0.py`, `run_screen.py`, `run_patch_matrix.py`, `run_site_decomposition.py`, `run_site_decomposition_lib.py`, `run_distractor_edit_site.py`, `make_plots.py`
- Plots: `research/plots/2026-07-19_stage_a_{screen_eligibility, margin_recovery_matrix, js_recovery_matrix, site_decomposition}.png`
