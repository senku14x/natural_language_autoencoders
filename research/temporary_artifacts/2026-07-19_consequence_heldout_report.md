# EXP-1 final-position layer sweep + EXP-2 held-out-value generalization

**Date:** 2026-07-19 · **Branch:** `counterfactual_nla_v2` · **Phase:** exploration
Probe-only. Predictions registered in each script header before running. `test_value` used with explicit user approval. Data `data/artifacts/v1/consequence_heldout/` + `position_sweep/`; scripts `research/exploratory/consequence_and_heldout.py`, `consequence_heldout_diag.py`, `plots_consequence_heldout.py`.

Both headline numbers were surprising, so each was checked with a diagnostic before any conclusion — and both raw numbers turned out to be misleading on their own.

---

## EXP-1 — Where does the behavioral consequence become readable?

**Hypothesis tested:** the edit-site L20 delta was behaviorally blind (target-vs-distractor AUC ≈ 0.5). Is that blindness specific to the edit site, or does the *final* position — which has attended to the query — encode relevance, and at what depth?

**What was run:** final-position hidden-state delta (h_cf − h_base) at every layer (0–28) for all 1,656 eligible dev change + distractor rows. Metric: target-vs-distractor discriminability (ROC AUC, GroupKFold by family), by query order. Diagnostic D1: the same, but on delta **norm** (magnitude), since the main sweep unit-normalized it away.

**Results:**
- **Direction (unit-normalized delta): AUC ≈ 0.5 at every layer, both query orders.** There is no readable "this edit is relevant" *direction* anywhere in the network.
- **Magnitude (delta norm): rises sharply and late** — L20 0.667, L21 0.67, L22 0.71, L23 0.88, **L24 0.971 → L28 0.985** (best L27 0.991); median target norm 154 vs distractor 47 at L28.

**What it answers:** the consequence *is* present at the final position, but only as **answer-change magnitude, forming in late layers (L24–28)** — i.e. answer formation. There is no abstract relevance representation at any layer; a distractor edit and a target edit are indistinguishable by *direction* everywhere, and separable by *norm* only once the output token is being written. This ties the whole arc together: Stage A found the final-position L20 state ~1.4% causally sufficient precisely because the answer forms at L24–28, not L20. Plot `plots/2026-07-19_final_layer_sweep.png`.

**Caveat / near-tautology:** "target = answer changed = big final-state delta" is close to definitional (eligible targets moved the answer by construction). So EXP-1's honest content is the *localization* (L24–28, magnitude) and the *negative* (no relevance direction anywhere), not a claim that the model computes an interesting "relevance" variable.

---

## EXP-2 — Does the value reading generalize to unseen values? (lookup vs general basis)

**Hypothesis tested:** is the transition reading a per-training-value lookup (breaks on unseen values) or a value-general representation (survives them)?

**What was run:** ridge map delta → new-value **token embedding**, trained on dev colors, top-1 retrieval on `test_value` rows whose color was never in training (charcoal, cream, green, lavender, turquoise; n=136 target rows; 42 candidate colors). Diagnostic D2: leave-one-family-out **1-NN among held-out rows** (predict the unseen color from *other* held-out deltas) — tests whether unseen-value deltas are internally consistent, independent of any decoder target.

**Results:**
- **Embedding-retrieval: held-out top-1 = 0.000** — but this was a **wrong-target artifact**: seen values scored only 0.388 against the same input-embedding target, so the L20 delta simply doesn't live in the input-embedding space.
- **D2 (the right test): unseen-color deltas are internally consistent** — held-out 1-NN **0.890** (5 colors, chance 0.20), *higher* than the seen-color reference (0.564, 37 colors, chance 0.027; higher partly because fewer held-out classes).

**What it answers:** unseen values **are** consistently and separably represented — a held-out color's delta reliably matches another instance of the same held-out color across different families/prompts. So the value representation is a **general, structured code, not a brittle per-training-value lookup**. What a *seen-trained closed-set decoder* cannot do is put a *name* on an unseen value — that is a decoder-vocabulary limitation, not a representation limitation. Plot `plots/2026-07-19_heldout_value_retrieval.png`.

---

## Combined read (this is still the token-identity story, now fully mapped)

- **Value / transition token identity:** robust, position-invariant, and general enough to place *unseen* values consistently (0.89 1-NN). A more diverse warm-start will very likely read transitions — including new values — reliably.
- **Behavioral consequence / relevance:** **not** encoded as a readable direction at any site or layer. It exists only as answer-change *magnitude* at the final position in late layers (L24–28) — which is "the output moved," and which the AV interface *destroys* by rescaling every injected vector to norm 150.
- Net: everything remains consistent with "the delta is the two value-token identities." Neither more prompts, more positions, nor unseen values changes that — they sharpen it. The consequence is a late-layer output-magnitude phenomenon, dissociated from the value-identity content the AV can read.

## Method lesson (recorded)

Both raw numbers were misleading: EXP-1 by normalizing away the operative (magnitude) signal, EXP-2 by scoring against the wrong target space. Neither would have been caught without the direction-vs-magnitude split and the decoder-free 1-NN control. Register the positive/negative controls *and the metric choice* before reading any single number (context doc §4.5).

Forward-looking implications for the next warm-start are in `research/docs/NEW_WARMSTART_CRUXES.md`.
