# Amendment 1 — caption schema: reopened
**Date:** 2026-07-18
**Status:** supersedes §3 (caption schema), the verbalization gates in §9, and the Level-4 gate wording in `COUNTERFACTUAL_DIFFERENCE_NLA_V1_DECISION.md`
**Does not change:** Stage A, the Level 0–5 ladder, gating discipline, split construction, or the DPI bound on templated families
---
## 1. Why this is being reopened
The single-field schema (`PREDICTED_OUTPUT_CHANGE: A -> B`) was chosen deliberately to minimize researcher degrees of freedom, and that reasoning still holds. Three pieces of evidence that postdate the decision change the balance.
**1.1 The AV interface removes magnitude.** The released sidecar sets `injection_scale: 150.0` — every injected vector is rescaled to that L2 norm — and the AR loss is direction-only. Consequence: a small distractor delta and a large target delta arrive at the AV *identical in norm*. `NO_CHANGE` must therefore be readable from direction alone. This is good for the claim (it removes a magnitude shortcut) but it means the schema has to give the AV something direction-shaped to be right about.
**1.2 A single ordered pair makes the frozen reader a lookup table by construction.** The canonicalized caption contains one ordered transition and nothing else, so a caption-only reader is a function of the transition label alone; at MSE-optimum it *is* the ordered value-transition mean. This is analytic, not an empirical outcome. Level 4's end-to-end composition is therefore arithmetic given Level 3, and its genuinely new evidence reduces to two arms (gold-caption causal efficacy; error semantics under wrong/reverse/shuffled captions).
**1.3 `NO_CHANGE` is under-determined in the `other_slot` cell.** When the non-queried binding changes (`mint → gray` while `jeck` is queried), the caption `NO_CHANGE` discards the fact that something changed. An AV can produce it by reading only *"the answer did not move"* — which the output-direction readout and a raw-norm classifier both supply for free. The cell we identified as carrying the informative weight is the one where the schema is weakest.
Additionally, `A -> B` is a severe distribution shift from a checkpoint trained to emit prose. If the stated reason for warm-starting is that the checkpoint already knows how to describe activations, the caption format should be testable against that claim rather than assumed away.
---
## 2. Two axes, separated
The previous discussion merged these. They must be varied independently or a positive result is uninterpretable.
| Axis | Levels | What it tests |
|---|---|---|
| **Format** (surface) | arrow / sentence frame | Does staying near the checkpoint's native output distribution help? Same information either way. |
| **Content** (information) | transition-only / transition + entity | Changes what is being tested. Fixes 1.3. |
**Neither axis is free-form or LLM-varied.** Both renderers are deterministic string functions of stored metadata. Surface variation sampled independently of Δ contributes loss the AV cannot reduce — it is noise in the objective, and it adds a parse-failure error mode on top of reading failure that the metric cannot separate. If output-distribution collapse becomes a problem, the fix is **rehearsal** (mixing a small fraction of original-distribution NLA explanations into the SFT set), not paraphrase.
### 2.1 Renderers
```
ARROW_TRANSITION:     brown -> coral
ARROW_TRANSITION:     NO_CHANGE
SENTENCE_TRANSITION:  The value changes from brown to coral.
SENTENCE_TRANSITION:  The queried value is unchanged.
ARROW_ENTITY:         jeck: brown -> coral
ARROW_ENTITY:         florp: mint -> gray | queried: jeck unaffected
SENTENCE_ENTITY:      The color assigned to jeck changes from brown to coral.
SENTENCE_ENTITY:      The color assigned to florp changes from mint to gray;
                      the queried entity jeck is unaffected.
```
---
## 3. Slot inventory and groundedness
A slot may appear in the schema only if the delta can carry it. Status as of now:
| Slot | Grounded? | Basis |
|---|---|---|
| `old -> new` transition | Yes | Stage A certifies the delta moves the answer distribution toward the counterfactual |
| `entity` | **Undetermined — gated on Stage B** | See §4 |
| unaffected-entity clause | Probably not; open | Prior tuple-mean result suggests Δ is a function of (entity, old, new); the nonlinearity caveat leaves it empirically open. Treat as a measurement, not a caption field, until tested. |
| magnitude / "large change" | **No — structurally impossible** | `injection_scale` rescales every input to norm 150. Any magnitude language is unfounded by construction. |
**Hard rule:** no caption in any v1 variant may contain magnitude, intensity, or confidence language. The interface provably removes the information.
---
## 4. Content is gated on Stage B, not chosen now
Whether the entity slot is groundable is exactly the value-direction rung of the mean hierarchy (backlog §2.1). Run it first.
| Stage B outcome | Content decision |
|---|---|
| `E[Δ \| old, new]` ties per-example under JS + top-k | Entity is not in the delta. Ship transition-only. An entity slot would be a deliberate confabulation field. |
| Entity-conditioned mean needed; value-direction insufficient | Entity slot is grounded. Ship transition + entity, and the `other_slot` cell becomes a real test of *which* entity changed. |
| Ambiguous / underpowered | Ship transition-only; log the entity slot as a deferred extension with the specific number that would license it. |
Evaluate the hierarchy under **JS-to-counterfactual and top-k overlap**, not answer margin alone — margin is spoofable by pure token promotion, so a margin-based tie tells you nothing about whether the entity is represented.
---
## 5. Format is tested as a 2×2 with initialization
Four LoRA runs, identical in every respect except the two factors.
```
{released-init, vanilla-init} × {arrow, sentence}
```
Read-off:
| Result | Interpretation |
|---|---|
| Sentence helps **only** under released-init | Warm-start transfer is real and format-sensitive; quantifies what the checkpoint buys |
| Sentence helps under **both** inits | Optimization effect (longer targets, easier credit assignment), unrelated to the checkpoint |
| **No difference** either way | The warm-start contributes little — bears directly on released-checkpoint vs delta-native-from-scratch |
The vanilla-init control already existed in §9 as a standalone arm; this crosses it with format for one extra run and strictly more information.
**Held fixed across all four:** data, splits, seeds, LoRA rank/alpha/target modules, optimizer, token budget, and number of steps. Only the renderer and the initialization differ.
**Per-format baselines are mandatory.** Run no-activation and shuffled-Δ separately for *each* format. A fluent sentence frame may be easier to produce from prior alone than an arrow string, so the majority-transition floor is format-dependent; comparing formats against a shared baseline would manufacture a difference.
---
## 6. Gate amendments
Replaces the single-field verbalization gates in §9. Thresholds below are **proposed starting values, to be finalized on dev data before any test-split inspection.**
| Gate | Threshold | Note |
|---|---|---|
| Transition-slot exact accuracy (macro over families, held-out) | ≥ 0.90 | Same quantity as the previous single-field gate |
| Entity-slot exact accuracy | ≥ 0.90 | Applies only if §4 licenses the entity slot |
| Joint all-slot exact accuracy | ≥ 0.85 | Strictly harder than any single slot |
| **Distractor `NO_CHANGE` accuracy** | ≥ 0.90 | **First-class gate.** Distractor edits only — nonzero delta, no behavioral change on the queried field |
| `other_slot` correct changed-entity identification | ≥ 0.80 | Only under entity content; this is the cell that separates consequence-reading from answer-movement-reading |
| Reverse consistency | ≥ 0.90 | Unchanged |
| Real vs shuffled Δ | **≥ 40 points above the measured no-activation prior for that format** | Replaces the absolute ≥50pp figure — the prior depends on value-set size and on format |
| No-activation output | At the balanced-prior baseline | Per format |
`NULL_AA` is a plumbing check and is excluded from all accuracy gates. `plumbing_only: true` rows must be filtered before scoring.
Bootstrap unit is the **family**, for every gate, including the reader gates (this also resolves the families-vs-examples inconsistency between the substrate and reader gates in §9).
---
## 7. Level-4 rewording
Replace the end-to-end composition gate with the two arms that carry independent evidence:
1. **Gold-caption arm** — a canonical caption decodes through the frozen reader into a causally effective patch, measured against generic output-token steering.
2. **Error-semantics arm** — wrong, reverse, and shuffled captions mis-steer in the corresponding directions.
The composed AV→reader→patch number remains reportable but is arithmetic given Level 3 plus these two arms, and must not be presented as independent evidence.
**Canonicalization requirement.** Both renderers must map to the same internal representation before the reader sees anything, or the reader arm is confounded by format. Write the parser alongside each renderer and unit-test `parse(render(x)) == x` per format, plus `canonicalize(arrow(x)) == canonicalize(sentence(x))`.
**Backlog addition:** an `h_base`-conditioned reader ("apply the described change to this state") is the first reader variant whose output can exceed a prototype table. It requires its own no-caption control, since state conditioning creates a bypass around the language channel.
---
## 8. Dataset implications
**No regeneration required.** All four renderers are deterministic functions of metadata already in the frozen manifest. Emit all four caption columns now (`caption_arrow_transition`, `caption_sentence_transition`, `caption_arrow_entity`, `caption_sentence_entity`), each with its own sha256. Content selection happens at training time by choosing a column; format selection likewise.
Add to the manifest: renderer version string, and per-column round-trip test status.
---
## 9. Registered predictions
Recorded before running, for calibration:
1. Sentence frame converges faster but reaches similar final held-out accuracy.
2. The released-init advantage over vanilla-init is smaller than the transfer story assumes.
3. If the arrow format does *dramatically* worse under released-init specifically, that is real evidence the warm-start matters — and argues for moving further toward native prose in later stages rather than stopping at a fixed frame.
4. Distractor `NO_CHANGE` accuracy is the lowest of all gates.
---
## 10. What this does not buy
With deterministic templates the reader remains a lookup table — over (entity, old, new) instead of (old, new). Richer captions do not escape the DPI bound on templated families. The language claim still lives only in held-out **values** and in derived-property families where the caption expresses something absent from the metadata tuple. This amendment improves what v1 *tests*; it does not upgrade what v1 can *claim*.
