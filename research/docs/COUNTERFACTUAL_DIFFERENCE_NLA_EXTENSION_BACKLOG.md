# Counterfactual Difference NLA: gated extension backlog

**Date:** 2026-07-18  
**Status:** audited research menu; not part of the v1 commitment  
**Core plan:** [`COUNTERFACTUAL_DIFFERENCE_NLA_V1_DECISION.md`](./COUNTERFACTUAL_DIFFERENCE_NLA_V1_DECISION.md)

**Later public-entity/parametric-knowledge branch:** [`PARAMETRIC_PRIOR_ENTITY_SWAP_EXTENSION.md`](./PARAMETRIC_PRIOR_ENTITY_SWAP_EXTENSION.md). Common-name identity binding is part of v1 itself.

## 1. Scope rule

Do not reopen or enlarge the boring v1 test. New branches are triggered by evidence:

- Run cheap diagnostics and controls immediately when they change the interpretation of v1.
- Run availability, calibration, and distributional-behavior experiments only after the final-position verbalizer is activation-grounded.
- Run new concepts only after the binding result survives null, reverse, shuffle, prototype, norm, and output-direction baselines.
- Run multi-edit, open-ended, or cross-source experiments only after a reliable single-change channel exists.

The aim is information gain, not accumulating experiments.

## 2. Additions that belong in v1 now

These refine the existing test without changing its claim.

### 2.1 Mean-delta hierarchy

Compare, using leave-one-family-out estimates:

1. global mean delta;
2. ordered value-transition mean, `E[delta | old, new]`;
3. entity-conditioned transition mean, `E[delta | entity, old, new]`;
4. per-example delta.

Evaluate answer-margin recovery, Jensen-Shannon movement toward the counterfactual distribution, top-k overlap, and raw logit changes.

Interpretation constraint: if a mean patch ties a per-example patch, the coarser patch is sufficient for the tested behavioral metric. It does **not** prove that the individual delta contains no additional information.

### 2.2 Token-promotion check

An answer margin can be recovered by merely increasing the new answer token. Include:

- an explicit output-token direction;
- ordered value-transition means;
- full-distribution movement toward the counterfactual;
- local top-k/neighborhood changes.

A pure promotion result is acceptable for the boring pass, but it caps the claim at reading an answer-transition code.

### 2.3 Scalar and interface baselines

- Train a norm-only consequence classifier.
- Measure delta norm by condition without assuming target edits are larger.
- Numerically verify whether the AV interface preserves or removes vector magnitude.
- Compare the released AV prompt with exactly one preregistered contrastive prompt on development data.

Geometry is diagnostic. Distance from ordinary activations cannot by itself distinguish domain shift from unreadability.

### 2.4 Reader positive control

Before interpreting delta-reader failure, test whether a canonical counterfactual-state caption can produce a causally effective state replacement through the same reader and patching interface. Control this against generic output-token steering.

## 3. Immediate post-v1 extensions

### Extension A: query availability and propagation

**Trigger:** the final-position AV passes activation-grounding controls.

1. **Hypothesis:** a calibrated reader reports a query-specific consequence only after query identity is causally available in the supplied state.
2. **Design:** cross query order (`query last`, `query first`) with edit relevance (`target`, `distractor`). For the query-last condition, pair an identical prefix and edit delta with balanced later queries.
3. **Expected if true:** query-last edit-site captions return `NOT_IDENTIFIABLE_FROM_THIS_STATE`; query-first target and distractor edits separate; later positions increasingly support the measured consequence.
4. **Boring alternative:** the AV repeats the local edit regardless of query availability, or query-first states still contain no readable consequence.
5. **Baselines:** per-position linear/MLP probes, pointwise AV descriptions, output-direction readouts, no/shuffled activation, and balanced query priors.
6. **Sanity checks:** verify query-last edit activations are numerically identical before the later query; separately report task eligibility and patching success for each template.
7. **Failure modes:** query-first grammar changes model competence; position-dependent AV performance is mistaken for propagation; later prompt text leaks into evaluation.
8. **Why now:** this turns unsupported consequence claims into an objectively measurable error rather than a subjective confabulation judgment.

After the 2x2 result, run a fixed-layer position sweep. Validate ordinary-state AV performance at each position before interpreting delta failures; position changes can also create reader distribution shift.

### Extension B: distributional behavioral changes

**Trigger:** the AV reliably reports ordinary argmax-changing deltas.

Construct causally measured conditions:

1. argmax changes;
2. argmax stays fixed while target confidence increases;
3. argmax stays fixed while target confidence decreases;
4. a local edit produces negligible output-distribution change.

Optional later condition: remove a binding and measure fallback toward a prior, using length- and alignment-controlled prompts.

1. **Hypothesis:** the verbalizer reads changes in the model's output distribution rather than only whether a new token becomes top-1.
2. **Expected if true:** it distinguishes the four conditions and predicts the measured confidence direction.
3. **Boring alternatives:** captions are determined by `W_U delta`, delta norm, or an argmax-flip bit.
4. **Baselines:** norm-only, output-direction, probe, transition means, distractor edits, and shuffled deltas.
5. **Sanity checks:** derive labels from the patched output distribution—not from the intended prompt edit. Report target-set probability, margin, entropy, JS divergence, and local top-k behavior.
6. **Failure modes:** global entropy changes for irrelevant reasons; condition classes differ systematically in norm or tokenization.
7. **Why now:** this is the cleanest test of whether “behavioral difference” means more than answer substitution.

### Extension C: derived relational consequence

**Trigger:** direct binding succeeds but is explained well by a value-transition or output-token direction.

Example query-first family:

```text
Question: What shape is Tob?
Tob is a fep.
Every fep is round.
Answer:
```

Edit `round -> square` and test an edit-position delta after the queried entity and membership fact are already available.

1. **Hypothesis:** the delta can support a derived answer change involving an entity not named in the edited rule.
2. **Expected if true:** the AV reports `Tob: round -> square`, not merely `feps: round -> square`, on compositional holdouts.
3. **Boring alternatives:** it repeats the edited rule, succeeds only at the final answer position, or memorizes property-transition classes.
4. **Baselines:** supervised probes, value-transition means, output directions, pointwise captions, distractor-class edits, and swapped memberships.
5. **Sanity checks:** causally validate the edit-site delta before training; hold out entity × class × property products and templates.
6. **Failure modes:** unnatural query-first syntax creates a separate task artifact; the final-position representation collapses again to answer promotion.
7. **Why now:** this tests transfer beyond direct lexical binding while retaining exact ground truth.

A probe can also learn the derived mapping. Success establishes compositional activation reading, not a language advantage over classifiers.

### Extension D: calibrated abstention

**Trigger:** Extension A establishes a reliable availability contrast.

Train the structured output to distinguish:

- a measured consequence;
- `NO_CHANGE`;
- `NOT_IDENTIFIABLE_FROM_THIS_STATE`.

Evaluate selective accuracy, abstention rate, and risk-coverage curves under balanced conditions. The last label means only that the requested fact is not identifiable from the supplied state by causal construction. It does not distinguish “the model has not decided” from “the information exists but this reader failed.”

### Extension E: detection threshold and magnitude channel

**Trigger:** a reliable signal exists and sensitivity limits matter.

Vary separately:

- target-model patch strength;
- semantic distance between endpoints;
- number of relevant changes;
- AV-input magnitude, only if the interface preserves it.

If the AV normalizes directions, add an explicit scalar magnitude channel before asking it to describe change size. Report a detection curve, not a single cherry-picked scale. Repeated caption agreement may measure stability, but is not calibrated confidence until validated against correctness.

## 4. Later extensions

### 4.1 Multi-edit interference

Run only after single-edit reliability. Construct a task where two edits both affect a defined two-item behavior. Vary their relative causal strengths and score exact set precision/recall. A one-answer task is invalid because an unqueried second edit is behaviorally irrelevant.

This tests masking, limited caption capacity, and whether the model reports only the louder transition. Include single-edit, union-of-prototype, shuffled-pair, and reverse controls.

### 4.2 Independent-reader robustness

Train at least two reader seeds independently of AV optimization. Parse and canonicalize AV outputs before both readers. Cross-reader agreement in causal direction is stronger than success through one co-adapted channel, but incompatible readers remain an interface failure rather than evidence against language in general.

### 4.3 Cross-source deltas

Evaluate prompt-induced, steering-induced, and fine-tuning/model-difference activations as distinct domains—not as a presumed naturalness ladder. Every source needs its own causal certification. A one-layer delta from a fine-tuned model is not automatically replayable in the base model because downstream weights also changed.

### 4.4 Natural-pair semi-supervision

Pointwise caption differencing may generate noisy labels for natural prompt pairs. Use them only as silver training data after controlled validation. Never evaluate against the same teacher captions or treat them as ground truth; retain constructed counterfactuals for the test set.

### 4.5 Open-ended descriptions

Only when one-line structured fields are reliable should the project add longer prose, claim ordering, field masking, or causal-priority training. Any deletion or substitution evaluation must match the reader's training distribution; matched length alone does not guarantee an in-distribution caption.

## 5. Catalogue items not adopted as written

- **Blind-judge behavioral suffix prediction as the headline:** exact field accuracy is cleaner for rigid captions, and a judge adds error. Suffix prediction is secondary when captions become free-form.
- **Dropping the causal reader:** an AV-only result is a behavioral-difference verbalizer, not a complete NLA language channel.
- **Screen-rejected examples as automatic `NO_CHANGE` ground truth:** they may reflect uncertainty, tokenization, weak deltas, or selection effects. Use measured patch behavior and matched exploratory strata.
- **Direct-versus-composed endpoint deltas:** `h_blue - h_red = (h_green - h_red) + (h_blue - h_green)` is algebraically exact for identical endpoint states. Different histories avoid the tautology but introduce context differences.
- **Fixed chance, skyline, or final-token percentages:** recompute every baseline for the actual task.
- **Probe failure implies information absence:** it establishes failure of the tested probe class under the chosen data and split.
- **Prototype equivalence defines representational content:** it establishes sufficiency for the measured behavior, not what else the individual delta contains.
- **Nearest-neighbor geometry diagnoses a null:** geometry can reveal distribution shift but cannot assign the cause of AV failure.
- **Differencing universally erases unchanged context:** exact shared additive components cancel, while context-dependent interactions may remain. Test unwanted-context decoding empirically.
- **Consensus equals confidence:** agreement can repeat a shared bias. Validate precision at coverage before using it as confidence.
- **Arbitrary AV-input scale sweeps:** meaningless if normalization removes magnitude; target-model patch strength is a separate intervention.
- **Prompt-visible text baseline as a fair activation baseline:** seeing both prompts reveals the edit and is a skyline, not a matched prior.
- **Bootstrapped captions as truth:** they create teacher-confirmation loops.
- **The relational family is uniquely immune to probes:** a supervised decoder can learn it too.
- **Broad novelty or universal confabulation claims:** require a dedicated literature review and evidence beyond this controlled task.

## 6. Decision sequence

1. Run the value-transition mean and output-direction audit before AV training.
2. Complete the boring final-position test without expanding its caption.
3. If it passes, run query availability × target/distractor edit.
4. Then test confidence-only distribution changes.
5. If the binding task is dominated by answer-transition prototypes, move to the derived relational family.
6. Only after reliable single-change reading, test calibrated abstention and detection thresholds.
7. Multi-edit and cross-source transfer are last.

At every step, a negative result should identify which claim failed rather than trigger more architecture by default.
