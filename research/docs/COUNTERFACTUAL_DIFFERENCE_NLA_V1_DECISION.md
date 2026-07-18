# Counterfactual Difference NLA: v1 research decision

**Date:** 2026-07-18  
**Status:** finalized scope for the first controlled experiment; not an empirical result  
**Supersedes for v1:** the "Temporal NLA" framing in the earlier proposal

## Retrieval summary

We are not initially studying temporal trajectories, a two-site handoff, or broad open-ended concepts. The first project is a deliberately boring controlled test, evaluated first with synthetic labels and then with common real names already familiar to the pretrained model:

> Given only a causally effective final-position activation difference that changes a binding answer from `A` to `B`, can an activation-only verbalizer report `A -> B` on held-out examples, and can a separately trained reader turn the canonicalized report back into a patch that shifts the model in that stated direction?

Use **Counterfactual Difference Verbalizer** for an AV-only system. Reserve **Counterfactual Difference NLA** for the AV plus a natural-language bottleneck and an independently trained/frozen reader whose decoded patch is evaluated causally.

Passing the first test would establish only a controlled proof of concept. It would not establish open-ended interpretation, mechanism understanding, temporal reasoning, safety relevance, or an advantage of language over simpler classifiers.

## 1. Final problem statement

Current Natural Language Autoencoders train an activation verbalizer and reconstructor to preserve a full activation through a text bottleneck. Full-state reconstruction is useful, but reconstruction improvement need not prioritize small behaviorally important differences, and high reconstruction quality can coexist with inaccurate or confabulated prose.

This project asks whether a behaviorally relevant **change** can instead be the object of the language bottleneck. We begin with matched prompt pairs from a controlled binding task. The first stratum uses synthetic symbols and colors for continuity with the existing patching result. The second committed stratum replaces arbitrary values with common personal names that the pretrained model already represents. For each pair, we compute a natural prompt-induced activation difference and first verify by patching that it shifts the model's output toward the paired counterfactual. We then ask whether an activation-only verbalizer can describe the directed output change on held-out examples. Finally, a separately trained reader must convert the parsed and deterministically re-rendered description into a patch that reproduces the stated directional behavioral shift.

The initial claim is deliberately narrow:

> On a controlled binding task, an activation-only language interface can read and causally replay a known answer-changing activation delta.

## 2. Why this is the right reframe

- The user's Qwen3-8B patching result is prior evidence that the controlled causal substrate is plausible. It is not a substitute for checking the exact Qwen2.5-7B checkpoint, layer, position, and hook contract used by the released NLA.
- The first experiment does not require time, a trajectory, or a two-site transition. Calling it "Temporal NLA" overstates the method.
- Patching is not the proposed method. It certifies that the input delta is causally effective for the selected behavioral metric in the tested setup.
- A supervised `delta -> caption` model is a useful feasibility test, but by itself it is a specialized activation decoder, not a complete NLA.
- Difference-vector verbalization already has adjacent Activation Oracle and steering-vector precedent. The candidate contribution is the combination of a natural prompt-induced causal delta, exact relational labels, information-availability controls, and causal replay through language.

## 3. Exact first task

### Target

- Target model: Qwen2.5-7B-Instruct, with exact revision pinned.
- Primary layer: the released NLA's layer 20, subject to interface parity tests.
- Primary position: the final prompt position immediately before answer generation.
- Qwen3-8B remains prior evidence and a later replication target; do not mix its activation space with Qwen2.5 components.

### Prompt pair

Illustrative base prompt:

```text
In this task, dax means red.
In this task, wug means blue.
What color is dax?
Answer:
```

Counterfactual:

```text
In this task, dax means green.
In this task, wug means blue.
What color is dax?
Answer:
```

At the chosen layer and final position:

```text
delta_final = h_counterfactual - h_base
patched_base = h_base + delta_final
```

The primary behavioral metric is the counterfactual logit margin:

```text
margin = logit(new_answer) - logit(old_answer)
```

Report normalized margin recovery, correct-direction rate, raw old/new logits, and per-example distributions—not only means. Also report improvement in Jensen-Shannon distance to the counterfactual distribution and local top-k overlap. These distributional checks do not replace the answer-margin target; they reveal when a patch merely promotes the new answer token without reproducing the surrounding counterfactual distribution.

### Caption

The required first field is intentionally minimal:

```text
PREDICTED_OUTPUT_CHANGE: red -> green
```

Allowed control labels are:

```text
PREDICTED_OUTPUT_CHANGE: NO_CHANGE
PREDICTED_OUTPUT_CHANGE: NOT_IDENTIFIABLE_FROM_THIS_STATE
```

An optional `EDIT_DESCRIPTION` field may be scored separately. It must not gate the behavioral proof of concept because a final-position delta may reliably encode the answer transition without retaining the edited entity.

Do not initially require `PERSISTENT`, `UNCHANGED`, distractor identity, task identity, or free-form rationale. A difference need not identify shared state, and requiring these fields invites unsupported prose.

### Committed name-familiarity condition

The common-name condition is part of v1 rather than a later open-ended entity audit. It tests whether the channel transfers from arbitrary symbols to familiar lexical identities without requiring biography or celebrity knowledge.

Illustrative target-query pair:

```text
My name is Alice.
I live in Paris.
Question: What is my name?
Answer:
```

```text
My name is Priya.
I live in Paris.
Question: What is my name?
Answer:
```

Required caption:

```text
PREDICTED_OUTPUT_CHANGE: Alice -> Priya
```

The matched distractor condition asks where the speaker lives. The name changes while the measured city answer should not:

```text
PREDICTED_OUTPUT_CHANGE: NO_CHANGE
```

For the later edit-site availability test, use both query orders:

- query last: `My name is Alice. I live in Paris. Question: ...`;
- query first: `Question: ... My name is Alice. I live in Paris. Answer:`.

The query-last name-site delta is identical before different later queries and therefore cannot identify whether the name change will affect the requested answer. Query-first makes the query available before the name is processed.

Construct the familiarity ladder as:

1. synthetic or nonce labels for pipeline calibration;
2. high-familiarity common names as the committed natural-value condition;
3. lower-familiarity and variable-token real names as robustness;
4. public-entity attributes only as a later, separately controlled branch.

Estimate familiarity with the frozen target model's name surprisal rather than relying only on intuition about which names are common. Initially require matched Qwen token counts and equal total prompt lengths. Split by canonical name before generating warm-start captions; keep reverse pairs, aliases, and all templates involving a name in the same split.

Do not train or grade unqueried demographic, nationality, occupation, or personality claims from common names. This condition establishes familiar-identity binding, not rich person understanding.

## 4. The causal-attention constraint

In a decoder-only model, an activation at an edit token cannot depend on a query that appears later. Therefore, under a query-last template, the same edit-position delta can be followed by different future queries with different behavioral consequences. The query-specific effect is formally not identifiable from that supplied state.

Consequences for the design:

- Use `delta_final` for the intentionally boring first behavioral test.
- At a query-last edit site, ask only for the local update or require `NOT_IDENTIFIABLE_FROM_THIS_STATE` for the future query-specific effect.
- After the boring test passes, compare query-last with query-first templates. Query-first makes query identity available before the edit, although it does not guarantee that the consequence is encoded.
- A fixed-layer position sweep can then measure where the answer consequence first becomes readable.

This availability experiment is the first extension within the binding task; it is not a prerequisite for the initial pass.

## 5. Claim ladder

### Level 0: implementation validity

The official activation injection, normalization, layer indexing, token position, patching hook, batch dimensions, and prompt grammar are reproduced. Ordinary-activation AV/AR parity is checked before any delta result is trusted.

### Level 1: causal substrate

On the exact Qwen2.5 setup, the real prompt-induced delta shifts the output margin toward the paired counterfactual. Reverse deltas reverse the direction; shuffled, unrelated, and matched-norm random controls do not reproduce the effect.

### Level 2: delta decodability

A linear probe, small MLP, norm-only classifier, mean-delta hierarchy, and output/logit-direction readout are evaluated. The mean hierarchy compares a global mean, an ordered value-transition mean, an entity-conditioned transition mean, and the per-example delta. These establish baselines, causal sufficiency for a selected metric, or decodability under the split—not mechanism, information absence, or model use.

### Level 3: activation-grounded verbalization — the core boring test

Given only the delta, the AV emits the exact ordered output change on held-out examples. Accuracy collapses under no-activation and shuffled-delta controls, reverses under reverse deltas, and returns `NO_CHANGE` on null and behaviorally irrelevant edits.

An AV-only success is called a **Counterfactual Difference Verbalizer**, not yet a full NLA result.

### Level 4: causal language channel

A reader is trained separately on canonical caption-to-delta or caption-to-patch examples and frozen. The generated AV text is parsed into declared fields and deterministically re-rendered before reaching the reader. This blocks free wording, punctuation, or hidden strings from acting as a private code.

The decoded patch must shift behavior in the direction stated by the caption. Wrong, reversed, shuffled, and random-string captions must fail or have the corresponding wrong/reversed effect.

### Level 5: within-task generalization

Hold out state families, templates, entity-value combinations, ordered value transitions, and complete common-name identities. Report synthetic and common-name strata separately before pooling. This tests increasingly strong composition and transfer from arbitrary to pretrained lexical values without changing the underlying binding task.

### Level 6: new concepts

Only after Levels 1–4 pass should the method move to rule replacement, arithmetic correction, tool-result changes, natural prompt edits, steering, fine-tuning/model differences, or safety-relevant behaviors.

## 6. Execution order

### Stage A: exact-model causal check

Do a small Qwen2.5 replication rather than reopening the entire patch-localization project. Choose the site on development data, freeze it, and evaluate all eligible test examples rather than reporting only strong oracle successes.

### Stage B: no-training diagnostics

Run before SFT:

1. ordinary-activation AV/AR parity;
2. `AV(h_base)` versus `AV(h_counterfactual)` with deterministic caption differencing;
3. zero-shot AV on normalized and contract-valid deltas;
4. linear probe, small MLP, and norm-only classifier on delta-to-label;
5. global-mean, ordered value-transition-mean, entity-conditioned-mean, and per-example delta hierarchy;
6. output/logit-direction readout at the final site;
7. no-activation/template-prior predictor;
8. delta norm and geometry diagnostics;
9. a preregistered two-prompt comparison: the released AV prompt and one explicitly contrastive AV prompt.

Verify whether the AV interface normalizes injected activations before interpreting any scale sweep. Target-model patch strength and AV-input scale are different interventions.

### Stage C: supervised verbalizer feasibility

Lightly adapt an activation-only AV to the rigid caption. Generate warm-start records for both the synthetic and common-name strata, with name-disjoint splits fixed before caption generation. Grade fields exactly; do not use an LLM judge for the primary metric. Report a synthetic-only warm-start arm and a matched-budget synthetic-plus-common-name arm so the contribution of familiar values is measurable. This stage tests whether a language model can extract the known transition from the delta.

### Stage D: independent causal reader

Before delta decoding, use a canonical counterfactual-state caption as a state-level positive control for the reader and patching interface. Then train a reader on canonical gold change captions, freeze it, and pass only parsed and re-rendered AV fields. Evaluate caption-to-patch behavioral recovery. This is the first stage that earns the controlled **Counterfactual Difference NLA** label.

RL, joint AV/AR optimization, causal reward, and unsupervised/open-ended training are later method-development questions, not prerequisites for establishing that the basic signal exists.

## 7. Experiment cards

### Experiment 1: exact-model causal substrate

1. **Hypothesis:** the selected Qwen2.5 layer-20 final-position delta is sufficient to move the old-versus-new answer margin toward the paired counterfactual.
2. **If true:** real deltas move the margin in the correct direction and reverse deltas reverse it.
3. **Boring alternatives:** generic norm effects, answer-token steering, a stable transition prototype, or test-set selection explain the result.
4. **Baselines/controls:** shuffled, unrelated, matched-norm random, output-token direction, global/value-transition/entity-conditioned means, null, reverse, and distractor-edit deltas. Compare both answer-margin recovery and full-distribution movement toward the counterfactual.
5. **Sanity checks:** exact hook/position parity, alpha zero identity, batch-size-one parity, tokenization checks, and raw-logit inspection.
6. **Likely failure modes:** wrong layer convention, wrong token position, hidden normalization, broadcasting/slicing bugs, or Qwen3 results not transferring to Qwen2.5.
7. **Why now:** every downstream caption claim is meaningless if the supplied delta is not causally effective on the exact target setup.

### Experiment 2: activation-grounded behavioral verbalization

1. **Hypothesis:** an activation-only AV can recover the ordered output transition from `delta_final` on held-out synthetic families and complete held-out common names.
2. **If true:** exact field accuracy is high in both reported strata, reverse deltas reverse the caption, null/distractor edits yield `NO_CHANGE`, and no/shuffled activation performance collapses.
3. **Boring alternatives:** language priors, label imbalance, output-direction readout, transition-prototype classification, metadata leakage, or ordinary pointwise NLA differencing solve the task.
4. **Baselines/controls:** no/zero/shuffled activation, balanced labels, pointwise differencing, linear probe, small MLP, prototype classifier, and logit/readout baseline.
5. **Sanity checks:** ordinary-activation AV parity, exact deterministic parsing, hidden prompt metadata, held-out state families, and manual inspection of random—not selected—examples.
6. **Likely failure modes:** delta distribution is outside the AV interface, the AV quotes a plausible majority transition, endpoint/reverse leakage, or exact values are decodable but not extracted by the AV.
7. **Why now:** this is the smallest direct test of the user's actual goal and determines whether training an AR or broader method is warranted.

### Experiment 3: causal use of the language bottleneck

1. **Hypothesis:** the canonical semantic fields produced by the AV are sufficient for an independent reader to construct a patch with the stated directional effect.
2. **If true:** generated-caption patches retain a substantial fraction of gold-caption recovery; wrong and reverse captions produce wrong or reverse effects.
3. **Boring alternatives:** the reader memorizes one transition prototype per label, exploits uncontrolled wording, or adds generic answer promotion.
4. **Baselines/controls:** gold, wrong, reverse, shuffled, matched-format random-string, transition-prototype, and output-direction patches.
5. **Sanity checks:** train the reader separately, freeze it before AV evaluation, parse and deterministically re-render every caption, and verify behavior as well as vector similarity.
6. **Likely failure modes:** private code, reader-caption distribution mismatch, prompt-specific information lost by canonicalization, or good delta reconstruction that does not recover behavior.
7. **Why now:** this separates a prose classifier from a causally usable natural-language channel and is the minimum extra evidence needed for the NLA label.

## 8. Minimum controls

### Causal target controls

- real `A -> B` delta;
- reverse `B -> A` delta;
- null `A -> A` delta;
- independently shuffled delta;
- unrelated real delta;
- matched-norm random direction;
- answer/output-token direction;
- global-mean, ordered value-transition-mean, and entity-conditioned transition-mean deltas;
- target edit versus distractor edit.

### Activation-grounding controls

- no activation;
- zero activation;
- independently shuffled activation-label pairing;
- balanced label frequencies;
- prompt and metadata hidden from the AV;
- ordinary pointwise NLA descriptions plus a deterministic differencer.

### Language-channel controls

- gold canonical caption;
- AV caption after parse and canonical re-render;
- wrong-transition caption;
- reversed caption;
- shuffled caption;
- matched-format random string;
- transition prototype patch;
- independent frozen reader.

### Split discipline

- group state families so reverse edges and shared endpoints do not leak causally across splits;
- group common-name examples by canonical name, keeping aliases, reverse pairs, and all templates involving that name in one split;
- freeze templates and semantic tuples for the test set before training;
- report both ordinary held-out contexts and harder compositional holdouts;
- bootstrap confidence intervals over independent state families, not individual near-duplicate prompts.

## 9. Provisional pass/fail gates

These are practical starting gates, not field standards. They should be frozen after development inspection and before the final held-out run.

### Causal substrate

- median normalized counterfactual-margin recovery at least `0.50`;
- correct-direction margin movement on at least `80%` of held-out families;
- shuffled, unrelated, and matched-norm random controls have median recovery below `0.10`;
- reverse delta reverses the effect.

### Verbalization

- apply and report the gates separately for synthetic and common-name strata; do not let a large easy stratum hide failure on the other;
- macro exact ordered-transition accuracy at least `90%`;
- reverse consistency and null calibration at least `90%`;
- real-delta accuracy exceeds independently shuffled-delta accuracy by at least `50` percentage points;
- no/zero-activation output is at the balanced-prior baseline;
- family-bootstrap confidence intervals and all raw failure categories are reported.

### Causal language channel

- gold-caption reader patch has positive median behavioral recovery and correct direction on at least `70%` of held-out examples;
- the generated, parsed, canonicalized caption retains at least `80%` of the gold-caption behavioral recovery;
- wrong and shuffled captions are materially worse;
- reversed captions reverse the direction.

If the exact numerical gates look unreasonable on development data, change them once with a written justification, then freeze them. Do not tune them after viewing the test set.

## 10. How to interpret the likely outcomes

- **Probe succeeds, AV fails:** the transition is decodable but this verbalizer does not extract it. This supports changing the reader/training, not claiming the delta lacks information.
- **AV succeeds, activation controls fail to collapse:** task-prior captioning or leakage, not activation interpretation.
- **AV succeeds, AR fails:** a behavioral-change verbalizer exists, but not a causal NLA language channel.
- **AR succeeds only on gold captions:** text can code a patch, but the AV has not read the delta.
- **All first gates pass:** a boring controlled proof of concept.
- **A value-transition mean ties the method:** the proof of concept survives, but the tested behavior may require only a stable answer-transition direction. This does not prove that individual deltas contain no additional information.
- **Output/logit direction ties the method:** the final-site result is answer promotion, which is acceptable for v1 but not evidence of upstream mechanism understanding.
- **Pointwise NLA differencing ties the method:** no evidence that a dedicated delta architecture is necessary; the contribution must become reliability, calibration, compactness, or earlier localization.
- **Reader memorizes one patch per label:** language is a discrete transition code, not prompt-specific reconstruction. Report this plainly.

## 11. Audit of the shared Claude responses

Keep:

- the causal-attention/non-identifiability correction;
- removing `PERSISTENT` and `UNCHANGED` from v1;
- running pointwise differencing and probes before training;
- exact structured grading instead of gist or an LLM judge;
- treating post-hoc field deletion as OOD, not clean semantic ablation;
- checking delta geometry and a declared semantic positive control;
- the warning that arbitrary optimized activations can dissociate AV text from model behavior.

Soften or reject:

- deltas do not cancel every nuisance "by construction";
- the NLA objective does not literally assign a variance weight to each sentence;
- the existing evidence does not prove reconstruction and useful surfacing are universally uncorrelated or anti-correlated;
- answer-containing text correlations do not prove a universal gradient preference;
- a behavior label avoids judges only in this rigid synthetic setting;
- consensus among samples is not calibrated confidence;
- one reader's failure cannot distinguish absent information from unreadable information;
- an externally released reader is not automatically compatible or an independent control without exact interface verification.

## 12. Deferred work

Remove from the v1 headline and main execution path:

- Temporal NLA branding;
- mandatory two-site synergy or handoff;
- low-rank potential models and endpoint algebra;
- state-family cycle losses;
- learned magnitude channels unless a diagnostic demands them;
- joint sitewise AV architecture;
- RL and explanation-ordering objectives;
- long trajectories and predictive innovation;
- multi-edit interference;
- naturalistic and safety claims.

After the boring pass, scale in this order:

1. harder held-out transitions and templates within binding;
2. query-order and fixed-layer position-sweep availability calibration;
3. one second concept with a different mechanism;
4. natural prompt-induced changes;
5. steering, LoRA, or model-difference deltas;
6. open-ended captions and causal-priority training.

## 13. Mentor-ready result

The first result worth showing is:

> On unseen synthetic bindings and complete held-out common names, an activation-only verbalizer correctly reports the ordered output change encoded by a causal activation delta; this accuracy collapses under no-activation and shuffled-delta controls, reverses under reverse deltas, and returns no change when the edited name is irrelevant to the queried field. A separately trained frozen reader converts the parsed and canonicalized report into a patch that shifts the model's output in the stated direction. Simple probes, transition prototypes, output directions, familiarity/norm baselines, and pointwise NLA differencing bound the strength of the claim.

## 14. Source map

- Original NLA paper: <https://transformer-circuits.pub/2026/nla/>
- NLA technical workspace/appendix: <https://transformer-circuits.pub/2026/workspace/>
- NLA robustness: <https://turntrout.com/natural-language-autoencoder-robustness>
- Internal-chain-of-thought evaluation: <https://www.lesswrong.com/posts/QQQAcKuWK6k98FivY/can-activation-verbalizers-surface-an-internal-chain-of-1>
- NLA summarizer analysis: <https://www.lesswrong.com/posts/4nZwNPyfoadAxPH98/natural-language-autoencoders-are-summarizers-but-do-they>
- Cycle-consistent activation oracles: <https://www.lesswrong.com/posts/Nf2sKaNNdxE2ssxbp/cycle-consistent-activation-oracles-1>
- NLA adversarial stress test: <https://www.lesswrong.com/posts/3oRQxpxn4LiWKwbeQ/can-you-hide-from-a-natural-language-autoencoder>
- Earlier proposal: `/Users/vishesh/Desktop/temporal nlas/TEMPORAL_NLA_RESEARCH_PROPOSAL.md`
- Retrospective/handoff: `/Users/vishesh/Desktop/temporal nlas/META_MODEL_INTERPRETABILITY_RETROSPECTIVE_AND_TEMPORAL_NLA_HANDOFF_2026-07-18.md`
- Durable field context: `NLA_JSPACE_RESEARCH_CONTEXT.md`
