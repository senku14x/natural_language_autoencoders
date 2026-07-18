# Parametric-prior entity swaps: controlled extension design

**Date:** 2026-07-18  
**Status:** later public-entity/attribute branch. Common-name identity binding is now committed directly in the v1 proposal; this document covers the richer case where names invoke person-specific parametric knowledge.  
**Parent plan:** [`COUNTERFACTUAL_DIFFERENCE_NLA_V1_DECISION.md`](./COUNTERFACTUAL_DIFFERENCE_NLA_V1_DECISION.md)

## 1. Verdict

Famous-name swaps are a useful extension, but they are not simply a harder version of nonce bindings. They change the source of information:

- nonce bindings primarily test information introduced in the prompt;
- famous entities can activate associations already stored in model weights;
- the AV inherits much of the target model's parametric knowledge, so it can decode a name and then supply attributes from its own weights.

Therefore, a rich and factually correct caption is not enough. The experiment must distinguish:

1. **identity lookup:** `delta -> entity name -> attributes supplied by AV weights`;
2. **activation-grounded reading:** `delta -> the target model's measured context-specific behavioral change`.

The precise claim worth testing is:

> Conditional on entity identity and a name-to-attribute lookup baseline, does the activation delta contain additional model-specific information about which property or downstream behavior changed?

## 2. Terminology correction

Call these **parametric-prior entity swaps**, not “counterfactuals already stored in the weights.” The weights may contain prior knowledge about both entities. They do not necessarily contain the directed `Vishesh -> Zendaya` transition or the context-specific behavioral effect being studied.

The pair

```text
My name is Vishesh.
My name is Zendaya.
```

is an identity swap, not yet a behavioral counterfactual. It needs a specified and measured query or continuation before a behavioral consequence is defined.

## 3. Recommended progression

### Family E0: smoke test only

Use `Vishesh -> Zendaya` to verify tokenization, activation extraction, and whether the zero-shot AV notices an identity change. Do not use it as evidence for rich behavioral interpretation: the pair is badly confounded by familiarity, token count, frequency, and the availability of public associations.

### Family E1: matched known-to-known swaps

Use pairs of public figures matched as closely as practical on:

- Qwen token count;
- target-model name surprisal as a frequency/familiarity proxy;
- fame proxy;
- era;
- broad domain or occupation;
- prompt length and template.

Vary one registered query attribute at a time where possible. Natural entities will never differ along exactly one dimension, so call this matching and stratification—not perfect attribute isolation.

### Family E2: crossed name × contextual-profile design

This is the central discriminating experiment. Vary the name and an explicitly stated contextual attribute independently.

Example factors:

- name: `Zendaya` versus another matched public figure;
- stated occupation: `actor` versus `geologist`;
- relation queried: occupation versus an irrelevant attribute.

For the first exact probe, prefer a curated finite relation such as `BIRTH_COUNTRY` with randomized semantic options. Occupation can be the second relation. Pronouns are a poor primary target because several continuations may be grammatical and predictions can reflect stereotype or style rather than a unique fact.

The four cells include:

1. name changes, stated occupation fixed;
2. stated occupation changes, name fixed;
3. both change;
4. neither changes.

Register the four cells by whether entity identity (`N`) and measured behavior (`B`) change:

| Cell | Name changes? | Measured output changes? | Required behavioral field |
|---|---:|---:|---|
| `N0B0` | no | no | `NO_CHANGE` |
| `N1B0` | yes | no | `NO_CHANGE` |
| `N0B1` | no | yes | exact `A -> B` |
| `N1B1` | yes | yes | exact `A -> B` |

Subdivide `N1B1` into **congruent** cases, where the name prior and measured context-specific behavior agree, and **conflict** cases, where they disagree. The conflict cell is decisive: a name-lookup system follows the ordinary entity prior, while an activation-grounded system should follow the target model's measured contextual behavior.

Use fictional-profile language so counterfactual attributes are explicit:

```text
Question: In this fictional profile, what is the person's occupation?
Options: actor, athlete, geologist, politician.
Profile: Zendaya works as a geologist.
Person: Zendaya.
Answer:
```

Randomize option order and derive labels from the target model's measured distribution. Include only examples where the model behaviorally follows the contextual profile, and report rejection rates by entity, relation, and congruence condition.

Predictions:

- a name-only swap with occupation held fixed should yield `NO_CHANGE` in measured occupation behavior;
- an occupation swap with name held fixed should yield the measured occupation change;
- if the AV reports the famous person's real occupation despite a successfully followed contextual override, it is using parametric lookup rather than reading the target behavior.

### Family E3: model-error and uncertainty challenge set

Cases where the target model disagrees with an external factual reference can be useful, but they are not a clean primary discriminator. Because the AV inherits the same base weights, it may share the target's misconception.

Use these cases only after E2, and compare against the explicit identity-to-lookup baseline. Grade the target model's measured output separately from external factual correctness.

The informative error subset is not merely `target answer != reference answer`. It is:

```text
TARGET_MODEL_ANSWER != AV_LOOKUP_ANSWER
```

If the target and AV share the same misconception, that example cannot distinguish activation reading from shared parametric lookup. Freeze the eligible discordant pool once and report its size; do not keep mining errors until the desired behavior appears.

### Family E4: multi-relation fan-out

For the same entity swap, test separate query-conditioned deltas for occupation, nationality, domain, and a carefully defined pronoun candidate set. This checks whether the behavioral effect changes with the queried relation rather than collapsing to one answer-token direction.

Do not ask one pre-query entity activation to report which later relation will be queried. Under query-last ordering, that consequence is not identifiable from the supplied state. Use query-first prompts for entity-site consequences, or use final pre-answer activations.

## 4. Prompt and output design

### Query-first entity-site template

```text
Question: Which occupation is most associated with the named person?
Options: actor, athlete, musician, politician.
Person: Zendaya.
Answer:
```

The query and candidate relation are available before the entity boundary. This makes a relation-specific effect possible at the entity site, though it still must be causally and empirically demonstrated.

### Query-last availability control

```text
Person: Zendaya.
Question: Which occupation is most associated with the named person?
Options: actor, athlete, musician, politician.
Answer:
```

At the entity site, pair the identical prefix and delta with balanced later queries. The correct query-specific label is:

```text
BEHAVIORAL_EFFECT: NOT_IDENTIFIABLE_FROM_THIS_STATE
```

### Structured warm-start caption

```text
ENTITY_CHANGE: Vishesh -> Zendaya
QUERY_RELATION: occupation
MEASURED_BEHAVIOR_CHANGE: <old_choice> -> actor
EFFECT_CLASS: ARGMAX_CHANGE
```

Other allowed effect classes:

```text
EFFECT_CLASS: NO_CHANGE
EFFECT_CLASS: CONFIDENCE_INCREASE
EFFECT_CLASS: CONFIDENCE_DECREASE
EFFECT_CLASS: NOT_IDENTIFIABLE_FROM_THIS_STATE
```

Do not initially train or grade a free-form biography. Do not include unqueried occupation, nationality, gender, era, or register claims. Every required field must have a deterministic label or a measured behavioral target.

## 5. The critical lookup baseline

Implement the boring alternative directly:

1. decode the old and new entity identities from the delta;
2. give those identities plus the registered query relation to a frozen text-only model;
3. ask it to predict the behavioral effect.

Query the final trained AV weights without activation injection as well as the frozen target backbone. The final AV's own lookup is the more direct competing explanation because adaptation may have changed its factual prior.

For an upper-bound version, give the baseline the gold entity identities. This is not a fair activation baseline; it is an explicit test of the explanation “the AV only recognized the name and looked up the rest.”

Interpretation:

- if this identity-to-lookup pipeline ties the full AV on known-to-known swaps, there is no evidence that the inspected delta carries richer attribute content;
- if the full AV follows measured contextual overrides or target-specific errors where the lookup baseline follows the ordinary entity prior, that is evidence for additional activation-grounded information;
- if both systems fail, the task or behavioral substrate is not established.

## 6. Minimum controls

### Activation and causal controls

- zero, no-activation, shuffled, unrelated, reverse, and matched-norm deltas;
- output-token direction and ordered answer-transition mean;
- name-only delta from a minimal identity prompt;
- entity embedding or very-early-state identity control, only through a contract-valid mapping;
- causal patch recovery at each candidate site;
- full-distribution movement, not only old-versus-new margin.

### Lookup and context controls

- gold-name-to-attribute text-only lookup;
- decoded-name-to-attribute lookup;
- same name with conflicting contextual profiles;
- different names with the same contextual profile;
- congruent and incongruent name/profile combinations;
- irrelevant-relation and distractor-attribute queries;
- randomized option order;
- prompt-visible system reported only as a skyline.

### Language-channel controls

- exact structured fields;
- independent frozen reader;
- parsed and canonicalized AV output;
- wrong, reverse, shuffled, and random-format captions;
- state-level reader positive control before delta decoding.

## 7. Tokenization and position policy

Famous names often have different token lengths. Last-subtoken extraction does not solve downstream positional misalignment.

For the first entity experiment:

1. pin the exact Qwen tokenizer and chat template;
2. require equal subtoken counts and equal total prompt lengths within each pair;
3. require tokenizer-exact answer candidates, including leading-space variants;
4. causally test the entire aligned entity span, then compare single-vector AV candidate sites on development data:
   - last entity subtoken;
   - the immediately following punctuation/delimiter token;
   - query token or relation token;
   - final pre-answer token;
5. freeze the site before held-out evaluation;
6. report causal eligibility and rejection rates per site.

The following delimiter is especially useful because it is the same surface token after the full entity span and can aggregate the name. It is still an empirical candidate, not an assumed solution.

Only after equal-length pairs work should variable-length names be introduced as a robustness condition. Do not redesign the AV for spans before seeing a signal.

## 8. Warm-start construction and split discipline

Record at least:

```text
pair_id
condition_class
base_entity
counterfactual_entity
query_relation
base_context_attribute
counterfactual_context_attribute
congruence_class
prompt_order
base_name_token_ids
counterfactual_name_token_ids
candidate_site
base_distribution
counterfactual_distribution
patched_distribution
measured_effect_label
target_model_answer
av_lookup_answer
external_metadata_for_stratification_only
```

Split before generating captions:

- entity-disjoint train/development/test sets;
- keep both directions and all paraphrases of an entity pair in one split;
- hold out name × relation × contextual-profile combinations;
- balance answer choices, effect classes, option positions, and congruence;
- balance the `N0B0`, `N1B0`, `N0B1`, `N1B1_CONGRUENT`, and `N1B1_CONFLICT` cells;
- keep prompt-template holdouts;
- later add a relation-held-out evaluation.

External factual databases may be used to curate and stratify examples. The behavioral caption must be labeled from the target model and patched distributions, not assumed from real-world truth.

Do not generate rich attribute text for held-out entities and then call them held out. The warm start itself is part of the training data and must respect entity-level splits.

## 9. Experiment card

1. **Hypothesis:** after controlling for entity identity and shared parametric lookup, an activation-only verbalizer can report the target model's context-specific behavioral change caused by an entity/profile delta.
2. **Expected if true:** the AV tracks measured contextual overrides, returns `NO_CHANGE` for behaviorally irrelevant name swaps, reverses under reverse deltas, and beats identity-to-lookup on conflict cases.
3. **Boring alternatives:** name recognition plus lookup, token frequency, name length, option-position priors, output-token promotion, or transition prototypes explain the result.
4. **Baselines:** identity-to-lookup, name-only delta, norm-only classifier, output directions, mean deltas, probes, pointwise caption differencing, and activation nulls.
5. **Sanity checks:** exact tokenizer checks, behaviorally screen every prompt family, inspect random examples, report rejected examples, and verify patching before AV evaluation.
6. **Failure modes:** the target ignores contextual overrides, entity familiarity dominates splits, the AV reproduces shared misconceptions, unequal positions contaminate deltas, or the caption asks for properties not identifiable at the chosen site.
7. **Why this is worth doing:** it tests whether the language channel captures model-specific consequences beyond decoding a finite synthetic tuple, while retaining exact behavioral evaluation.

## 10. Continue, narrow, and stop rules

### Continue

- the exact target model follows enough contextual overrides to create balanced held-out cells;
- causally effective deltas exist at a frozen site;
- the AV beats activation controls and the identity-to-lookup baseline on contextual-conflict cases;
- performance transfers to held-out entities and templates;
- a frozen reader causally replays the canonicalized behavioral field.

### Narrow the claim

- known-to-known swaps work, but identity-to-lookup ties the AV: report entity decoding plus shared-prior inference;
- only final-position deltas work: report answer-transition reading, not upstream retrieval;
- contextual profiles work only when congruent with parametric knowledge: report a prior-dominated limitation;
- probes succeed but AV fails: report an extraction deficit.

### Stop this family

- the target model does not reliably follow the controlled prompts;
- no selected activation delta causally changes the registered behavior;
- results disappear under entity-disjoint or token-count-matched splits;
- captions remain rich but fail exact behavioral fields;
- performance is fully explained by name lookup, option priors, or output-token directions.

## 11. Literature boundary

- [ROME](https://rome.baulab.info/) found factual-recall effects in GPT-style models at middle-layer MLPs while processing the last subject token, with late attention effects near the final token. This supplies candidate sites, not a Qwen-specific convention.
- [MEMIT](https://arxiv.org/abs/2210.07229) extends weight editing to many stored subject–relation–object associations. It studies parameter edits, not prompt-induced activation differences.
- [CounterFact](https://ar5iv.labs.arxiv.org/html/2202.05262#S3.SS3) replaces known factual objects with deliberately difficult false targets and evaluates efficacy, generalization, and specificity. It does not show that the replacement association was already stored.
- [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191) provides causal evidence for in-context entity–attribute binding, mainly with controlled single-token entities and attributes. It does not establish that the last subtoken is universally sufficient for multi-token famous names in Qwen.
