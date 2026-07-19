# Research artifacts log

Running log of research reports and result analyses for the J-space / NLA
project. Newest entry first. This file is the temporary home for anything that
would otherwise live in a lab notebook: experiment reports, result analyses,
negative results, and decision records.

Entry format:

```
## YYYY-MM-DD — short title

- Phase: ideation | exploration | validation | execution | distillation
- Question: what uncertainty this was meant to reduce
- Setup: exact model, layer(s), positions, data, seeds, configs (or a link to
  the config/commit)
- Observations: what was measured, with evidence class per the context doc
  (observation / recurring pattern / supported claim / causal claim)
- Interpretation: kept separate from observations; name the boring
  alternatives not yet ruled out
- Plots: links into research/plots/
- Next: what this changes about the plan
```

Negative and ambiguous results get entries too — they are usually the ones
that save the most compute later.

---

## 2026-07-19 — Session orientation: docs-vs-code audit, frozen-dataset checks, literature pass (no GPU work)

- Phase: exploration (orientation; no experiments run)
- Question: is the project state internally consistent (docs ↔ code ↔ frozen
  artifacts ↔ literature) before Stage A commits GPU time?
- Setup: fresh H100 SXM 80GB instance (sm_90, driver 580.105.08, CUDA 12.8,
  torch 2.7.0+cu128, nvcc 12.8.93). Persistent NFS volume
  `/lambda/nfs/cot-oracle` was **empty** at session start — contrary to the
  session brief, no repo/dataset existed on the instance; cloned fresh from
  `senku14x/natural_language_autoencoders@counterfactual_nla` (HEAD c0db220).
  `HF_HOME` unset → caches would land on the ephemeral root disk (flagged; fix
  before model downloads). No model weights downloaded.

### Observations (all observation-level; no behavioral or activation data exists yet)

- **Frozen dataset verifies.** 58,320 rows / 7,290 semantic pairs / 1,240
  families / 2,514 ordered transitions, matching manifest and prior entries.
  Cells 19,360 × {TARGET, DISTRACTOR, REVERSE} + 240 NULL_AA; variants
  perfectly balanced (29,160 per level of query_order / format / preamble).
  All four Amendment-1 caption columns present; dataset-wide re-verification
  of round-trip, cross-format canonical equality, sha256s, and
  legacy-caption consistency: **0 failures in 58,320 rows**. Tokenizer-independent
  tests: 65 passed, 1 skipped (integration test needs the pinned tokenizer,
  which is gitignored — not in the clone).
- **Held-out-value splits are real**: test_value (4 colors: charcoal,
  lavender, cream, green; 2,880 rows, stratum S only), test_entity (5 nonces),
  test_name (10 complete names, N only), plus unordered-pair transition
  holdouts. Survivors from the tokenizer audit: colors 42/52, names 91/124,
  cities 80/85, nonces 24/24 (all four surface forms single-token).
- **Position-floor exposure**: no-preamble rows have final_pos 18–59
  (mean 38) — below the released AV's `_MIN_POSITION = 50` training floor;
  preamble rows sit at 90–131. Half the dataset is OOD in position for the
  released checkpoint by construction; Stage B.1 parity must be reported per
  preamble condition.
- **Transition multiplicity is skewed by kind** (plot 2): city transitions
  (the R4/R6 deviation rows) repeat up to 10× (median 5), colors median 2,
  names mostly singletons. Transition-prototype baselines will be strongest
  exactly on the deviation rows — watch this in Stage B/C readouts.
- **Qwen3-8B prior patching artifacts: still absent** (no logs, model/layer/
  position/patch formula anywhere in `research/`). This is at least the third
  flag; the Amendment 1 §11.2 citation embargo remains in force.
- Plots: `plots/2026-07-19_v1_dataset_composition.png`,
  `plots/2026-07-19_v1_transition_distribution.png`.

### Contradictions and gaps found (docs ↔ docs, docs ↔ code)

1. `research/CLAUDE.md` (workspace-paper row) references `research/PLAN.md`
   ("per-paper priorities") — **file does not exist**.
2. `AO_NLA_JSPACE_METHODS_LEARNINGS.md` still not provided (declared, but the
   context doc §10 instructs reading it alongside).
3. **docs/design.md §2 vs nla/config.py:176–182 on injection_scale default**:
   design doc says absent key ⇒ `sqrt_d_model`; code resolves absent ⇒ `None`
   ⇒ hard assert in train_actor. Code wins (this is the known embed-scale
   comment/config contradiction; `mse_scale` does default to sqrt_d_model).
4. `nla/injection.py` docstring uses marker char ㊗ (U+3297); the actual
   sidecar char is ㈎ (U+320E, id 149705). Stale comment, harmless.
5. `research/data/README.md` "STATUS: blocked on tokenizer access" is stale —
   generation completed 2026-07-18/19 at the pinned SHA.
6. **Gating-order tension**: context doc §5.8 and the CLAUDE.md evidence
   discipline say "gold-caption/AR gate before AV training", but the v1 doc
   runs Stage C (AV SFT) before Stage D (reader). In practice Amendment 1
   §1.2 dissolves most of this: with a single-transition canonical caption the
   MSE-optimal reader IS the ordered value-transition mean, so the
   "gold-caption reader gate" is materially the Stage B mean-hierarchy rung
   E[Δ|old,new] — which does run before Stage C. The formal stage wording
   should still be reconciled (one sentence in the v1 doc).
7. Session brief calls the scale resolver `resolve_embed_scale`; that function
   exists only in `nla_inference.py` (arch embedding multiplier: 1.0 for
   Qwen). The training-side scale resolver is `resolve_target_scale`
   (`nla/schema.py`). Naming only.
8. One PDF referenced in the session instructions ("in the docs directory")
   is absent from the repo and has no public copy. Need the file from the
   user; per the confidentiality rule it must not be registered, cited, or
   committed until provided and cleared.
9. `normalize_activation` clamps only exactly-zero norms (1e-12); a genuinely
   tiny-but-nonzero delta is amplified to norm 150. The epsilon-guard policy
   the CLAUDE.md calls for exists nowhere in code yet — it is a GPU-stage
   TODO, not a dataset problem (NULL_AA deltas are exactly zero by prompt
   identity).

### Injection contract as verified in code (constrains Stage A/B)

- Layer convention: `layer_index=20` = output of decoder block 20
  (post-residual-add) = HF `hidden_states[21]`; AR truncates to blocks 0..20
  inclusive with final-LN → Identity, extraction suffix-anchored at
  `tokens[-1]`.
- AV injection: embedding-row replacement at the ㈎ marker (neighbor-checked
  scan inside the hook), vector rescaled to L2 = `injection_scale` (released
  Qwen sidecar: 150.0; ambient sqrt_d would be ≈59.9). AR loss direction-only
  (`mse_scale` = sqrt_d on both sides). Magnitude is destroyed at the
  interface — Amendment 1's magnitude ban is structurally correct.
- Community forks (EasyNLA, nanoNLA) use **additive norm-matched injection at
  an early layer** instead — a different interface; their results do not
  transfer to our checkpoints' interface without re-verification.

### Literature constraints registered (sources added to research/CLAUDE.md)

- Zhang & Turner: implausible-initialized NLA reaches FVE 0.68 vs 0.70
  control while end-of-RL plausibility is 0.7% vs 7.6% — reconstruction
  quality cannot certify caption truth. (Grounds exact grading + Stage C
  activation-dependence controls.)
- oakhu & ryan_greenblatt: Qwen2.5-7B NLA showed **FVU > 1** on OOD short
  math prompts (worse than predicting the mean), poor cross-layer transfer,
  final-token positions most informative; steering "corrections" at the
  random-perturbation rate (17%). (Grounds Stage B.1 parity as load-bearing;
  expect near-zero zero-shot delta reading; adaptation, not prompting, is the
  likely path.)
- Anurin: snippet ablation drops FVE 0.77 → −0.76; cross-family transfer
  0.51–0.68 — reconstruction rides heavily on context snippets.
  (Grounds the no-activation/shuffled/text-inversion control family; our
  AV sees only the delta, so the residual leak channel is template metadata,
  which the crossed/reverse cells decorrelate.)
- Chalnev: linear probes beat decoder extraction by 20–49pp on
  classification. ("Probe succeeds, AV fails" is the expected default; v1 doc
  §10 already encodes the correct interpretation.)
- Prabhu: 81.4% explanation-flip with 99.6% behavior preservation.
  (No monitor claims; adversarial eval is a Level-6 concern.)
- loops: explanations shrink 28–40% at ~0.015 FVE cost. (Supports rigid
  one-line captions; per-sentence faithfulness claims are weak.)
- Karvonen AOs + "Hard to use": AOs recover fine-tuned info absent from text,
  but 49.4% vague responses, confabulated constants, text-inversion
  dependence, AUC 0.55 vs probe 0.77 on sycophancy. (The AO baseline family
  is strong on decodability but weak on reliability; our contribution must be
  exact relational labels + causal replay, not "reading deltas" per se.)
- Bauer et al. (AObench): the Activation Sensitivity task (same tokens,
  different upstream context) is direct precedent for our crossed pairs; on
  Qwen3-8B "every run that did NLA-style injection performed significantly
  worse" than additive norm-matched injection for QA-style reading. (If the
  released interface fails on deltas, an AO-style interface is the next
  boring rung, not a bigger model.)
- NLA paper: activations unit-normalized in the loss; steering via AR
  succeeds only ~50% even in-house on Opus. (Stage D causal replay is
  genuinely untested territory; the workspace J-lens results are
  Claude-specific and motivate framing only.)

- Interpretation (kept separate): the project's docs are in an unusually
  consistent, pre-registered state; the two real debts before GPU spend are
  the missing Qwen3-8B artifacts (embargo already handles citation risk) and
  the Stage C/D vs §5.8 ordering sentence. Nothing found invalidates the
  frozen dataset.
- Next: recommend Stage A (target-model-only causal check: Level-0 plumbing
  parity, then the patch-control matrix on dev families) — it needs only
  Qwen2.5-7B-Instruct at the pinned revision, no NLA checkpoints, and its
  margin-recovery number gates everything downstream. Set `HF_HOME` to the
  NFS volume first.

- Phase: execution
- What changed (all four review decisions accepted by user):
  1. Distractor entity forms now carry the behavioral claim only
     ("`| answer unaffected`"); queried-entity NAME reserved. Renderer
     `amendment1-r2`; r1 naming forms are parse errors.
  2. Stage B gains the queried-entity probe (amendment §4): decode query
     identity from Δ_final across edit-matched crossed pairs; decision rule
     recorded. Without it the naming slot is permanently undecidable.
  3. Tuple-mean result de-scoped: licenses nothing at final position (Stage A
     passing entails Δ_final is query-dependent). Qwen3-8B result citation
     EMBARGOED until its logs are imported into research/.
  4. §6 gate split (real-Δ ≥ prior+40pp; shuffled-Δ collapses to prior) and
     vanilla-init pinned to identical sidecar mechanics, fresh weights only.
- Verification: regeneration reproduced all 58,320 pairs with pair_id,
  input_ids, legacy caption, and both transition columns unchanged; only
  entity-content columns changed. 66/66 tests pass.
- Artifacts: `research/data/artifacts/v1/` refreshed.

## 2026-07-18 — Amendment 1 applied: four caption columns added to frozen v1

- Phase: execution
- Question: implement `CAPTION_SCHEMA_AMENDMENT_1.md` §8 — emit
  `{arrow, sentence} × {transition, entity}` renders without regenerating
  pairs.
- Setup: renderer `amendment1-r1` (`ctf_data/rich_captions.py`); same config,
  seed, tokenizer as the v1 generation entry below.
- Observations: regeneration reproduced all 58,320 pairs bit-identically
  (pair_id, legacy caption, base_input_ids verified against the frozen
  parquet); 8 new columns added (4 captions + 4 sha256); round-trip +
  cross-format canonical equality verified per semantic pair at generation
  time; 64/64 tests pass.
- Interpretation: none — infrastructure.
- Flags: (1) implemented the amendment's §2.1 distractor forms verbatim,
  including the queried-entity naming clause, despite the tension with its
  own §3 (clause = "measurement, not caption field, until tested") — see
  README; (2) the §3 "prior tuple-mean result" needs its measurement site
  pinned (edit-site vs final-position) before it can inform the
  unaffected-clause decision.
- Artifacts: `research/data/artifacts/v1/` refreshed (parquet 3.4 MB,
  manifest with renderer version + verification status).
- Next: unchanged — GPU Stage A/B; content decision resolves per amendment §4
  after the mean hierarchy runs.

## 2026-07-18 — v1 prompt-pair dataset generated (tokenizer-only stage)

- Phase: execution
- Question: build the frozen CPU-stage dataset the GPU stage consumes; learn
  which answer values survive single-token filtering (gates the held-out-value
  split).
- Setup: `research/data/configs/v1.yaml`, seed 20260718, tokenizer
  Qwen/Qwen2.5-7B-Instruct @ `a09a35458c702b33eeacc393d103063234e8bc28`
  (local files, sha256s in manifest), config_hash `0d85e5a602c4a4aa…`, code
  commit `2eed5ad`+fixes. transformers 5.14.1.
- Observations (counts, not claims):
  - Audit: colors 42/52 single-token survivors (gate ≥16 passed), names
    91/124, cities 80/85, nonces 24/24. Attrited names incl. Priya, Fatima,
    Yuki, Kenji, Layla, Ravi, Zara (multi-token).
  - Full run: 58,320 rows = 7,290 semantic pairs × 8 variants (query order ×
    raw/chat × ±preamble), 1,240 families, 2,514 unique ordered transitions,
    TARGET:DISTRACTOR:REVERSE = 19,360 each, NULL_AA 240 (plumbing-only).
    Zero invariant rejections. Splits: train 36,240 / dev 4,800 /
    test_context 4,800 / test_transition 4,800 / test_value 2,880 /
    test_entity 2,880 / test_name 1,920 rows.
  - Preamble variant pushes read site to final_pos ≥ 90 (≥52 required).
  - 49/49 tests pass incl. tokenizer integration; causal-masking prefix
    check asserted dataset-wide.
- Interpretation: none — this is infrastructure; no behavioral or activation
  measurements exist yet. The behavioral screen (GPU stage) determines
  eligibility; expect attrition, surplus is ~3×.
- Open decisions flagged: NOT_IDENTIFIABLE label (parser-valid, unused);
  preamble text (provisional); name survivor list (user review); slot2 TARGET
  rows included (README §decision 1, kill switch documented).
- Artifacts: `research/data/artifacts/v1/` (pairs.parquet 2.6 MB frozen,
  manifest.json, values_audit.json, rejections.json, smoke/full stdout).
- Next: GPU stage A/B — behavioral eligibility screen, format freeze
  (raw vs chat), site freeze, patch-control matrix.
