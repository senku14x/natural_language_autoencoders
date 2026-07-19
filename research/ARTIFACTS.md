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

## 2026-07-19 — SFT: LoRA-adapted released AV emits `old -> new` at 0.997 (shuffled-Δ control 0.000; unseen transitions of seen values 0.993)

- Phase: execution/feasibility (work item 2; Stage-C-shaped but NOT the
  registered Stage C 2×2 — one format × one init, no vanilla arm). Predictions
  + frozen hyperparameters registered before running
  (`temporary_artifacts/2026-07-19_sft_predictions.md`); report
  `temporary_artifacts/2026-07-19_sft_transition_report.md`; scripts
  `research/exploratory/sft_{extract_train_deltas,transition_av}.py`; data +
  adapters `data/artifacts/v1/sft_transition/`.
- Setup: 4,348 eligible train raw/pre change rows (72.5%; screen run on this
  instance — train was never screened; stratum split N 99.6%/S 54.4% matches
  frozen dev pattern), Δ at edit_pos, canonical-shape forwards; caption
  `caption_arrow_transition` + EOS after the unmodified sidecar AV prompt;
  LoRA r=64 α=128, lr 2e-5, 3 epochs, batch 32, seed 20260719; identical
  shuffled-Δ run (permutation transition-coincidence 0.000). Eval: 594 frozen-
  eligible dev raw/pre change rows (families disjoint from train; all values
  train-seen; 46.5% of rows have train-UNSEEN ordered transitions), greedy,
  strict fullmatch parse. test_value untouched. Dev screen agreement across
  torch versions 0.9925.
- Observations (dev, family-bootstrap 95% CI):
  - **real-SFT × real Δ pair-exact 0.997 [0.992, 1.000]** (S 1.000, N 0.994;
    old 0.997 / new 0.998; parse 1.000; errors = 2 generations of one pair).
  - **Unseen-transition rows 0.993** (276 rows) — composes per-endpoint, not a
    per-pair lookup.
  - **shuffled-SFT × real Δ 0.000** (= analytic Δ-ignoring floor 0.003; greedy
    collapsed to one caption 583/594; train loss pinned at ~1.95 nats ≈ prior
    entropy vs real ~0.001).
  - **real-SFT × permuted Δ: own-label 0.002, donor-label 0.997** — emits the
    injected vector's transition; activation dependence total.
  - Registered gate analog real − shuffled = +0.997 ≫ 40pp. Prediction misses
    owned: pair-exact predicted 0.82 [0.65,0.92], observed 0.997 — second
    consecutive under-prediction of token-identity readability; probe numbers
    are not a ceiling.
- Interpretation (separate): feasibility settled — the released AV LoRA-adapts
  to a **bidirectional, per-value-compositional token-identity read** of the
  edit-site delta (old side = the sign-negative component per 1a). Caps stand:
  seen values only, deterministic template ⇒ reader-is-lookup at the VALUE
  level (Amendment 1 §10, cruxes §C3); not a language, consequence, or
  held-out-value result; Stage C not marked passed (2×2 unrun).
- Plot: `plots/2026-07-19_sft_transition_accuracy.png` (accuracy by arm ×
  stratum; analytic floor, trivial-decoder 0.04, zero-shot 0.74 on same axes;
  loss curves).
- Next: user review. Highest-value cheap follow-up: registered, user-approved
  `test_value` eval with the saved adapter — it is the last unspent split.

## 2026-07-19 — Exploratory: trivial-decoder baseline — the unembedding CANNOT read the delta (0.04 top-5 vs AV 0.74); "logit-lens-like" retired

- Phase: exploration (baseline; work item 1b). Predictions + decision rule
  registered before running
  (`temporary_artifacts/2026-07-19_negdelta_logitlens_predictions.md`); report
  `temporary_artifacts/2026-07-19_trivial_decoder_report.md`; script
  `research/exploratory/trivial_decoder_baseline.py`; data
  `zeroshot_av/trivial_decoder.parquet`, `trivial_decoder_top5.json`.
- Question: does `W_U·Δ` / `W_U·RMSNorm(Δ)` recover the new value at anything
  near the AV's 0.74 (the zero-shot report asserted "logit-lens-like" 4× without
  measuring it)?
- Setup: same 50 pairs / saved edit-site states / token-level translation of the
  string rule (audited surface-form ids); exact shuffled donors from
  `scored.parquet`; state positive controls; both norm conventions; no forwards
  (only `lm_head.weight` + `model.norm.weight`; untied verified; plumbing check
  passed).
- Observations (dev, this setup): best convention top-5 new|+Δ **0.04** (top-1
  0.02); raw 0.00; −Δ→old 0.00; **state controls also 0.00** (h_cf→new median
  rank 15,022/152,064; Δ 11,900; shuffled donor 68,386); top-5 decodes are
  mid-layer noise tokens. AV references on the same vectors: 0.740 / 0.696.
  Prediction miss in the informative direction (predicted lens top-5 ~0.45, gave
  20% to "AV ≫ lens").
- **Registered decision rule fires on the AV ≫ lens branch**: the AV decodes
  something the trivial unembedding readout does not — the first result in the
  arc a trivial baseline fails to reproduce. The value code at edit-site L20 is
  in neither the unembedding basis (states fail too) nor the input-embedding
  basis (EXP-2 null, prior session): a "middle" code readable by trained probe
  (0.96/0.87) and trained AV (0.74) only.
- Interpretation (separate): retire "logit-lens-like" from the zero-shot
  report's framing. Does NOT establish AV > trained probe (0.74 < 0.87–0.96,
  consistent with the probes-ceiling literature) and does not upgrade the
  content — still a token-identity code (Amendment 1 §10 cap).
- Plot: `plots/2026-07-19_trivial_decoder_vs_av.png` (AV, lens, floors, probe
  ceiling, same axes, both directions).
- Next: user review. Feeds the SFT writeup: "decodes a non-trivially-readable
  token-identity code," not "reads the transition."

## 2026-07-19 — Exploratory: −Δ arm — the AV's old/new asymmetry is a SIGN CONVENTION (−Δ names old 0.70, new 0.00)

- Phase: exploration (work item 1a; released AV, no fine-tuning). Predictions
  registered first (same predictions file as above; P1–P3); report
  `temporary_artifacts/2026-07-19_negdelta_av_report.md`; script
  `research/exploratory/negdelta_av_arm.py`; data
  `zeroshot_av/negdelta_{scored.parquet,explanations.json,summary.json}`.
- Question: probe decodes both endpoints from Δ (~0.96/0.96) but the AV names
  new 0.74 / old 0.00 — is the old side AV-inaccessible or is it a sign
  convention?
- Setup: same 50 pairs / saved `h_edit_{base,cf}.npy` / string rule / seed
  20260719; −Δ injected, 5 samples × 50 pairs @ T=1; sidecar-driven; new
  instance (torch 2.12.0+cu130 — AV generation only; activations are the saved
  arrays). Pre-registered fact: REVERSE pairs are exact prompt swaps of their
  TARGET partners (200/200 verified) ⇒ −Δ(row) is bitwise the partner row's
  real Δ ⇒ outcome heavily favored a priori (registered as such).
- Observations (dev, this setup; pair-bootstrap 95% CI): **−Δ names old 0.696
  [0.580, 0.804], new 0.000**; strata S 0.854 / N 0.525 (mirror of real arm
  0.892/0.575); paired (−Δ old) − (real new) = −0.044 [−0.176, +0.092];
  entity mention 0.000; same value-word-in-confabulated-prose pattern (random
  inspection in report).
- Interpretation (separate): **sign convention confirmed** — the AV names
  whichever token identity sits in the positive direction; both endpoints are
  AV-accessible, one per sign. The SFT therefore teaches a *bidirectional read
  of a code already decodable one direction at a time*; an SFT gain on the old
  field must not be sold as "learning the transition." Scope cap unchanged
  (token identity; Amendment 1 §10, cruxes §C3).
- Plot: `plots/2026-07-19_negdelta_av_mention_rates.png` (all 7 arms, floor
  drawn).
- Next: user review; proceed to work item 2 (SFT) per session brief.

## 2026-07-19 — Standing instructions registered; retrospective import still blocked (file absent)

- Phase: execution (work item 0, partial). The user's Part 0 standing
  instructions are registered verbatim as
  `research/docs/STANDING_INSTRUCTIONS.md` and linked as item 0 of the
  `research/CLAUDE.md` read order.
- **The Qwen3-8B retrospective file
  (`META_MODEL_INTERPRETABILITY_RETROSPECTIVE_AND_TEMPORAL_NLA_HANDOFF_2026-07-18.md`)
  is NOT on this instance** (searched repo, home, scratchpad, filesystem-wide) —
  sixth absence flag across sessions. Import, the §11.2 provenance entry, and
  the two establishing facts (predicted-Stage-A; edit+final redundancy closure)
  remain blocked until the user re-shares the file. Not fabricated; nothing
  registered.
- Environment note for this instance: `/workspace` is **not** volume-backed
  (`workspace_is_volume: false`) — nothing survives recycle/destroy; HF_HOME
  = `/workspace/.hf_home`; torch 2.12.0+cu130 (prior sessions 2.7.0+cu128);
  transformers pinned 5.14.1; pytest 65 passed / 1 skipped; pinned tokenizer
  re-fetched at the dataset SHA.

## 2026-07-19 — Exploratory: consequence is late-final MAGNITUDE only; unseen values ARE consistently represented

- Phase: exploration. Probe-only; test_value used with explicit user approval.
  Reports `temporary_artifacts/2026-07-19_consequence_heldout_report.md` and
  forward-looking `docs/NEW_WARMSTART_CRUXES.md`; scripts
  `research/exploratory/{consequence_and_heldout,consequence_heldout_diag,
  plots_consequence_heldout}.py`; data `data/artifacts/v1/consequence_heldout/`.
- EXP-1 (final-position layer sweep, 1,656 dev change+distractor rows,
  target-vs-distractor AUC by layer): **direction (unit-normalized) ≈ 0.5 at
  every layer, both query orders** — no readable relevance direction anywhere.
  Diagnostic D1 (delta NORM): rises late — L20 0.667, L24 0.971, L28 0.985
  (median target norm 154 vs distractor 47). Consequence = answer-change
  MAGNITUDE forming at L24–28 (answer formation), which the AV interface
  erases (injection_scale=150). Ties to Stage A (final-L20 1.4% causal because
  the answer forms later).
- EXP-2 (held-out-value generalization): embedding-retrieval held-out top1
  **0.000** was a WRONG-TARGET artifact (seen values only 0.388 vs input
  embedding). Diagnostic D2 (decoder-free leave-family-out 1-NN among held-out
  rows): **0.890** (5 unseen colors, chance 0.20) > seen reference 0.564.
  Unseen values ARE consistently/separably represented — a general token code,
  not a per-value lookup; a closed-vocab decoder just can't NAME them.
- Interpretation (separate): still the token-identity story, now fully mapped.
  Value/transition identity = general, position-invariant, extends to unseen
  values → a diverse warm-start will read old→new robustly. Consequence is not
  a readable direction at any site; only late-final answer magnitude (erased by
  the interface). Real progress = derived-relation families (make consequence-
  reading distinguishable from token-naming) + a magnitude/availability channel
  (to ground NO_CHANGE) — see NEW_WARMSTART_CRUXES.md.
- Method lesson recorded: both raw numbers were misleading (normalized away the
  magnitude signal; scored against the wrong target space) — caught only by the
  direction-vs-magnitude split and a decoder-free control.
- Plots: `plots/2026-07-19_final_layer_sweep.png`,
  `plots/2026-07-19_heldout_value_retrieval.png`.

## 2026-07-19 — Exploratory: position-invariant value basis (confirmed) but edit-site delta is behaviorally BLIND

- Phase: exploration. Follow-on to the zero-shot AV read; probe-only, no new
  prompts, **test_value untouched**. Report
  `temporary_artifacts/2026-07-19_position_sweep_report.md`; script
  `research/exploratory/probe_position_sweep.py`; data
  `data/artifacts/v1/position_sweep/`.
- Question: (A) tighten CIs / more prompts; (B) is the value basis
  position-invariant across edit positions; (C) does the edit-site delta
  encode behavioral relevance or only token identity?
- Setup: edit-position L20 states for all 1,656 eligible dev change +
  distractor rows across cells (edit-pos 3–122). Linear probe (PCA→logreg,
  GroupKFold by family, direction-only).
- Observations (dev, this setup):
  - (A) old~delta ≈ new~delta, tightened: colors 0.945 [0.91,0.97],
    names 0.870 [0.83,0.95] (names up from 0.76 with more data).
  - (B) **basis is position-invariant** — cross-position transfer (colors,
    new value) 0.92–1.00 off-diagonal for delta AND h_cf state, position 3→122;
    early positions do NOT degrade. Confirms the value representation is a
    stable position-general direction (= token identity).
  - (C) **edit-site delta is behaviorally blind** — target-vs-distractor
    discriminability at the edit site AUC 0.508 (query_last) / 0.473
    (query_first) / 0.495 (both), i.e. **chance in both query orders**, while
    the edited value is decodable from distractor deltas (colors 0.95, names
    0.93). Which value changed is present (~95%); whether it matters to the
    answer is absent.
- Interpretation (separate): basis-invariance confirms the BORING reading
  (stable token identity), not consequence-reading. (C) directly demonstrates
  the "names the changed token, not the consequence" alternative (v1 §5.4),
  holding even under query-first. Clean dissociation with Stage A: edit site =
  what changed (identity, causally sufficient because edited==answer for
  TARGET), final site = that the output differs (consequence, computed
  downstream); neither single site reads a behavioral difference — as v1 §4's
  causal-attention argument predicts. Consequence for "verbalise the diff":
  YES for the transition slot (old→new, robust/position-invariant/held-out-
  family) — on-plan since v1 captions are transition-only — but the claim is
  firmly capped at an answer-transition token code; the language-earns-its-keep
  question lives entirely in held-out VALUES and derived-relation families.
- Plots: `plots/2026-07-19_position_transfer.png`,
  `plots/2026-07-19_target_distractor_auc.png`.
- Next: user review. Not committed. Real discriminators remain held-out values
  (test_value — untouched) and derived families; the edit-site delta will not
  ground a consequence/NO_CHANGE claim.

## 2026-07-19 — Exploratory: zero-shot AV reads new-value TOKEN IDENTITY from edit-site deltas (74% vs 0% floor)

- Phase: exploration. **Not Stage B, not a verbalizer result.** One cheap
  question on the RELEASED AV, no fine-tuning. Predictions registered first
  (`temporary_artifacts/2026-07-19_zeroshot_av_predictions.md`); report
  `temporary_artifacts/2026-07-19_zeroshot_av_report.md`.
- Question: does the released AV (`kitft/nla-qwen2.5-7b-L20-av`) mention the
  new value when handed the edit-site L20 delta?
- Setup: target Qwen2.5-7B @ `a09a3545…`, L20 = hidden_states[21]; AV sidecar
  (㈎ 149705, injection_scale 150, embed_scale 1.0, all from nla_meta.yaml).
  Read site = **edit-token position** (Stage A's h_*.npy were final-position;
  re-extracted here). 50 random dev/raw+preamble/eligible change pairs (seed
  20260719; S=26/N=24; query 46 last/4 first — a groupby.first() artifact).
  6 arms × 5 samples @ T=1 = 1,450 generations. String-match scoring, no LLM
  judge. Serving: repo pure injection funcs + transformers-native
  generate(inputs_embeds=) (identical forward math to SGLang; Qwen needs no
  patch). **Tokenization fix**: one-step apply_chat_template(tokenize=True)
  lets NFKC rewrite ㈎→"(가)" and drops the injection token; used the
  two-step render→encode path (per the user's Validating-NLAs infra doc).
- Observations (dev, this setup; new-value mention rate, pair-bootstrap CI):
  h_cf state **1.00** (positive control) · real Δ **0.74** [0.64,0.84] ·
  E[Δ|old,new] mean 0.62 · h_base 0.02 (names OLD 1.00) · shuffled Δ **0.00**
  · random **0.00**. Real names old 0.00. **Headline real−shuffled = +0.74**
  [+0.64,+0.84]. Colors 0.89 ≫ names 0.58. Edited-entity mention ~0 in every
  arm. Matches verified genuine in context ("The pink color", "Final token
  yellow", "My name is Oliver"); ambiguous dual-meaning words only 39/185
  hits (real still 58% excluding all of them). Shuffled names the DONOR
  pair's value (other-value rate ≈1.0), confirming the mechanism.
- Interpretation (separate): **token-identity readout at the edit position**
  (the AV literally reports "Final token X"), i.e. the "names the changed
  token" boring alternative made concrete — NOT consequence/transition
  reading. Establishes the instrument IS sensitive here (contra final-position
  pessimism and the oakhu noisiness prior) and that the edit-site delta
  carries recoverable new-value token identity above a clean 0% floor. Does
  NOT separate "names edited token" from "reads the answer transition"
  (TARGET rows: edited==queried) and licenses nothing about captions/gates/
  Stage B-C or anything causal. Surrounding prose is confabulated; only the
  named token carries signal. **Prediction miss owned**: predicted real 0.15 /
  h_cf 0.40 / headline +0.10 — over-applied the oakhu prior (measured for
  subtle final-position diffs, not single-token naming at the token's own
  position).
- Plot: `plots/2026-07-19_zeroshot_av_mention_rates.png`. Data:
  `data/artifacts/v1/zeroshot_av/` (all_explanations.json, scored.parquet,
  arm_summary.parquet, headline.json, random20.json, h_edit_*.npy).
- Qwen3-8B retrospective: **still not on this instance** (not in repo/home/
  scratchpad/attachment) — could not import; §11.2 provenance unsatisfied
  here. Not fabricated. Re-share to import (narrative, not citable as result).
- Next: user review. This is decodability-of-a-token evidence only; the
  substrate/site decisions and Stage B plan are unchanged by it.
- Follow-up probe (same day): **value_old IS linearly recoverable from the
  delta** — linear probe (PCA→logreg, GroupKFold by family), old~delta 0.96
  (colors) / 0.76 (names) ≈ new~delta 0.96/0.76, both ≫ chance (0.029/0.009);
  single states give only their own endpoint (neg ctrls at chance); label-perm
  at chance. So the AV's 0% zero-shot old-mention is a decoding limit, not
  information absence — the transition (both endpoints) is in the delta. Still
  only decodability of two token identities on dev/shared-vocabulary;
  reader-is-lookup and held-out-value caveats bind. Script
  `research/exploratory/oldvalue_probe.py`; plot
  `plots/2026-07-19_oldvalue_probe.png`; `zeroshot_av/oldvalue_probe.json`.

## 2026-07-19 — Stage A: causal substrate FAILS at the final position, PASSES at the edit site (dev split)

- Phase: validation (Experiment 1, v1 doc §7; Level 0–1 of the claim ladder)
- Question: is the pre-registered layer-20 final-position delta causally
  sufficient for the answer change (gate: median normalized margin recovery
  ≥ 0.50, direction ≥ 80%, controls < 0.10)?
- Setup: Qwen2.5-7B-Instruct @ `a09a3545…` (bf16, sdpa), L20 = block-20
  output = hidden_states[21], dev split only (4,800 rows + 240 NULL_AA),
  seed 20260719, transformers 5.14.1 (generation-env match), H100 80GB.
  **Canonical-shape forward policy**: all measurement forwards at (B=64,
  L=131, left-padded, explicit position_ids) — forward results are
  deterministic given (input, batch shape) but differ across shapes by up to
  ~0.6 logit / ~2.65 h20-norm units (≈15% of the median final-position delta
  norm); under the fixed shape, repeats are bitwise identical and all 240
  NULL_AA deltas are exactly zero. Scripts in `research/stage_a/`; artifacts
  in `research/data/artifacts/v1/stage_a/`; report in
  `temporary_artifacts/2026-07-19_stage_a_report.md`.
- Observations (evidence class: supported empirical claims on dev under the
  stated setup; per-cell family-bootstrap CIs in the report):
  1. Level-0 all pass: frozen ids re-encode; hook ≡ hidden_states[21] (0.0);
     zero patch exact identity; bit-identical dataset regeneration (66/66
     tests, config_hash match); provisional ε = 3.97.
  2. Behavioral screen: raw/pre 74.3% eligible (594/800, both strata),
     raw/nopre 32.8% (S only), chat/pre 20.0%, **chat/nopre 0.0%** (model
     opens "The/Your/Based…"). The PROVISIONAL preamble is load-bearing.
     Margin gaps median 32–53 logits. 1,016 eligible change rows,
     100 families.
  3. **Final-position substrate FAILS the gate**: real-Δ median recovery
     0.014 (0/1,016 rows > 0.5; p99 0.17; direction 75.7%); JS-to-cf
     0.982→0.981; controls: unrelated/random 0.002 (direction ~55%), zero
     exact 0, W_U answer-direction 0.100. Real beats unspecific ~7× — a
     specific but tiny signal. Since h_base+Δ_final = h_cf bitwise at that
     position: the final-position L20 state carries ~1.4% of the answer;
     the rest flows through layers 21–27 attention from earlier positions.
  4. **Site decomposition**: ALL-positions patch = exactly 1.000 (positive
     control; machinery certified). **Edit-token-only = 0.983 median, 100%
     of rows > 0.5, 100% direction-correct**, in all three live cells
     (per-cell medians 0.974–0.987, tight family-bootstrap CIs), both
     strata, both query orders; JS recovery 0.995; top-10 overlap 0.5→0.9.
     post-edit-excl-edit 0.017. Controls at the edit site: random 0.019,
     reverse −0.025 (floor-limited), same-transition other pair 0.955
     (n=78), **unrelated real Δ 0.169 — exceeds the §9 control bound 0.10**
     (nonspecific old-binding disruption; specific:nonspecific ≈ 6:1).
  5. Distractor arm at the edit site: same-norm (~76) distractor deltas →
     99.2% answer retention, JS-to-base 0.0006 — the site is
     value-binding-selective, not norm-sensitive. Final-position distractor
     arm likewise inert (99.4%).
- Interpretation (separate): v1's premise fails at its pre-registered site
  and holds one position left; this is the gating order doing its job.
  Caption-grounding consequence (already anticipated by v1 §4): the
  transition slot is groundable from an edit-site delta in any query order;
  the behavioral "answer unaffected" claim only in query-first cells. The
  same-transition prototype nearly ties the own-pair delta → §10's
  "transition-code" outcome is likely; per-example-vs-prototype goes to
  Stage B's mean hierarchy at train-scale coverage (dev n=78 is thin).
  Token-identity-vs-transition content is NOT yet distinguished. No AV/AR
  claims of any kind are made here.
- Plots: `plots/2026-07-19_stage_a_screen_eligibility.png`,
  `…_stage_a_margin_recovery_matrix.png`, `…_stage_a_js_recovery_matrix.png`,
  `…_stage_a_site_decomposition.png`.
- Next (user decisions): freeze substrate = edit site, L20, raw/preamble
  primary cell; rule on the §9 unrelated-control bound; preamble sign-off
  (now empirically load-bearing: 74% vs 33% eligibility); then Stage B
  diagnostics on edit-site deltas. Test split untouched.

## 2026-07-19 — Orientation #3 (fresh instance): spot-check audit of both prior orientations, sidecar verified from HF release, methods doc arrived (no GPU work)

- Phase: exploration (orientation; no experiments run). Mode per session brief:
  absorb + spot-check + extend, not re-derive.
- Question: do the prior orientations' findings survive independent spot checks
  against the parquet, the code, and the primary sources; and what changed on
  this (new) instance before Stage A can run?
- Setup: **new** fresh H100 SXM 80GB instance (sm_90, driver 580.105.08, CUDA
  12.8/nvcc 12.8.93, torch 2.7.0+cu128 preinstalled, Python 3.12.3).
  Persistent volume is `/home/ubuntu/counterfactualnlas` (virtiofs) — **not**
  the prior report's `/lambda/nfs/cot-oracle`; volume was empty, repo cloned
  fresh at `counterfactual_nla_v2` (HEAD 639300a). `HF_HOME` unset (flag
  stands). **transformers absent on this instance** (prior fingerprint stale);
  pyarrow 25.0.0 / pandas 2.1.4 / pytest installed user-level this session.
  Egress open. Tests: 65 passed, 1 skipped (`-p no:libtmux`; pinned tokenizer
  still absent from clone).
- Observations (all observation-level):
  - Frozen dataset spot-checked from the parquet: every prior number
    confirmed exactly — 58,320/7,290/1,240/2,514 (edit transitions);
    cells, variant balance, splits, distractor flavors; held-out sets
    {charcoal, cream, green, lavender} S-only / 10 names N-only / 5 nonces,
    zero train leakage; NULL_AA 240 all plumbing_only; position floor per
    format×preamble cell (100% / 29.3% / 0% / 0% below 50; 18,848 = 32.3%
    total); values audit 42/52, 91/124, 80/85, 24/24.
  - Caption audit re-run dataset-wide with the repo renderer/parser: 58,320
    rows × (round-trip ×4 + cross-format canonical equality + sha256 ×4) →
    **0 failures**. Link integrity: 38,720 crossed + 19,360 reverse row-level
    links (= 4,840/2,420 semantic ×8), 0 dangling, 0 cross-split.
  - Injection contract re-confirmed in code (replacement at ㈎ w/ neighbor
    check; config.py absent-injection_scale → None → assert; schema.py:82–84
    docstring wrong for injection_scale; 1e-12 clamp / missing epsilon guard;
    `_MIN_POSITION=50`; AR 21-block final-LN→Identity Linear(d,d) tokens[-1]).
    **New:** released sidecar fetched from HF
    (`kitft/nla-qwen2.5-7b-L20-av/raw/main/nla_meta.yaml`, no weights):
    `injection_scale: 150.0`, `mse_scale: 59.8665… = √3584` exactly, ㈎ id
    149705, neighbors 29/522, layer 20, exact AV/AR prompt templates —
    first verification against the released artifact itself.
  - Literature spot-checks (8 sources, targeted verbatim extraction):
    turntrout, oakhu, Anurin, Prabhu, loops, Chalnev, Realmbird, nanoNLA
    README — all prior figures **exact** (details in the standalone report).
    All ten working conclusions in the session brief survive; none overstated.
  - Qwen3-8B patching artifacts: **still absent** — fifth flag; §11.2 embargo
    remains in force. Both README open decisions (NOT_IDENTIFIABLE label;
    preamble text) still open.
- Corrections/updates made this session:
  1. Restored the missing `##` header of the Orientation #1 entry (same
     failure mode Orientation #2 fixed for the Amendment-1 entry — second
     occurrence in two days; paste the header template first when appending).
  2. `research/CLAUDE.md` item 5: `AO_NLA_JSPACE_METHODS_LEARNINGS.md` is now
     **provided** (commit 639300a, user upload post-Orientation-#2). Read in
     full; its five sources are all published; content consistent with the
     context doc.
  3. `research/CLAUDE.md` Chalnev row: "25–33pp on gender/number" is the
     **mean-over-tokens** condition (25.4/33.3pp); last-token gaps are
     19.7/7.7pp. Range ~3–49pp unchanged.
  4. Flagged, not changed: the oakhu row's "FVU > 1" and "poor cross-layer
     generalization" phrasings could not be re-verified verbatim in the post
     body (the confirmed "rock" result carries the same design consequence);
     possible appendix content or gloss — reword or re-check when convenient.
- Interpretation (kept separate): three independent audits now agree on every
  frozen number; the plan's exposure is execution discipline, not design gaps.
  New instance facts (volume path, missing transformers) are the only
  regressions; both are setup items, not blockers.
- Plots: none new (verification-only session; prior sessions' three plots
  stand).
- Standalone report: `temporary_artifacts/2026-07-19_orientation3_report.md`.
- Next: Stage A, pending user confirmation. Recommended order in the report
  §8; wanted first: preamble sign-off (cheapest, blocks regeneration risk),
  name survivors, slot2 R4/R6, Qwen3-8B logs, and the three pre-registrations
  (unrelated-delta excludes same-transition; margin/AUC scoring; per-cell
  parity expectations) — plus pin transformers==5.14.1 and set HF_HOME to the
  persistent volume before any download.

## 2026-07-19 — Orientation #2 (fresh session): independent re-verification, new gaps, literature corrections (no GPU work)

- Phase: exploration (orientation; no experiments run). Independent re-check of
  the same-day orientation entry below — findings there were re-derived from
  scratch, not trusted.
- Question: does the project state survive a second, independent audit; and do
  the ten working conclusions in the session brief survive the primary sources?
- Setup: same H100 SXM 80GB instance state (driver 580.105.08, CUDA 12.8/nvcc
  12.8.93, torch 2.7.0+cu128, capability sm_90, 78.7 GiB VRAM free). Repo now
  present on the persistent NFS volume at `counterfactual_nla` (HEAD d6c4482,
  clean, 1 commit ahead of origin). `HF_HOME` still unset (flag stands — set to
  the NFS volume before any model download). **Egress is open on this
  instance** (transformer-circuits, LW/GreaterWrong, arXiv, GitHub, HF all
  reachable) — unlike the generation session's proxy block; Stage A downloads
  are unblocked. `pytest` 9.1.1 aborts at collection because the *system*
  libtmux pytest plugin applies a mark to a fixture; run with `-p no:libtmux`
  → 65 passed, 1 skipped (integration test needs the pinned tokenizer, which
  is gitignored and not in the clone).

### Re-verification results (all confirmed independently, observation-level)

- Frozen dataset re-verified from the parquet, not the manifest: 58,320 rows /
  7,290 semantic pairs / 1,240 families / 2,514 unique ordered transitions;
  cells 19,360×{TARGET,DISTRACTOR,REVERSE}+240 NULL_AA; variants perfectly
  balanced (29,160 per level of query_order/format/preamble); distractor
  flavors 9,680+9,680; splits and pools match the manifest; no value/name/
  entity leakage into train (test_value = {charcoal, lavender, cream, green},
  S-only; test_name = 10 names, N-only; test_entity = 5 nonces).
- Caption re-verification, all 58,320 rows, five checks (round-trip re-render,
  cross-format canonical equality, sha256, caption-vs-answer semantics,
  edited-entity token identity): **0 failures**. The last two checks are new
  (not in the prior entry). NULL_AA rows are identical prompts (delta exactly
  zero) and all plumbing_only.
- Link integrity (new check): 4,840 `crossed_with` + 2,420 `reverse_of` links;
  0 dangling; all crossed pairs share the edit and differ in query; all
  reverse pairs swap values; no link crosses a split boundary.
- Injection contract re-read in code, matches the prior entry: layer 20 =
  block-20 output = HF hidden_states[21] (`extractors.py` hook); AV =
  embedding-row replacement at ㈎ id 149705 with neighbor check
  (`injection.py`); vector rescaled to L2=150 from the sidecar (config.py:
  absent key → None → assert, no sqrt_d default); AR = truncated 21-block
  backbone, final-LN→Identity, Linear(d,d) value head, last-token anchored,
  direction-only 2(1−cos); `resolve_embed_scale` = arch multiplier (1.0 Qwen)
  in nla_inference.py only.
- Qwen3-8B prior patching artifacts: **still absent** (no logs, model/layer/
  site/patch formula anywhere in research/). Fourth flag; Amendment 1 §11.2
  citation embargo remains in force. v1 doc §14 sources them to two files on
  the user's local machine (`/Users/vishesh/Desktop/temporal nlas/…`).

### New gaps found this session (docs ↔ docs, docs ↔ code, entry ↔ source)

1. **ARTIFACTS.md missing entry header** — the Amendment-1 §11 block (commit
   c0db220) had no `## date — title` line and read as a continuation of the
   orientation entry. Restored above.
2. **`nla/schema.py:82–84` docstring contradicts `nla/config.py:176–182`** —
   `resolve_target_scale`'s docstring says "Key-absent in sidecar is NOT None
   — config.py supplies sqrt_d_model as the default to .get()". True only for
   `mse_scale`; false for `injection_scale`. Second instance of the known
   design.md §2 embed-scale contradiction, this time in the shared schema
   module both readers import.
3. **Position-floor statement sharpened** (prior entry said "half the dataset
   is OOD in position"): by format×preamble cell (14,580 rows each) —
   raw/no-preamble final_pos 18–30, **100% below** the released AV's
   `_MIN_POSITION=50`; chat/no-preamble 47–59, **29.3% below** (straddles);
   both preamble cells ≥90, 0% below. Total below floor: 18,848 rows (32.3%).
   Stage B.1 parity must therefore be reported per **format×preamble cell**
   (4 cells), not per preamble (2). Plot:
   `plots/2026-07-19_v1_final_pos_vs_training_floor.png`.
4. **Two different transition-multiplicity statistics** (prior plot/entry used
   answer-transitions: TARGET+REVERSE pairs only; city max 10, median 5, names
   92% singletons — verified correct for that quantity). The statistic that
   feeds the `E[Δ|old,new]` prototype baseline is **edit**-transition
   multiplicity including DISTRACTOR cells: city max 15 (Tokyo↔Singapore),
   median 5; colors median 3, max 9; names median 2, 46% singletons. Both are
   over the same 2,514 unique ordered transitions. Implication for §8
   controls: an "unrelated real delta" control must be defined to exclude
   same-transition pairs, or it is contaminated by the shared prototype and
   is not a null.
5. **Prior entry's Bauer quote lacked its own qualifier** — the paper says
   "We do **not** have a formal ablation for this" immediately before the
   "every run that did NLA-style injection performed significantly worse"
   sentence, and nanoNLA's README now says the maintainer is no longer
   confident additive is better. Additive-vs-replacement is unsettled;
   registry row updated.
6. **Jakkli AO numbers need the Bauer §A.1 rebuttal attached** — near-chance
   sycophancy AUC is "largely a calibration artifact" (AOs default to "No");
   Yes/No token-logit margin gives 0.83 ROC AUC. Consequence for us: **score
   AO/probe baselines on logit margins or AUC, never sampled-string
   accuracy** — otherwise we sandbag the boring baseline. Registry row
   updated.
7. **Chalnev gap range corrected** in the registry: probes beat decoder
   extraction on *every* task but by ~3–49pp depending on task (AG News ~3pp;
   Language ID ~44–49pp), not uniformly 20–49pp.
8. **Frozen-artifact reproducibility is not currently executable**: the
   pinned tokenizer files are gitignored and absent from this clone, so the
   README's bit-identical-regeneration check and the integration test cannot
   run until the pinned-SHA tokenizer is re-fetched (egress is open, so this
   is cheap — fold into Stage A setup).

### Ten working conclusions from the session brief vs primary sources

All ten survive, with these calibration notes: (1) turntrout exact — FVE 0.68
vs 0.70, plausibility 99.3% implausible, 21%→7.6% under RL for plausible-init,
0.08%→0.7% for implausible-init. (2) The paper itself says surfacing terms
"generally increase … with an unexplained spike" attributed to possible
string-matching noise — "moves erratically" is Anurin's (and our) gloss of the
same figure; the brief's own "suggestive, not established" caveat is the
paper-accurate reading. (3) loops exact (λ=0.001: −28% tokens at +0.007 FVE;
λ=0.002: −40% at −0.015; λ=0.03 degenerates to tail-repetition) plus
turntrout's last-paragraph-only NLA at FVE 0.67 vs 0.70 control. (4) Anurin
exact — snippets→empty −0.09, →"REDACTED" −0.76 FVE (below mean baseline);
on-policy constrained sampling (quote bans) barely hurts (0.774→0.757);
truncated/masked-caption reader training was proposed publicly in the loops
comments (Karvonen: random-K prefix to the AR). (5) oakhu exact — for Qwen the
mean-direction "rock" wins **dataset-wide at the trained layer**, all three
NLAs lose to the within-variant rock, Qwen's NLA does not detect
problem-constant changes while Gemma-27B's does; steering "corrections" 11/65
vs 10/65 with random vectors. (6) Chalnev — structure-preserved
entity-substitution confirmed verbatim (Sarah/marbles for Alice/apples);
under-extraction vs probes confirmed with the task-dependence caveat above.
(7) Realmbird — on our exact kitft L20 checkpoints: answer-mention rises
toward the final answer; higher counterfactual importance ↔ lower
reconstruction loss. (8) Prabhu exact — 81.4% explanation flip, 99.6% answer
label unchanged, per-sample optimized vectors (Qwen2.5-7B L20). (9) Fully
grounded: Anurin replicates Qwen2.5-7B-L20 round-trip FVE 0.748 vs paper
appendix 0.752 on held-out UltraFineWeb, shuffled-verbalization FVE < −0.6;
turntrout independently gets 0.70 at 20k docs (vs Anthropic's 0.75 at 100k).
(10) is our own plotting norm; every strong source above draws its trivial
baseline (rock bars, shuffle nulls, no-activation floors).

Additional constraints registered from the papers proper: AV scaling factor
α ≈ 75th-percentile activation norm at the target layer (explains sidecar 150
≫ ambient √d≈59.9); AR trained without chat formatting on the `<summary>`
suffix; Anurin: the AR value head is near-identity (+0.007 FVE) on Gemma-27B;
Anurin's zero-shot diff-of-means verbalization (anger/pirate/Russian visible,
SAE directions mostly not) is the closest public precedent for our Stage B.3
zero-shot delta reading — qualitative only, no exact fields; AObench's
"Activation Sensitivity" task (same tokens, different upstream context) is
direct precedent for our crossed pairs.

### Interpretation (kept separate)

The frozen dataset and injection-contract facts are solid under independent
re-derivation; nothing found invalidates any frozen artifact. The genuinely
open debts before GPU spend are unchanged (Qwen3-8B artifact import; one
reconciliation sentence for §5.8-vs-Stage-C/D wording) plus three new small
ones from this session: define the "unrelated delta" control to exclude
same-transition pairs (finding 4), commit to margin/AUC scoring for AO/probe
baselines (finding 6), and pre-register the expected parity outcome per
format×preamble cell before Stage B.1 (finding 3) so a raw/no-preamble parity
failure is not spent as if it were news.

- Plots: `plots/2026-07-19_v1_final_pos_vs_training_floor.png` (new);
  prior session's two plots re-verified against independent recomputation.
- Standalone report: `temporary_artifacts/2026-07-19_orientation_report.md`.
- Next: Stage A unchanged as the recommended next action (see chat summary);
  before GPU spend, user sign-offs wanted on: preamble text (PROVISIONAL),
  name-survivor list, slot2 R4/R6 retention, and (if available) the Qwen3-8B
  patching logs for import.

## 2026-07-19 — Orientation #1 (fresh session): docs-vs-code audit, frozen-dataset verification, literature registration (no GPU work)

*(Header restored 2026-07-19 by Orientation #3 — it was missing from the
original commit and this entry read as a continuation of the one above.)*

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

## 2026-07-18 — Amendment 1 §11 revisions applied (renderer r2)

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
