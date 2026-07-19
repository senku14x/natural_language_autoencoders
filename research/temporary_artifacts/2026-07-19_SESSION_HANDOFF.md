# Session handoff — 2026-07-19 (Stage A + exploratory arc)

**Branch:** `counterfactual_nla_v2` · **Everything below is committed & pushed** (HEAD `5359ec7`). This is the detailed record of what was run, in order, with the numbers and the decision state, so the next session can pick up without re-running.

---

## 0. One-paragraph summary

On Qwen2.5-7B-Instruct / L20 / the frozen v1 binding task (dev split), the pre-registered **final-position** causal substrate **fails** its gate (median margin recovery 0.014), while the **edit-token position passes** decisively (0.983). Exploring the edit-site delta: it is a **general, position-invariant, token-identity code** for the value transition — old and new value each ~0.95/0.87 linearly decodable, stable across edit positions 3→122, and consistent even for **unseen values** (0.89 leave-family-out 1-NN). The released AV, zero-shot, names the new value 74% (vs 0% shuffled/random) — literally reporting "Final token X". But the delta is **behaviorally blind**: it cannot tell a consequential edit from an inert one (target-vs-distractor AUC ≈ 0.5 at the edit site *and* at the final position, every layer, both query orders). The behavioral consequence exists only as **answer-change magnitude at the final position, layers 24–28** (norm-AUC → 0.99) — which the AV interface erases by rescaling every injected vector to norm 150. Net: the channel reads *which token changed*, not *whether it matters*; and in this dataset the two are confounded because for TARGET rows the edited value **is** the queried answer.

---

## 1. Environment (this instance; persistent volume survives)

- H100 80GB (sm_90), driver 580.105.08, CUDA 12.8, torch 2.7.0+cu128, Python 3.12.3.
- Persistent NFS volume: `/home/ubuntu/counterfactualnlas` (= `/lambda/nfs/counterfactualnlas`). Repo at `<vol>/natural_language_autoencoders`. `HF_HOME=<vol>/hf_home` (persisted in `~/.profile`).
- On the volume already: Qwen2.5-7B-Instruct @ `a09a35458c702b33eeacc393d103063234e8bc28` (bf16), pinned tokenizer at that SHA (`research/data/tokenizer/qwen25`), AV `kitft/nla-qwen2.5-7b-L20-av` (snapshot `b884691`).
- Per-instance (reinstall on a fresh box): `transformers==5.14.1` (generation-env match — pin it), `accelerate safetensors matplotlib scikit-learn pandas pyarrow orjson httpx`. gh + hf CLIs authenticated this session.

## 2. Interface gotchas verified this session (these bite silently)

1. **NFKC eats the injection token.** `apply_chat_template(tokenize=True)` NFKC-normalizes ㈎ (U+320E → "(가)") and **drops** the injection token on transformers 5.14.1. Use the two-step `apply_chat_template(tokenize=False)` → `tokenizer(rendered, add_special_tokens=False)` (matches the project's `Validating-NLAs` infra doc). Injection then lands correctly (verified: pos 111/125, neighbors 29/522).
2. **Forwards are batch-shape-dependent** (bf16): different padding length / batch size shift logits ~0.6 and h20 norm ~2.6 units (≈15% of the median final-position delta). All Stage A/exploratory forwards use a **canonical shape (B=64, L=131, left-padded, explicit position_ids)**; under it, repeats are bitwise identical and NULL_AA deltas are exactly zero.
3. **Serving:** injection is client-side, so generation runs **in-process** via `generate(inputs_embeds=…)` reusing the repo's own pure functions (`nla_inference.normalize_activation` / `inject_at_marked_positions` / `resolve_embed_scale`). Qwen needs no SGLang patch. Same vectors drop into `NLAClient` for SGLang-parity if wanted.
4. **The AV interface destroys magnitude** (`injection_scale=150`; AR loss direction-only). Anything that lives in the vector's norm (e.g., the behavioral consequence — see §4) is invisible to the AV.

## 3. Stage A (validation; committed, gated) — `temporary_artifacts/2026-07-19_stage_a_report.md`

- **Level-0 plumbing:** all pass. Frozen `input_ids` re-encode exactly; hook `layers[20]` ≡ `hidden_states[21]` (0.0); zero patch exact identity; bit-identical dataset regeneration (66/66 tests, config_hash match).
- **Behavioral screen (dev):** raw/preamble **74.3%** eligible (both strata) — the substrate cell; raw/nopre 32.8% (colors only); chat/pre 20.0%; **chat/nopre 0.0%** (the chat model opens "The/Your/Based…"). The PROVISIONAL preamble is **load-bearing**. 1,016 eligible change rows, 100 families.
- **Patch-control matrix (final position):** real-Δ median margin recovery **0.014** (0/1016 rows >0.5; direction 75.7%) — **FAILS** the ≥0.50 gate. Controls ~0.
- **Site decomposition:** all-positions patch = **1.000** (positive control); **edit-token-only = 0.983** (100% >0.5, 100% direction), per-cell 0.974–0.987, both strata, both query orders. post-edit-excl-edit 0.017. Distractor edit-site deltas are inert (99.2% answer retention). **Edit site PASSES** (one flagged exception: unrelated-real-delta control 0.169 > the §9 0.10 bound — user ruling pending).
- **Outcome:** substrate = **edit-token position, L20, raw/preamble primary cell** (no dataset regeneration needed; edit positions 74–98 are above the AV's position-50 floor). Recommended freeze pending user sign-off.

## 4. Exploratory arc (all committed; probe-only unless noted)

Reports: `2026-07-19_zeroshot_av_report.md`, `2026-07-19_position_sweep_report.md`, `2026-07-19_consequence_heldout_report.md`. Predictions were registered before each run.

1. **Zero-shot AV read** (`exploratory/zeroshot_av_edit_site.py`): released AV, no fine-tuning, edit-site delta → **new-value mention 74%** [64,84] vs shuffled 0% vs random 0%; h_cf state 100% new, h_base 100% old. **Token-identity readout** ("Final token X"), surrounding prose confabulated. All 1,450 explanations dumped; 20 random verbatim in the report.
2. **Old-value probe** (`exploratory/oldvalue_probe.py`): value_old **is** linearly in the delta — old~delta 0.96 (colors) / 0.76 (names) ≈ new~delta; single states give only their own endpoint; label-perm at chance. The AV's 0% zero-shot old-mention is a *decoding* limit, not information absence.
3. **Position sweep + discriminator** (`exploratory/probe_position_sweep.py`): scaled probe old≈new (colors 0.945, names 0.870, tighter CIs). **Value basis is position-invariant** — cross-position transfer 0.92–1.00 (edit-pos 3→122, delta and state). **Discriminator: edit-site delta is behaviorally blind** — target-vs-distractor AUC 0.51/0.47/0.49 (query_last/first/both), i.e. chance; edited value still decodable from distractor deltas (~0.95).
4. **Consequence layer-sweep + held-out values** (`exploratory/consequence_and_heldout.py` + `consequence_heldout_diag.py`):
   - **EXP-1:** no relevance *direction* at any layer (AUC~0.5); consequence is answer-change **magnitude**, forming **late** (norm-AUC L20 0.67 → L24 0.97 → L28 0.99; median target norm 154 vs distractor 47).
   - **EXP-2:** embedding-retrieval held-out 0.00 was a **wrong-target artifact** (seen values only 0.39 vs input embedding); decoder-free 1-NN shows **unseen values are consistently represented (0.89, chance 0.20)** — a general token code, not a per-value lookup; a closed-vocab decoder just can't *name* them.
   - **Method lesson:** both raw numbers were misleading (normalized away magnitude; wrong target space) — caught only by the direction-vs-magnitude split and a decoder-free control.

Plots: `plots/2026-07-19_stage_a_*.png`, `…_zeroshot_av_mention_rates.png`, `…_oldvalue_probe.png`, `…_position_transfer.png`, `…_target_distractor_auc.png`, `…_final_layer_sweep.png`, `…_heldout_value_retrieval.png`.

## 5. What it means (interpretation, kept separate)

- The only "difference" cleanly in the activation is the **token transition** (old→new), which is read robustly and generalizes to unseen values. There is **no separate behavioral-difference object** to probe: the behavior diff is either identical to the token (TARGET rows: edited value == queried answer) or a downstream, query-conditioned output change (late-final magnitude) that the AV can't see.
- This is **not** a "counterfactual datasets have no behavior diff" fact — it's *this design* not isolating one. **Derived-relation families** (edit a rule, query an inheriting entity whose name isn't in the edit) would put the consequence somewhere other than the edited token, making consequence-reading separable from token-naming. That is the highest-value new case type.

## 6. Open decisions / debts (for the user)

- **Stage A freezes** (substrate = edit-site/L20/raw-preamble; §9 unrelated-control 0.10 bound keep-or-rejustify; preamble sign-off — now empirically load-bearing).
- **Qwen3-8B retrospective still absent** on this instance (not in repo/home/scratchpad/attachment) — could not import; §11.2 provenance unsatisfied. Not fabricated. Re-share to import (narrative, not citable as result).
- Frozen-dataset open items unchanged: `NOT_IDENTIFIABLE_FROM_THIS_STATE` label; name-survivor review; slot2 R4/R6 retention.

## 7. Recommended next steps (not run)

1. **Derived-relation mini-set** + the same edit-site probe — the decisive test of whether *any* consequence is readable when it isn't just the token. (Highest value.)
2. **Within-prompt position sweep** — read the delta at *every* token position for fixed prompts (we only checked edit + final; interior positions unprobed, pre-edit are exactly zero). Cheap; closes the "did we check positions" gap.
3. For the new warm-start: follow `research/docs/NEW_WARMSTART_CRUXES.md` (cruxes, fixes, go/no-go probe checklist). Headline: leverage is derived-relation families + a magnitude/availability channel, not more transition data.

## 8. Provenance

All scripts under `research/stage_a/` and `research/exploratory/`; artifacts under `research/data/artifacts/v1/{stage_a,zeroshot_av,position_sweep,consequence_heldout}/` (large activation/logit caches gitignored as regenerable). Author/committer `senku14x <visheshgupta14x@gmail.com>`, Claude Fable 5 co-author.
