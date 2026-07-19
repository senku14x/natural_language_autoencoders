# J-space / NLA research project — session instructions

This directory holds a mechanistic-interpretability research project built on
top of the NLA scaffold in this repo (fork of
[kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)).
The root `CLAUDE.md` still governs anything touching `nla/`, `miles/`, configs,
or the training pipeline — do not edit the root `CLAUDE.md`, and do not edit
files under `miles/`.

## Read this first, every session

Before doing any research work in a session, read in order:

1. `research/docs/NLA_JSPACE_RESEARCH_CONTEXT.md` — durable methods context
   for NLA, J-lens/J-space, and the behavioral-difference project framing.
   Sections 4 (methodological takeaways: four-claims separation, evidence
   ladder, mandatory baselines) and 5 (project implications: gating order,
   boring alternatives, continue/pivot/stop rules) are binding on experiment
   design, not just background.
2. `research/docs/COUNTERFACTUAL_DIFFERENCE_NLA_V1_DECISION.md` — **the active
   research plan.** Finalized v1 scope: counterfactual difference verbalization
   on a controlled binding task (Qwen2.5-7B-Instruct, layer 20, final
   position), claim ladder L0–L6, execution stages A–D, minimum controls, and
   provisional pass/fail gates. Work follows this document, as amended by:
   `research/docs/CAPTION_SCHEMA_AMENDMENT_1.md` — supersedes the caption
   schema (§3), verbalization gates (§9), and Level-4 wording: two caption
   axes (format: arrow/sentence × content: transition/entity), content gated
   on Stage B mean-hierarchy results, format tested as a 2×2 with
   initialization, registered predictions in its §9.
3. `research/docs/COUNTERFACTUAL_DIFFERENCE_NLA_EXTENSION_BACKLOG.md` — gated
   extension menu (availability/propagation, distributional changes, derived
   relations, abstention, magnitude). §1 scope rule: extensions are triggered
   by evidence, never opened preemptively. §2 lists refinements that belong in
   v1 now.
4. `research/docs/PARAMETRIC_PRIOR_ENTITY_SWAP_EXTENSION.md` — later
   public-entity branch (parametric-prior entity swaps, crossed
   name × contextual-profile design, identity-to-lookup baseline). Not part of
   v1; common-name identity binding is in v1 itself.
5. `research/docs/AO_NLA_JSPACE_METHODS_LEARNINGS.md` — companion extended
   methods review referenced by the context doc. **Not yet provided** — if a
   task depends on it, ask for it rather than guessing its contents.
6. `research/ARTIFACTS.md` — running log of reports and result analyses; check
   it to see what has already been run and found.

## Repo facts that constrain the plan (verified in code, 2026-07-18)

- Released Qwen NLA: `kitft/nla-qwen2.5-7b-L20-av` / `-ar` on HF
  (`kitft/nla-models` collection), target Qwen2.5-7B-Instruct, layer 20/28,
  `d_model` 3584. Sidecar `nla_meta.yaml` ships prompt template, injection
  token (`㈎` U+320E, id 149705, inside `<concept>…</concept>`), and scales —
  load from sidecar, never hardcode (root `CLAUDE.md` invariant).
- **AV input normalizes**: `injection_scale: 150.0` = the L2 norm every
  injected vector is rescaled to (`nla/config.py`). Delta magnitude is
  destroyed at the AV interface; `NO_CHANGE` must be readable from direction
  alone; near-zero deltas need an epsilon guard before rescale.
- **AR / loss are direction-only**: `mse_scale` normalizes both prediction and
  gold before MSE (= `2(1 − cos)`).
- Released AV/AR trained on fineweb-style *document* activations at token
  positions ≥ 50 (`_MIN_POSITION` in `nla/datagen/stage0_extract.py`), no chat
  template. Short task prompts are OOD in both position and style — the
  ordinary-activation parity check (Stage B.1) is load-bearing before any
  delta result is interpreted.

## Required reading (external sources)

Papers (read before designing experiments that depend on them):

| Source | Link | Role |
|---|---|---|
| NLA paper | https://transformer-circuits.pub/2026/nla/ | Verbalizer/reconstructor architecture, activation injection, RL training pattern we reuse. |
| Workspace paper | https://transformer-circuits.pub/2026/workspace/ | Main text for the Jacobian lens (J-lens) and the workspace concept. |
| Workspace appendix: "Extending the Jacobian lens to multi-token concepts" | (appendix of the workspace paper above) | Template lens → oracle lens, four training stages. Per-paper priorities are set by `research/PLAN.md`. |
| Activation Oracles | https://arxiv.org/abs/2512.15674 · https://alignment.anthropic.com/2025/activation-oracles/ | Karvonen et al. 2025 — LatentQA-style general-purpose activation explainers; the AO baseline family. |
| Building Better Activation Oracles | https://arxiv.org/abs/2606.02609 | Bauer et al. 2026 — on-policy AO training, improved injection formula, multi-layer feeds, and AObench (eval suite). |
| Current Activation Oracles Are Hard To Use | https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use | Practitioner-side failure modes of current AOs; motivates the usability bar. |
| NLA robustness (Zhang & Turner) | https://turntrout.com/natural-language-autoencoder-robustness | Implausible-init NLA reaches FVE 0.68 vs 0.70 control while plausibility stays 0.7% vs 7.6% — reconstruction cannot certify caption truth. Grounds the exact-grading + activation-dependence controls. |
| Internal chain-of-thought eval (oakhu & ryan_greenblatt) | https://www.lesswrong.com/posts/QQQAcKuWK6k98FivY/can-activation-verbalizers-surface-an-internal-chain-of-1 | Qwen2.5-7B NLA showed FVU > 1 on OOD short prompts; poor cross-layer generalization; final-token most informative. Grounds the Stage B.1 parity gate. |
| Cycle-consistent activation oracles (Chalnev) | https://www.lesswrong.com/posts/Nf2sKaNNdxE2ssxbp/cycle-consistent-activation-oracles-1 | Probes beat decoder extraction by 20–49pp; cycle consistency ≠ faithfulness. Grounds the probe-vs-AV interpretation rule (§10 of the v1 doc). |
| NLA thought anchors (Realmbird) | https://www.lesswrong.com/posts/6HnnMHRoJLff46kgw/nla-thought-anchors | GSM8K pipeline on our exact kitft L20 checkpoints; answer-containment and reconstruction-loss observations. |
| NLA length penalties (loops) | https://www.lesswrong.com/posts/NazprRfWJ4qkwcSro/nla-explanations-can-be-shortened-without-harming | 28–40% shorter explanations at ~0.015 FVE cost — heavy redundancy; supports rigid one-line captions over prose. |
| Hiding from an NLA (Prabhu) | https://www.lesswrong.com/posts/3oRQxpxn4LiWKwbeQ/can-you-hide-from-a-natural-language-autoencoder | 81.4% explanation-flip with 99.6% behavior preservation via prefix-tuning — verbalizers are attackable; blocks any monitor claim without adversarial eval. |
| NLAs as summarizers (Anurin) | https://www.lesswrong.com/posts/4nZwNPyfoadAxPH98/natural-language-autoencoders-are-summarizers-but-do-they | Snippet ablation drops FVE 0.77 → −0.76; cross-family transfer 0.51–0.68. Grounds the text-inversion / decoder-prior control family. |
| Matryoshka NLAs | (PDF not yet in repo — no public copy found; ask user) | Referenced in session instructions; likely the importance-ordered-explanation line behind the syvb nanoNLA collection. |

Code scaffolds:

| Repo | Link | Role |
|---|---|---|
| natural_language_autoencoders | https://github.com/kitft/natural_language_autoencoders | Main scaffold (this repo's upstream): data gen, SFT, GRPO RL, activation injection via `input_embeds`, checkpoint conversion. |
| nla-inference | https://github.com/kitft/nla-inference | Lightweight inference client; reference for injection mechanics. |
| jacobian-lens | https://github.com/anthropics/jacobian-lens | Apache-2.0 J-lens reference (fit/apply/visualize on HF decoders). Baseline + layer-selection sanity checks; not on the critical path. |
| EasyNLA (asherps) | https://github.com/asherps/EasyNLA | Distributed nanoNLA fork, Qwen3-8B L24. **Different injection default from official**: Karvonen-style additive norm-matched injection at an early layer, not embedding replacement at the marker token. |
| nanoNLA (ceselder) | https://github.com/ceselder/nanoNLA | Minimal single-GPU NLA reimplementation (Qwen3-8B, LoRA, HF generate). Same additive-injection deviation as EasyNLA. |
| nla-thought-anchors (Realmbird) | https://github.com/Realmbird/nla-thought-anchors | Working SGLang pipeline for our exact kitft L20 AV/AR checkpoints (radix cache off, batch/OOM workarounds documented). No license file — ask before reusing code. |
| syvb HF collection | https://huggingface.co/syvb | Qwen3-8B L24 nanoNLA AV/AR/RL-LoRA checkpoints plus released completions/results datasets — an independent checkpoint family for later replication. |

Interactive demos (intuition, not evidence): https://www.neuronpedia.org/jlens
and the NLA demo linked from
https://www.anthropic.com/research/natural-language-autoencoders

## Working conventions

- **Branch**: all work goes on `counterfactual_nla`. Never push to any other
  branch without the user's explicit permission.
- **Commits**: author/committer `senku14x <visheshgupta14x@gmail.com>`, with a
  `Co-Authored-By` trailer naming the Claude model that did the work.
- **Artifacts**: every research report or result analysis gets a dated entry
  in `research/ARTIFACTS.md` (format described there). Plots go under
  `research/plots/` and are linked from the entry. Prefer a plot over a table
  of numbers whenever the shape of the data carries the point.
- **When unsure, ask the user.** In particular: scope changes, anything
  destructive, and any deviation from the gating order below.

## Evidence discipline (summary — full version in the context doc)

- Keep observations, interpretations, and speculation explicitly separated in
  every writeup; scope claims to the tested model/layer/position/prompts.
- Never collapse the four claims: decodability ≠ reconstruction ≠ causal use ≠
  semantic faithfulness (context doc §4.1).
- Execution order is gated (context doc §5.8): freeze the causal patching
  setup → run the patch-control matrix → gold-caption reconstruction gate →
  only then train the AV → activation-dependence controls → held-out splits.
  Do not train an AV before the gold-caption/AR gate passes.
- Every positive result must be run against the boring alternatives in §5.4
  (metadata lookup, text inversion, decoder-prior leakage, token promotion,
  norm damage, template leakage) before being reported as more than an
  observation.
- Measurement instruments (NLA, AO, LLM judges) are untrusted until they pass
  a relevant positive control; a null without a sensitivity check is not
  evidence of absence.
