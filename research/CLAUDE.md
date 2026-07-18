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
2. `research/docs/AO_NLA_JSPACE_METHODS_LEARNINGS.md` — companion extended
   methods review referenced by the context doc. **Not yet provided** — if a
   task depends on it, ask for it rather than guessing its contents.
3. `research/PLAN.md` — the active research plan. **Not yet written**; it will
   be added when the user shares it. Until then, do not commit to a specific
   experimental direction beyond what the context doc establishes.
4. `research/ARTIFACTS.md` — running log of reports and result analyses; check
   it to see what has already been run and found.

## Required reading (external sources)

Papers (read before designing experiments that depend on them):

| Source | Link | Role |
|---|---|---|
| NLA paper | https://transformer-circuits.pub/2026/nla/ | Verbalizer/reconstructor architecture, activation injection, RL training pattern we reuse. Read fully. |
| Workspace paper | https://transformer-circuits.pub/2026/workspace/ | Main text for the Jacobian lens (J-lens) and the workspace concept. |
| Workspace appendix: "Extending the Jacobian lens to multi-token concepts" | (appendix of the workspace paper above) | **Primary spec for this build** — template lens → oracle lens, four training stages. Read very carefully. |
| Activation Oracles | https://arxiv.org/abs/2512.15674 · https://alignment.anthropic.com/2025/activation-oracles/ | Karvonen et al. 2025 — LatentQA-style general-purpose activation explainers; the AO baseline family. |
| Building Better Activation Oracles | https://arxiv.org/abs/2606.02609 | Bauer et al. 2026 — on-policy AO training, improved injection formula, multi-layer feeds, and AObench (eval suite). |
| Current Activation Oracles Are Hard To Use | https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use | Practitioner-side failure modes of current AOs; motivates the usability bar. |

Code scaffolds:

| Repo | Link | Role |
|---|---|---|
| natural_language_autoencoders | https://github.com/kitft/natural_language_autoencoders | Main scaffold (this repo's upstream): data gen, SFT, GRPO RL, activation injection via `input_embeds`, checkpoint conversion. |
| nla-inference | https://github.com/kitft/nla-inference | Lightweight inference client; reference for injection mechanics. |
| jacobian-lens | https://github.com/anthropics/jacobian-lens | Apache-2.0 J-lens reference (fit/apply/visualize on HF decoders). Baseline + layer-selection sanity checks; not on the critical path. |

Interactive demos (intuition, not evidence): https://www.neuronpedia.org/jlens
and the NLA demo linked from
https://www.anthropic.com/research/natural-language-autoencoders

## Working conventions

- **Branch**: all work goes on `claude/mech-interp-research-a5hkvv`. Never
  push to any other branch without the user's explicit permission.
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
