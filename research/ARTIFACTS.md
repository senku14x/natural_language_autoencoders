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
