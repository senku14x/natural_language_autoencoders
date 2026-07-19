# Counterfactual Difference NLA v1 — dataset generation (tokenizer-only)

Builds the frozen prompt-pair dataset + manifest consumed by the GPU stage.
No model weights are loaded anywhere in this stage. Everything is
reproducible from a config file + seed; the "frozen artifact" is
(config + seed + code commit + manifest) — `pairs.parquet` regenerates
bit-identically from those, and the GPU box must verify `config_hash` and
counts against `manifest.json` after regenerating.

## Run

```bash
pip install "transformers>=4.45" tokenizers huggingface_hub numpy pyarrow pyyaml pytest

# audit only (survivor tables, gates):
python research/data/generate.py --config research/data/configs/v1.yaml --stage audit

# 20-family smoke run (hand-readable manifest on stdout):
python research/data/generate.py --config research/data/configs/smoke.yaml

# full v1:
python research/data/generate.py --config research/data/configs/v1.yaml

# tests (tokenizer-independent ones always run):
pytest research/data/tests -q
# integration test additionally needs:
NLA_DATA_TOKENIZER=Qwen/Qwen2.5-7B-Instruct pytest research/data/tests -q
```

## STATUS: blocked on tokenizer access

This session's egress policy denies `huggingface.co` (CONNECT 403 at the
proxy — an org/network policy, not an auth problem; an HF token does not
help because the TCP tunnel itself is refused, and Qwen2.5-7B-Instruct is
public anyway). Two unblock paths:

1. **Allowlist** `huggingface.co` (and the LFS CDN hosts `*.hf.co`) in the
   Claude Code environment's network settings, then rerun the pinned-SHA
   fetch + audit + smoke + full.
2. **Local fetch + upload**: on any machine with network, run

   ```python
   from huggingface_hub import snapshot_download, HfApi
   print("sha:", HfApi().model_info("Qwen/Qwen2.5-7B-Instruct").sha)
   snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                     allow_patterns=["tokenizer*", "vocab*", "merges*",
                                     "special_tokens_map.json"],
                     local_dir="qwen25_tok")
   ```

   and share the printed SHA plus the files in `qwen25_tok/`. The pipeline
   then loads them locally; the SHA goes into `tokenizer.revision`.

`configs/*.yaml` carry `revision: FILL_ME` until the SHA is known — the
pipeline refuses to run on a placeholder (irreproducible revisions are the
first thing the V1 doc's Level-0 gate forbids).

## Layout

```
configs/         v1.yaml, smoke.yaml
ctf_data/        inventory, audit, templates, captions, splits, pairs,
                 invariants, writer, pipeline
generate.py      argparse CLI
tests/           unit tests (incl. deliberately-broken fixtures) + integration
out/             generated artifacts (gitignored): pairs.parquet,
                 manifest.json, values_audit.json, rejections.json
```

## Design decisions that deviate from or refine the specs — flagged for review

1. **Slot2 TARGET rows (R4/R6) are generated**, including city→city captions
   in stratum N (a deviation from the V1 doc's minimal cell list). Reason:
   with query-crossed distractors only, every slot2-query state would carry
   the NO_CHANGE label, and query identity — readable from a final-position
   delta — would leak the label. R4/R6 give both query states both labels
   (`test_query_label_decorrelation`). Config-free kill switch: drop the R4/R6
   rows at the GPU stage by filtering `edited_slot == "slot2"` if you decide
   against them; the crossing/reverse links keep this filter clean.
2. **Both distractor flavors** (`crossed_query`, `other_slot`) exist so
   NO_CHANGE decorrelates from "which slot was edited" as well.
3. **Family-atomic rejection**: one violating row drops its whole family
   (balance and crossed/reverse partners stay intact). The rejection table
   records the violating rows; surplus quotas absorb the loss.
4. **`NOT_IDENTIFIABLE_FROM_THIS_STATE`** is parser-valid but never emitted
   (renderer refuses it). Open decision per spec §5 — awaiting user call.
5. **Preamble text is PROVISIONAL** (inventory.py) — neutral registry-prose,
   no overlap with any answer value, pushes `final_pos` ≥ 52. Swap requires
   only regeneration; awaiting user sign-off.
6. **Cities are not partitioned** — they are context/distractor-query answers,
   not a v1 generalization axis. (Names/colors/nonces/transitions are.)
7. **A `dev` split exists** alongside train (iid families) — the V1 doc's
   Stage A/B site-freezing and format-freezing screens need it.
8. **Chat variant**: the raw content (including `Answer:`) is wrapped
   unchanged as a user message with `add_generation_prompt=True`; the splice
   is asserted equal to `apply_chat_template` output. Encoding everywhere is
   `add_special_tokens=False`; the chat wrapper carries its own specials.
9. **Held-out transitions are held out as unordered pairs** (both directions
   travel together), stronger than the spec's "unseen ordered pair" and
   required by its own reverse-edges-stay-together rule.
10. **Storage**: Arrow `list<int32>` columns in zstd parquet (typed arrays
    with Arrow's offsets buffers; layout documented in the manifest), matching
    the repo's pyarrow convention rather than a bespoke flat-binary format.

## Amendment 1 caption columns (2026-07-18)

Per `research/docs/CAPTION_SCHEMA_AMENDMENT_1.md`, every row carries four
deterministic caption renders — `{arrow, sentence} × {transition, entity}` —
as `caption_arrow_transition`, `caption_sentence_transition`,
`caption_arrow_entity`, `caption_sentence_entity` (each with a `_sha256`
twin). Renderer version `amendment1-r1` (`ctf_data/rich_captions.py`);
round-trip and cross-format canonical equality are verified for every
semantic pair at generation time. The legacy `caption` column is the
unchanged pre-amendment single-field render. Content/format selection is a
training-time column choice; the entity-content decision is gated on Stage B.

**Resolved (amendment §11.1, renderer `amendment1-r2`):** distractor
entity-content forms carry the behavioral claim only —
"`| answer unaffected`" / "; the queried answer is unaffected." — grounded
by Stage A. The queried-entity NAME is a reserved slot, licensable only by
the Stage-B queried-entity probe (can query identity be decoded from
Δ_final across edit-matched crossed pairs?). The r1 naming forms are now
parse errors by design.

## Schema notes

`value_old`/`value_new` always describe THE EDIT (what changed in the
prompt); `answer_old`/`answer_new` describe the expected ANSWER — captions
derive from answers only (single code path), so DISTRACTOR rows caption
NO_CHANGE while still recording their edit. `value_distractor` is the base
value of the non-queried slot (the plausible-confusion candidate);
`answer_token_id_*` are leading-space single-token ids (the raw-format
candidates; per-form ids for all four surface forms live in
`values_audit.json` for the GPU stage's chat-format screens).
`crossed_with` / `reverse_of` link semantic ids for the causal-masking check
and paired analyses. GPU-stage fields (`behavioral_screen`,
`oracle_patch_metrics`, `delta_norms`) are explicit nulls.
