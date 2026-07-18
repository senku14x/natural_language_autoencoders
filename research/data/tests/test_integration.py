"""End-to-end micro-run through the real tokenizer.

Skipped unless the Qwen tokenizer is reachable: set NLA_DATA_TOKENIZER to a
local path (or HF id if the hub is reachable). This is the test that exercises
tokenization, edit_pos discovery, the causal-masking check, and caption
round-trips on real ids.
"""

import os

import numpy as np
import pytest

TOK_SRC = os.environ.get("NLA_DATA_TOKENIZER", "")
pytestmark = pytest.mark.skipif(
    not TOK_SRC, reason="NLA_DATA_TOKENIZER not set (hub blocked or no local copy)")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_SRC)


def test_micro_end_to_end(tok):
    from ctf_data import audit as audit_mod
    from ctf_data import captions as cap
    from ctf_data import invariants as inv
    from ctf_data.inventory import PREAMBLE_TEXT
    from ctf_data.pairs import build_family, expand_variants
    from ctf_data.pipeline import chat_wrap_parts, tokenize_rows
    from ctf_data.splits import build_pools

    audits = audit_mod.audit_values(tok, {
        "color": ["red", "green", "blue", "gold", "purple", "orange"],
        "name": ["Alice", "Priya", "John", "Maria"],
        "city": ["Paris", "Berlin", "Tokyo"],
        "nonce": ["dax", "wug"]})
    colors = [a.value for a in audit_mod.survivors(audits, "color")]
    assert len(colors) >= 4, "basic colors must be single-token"

    rng = np.random.default_rng(0)
    pools = build_pools(
        rng, colors=colors,
        names=[a.value for a in audit_mod.survivors(audits, "name")],
        cities=[a.value for a in audit_mod.survivors(audits, "city")],
        nonces=["dax", "wug"],
        holdout_cfg={"colors": 1, "color_transitions": 1, "nonces": 0,
                     "names": 1, "name_transitions": 1},
        s_templates=["s1"], s_context_templates=["s4"],
        n_templates=["n1"], n_context_templates=["n4"])

    A, B, d, dp = pools.train_colors[:4]
    fam = build_family(stratum="S", template_id="s1", split="train",
                       split_axis="iid", entities=("dax", "wug"),
                       A=A, B=B, d=d, dp=dp, bundle_idx=0)
    rows = []
    for s in fam:
        rows.extend(expand_variants(s, PREAMBLE_TEXT))
    prefix, suffix = chat_wrap_parts(tok)
    rows = tokenize_rows(rows, tok, prefix, suffix, pools, PREAMBLE_TEXT)

    answer_ids = {a.value: a.leading_space_id
                  for kind in ("color", "name", "city")
                  for a in audit_mod.survivors(audits, kind)}
    for row in rows:
        problems = inv.check_variant_row(row, answer_ids=answer_ids,
                                         audited_single_token=set(answer_ids))
        assert problems == [], f"{row.pair_id}: {problems}"
        parsed = cap.parse(row.semantic.caption())
        assert cap.render(parsed) == row.semantic.caption()

    by_key = {(r.semantic.semantic_id, r.query_order, r.prompt_format, r.preamble): r
              for r in rows}
    assert inv.check_crossings(by_key) == []

    # chat splice must equal transformers' own chat rendering
    chat_row = next(r for r in rows if r.prompt_format == "chat")
    ref = tok.apply_chat_template([{"role": "user", "content": chat_row.base_content}],
                                  tokenize=True, add_generation_prompt=True)
    assert ref == chat_row.base_input_ids
