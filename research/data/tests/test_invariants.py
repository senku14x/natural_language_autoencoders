"""Invariant tests with deliberately-broken fixtures (spec §8).

Fixtures are synthetic — no tokenizer needed. A clean row passes; each broken
fixture must be flagged with the matching reason.
"""

import pytest

from ctf_data import invariants as inv
from ctf_data.pairs import Semantic, VariantRow

ANSWER_IDS = {"red": 1001, "green": 1002, "blue": 1003, "gold": 1004,
              "Alice": 2001, "Priya": 2002, "Paris": 3001, "Berlin": 3002}
AUDITED = set(ANSWER_IDS)


def mk_sem(**kw):
    base = dict(
        semantic_id="sem1", family_id="fam1", stratum="S", cell="TARGET_EDIT",
        distractor_flavor="", split="train", split_axis="iid", template_id="s1",
        entity_slot1="dax", entity_slot2="wug", base_vals=("red", "blue"),
        cf_vals=("green", "blue"), edited_slot="slot1", queried_slot="slot1",
        value_old="red", value_new="green", value_distractor="blue",
        answer_old="red", answer_new="green")
    base.update(kw)
    return Semantic(**base)


def mk_row(sem=None, base_ids=None, cf_ids=None, edit_pos=3, **kw):
    sem = sem or mk_sem()
    base_ids = base_ids if base_ids is not None else [10, 11, 12, 1001, 14, 15]
    cf_ids = cf_ids if cf_ids is not None else [10, 11, 12, 1002, 14, 15]
    fields = dict(
        pair_id="p1", semantic=sem, query_order="query_last", prompt_format="raw",
        preamble=False, base_prompt="x", cf_prompt="y", base_content="x",
        cf_content="y", base_input_ids=base_ids, cf_input_ids=cf_ids,
        edit_pos=edit_pos, final_pos=len(base_ids) - 1,
        edit_token_decoded_base=" red", edit_token_decoded_cf=" green",
        final_token_decoded=":")
    fields.update(kw)
    return VariantRow(**fields)


def check(row):
    return inv.check_variant_row(row, answer_ids=ANSWER_IDS,
                                 audited_single_token=AUDITED)


def test_clean_row_passes():
    assert check(mk_row()) == []


def test_len_mismatch_rejected():                       # invariant 1
    row = mk_row(cf_ids=[10, 11, 12, 1002, 14])
    assert any("len_mismatch" in p for p in check(row))


def test_two_diffs_rejected():                          # invariant 2
    row = mk_row(cf_ids=[10, 99, 12, 1002, 14, 15])
    assert any("diff_count=2" in p for p in check(row))


def test_zero_diffs_on_edit_cell_rejected():            # invariant 2
    row = mk_row(cf_ids=[10, 11, 12, 1001, 14, 15])
    assert any("diff_count=0" in p for p in check(row))


def test_edit_pos_mismatch_rejected():                  # invariant 3
    row = mk_row(edit_pos=2)
    assert any("edit_pos_mismatch" in p for p in check(row))


def test_wrong_decoded_edit_token_rejected():           # invariant 4
    row = mk_row(edit_token_decoded_base=" blue")
    assert any("edit_decode_base" in p for p in check(row))


def test_final_pos_must_be_last_index():                # invariant 5
    row = mk_row(final_pos=2)
    assert any("final_pos" in p for p in check(row))


def test_unaudited_answer_value_rejected():             # invariant 6
    row = mk_row(sem=mk_sem(answer_new="chartreuse", value_new="chartreuse"),
                 edit_token_decoded_cf=" chartreuse")
    assert any("not_audited" in p for p in check(row))


def test_equal_answers_on_transition_cell_rejected():   # invariant 6
    row = mk_row(sem=mk_sem(answer_new="red", value_new="red"),
                 cf_ids=[10, 11, 12, 1001, 14, 15],
                 edit_token_decoded_cf=" red")
    assert any("transition_cell_with_equal_answers" in p for p in check(row))


def test_null_aa_with_diff_rejected():
    sem = mk_sem(cell="NULL_AA", cf_vals=("red", "blue"), value_new="red",
                 answer_new="red", plumbing_only=True)
    row = mk_row(sem=sem, edit_token_decoded_cf=" red")  # cf differs -> violation
    assert any("null_pair_has_diffs" in p for p in check(row))


def test_null_aa_clean_passes():
    sem = mk_sem(cell="NULL_AA", cf_vals=("red", "blue"), value_new="red",
                 answer_new="red", plumbing_only=True)
    row = mk_row(sem=sem, cf_ids=[10, 11, 12, 1001, 14, 15],
                 edit_token_decoded_cf=" red")
    assert check(row) == []


def test_crossing_prefix_mismatch_detected():           # invariant 8
    s1 = mk_sem(semantic_id="a", crossed_with="b")
    s2 = mk_sem(semantic_id="b", crossed_with="a", queried_slot="slot2",
                cell="DISTRACTOR_EDIT", distractor_flavor="crossed_query",
                answer_old="blue", answer_new="blue", value_distractor="red")
    r1 = mk_row(sem=s1)
    r2 = mk_row(sem=s2, base_ids=[10, 99, 12, 1001, 14, 16],
                cf_ids=[10, 99, 12, 1002, 14, 16],
                edit_token_decoded_base=" red", edit_token_decoded_cf=" green")
    key = lambda r: (r.semantic.semantic_id, r.query_order, r.prompt_format, r.preamble)
    problems = inv.check_crossings({key(r1): r1, key(r2): r2})
    assert any("crossed_prefix_differs" in p for p in problems)


def test_crossing_identical_prefix_passes():
    s1 = mk_sem(semantic_id="a", crossed_with="b")
    s2 = mk_sem(semantic_id="b", crossed_with="a", queried_slot="slot2",
                cell="DISTRACTOR_EDIT", distractor_flavor="crossed_query",
                answer_old="blue", answer_new="blue", value_distractor="red")
    r1 = mk_row(sem=s1)
    r2 = mk_row(sem=s2, base_ids=[10, 11, 12, 1001, 14, 16],
                cf_ids=[10, 11, 12, 1002, 14, 16])
    key = lambda r: (r.semantic.semantic_id, r.query_order, r.prompt_format, r.preamble)
    assert inv.check_crossings({key(r1): r1, key(r2): r2}) == []


class FakePools:
    heldout_colors = ["gold"]
    heldout_names = []
    heldout_nonces = ["blicket"]
    heldout_color_pairs = [("red", "blue"), ("blue", "red")]
    heldout_name_pairs = []


def _rec(**kw):
    base = dict(family_id="f1", semantic_id="s1", split="train", cell="TARGET_EDIT",
                stratum="S", value_old="red", value_new="green",
                value_distractor="blue", answer_old="red", answer_new="green",
                entity_target="dax", entity_distractor="wug", reverse_of="")
    base.update(kw)
    return base


def test_heldout_value_in_train_detected():             # invariant 9
    recs = [_rec(value_new="gold", answer_new="gold")]
    assert any("heldout_value_in_train" in p
               for p in inv.check_dataset(recs, FakePools))


def test_family_in_two_splits_detected():               # invariant 10
    recs = [_rec(semantic_id="s1"), _rec(semantic_id="s2", split="dev")]
    assert any("family_in_two_splits" in p
               for p in inv.check_dataset(recs, FakePools))


def test_reverse_split_mismatch_detected():             # invariant 11
    recs = [_rec(semantic_id="fwd", family_id="f1"),
            _rec(semantic_id="rev", family_id="f2", split="dev",
                 cell="REVERSE", value_old="green", value_new="red",
                 answer_old="green", answer_new="red", reverse_of="fwd")]
    assert any("reverse_split_mismatch" in p
               for p in inv.check_dataset(recs, FakePools))


def test_heldout_transition_in_train_detected():
    recs = [_rec(value_old="red", value_new="blue", answer_old="red",
                 answer_new="blue")]
    assert any("heldout_transition_in_train" in p
               for p in inv.check_dataset(recs, FakePools))


def test_clean_dataset_passes():
    recs = [_rec()]
    assert inv.check_dataset(recs, FakePools) == []
