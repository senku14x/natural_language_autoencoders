"""Amendment 1 renderer tests: round-trips, cross-format canonicalization,
strictness, and the no-magnitude-language hard rule."""

import pytest

from ctf_data import rich_captions as rc
from ctf_data.pairs import build_family, build_null_family


def _s_family():
    return build_family(stratum="S", template_id="s1", split="train",
                        split_axis="iid", entities=("jeck", "florp"),
                        A="brown", B="coral", d="mint", dp="gray", bundle_idx=0)


def _n_family():
    return build_family(stratum="N", template_id="n1", split="train",
                        split_axis="iid", entities=(None, None),
                        A="Chen", B="Omar", d="Karachi", dp="Lahore",
                        bundle_idx=0)


def test_renders_match_amendment_examples():
    r1, r2, r3, r4, r5, r6 = _s_family()
    c1 = rc.render_all(r1)
    assert c1["caption_arrow_transition"] == "brown -> coral"
    assert c1["caption_sentence_transition"] == "The value changes from brown to coral."
    assert c1["caption_arrow_entity"] == "jeck: brown -> coral"
    assert c1["caption_sentence_entity"] == \
        "The color assigned to jeck changes from brown to coral."
    c3 = rc.render_all(r3)  # other_slot distractor: florp edited, jeck queried
    assert c3["caption_arrow_transition"] == "NO_CHANGE"
    assert c3["caption_arrow_entity"] == "florp: mint -> gray | queried: jeck unaffected"
    assert c3["caption_sentence_entity"] == \
        "The color assigned to florp changes from mint to gray; the queried entity jeck is unaffected."


def test_crossed_query_distractor_names_queried_entity():
    _, r2, *_ = _s_family()  # jeck edited, florp queried
    c2 = rc.render_all(r2)
    assert c2["caption_arrow_entity"] == "jeck: brown -> coral | queried: florp unaffected"
    assert c2["caption_arrow_transition"] == "NO_CHANGE"


def test_n_stratum_uses_slot_words():
    r1, r2, r3, r4, _, _ = _n_family()
    assert rc.render_all(r1)["caption_arrow_entity"] == "name: Chen -> Omar"
    assert rc.render_all(r1)["caption_sentence_entity"] == \
        "The name changes from Chen to Omar."
    assert rc.render_all(r3)["caption_sentence_entity"] == \
        "The city changes from Karachi to Lahore; the queried name is unaffected."


def test_null_renders_plain_no_change_in_all_formats():
    (s,) = build_null_family(stratum="S", template_id="s1", split="train",
                             entities=("jeck", "florp"), A="brown", d="mint",
                             bundle_idx=0)
    c = rc.render_all(s)
    assert c["caption_arrow_transition"] == "NO_CHANGE"
    assert c["caption_arrow_entity"] == "NO_CHANGE"
    assert c["caption_sentence_transition"] == "The queried value is unchanged."
    assert c["caption_sentence_entity"] == "The queried value is unchanged."


def test_round_trip_and_cross_canonical_for_every_cell():
    for fam in (_s_family(), _n_family()):
        for sem in fam:
            assert rc.verify_row(sem) == [], sem.cell


@pytest.mark.parametrize("parser,bad", [
    (rc.parse_arrow_transition, "brown ->  coral"),
    (rc.parse_arrow_transition, "NO CHANGE"),
    (rc.parse_sentence_transition, "The value changes from brown to coral"),
    (rc.parse_sentence_transition, "the value changes from brown to coral."),
    (rc.parse_arrow_entity, "jeck brown -> coral"),
    (rc.parse_arrow_entity, "jeck: brown -> coral | queried: florp"),
    (rc.parse_sentence_entity, "The colour assigned to jeck changes from brown to coral."),
    (rc.parse_sentence_entity, "The color assigned to jeck changes from brown to coral;"),
])
def test_parser_strictness(parser, bad):
    with pytest.raises(ValueError):
        parser(bad)


def test_sha_columns_present_and_distinct():
    r1 = _s_family()[0]
    c = rc.render_all(r1)
    shas = [c[k + "_sha256"] for k in rc.CAPTION_COLUMNS]
    assert all(len(s) == 64 for s in shas)
    assert len(set(shas)) == 4  # four distinct renders for a target row


def test_transition_level_collapses_distractor_to_no_change():
    _, r2, r3, _, _, _ = _s_family()
    for sem in (r2, r3):
        c = rc.render_all(sem)
        at = rc.parse_arrow_entity(c["caption_arrow_entity"])
        # entity level keeps the edit; transition level must not
        assert at.kind == "distractor_change"
        assert c["caption_arrow_transition"] == "NO_CHANGE"
        assert rc.canonical_transition(at) == ("no_change",)
