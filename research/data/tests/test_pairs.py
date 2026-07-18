"""Family construction semantics — no tokenizer needed."""

from ctf_data.pairs import build_family, build_null_family, expand_variants


def _fam():
    return build_family(stratum="S", template_id="s1", split="train",
                        split_axis="iid", entities=("dax", "wug"),
                        A="red", B="green", d="blue", dp="gold", bundle_idx=0)


def test_cell_structure_and_balance():
    fam = _fam()
    cells = [s.cell for s in fam]
    assert cells.count("TARGET_EDIT") == 2
    assert cells.count("DISTRACTOR_EDIT") == 2
    assert cells.count("REVERSE") == 2
    assert len({s.family_id for s in fam}) == 1
    assert len({s.semantic_id for s in fam}) == 6


def test_captions_follow_answers():
    fam = _fam()
    r1, r2, r3, r4, r5, r6 = fam
    assert r1.caption() == "PREDICTED_OUTPUT_CHANGE: red -> green"
    assert r2.caption() == "PREDICTED_OUTPUT_CHANGE: NO_CHANGE"
    assert r3.caption() == "PREDICTED_OUTPUT_CHANGE: NO_CHANGE"
    assert r4.caption() == "PREDICTED_OUTPUT_CHANGE: blue -> gold"
    assert r5.caption() == "PREDICTED_OUTPUT_CHANGE: green -> red"
    assert r6.caption() == "PREDICTED_OUTPUT_CHANGE: gold -> blue"


def test_query_label_decorrelation():
    """Both queried slots must carry both caption labels within a family —
    otherwise query identity (readable from the final position) leaks the
    label and the NO_CHANGE cell is a shortcut."""
    fam = _fam()
    by_slot = {}
    for s in fam:
        label = "no_change" if s.answer_old == s.answer_new else "transition"
        by_slot.setdefault(s.queried_slot, set()).add(label)
    assert by_slot["slot1"] == {"transition", "no_change"}
    assert by_slot["slot2"] == {"transition", "no_change"}


def test_crossed_and_reverse_links():
    r1, r2, r3, r4, r5, r6 = _fam()
    assert r1.crossed_with == r2.semantic_id and r2.crossed_with == r1.semantic_id
    assert r3.crossed_with == r4.semantic_id and r4.crossed_with == r3.semantic_id
    assert r5.reverse_of == r1.semantic_id
    assert r6.reverse_of == r4.semantic_id
    # crossed partners share the same edit
    assert (r1.value_old, r1.value_new) == (r2.value_old, r2.value_new)
    assert (r3.value_old, r3.value_new) == (r4.value_old, r4.value_new)


def test_distractor_flavors_present():
    fam = _fam()
    flavors = {s.distractor_flavor for s in fam if s.cell == "DISTRACTOR_EDIT"}
    assert flavors == {"crossed_query", "other_slot"}


def test_value_distractor_is_static_nonqueried_base_value():
    r1, r2, r3, r4, _, _ = _fam()
    assert r1.value_distractor == "blue"   # query slot1, other slot holds blue
    assert r2.value_distractor == "red"    # query slot2, other slot holds red
    assert r3.value_distractor == "blue"
    assert r4.value_distractor == "red"


def test_null_family():
    (s,) = build_null_family(stratum="S", template_id="s1", split="train",
                             entities=("dax", "wug"), A="red", d="blue",
                             bundle_idx=0)
    assert s.cell == "NULL_AA" and s.plumbing_only
    assert s.base_vals == s.cf_vals
    assert s.caption() == "PREDICTED_OUTPUT_CHANGE: NO_CHANGE"


def test_variant_expansion_is_eightfold():
    fam = _fam()
    rows = expand_variants(fam[0], "PRE. ")
    assert len(rows) == 8
    combos = {(r.query_order, r.prompt_format, r.preamble) for r in rows}
    assert len(combos) == 8
    for r in rows:
        assert r.base_content.startswith("PRE. ") == r.preamble
        assert r.base_content.rstrip().endswith("Answer:")


def test_n_stratum_family():
    fam = build_family(stratum="N", template_id="n1", split="train",
                       split_axis="iid", entities=(None, None),
                       A="Alice", B="Priya", d="Paris", dp="Berlin",
                       bundle_idx=0)
    r1, r2, r3, r4, r5, r6 = fam
    assert r1.caption() == "PREDICTED_OUTPUT_CHANGE: Alice -> Priya"
    assert r4.caption() == "PREDICTED_OUTPUT_CHANGE: Paris -> Berlin"
    assert r2.caption() == r3.caption() == "PREDICTED_OUTPUT_CHANGE: NO_CHANGE"
