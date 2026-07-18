import pytest

from ctf_data import captions as cap


def test_transition_round_trip():
    c = cap.Caption("transition", "red", "green")
    assert cap.parse(cap.render(c)) == c


def test_no_change_round_trip():
    c = cap.Caption("no_change")
    assert cap.parse(cap.render(c)) == c


def test_name_transition_round_trip():
    c = cap.Caption("transition", "Alice", "Priya")
    assert cap.parse(cap.render(c)) == c


def test_not_identifiable_is_parser_valid_but_not_renderable():
    parsed = cap.parse("PREDICTED_OUTPUT_CHANGE: NOT_IDENTIFIABLE_FROM_THIS_STATE")
    assert parsed.kind == "not_identifiable"
    with pytest.raises(ValueError):
        cap.render(cap.Caption("not_identifiable"))


def test_render_refuses_equal_endpoints():
    with pytest.raises(ValueError):
        cap.render(cap.Caption("transition", "red", "red"))


def test_render_refuses_reserved_and_nonalpha_values():
    with pytest.raises(ValueError):
        cap.render(cap.Caption("transition", "NO_CHANGE", "red"))
    with pytest.raises(ValueError):
        cap.render(cap.Caption("transition", "red green", "blue"))
    with pytest.raises(ValueError):
        cap.render(cap.Caption("transition", "red", "gr->een"))


@pytest.mark.parametrize("bad", [
    "PREDICTED_OUTPUT_CHANGE: red ->  green",   # double space
    "PREDICTED_OUTPUT_CHANGE:red -> green",     # missing space
    "predicted_output_change: red -> green",    # lowercase field
    "PREDICTED_OUTPUT_CHANGE: red -> green ",   # trailing space
    "PREDICTED_OUTPUT_CHANGE: red => green",    # wrong arrow
    "PREDICTED_OUTPUT_CHANGE: red -> green\nX", # trailing junk
    "PREDICTED_OUTPUT_CHANGE: NOCHANGE",
    "OUTPUT_CHANGE: red -> green",
])
def test_parser_strictness(bad):
    with pytest.raises(ValueError):
        cap.parse(bad)


def test_sha_stable():
    a = cap.sha256("PREDICTED_OUTPUT_CHANGE: red -> green")
    b = cap.sha256("PREDICTED_OUTPUT_CHANGE: red -> green")
    assert a == b and len(a) == 64
