"""Deterministic caption rendering and parsing.

One field. Exactly three admissible forms:

    PREDICTED_OUTPUT_CHANGE: <old> -> <new>
    PREDICTED_OUTPUT_CHANGE: NO_CHANGE
    PREDICTED_OUTPUT_CHANGE: NOT_IDENTIFIABLE_FROM_THIS_STATE

NOT_IDENTIFIABLE_FROM_THIS_STATE is parser-valid but NEVER emitted by v1
generation (open decision, surfaced to the user — see README). The renderer
refuses it so it cannot enter the dataset silently.

parse(render(x)) == x is unit-tested for every generated row.
"""

import hashlib
import re
from dataclasses import dataclass

FIELD = "PREDICTED_OUTPUT_CHANGE"
NO_CHANGE = "NO_CHANGE"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE_FROM_THIS_STATE"

# Values are audited single-token surfaces: alphabetic, no whitespace, and
# never equal to a reserved label. The parser enforces the same alphabet so
# free wording cannot round-trip.
_VALUE_RE = r"[A-Za-z]+"
_CAPTION_RE = re.compile(
    rf"^{FIELD}: (?:({_VALUE_RE}) -> ({_VALUE_RE})|{NO_CHANGE}|{NOT_IDENTIFIABLE})$"
)


@dataclass(frozen=True)
class Caption:
    kind: str                  # "transition" | "no_change" | "not_identifiable"
    old: str | None = None
    new: str | None = None


def render(caption: Caption) -> str:
    if caption.kind == "transition":
        if not caption.old or not caption.new:
            raise ValueError("transition caption requires old and new")
        for v in (caption.old, caption.new):
            if not re.fullmatch(_VALUE_RE, v):
                raise ValueError(f"value {v!r} not renderable (must be alphabetic)")
            if v in (NO_CHANGE, NOT_IDENTIFIABLE):
                raise ValueError(f"value {v!r} collides with a reserved label")
        if caption.old == caption.new:
            raise ValueError("transition caption with old == new; use NO_CHANGE")
        return f"{FIELD}: {caption.old} -> {caption.new}"
    if caption.kind == "no_change":
        return f"{FIELD}: {NO_CHANGE}"
    if caption.kind == "not_identifiable":
        raise ValueError(
            "NOT_IDENTIFIABLE_FROM_THIS_STATE is parser-valid but not emitted "
            "in v1 (open decision — ask before enabling)")
    raise ValueError(f"unknown caption kind {caption.kind!r}")


def parse(text: str) -> Caption:
    m = _CAPTION_RE.match(text)
    if not m:
        raise ValueError(f"unparseable caption: {text!r}")
    old, new = m.group(1), m.group(2)
    if old is not None:
        return Caption("transition", old, new)
    if text == f"{FIELD}: {NOT_IDENTIFIABLE}":
        return Caption("not_identifiable")
    return Caption("no_change")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
