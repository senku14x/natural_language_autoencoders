"""Amendment 1 caption renderers: {arrow, sentence} x {transition, entity}.

All four are deterministic string functions of stored metadata — no LLM, no
paraphrase, no sampled surface variation. Each has a strict parser
(parse(render(x)) == x) and both formats of a content level canonicalize to
the same internal representation (amendment §7 requirement).

Entity tokens: stratum S uses the nonce entity word of the edited/queried
slot; stratum N uses the slot words "name"/"city".

NULL_AA rows (no edit exists) render as the plain no-change form in every
renderer. The hard rule of amendment §3 holds by construction: no renderer
emits magnitude, intensity, or confidence language.

Post-review revision (amendment §11): the distractor forms carry the
behavioral claim only ("| answer unaffected") — grounded by Stage A. The
QUERIED-ENTITY NAME is a separate slot whose groundedness (is query identity
recoverable from Δ_final?) is gated on the Stage-B queried-entity probe; no
renderer emits it, and RichCaption.queried is reserved until that probe
licenses it.
"""

import hashlib
import re
from dataclasses import dataclass

RENDERER_VERSION = "amendment1-r2"

CAPTION_COLUMNS = (
    "caption_arrow_transition",
    "caption_sentence_transition",
    "caption_arrow_entity",
    "caption_sentence_entity",
)

_V = r"[A-Za-z]+"


@dataclass(frozen=True)
class RichCaption:
    """Canonical internal representation (the reader-facing object).

    kind: "change" (queried answer moves) | "distractor_change" (an edit
    exists but the queried answer does not move) | "no_change" (no edit).
    entity/queried are None at transition content level.
    """
    kind: str
    old: str | None = None
    new: str | None = None
    entity: str | None = None
    # reserved: never rendered/parsed in v1; licensable only by the Stage-B
    # queried-entity probe (amendment §11.1)
    queried: str | None = None


def _entity_token(sem, slot: str) -> str:
    if sem.stratum == "S":
        return sem.entity_slot1 if slot == "slot1" else sem.entity_slot2
    return "name" if slot == "slot1" else "city"


def rich_from_semantic(sem) -> RichCaption:
    if sem.cell == "NULL_AA":
        return RichCaption("no_change")
    ent = _entity_token(sem, sem.edited_slot)
    if sem.answer_old != sem.answer_new:          # TARGET_EDIT / REVERSE
        return RichCaption("change", sem.value_old, sem.value_new, ent)
    return RichCaption("distractor_change", sem.value_old, sem.value_new, ent)


# ---------------------------------------------------------------- transition

def render_arrow_transition(c: RichCaption) -> str:
    if c.kind == "change":
        return f"{c.old} -> {c.new}"
    return "NO_CHANGE"


def parse_arrow_transition(s: str) -> RichCaption:
    if s == "NO_CHANGE":
        return RichCaption("no_change")
    m = re.fullmatch(rf"({_V}) -> ({_V})", s)
    if not m:
        raise ValueError(f"unparseable arrow_transition: {s!r}")
    return RichCaption("change", m.group(1), m.group(2))


def render_sentence_transition(c: RichCaption) -> str:
    if c.kind == "change":
        return f"The value changes from {c.old} to {c.new}."
    return "The queried value is unchanged."


def parse_sentence_transition(s: str) -> RichCaption:
    if s == "The queried value is unchanged.":
        return RichCaption("no_change")
    m = re.fullmatch(rf"The value changes from ({_V}) to ({_V})\.", s)
    if not m:
        raise ValueError(f"unparseable sentence_transition: {s!r}")
    return RichCaption("change", m.group(1), m.group(2))


# -------------------------------------------------------------------- entity

def render_arrow_entity(c: RichCaption) -> str:
    if c.kind == "change":
        return f"{c.entity}: {c.old} -> {c.new}"
    if c.kind == "distractor_change":
        return f"{c.entity}: {c.old} -> {c.new} | answer unaffected"
    return "NO_CHANGE"


def parse_arrow_entity(s: str) -> RichCaption:
    if s == "NO_CHANGE":
        return RichCaption("no_change")
    m = re.fullmatch(rf"({_V}): ({_V}) -> ({_V})( \| answer unaffected)?", s)
    if not m:
        raise ValueError(f"unparseable arrow_entity: {s!r}")
    e, old, new, clause = m.groups()
    if clause is None:
        return RichCaption("change", old, new, e)
    return RichCaption("distractor_change", old, new, e)


def _sentence_entity_change(c: RichCaption, stratum: str) -> str:
    if stratum == "S":
        return f"The color assigned to {c.entity} changes from {c.old} to {c.new}"
    return f"The {c.entity} changes from {c.old} to {c.new}"


def render_sentence_entity(c: RichCaption, stratum: str) -> str:
    if c.kind == "change":
        return _sentence_entity_change(c, stratum) + "."
    if c.kind == "distractor_change":
        return _sentence_entity_change(c, stratum) + "; the queried answer is unaffected."
    return "The queried value is unchanged."


def parse_sentence_entity(s: str) -> RichCaption:
    if s == "The queried value is unchanged.":
        return RichCaption("no_change")
    pats = [
        # stratum S
        (rf"The color assigned to ({_V}) changes from ({_V}) to ({_V})"
         rf"(; the queried answer is unaffected)?\."),
        # stratum N (entity token is the slot word)
        (rf"The (name|city) changes from ({_V}) to ({_V})"
         rf"(; the queried answer is unaffected)?\."),
    ]
    for p in pats:
        m = re.fullmatch(p, s)
        if m:
            e, old, new, clause = m.groups()
            if clause is None:
                return RichCaption("change", old, new, e)
            return RichCaption("distractor_change", old, new, e)
    raise ValueError(f"unparseable sentence_entity: {s!r}")


# ------------------------------------------------------------------- helpers

def canonical_transition(c: RichCaption) -> tuple:
    """Content-level canonical form at transition level: entity info dropped,
    distractor_change collapses to the behavioral no-change claim."""
    if c.kind == "change":
        return ("change", c.old, c.new)
    return ("no_change",)


def canonical_entity(c: RichCaption) -> tuple:
    return (c.kind, c.old, c.new, c.entity)


def render_all(sem) -> dict[str, str]:
    """All four caption columns (+sha256 columns) for one semantic pair."""
    c = rich_from_semantic(sem)
    t = RichCaption(("change" if c.kind == "change" else "no_change"),
                    c.old if c.kind == "change" else None,
                    c.new if c.kind == "change" else None)
    out = {
        "caption_arrow_transition": render_arrow_transition(t),
        "caption_sentence_transition": render_sentence_transition(t),
        "caption_arrow_entity": render_arrow_entity(c),
        "caption_sentence_entity": render_sentence_entity(c, sem.stratum),
    }
    for k in list(out):
        out[k + "_sha256"] = hashlib.sha256(out[k].encode()).hexdigest()
    return out


def verify_row(sem) -> list[str]:
    """Round-trip + cross-format canonical equality for one semantic pair."""
    problems = []
    cols = render_all(sem)
    at = parse_arrow_transition(cols["caption_arrow_transition"])
    st = parse_sentence_transition(cols["caption_sentence_transition"])
    ae = parse_arrow_entity(cols["caption_arrow_entity"])
    se = parse_sentence_entity(cols["caption_sentence_entity"])
    if canonical_transition(at) != canonical_transition(st):
        problems.append("transition formats disagree after canonicalization")
    if canonical_entity(ae) != canonical_entity(se):
        problems.append("entity formats disagree after canonicalization")
    if render_arrow_transition(at) != cols["caption_arrow_transition"]:
        problems.append("arrow_transition round-trip failed")
    if render_sentence_transition(st) != cols["caption_sentence_transition"]:
        problems.append("sentence_transition round-trip failed")
    if render_arrow_entity(ae) != cols["caption_arrow_entity"]:
        problems.append("arrow_entity round-trip failed")
    if render_sentence_entity(se, sem.stratum) != cols["caption_sentence_entity"]:
        problems.append("sentence_entity round-trip failed")
    # amendment §3 hard rule, enforced mechanically
    banned = ("large", "small", "slightly", "strongly", "confident", "probably")
    for k in CAPTION_COLUMNS:
        low = cols[k].lower()
        if any(b in low.split() for b in banned):
            problems.append(f"banned magnitude/confidence language in {k}")
    return problems
