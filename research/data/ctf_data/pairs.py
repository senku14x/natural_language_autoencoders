"""Family/pair construction and variant expansion.

A FAMILY is one sampled bundle: (stratum, template, entity slots, base state
(A, d), slot1 transition A->B, slot2 transition d->d'). From it we derive six
semantic pairs (spec §3):

  R1 TARGET      edit slot1 A->B, query slot1, caption "A -> B"
  R2 DISTRACTOR  edit slot1 A->B, query slot2, caption NO_CHANGE   (crossed_with R1)
  R3 DISTRACTOR  edit slot2 d->d', query slot1, caption NO_CHANGE  (crossed_with R4)
  R4 TARGET      edit slot2 d->d', query slot2, caption "d -> d'"
  R5 REVERSE     of R1 (base state (B,d)), caption "B -> A"
  R6 REVERSE     of R4 (base state (A,d')), caption "d' -> d"

R2/R3 are the two distractor flavors (crossed_query / other_slot): together
they decorrelate "which slot was edited" and "which slot was queried" from
the NO_CHANGE label, so neither is a shortcut. R4/R6 exist for the same
reason: without them, every slot2-query state would carry NO_CHANGE and the
query identity readable from the final-position state would leak the label.
(R4 for stratum N means city transitions appear as captions — a deliberate,
config-flaggable deviation from the V1 doc's minimal cell table; see README.)

Every semantic pair expands into 8 variant rows:
  query_order  x  prompt_format  x  preamble   (2 x 2 x 2)
sharing its semantic_id and family_id. TARGET:DISTRACTOR is 1:1 per family
by construction. NULL_AA families contribute a single identical-prompt
semantic pair, capped globally, flagged plumbing_only.
"""

import hashlib
from dataclasses import dataclass, field, asdict

from . import captions as cap
from .templates import ALL_TEMPLATES

QUERY_ORDERS = ("query_last", "query_first")
FORMATS = ("raw", "chat")
PREAMBLES = (False, True)

SLOT_LABELS = {"S": ("slot1", "slot2"), "N": ("speaker_name", "home_city")}


def _h(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


@dataclass
class Semantic:
    """One semantic pair, pre-variant-expansion."""
    semantic_id: str
    family_id: str
    stratum: str
    cell: str                      # TARGET_EDIT | DISTRACTOR_EDIT | REVERSE | NULL_AA
    distractor_flavor: str         # crossed_query | other_slot | ""
    split: str
    split_axis: str
    template_id: str
    entity_slot1: str | None       # S: nonce word; N: None (slot is lexical)
    entity_slot2: str | None
    base_vals: tuple[str, str]     # (slot1, slot2) values in the BASE prompt
    cf_vals: tuple[str, str]       # values in the COUNTERFACTUAL prompt
    edited_slot: str               # "slot1" | "slot2"
    queried_slot: str
    value_old: str                 # edit old (== new for NULL_AA)
    value_new: str
    value_distractor: str          # base value of the non-queried slot
    answer_old: str                # expected base answer surface
    answer_new: str                # expected cf answer surface
    crossed_with: str = ""         # semantic_id of same-edit/other-query partner
    reverse_of: str = ""
    plumbing_only: bool = False

    def caption(self) -> str:
        if self.answer_old == self.answer_new:
            return cap.render(cap.Caption("no_change"))
        return cap.render(cap.Caption("transition", self.answer_old, self.answer_new))


def build_family(*, stratum: str, template_id: str, split: str, split_axis: str,
                 entities: tuple[str | None, str | None],
                 A: str, B: str, d: str, dp: str, bundle_idx: int) -> list[Semantic]:
    """Construct R1..R6 for one family. Values must already be pool-legal."""
    assert len({A, B, d, dp}) == 4 or stratum == "N", "S families need 4 distinct values"
    assert A != B and d != dp
    fam = _h(stratum, template_id, str(entities), A, B, d, dp, split, str(bundle_idx))

    def mk(i: int, cell: str, flavor: str, base, cfv, edited, queried,
           vold, vnew) -> Semantic:
        non_q = 1 if queried == "slot1" else 0
        return Semantic(
            semantic_id=_h(fam, str(i)), family_id=fam, stratum=stratum,
            cell=cell, distractor_flavor=flavor, split=split, split_axis=split_axis,
            template_id=template_id, entity_slot1=entities[0], entity_slot2=entities[1],
            base_vals=base, cf_vals=cfv, edited_slot=edited, queried_slot=queried,
            value_old=vold, value_new=vnew,
            value_distractor=base[non_q],
            answer_old=base[0 if queried == "slot1" else 1],
            answer_new=cfv[0 if queried == "slot1" else 1],
        )

    r1 = mk(1, "TARGET_EDIT", "", (A, d), (B, d), "slot1", "slot1", A, B)
    r2 = mk(2, "DISTRACTOR_EDIT", "crossed_query", (A, d), (B, d), "slot1", "slot2", A, B)
    r3 = mk(3, "DISTRACTOR_EDIT", "other_slot", (A, d), (A, dp), "slot2", "slot1", d, dp)
    r4 = mk(4, "TARGET_EDIT", "", (A, d), (A, dp), "slot2", "slot2", d, dp)
    r5 = mk(5, "REVERSE", "", (B, d), (A, d), "slot1", "slot1", B, A)
    r6 = mk(6, "REVERSE", "", (A, dp), (A, d), "slot2", "slot2", dp, d)
    r1.crossed_with, r2.crossed_with = r2.semantic_id, r1.semantic_id
    r3.crossed_with, r4.crossed_with = r4.semantic_id, r3.semantic_id
    r5.reverse_of, r6.reverse_of = r1.semantic_id, r4.semantic_id
    return [r1, r2, r3, r4, r5, r6]


def build_null_family(*, stratum: str, template_id: str, split: str,
                      entities: tuple[str | None, str | None],
                      A: str, d: str, bundle_idx: int) -> list[Semantic]:
    fam = _h("NULL", stratum, template_id, str(entities), A, d, split, str(bundle_idx))
    s = Semantic(
        semantic_id=_h(fam, "0"), family_id=fam, stratum=stratum,
        cell="NULL_AA", distractor_flavor="", split=split, split_axis="iid",
        template_id=template_id, entity_slot1=entities[0], entity_slot2=entities[1],
        base_vals=(A, d), cf_vals=(A, d), edited_slot="slot1", queried_slot="slot1",
        value_old=A, value_new=A, value_distractor=d,
        answer_old=A, answer_new=A, plumbing_only=True,
    )
    return [s]


def render_contents(sem: Semantic, query_order: str, preamble_text: str) -> tuple[str, str]:
    """Raw text content (pre chat-wrapping) for base and cf prompts."""
    tmpl = ALL_TEMPLATES[sem.template_id]

    def render(vals):
        return tmpl.render(
            slot1_entity=sem.entity_slot1, slot1_value=vals[0],
            slot2_entity=sem.entity_slot2, slot2_value=vals[1],
            query_slot="slot1" if sem.queried_slot in ("slot1", "speaker_name") else "slot2",
            query_order=query_order)

    base = preamble_text + render(sem.base_vals)
    cf = preamble_text + render(sem.cf_vals)
    return base, cf


@dataclass
class VariantRow:
    """One fully tokenized dataset row (a semantic pair under one variant)."""
    pair_id: str
    semantic: Semantic
    query_order: str
    prompt_format: str
    preamble: bool
    base_prompt: str
    cf_prompt: str
    base_content: str
    cf_content: str
    base_input_ids: list[int] = field(default_factory=list)
    cf_input_ids: list[int] = field(default_factory=list)
    edit_pos: int = -1
    final_pos: int = -1
    edit_token_decoded_base: str = ""
    edit_token_decoded_cf: str = ""
    final_token_decoded: str = ""

    def to_record(self, answer_ids: dict[str, int]) -> dict:
        s = self.semantic
        caption = s.caption()
        labels = SLOT_LABELS[s.stratum]
        ent = {
            "slot1": s.entity_slot1 if s.stratum == "S" else labels[0],
            "slot2": s.entity_slot2 if s.stratum == "S" else labels[1],
        }
        q, e = s.queried_slot, s.edited_slot
        return {
            "pair_id": self.pair_id,
            "semantic_id": s.semantic_id,
            "family_id": s.family_id,
            "stratum": s.stratum,
            "cell": s.cell,
            "distractor_flavor": s.distractor_flavor,
            "query_order": self.query_order,
            "prompt_format": self.prompt_format,
            "preamble": self.preamble,
            "split": s.split,
            "split_axis": s.split_axis,
            "template_id": s.template_id,
            "entity_target": ent[q],
            "entity_distractor": ent["slot2" if q == "slot1" else "slot1"],
            "edited_slot": e,
            "queried_slot": q,
            "value_old": s.value_old,
            "value_new": s.value_new,
            "value_distractor": s.value_distractor,
            "answer_old": s.answer_old,
            "answer_new": s.answer_new,
            "base_prompt": self.base_prompt,
            "cf_prompt": self.cf_prompt,
            "base_content": self.base_content,
            "cf_content": self.cf_content,
            "base_input_ids": self.base_input_ids,
            "cf_input_ids": self.cf_input_ids,
            "edit_pos": self.edit_pos,
            "final_pos": self.final_pos,
            "edit_token_decoded_base": self.edit_token_decoded_base,
            "edit_token_decoded_cf": self.edit_token_decoded_cf,
            "final_token_decoded": self.final_token_decoded,
            "answer_token_id_old": answer_ids[s.answer_old],
            "answer_token_id_new": answer_ids[s.answer_new],
            "answer_token_id_distractor": answer_ids[s.value_distractor],
            "caption": caption,
            "caption_sha256": cap.sha256(caption),
            "plumbing_only": s.plumbing_only,
            "crossed_with": s.crossed_with,
            "reverse_of": s.reverse_of,
            # GPU stage fills these; kept as explicit null placeholders.
            "behavioral_screen": None,
            "oracle_patch_metrics": None,
            "delta_norms": None,
        }


def expand_variants(sem: Semantic, preamble_text: str) -> list[VariantRow]:
    rows = []
    for order in QUERY_ORDERS:
        for fmt in FORMATS:
            for pre in PREAMBLES:
                pt = preamble_text if pre else ""
                base_c, cf_c = render_contents(sem, order, pt)
                rows.append(VariantRow(
                    pair_id=f"{sem.semantic_id}-{order}-{fmt}-{'pre' if pre else 'nopre'}",
                    semantic=sem, query_order=order, prompt_format=fmt, preamble=pre,
                    base_prompt="", cf_prompt="",  # filled after chat wrapping
                    base_content=base_c, cf_content=cf_c,
                ))
    return rows
