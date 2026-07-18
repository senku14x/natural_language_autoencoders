"""Per-pair and dataset-level invariants (spec §7). Fail loud, never repair.

Violations are returned as reason strings; the pipeline records them in the
rejection table and drops the ENTIRE family atomically (dropping single rows
would unbalance cells and orphan crossed/reverse partners — documented in
README). Dataset-level failures abort generation: they indicate a
construction bug, not a bad example.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rejection:
    family_id: str
    template_id: str
    stratum: str
    cell: str
    pair_id: str
    reason: str


@dataclass
class RejectionTable:
    rows: list[Rejection] = field(default_factory=list)

    def add(self, **kw):
        self.rows.append(Rejection(**kw))

    def summary(self) -> dict:
        by_reason, by_template, by_cell = {}, {}, {}
        for r in self.rows:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
            by_template[r.template_id] = by_template.get(r.template_id, 0) + 1
            by_cell[r.cell] = by_cell.get(r.cell, 0) + 1
        return {"total": len(self.rows), "by_reason": by_reason,
                "by_template": by_template, "by_cell": by_cell}


def check_variant_row(row, *, answer_ids: dict[str, int],
                      audited_single_token: set[str]) -> list[str]:
    """Spec §7 invariants 1–7 for one tokenized variant row."""
    s = row.semantic
    v = []
    b, c = row.base_input_ids, row.cf_input_ids

    # 1. equal length
    if len(b) != len(c):
        return [f"len_mismatch base={len(b)} cf={len(c)}"]  # nothing else is well-defined

    diffs = [i for i, (x, y) in enumerate(zip(b, c)) if x != y]

    # 2./3. exactly one differing token at edit_pos (0 for NULL_AA)
    if s.cell == "NULL_AA":
        if diffs:
            v.append(f"null_pair_has_diffs at {diffs[:5]}")
    else:
        if len(diffs) != 1:
            v.append(f"diff_count={len(diffs)} (expected 1) at {diffs[:5]}")
        elif diffs[0] != row.edit_pos:
            v.append(f"edit_pos_mismatch recorded={row.edit_pos} actual={diffs[0]}")

    # 4. decoded token at edit_pos matches intended values (leading-space form)
    if 0 <= row.edit_pos < len(b):
        if row.edit_token_decoded_base != f" {s.value_old}":
            v.append(f"edit_decode_base={row.edit_token_decoded_base!r} "
                     f"expected={' ' + s.value_old!r}")
        if row.edit_token_decoded_cf != f" {s.value_new}":
            v.append(f"edit_decode_cf={row.edit_token_decoded_cf!r} "
                     f"expected={' ' + s.value_new!r}")
    else:
        v.append(f"edit_pos_out_of_range {row.edit_pos}")

    # 5. final_pos is the last index
    if row.final_pos != len(b) - 1:
        v.append(f"final_pos={row.final_pos} != len-1={len(b) - 1}")

    # 6. answer candidates single-token, pairwise sane
    for val in (s.answer_old, s.answer_new, s.value_distractor):
        if val not in audited_single_token:
            v.append(f"answer_value_not_audited_single_token {val!r}")
        if val not in answer_ids:
            v.append(f"answer_value_missing_id {val!r}")
    if s.cell in ("TARGET_EDIT", "REVERSE"):
        if s.answer_old == s.answer_new:
            v.append("transition_cell_with_equal_answers")
    if s.answer_old == s.value_distractor and s.cell in ("TARGET_EDIT", "REVERSE"):
        v.append("answer_old_equals_distractor")

    # 7. suffix identity after the edit through final_pos (subsumed by 2, but
    # asserted explicitly per spec)
    if s.cell != "NULL_AA" and 0 <= row.edit_pos < len(b):
        if b[row.edit_pos + 1:] != c[row.edit_pos + 1:]:
            v.append("suffix_after_edit_differs")

    return v


def check_crossings(rows_by_key: dict) -> list[str]:
    """Spec §7.8 causal-masking check. For query_last rows, an example and its
    crossed partner (same edit, different later query) must have byte-identical
    input_ids up to and including edit_pos. Checked for both members and both
    prompt formats/preamble variants."""
    problems = []
    for (sem_id, order, fmt, pre), row in rows_by_key.items():
        if order != "query_last" or not row.semantic.crossed_with:
            continue
        partner = rows_by_key.get((row.semantic.crossed_with, order, fmt, pre))
        if partner is None:
            problems.append(f"missing_crossed_partner {sem_id} {fmt} pre={pre}")
            continue
        e1, e2 = row.edit_pos, partner.edit_pos
        if e1 != e2:
            problems.append(
                f"crossed_edit_pos_mismatch {sem_id}<->{partner.pair_id} {e1}!={e2}")
            continue
        a = np.asarray(row.base_input_ids[:e1 + 1])
        b = np.asarray(partner.base_input_ids[:e1 + 1])
        if not np.array_equal(a, b):
            problems.append(f"crossed_prefix_differs_base {sem_id} {fmt} pre={pre}")
        a = np.asarray(row.cf_input_ids[:e1 + 1])
        b = np.asarray(partner.cf_input_ids[:e1 + 1])
        if not np.array_equal(a, b):
            problems.append(f"crossed_prefix_differs_cf {sem_id} {fmt} pre={pre}")
    return problems


def check_dataset(records: list[dict], pools) -> list[str]:
    """Spec §7.9–11 plus split hygiene. Abort-level problems."""
    problems = []

    train_values: set[str] = set()
    fam_split: dict[str, str] = {}
    sem_split: dict[str, tuple[str, str]] = {}
    for r in records:
        if r["split"] in ("train", "dev"):
            train_values.update([r["value_old"], r["value_new"], r["value_distractor"],
                                 r["answer_old"], r["answer_new"]])
        prev = fam_split.setdefault(r["family_id"], r["split"])
        if prev != r["split"]:
            problems.append(f"family_in_two_splits {r['family_id']} {prev}/{r['split']}")
        sem_split[r["semantic_id"]] = (r["split"], r["reverse_of"])

    # 9. held-out values absent from train/dev (colors and names)
    for held in (set(pools.heldout_colors) | set(pools.heldout_names)):
        if held in train_values:
            problems.append(f"heldout_value_in_train {held!r}")

    # held-out transitions absent from train/dev target/reverse edits
    held_keys = ({tuple(sorted(p)) for p in pools.heldout_color_pairs}
                 | {tuple(sorted(p)) for p in pools.heldout_name_pairs})
    for r in records:
        if r["split"] in ("train", "dev") and r["cell"] in ("TARGET_EDIT", "REVERSE",
                                                            "DISTRACTOR_EDIT"):
            if r["value_old"] != r["value_new"]:
                if tuple(sorted((r["value_old"], r["value_new"]))) in held_keys:
                    problems.append(
                        f"heldout_transition_in_train {r['value_old']}->{r['value_new']}")
                    break

    # held-out entities / names absent from train/dev
    for r in records:
        if r["split"] in ("train", "dev"):
            if r["stratum"] == "S":
                for e in (r["entity_target"], r["entity_distractor"]):
                    if e in set(pools.heldout_nonces):
                        problems.append(f"heldout_entity_in_train {e!r}")
    # 11. reverse rows in same split as forward partner
    for sid, (split, rev_of) in sem_split.items():
        if rev_of and rev_of in sem_split and sem_split[rev_of][0] != split:
            problems.append(f"reverse_split_mismatch {sid}")

    return sorted(set(problems))
