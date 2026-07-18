"""Tokenizer audit — runs first, gates everything downstream.

For every candidate answer value, tokenize all four surface forms and keep
only values whose LEADING-SPACE form is a single token (answers are predicted
after "Answer:", so " value" is the form that matters).

Nonce entities are audited for record-keeping (token counts in isolation);
their within-pair consistency is enforced at the prompt level by the
exactly-one-differing-token invariant.
"""

from dataclasses import dataclass, field, asdict

FORMS = ("leading_space", "bare", "leading_space_cap", "bare_cap")


def _surface(value: str, form: str) -> str:
    cap = value[0].upper() + value[1:]
    return {
        "leading_space": f" {value}",
        "bare": value,
        "leading_space_cap": f" {cap}",
        "bare_cap": cap,
    }[form]


@dataclass
class ValueAudit:
    value: str
    kind: str                                  # color | name | city | nonce
    forms: dict = field(default_factory=dict)  # form -> {"ids": [...], "n": int}
    kept: bool = False
    reject_reason: str | None = None

    @property
    def leading_space_id(self) -> int:
        ids = self.forms["leading_space"]["ids"]
        assert len(ids) == 1, f"{self.value}: leading-space form is not single-token"
        return ids[0]


def audit_values(tokenizer, candidates: dict[str, list[str]]) -> dict[str, list[ValueAudit]]:
    """candidates: kind -> list of candidate strings. Returns kind -> audits."""
    out: dict[str, list[ValueAudit]] = {}
    for kind, values in candidates.items():
        audits = []
        for value in values:
            a = ValueAudit(value=value, kind=kind)
            for form in FORMS:
                ids = tokenizer.encode(_surface(value, form), add_special_tokens=False)
                a.forms[form] = {"ids": ids, "n": len(ids)}
            if kind == "nonce":
                # nonces are not answers; no single-token requirement
                a.kept = True
            elif a.forms["leading_space"]["n"] == 1:
                a.kept = True
            else:
                a.reject_reason = (
                    f"leading-space form is {a.forms['leading_space']['n']} tokens")
            audits.append(a)
        out[kind] = audits
    return out


def survivors(audits: dict[str, list[ValueAudit]], kind: str) -> list[ValueAudit]:
    return [a for a in audits[kind] if a.kept]


def audit_table_rows(audits: dict[str, list[ValueAudit]]) -> list[dict]:
    rows = []
    for kind in audits:
        for a in audits[kind]:
            for form in FORMS:
                rows.append({
                    "kind": kind, "value": a.value, "form": form,
                    "surface": _surface(a.value, form),
                    "n_tokens": a.forms[form]["n"],
                    "token_ids": a.forms[form]["ids"],
                    "kept": a.kept,
                    "reject_reason": a.reject_reason,
                })
    return rows


def audit_to_dict(audits: dict[str, list[ValueAudit]]) -> dict:
    return {kind: [asdict(a) for a in lst] for kind, lst in audits.items()}
