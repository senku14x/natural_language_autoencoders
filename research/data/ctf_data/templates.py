"""Prompt templates for both strata, both query orders.

A template renders (bindings, query) into a raw-completion prompt string.
Query-first puts the question line before the bindings; the trailing
"Answer:" line is always last so the final position sits immediately before
answer generation in both orders.

Slot semantics per stratum:
  S: slot1 = (entity1 -> color), slot2 = (entity2 -> color); query names one entity.
  N: slot1 = (name),             slot2 = (city);            query asks name or city.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    template_id: str
    stratum: str                 # "S" | "N"
    binding_lines: tuple[str, str]   # format strings for slot1 / slot2
    query_lines: dict[str, str]      # slot -> query format string ("slot1"/"slot2")

    def render(self, *, slot1_entity: str | None, slot1_value: str,
               slot2_entity: str | None, slot2_value: str,
               query_slot: str, query_order: str) -> str:
        b1 = self.binding_lines[0].format(e=slot1_entity, v=slot1_value)
        b2 = self.binding_lines[1].format(e=slot2_entity, v=slot2_value)
        q = self.query_lines[query_slot].format(
            e=slot1_entity if query_slot == "slot1" else slot2_entity)
        if query_order == "query_last":
            body = f"{b1}\n{b2}\n{q}"
        elif query_order == "query_first":
            body = f"{q}\n{b1}\n{b2}"
        else:
            raise ValueError(f"unknown query_order {query_order!r}")
        return f"{body}\nAnswer:"


S_TEMPLATES = {
    "s1": Template(
        "s1", "S",
        ("In this task, {e} means {v}.", "In this task, {e} means {v}."),
        {"slot1": "What color is {e}?", "slot2": "What color is {e}?"},
    ),
    "s2": Template(
        "s2", "S",
        ("For this exercise, {e} stands for {v}.",
         "For this exercise, {e} stands for {v}."),
        {"slot1": "What color is {e}?", "slot2": "What color is {e}?"},
    ),
    "s3": Template(
        "s3", "S",
        ("Here, the word {e} refers to {v}.",
         "Here, the word {e} refers to {v}."),
        {"slot1": "What color is {e}?", "slot2": "What color is {e}?"},
    ),
    # held out for test_context by default config
    "s4": Template(
        "s4", "S",
        ("Definition: {e} means {v}.", "Definition: {e} means {v}."),
        {"slot1": "Which color does {e} mean?",
         "slot2": "Which color does {e} mean?"},
    ),
}

N_TEMPLATES = {
    "n1": Template(
        "n1", "N",
        ("My name is {v}.", "I live in {v}."),
        {"slot1": "Question: What is my name?",
         "slot2": "Question: What city do I live in?"},
    ),
    "n2": Template(
        "n2", "N",
        ("I am {v}.", "My home is in {v}."),
        {"slot1": "Question: What is my name?",
         "slot2": "Question: What city is my home in?"},
    ),
    "n3": Template(
        "n3", "N",
        ("This is {v} speaking.", "I am calling from {v}."),
        {"slot1": "Question: What is my name?",
         "slot2": "Question: What city am I calling from?"},
    ),
    # held out for test_context by default config
    "n4": Template(
        "n4", "N",
        ("The speaker's name is {v}.", "The speaker lives in {v}."),
        {"slot1": "Question: What is the speaker's name?",
         "slot2": "Question: What city does the speaker live in?"},
    ),
}

ALL_TEMPLATES = {**S_TEMPLATES, **N_TEMPLATES}
