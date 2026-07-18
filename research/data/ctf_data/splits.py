"""Seeded partition of value inventories into train pools and held-out pools.

Split assignment happens at FAMILY CONSTRUCTION time from these pools —
before any caption is rendered — so no caption statistic can influence
membership. Held-out transitions are held out as UNORDERED pairs: if {A,B}
is held out, neither A->B nor B->A may appear in train (reverse edges travel
together).

Split axes produced downstream:
  train, dev           — iid families from train pools
  test_context         — held-out template, train values/entities/transitions
  test_transition      — held-out unordered value pair, both endpoints train-seen
  test_value   (S)     — a color never seen anywhere in train
  test_entity  (S)     — held-out nonce entities
  test_name    (N)     — complete held-out canonical names (both endpoints)
Cities are deliberately NOT partitioned: they are static context / distractor-
query answers, never a generalization axis in v1 (documented in README).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pools:
    train_colors: list[str]
    heldout_colors: list[str]
    train_color_pairs: list[tuple[str, str]]      # ordered, unordered-key not held out
    heldout_color_pairs: list[tuple[str, str]]    # ordered, unordered-key held out
    heldout_value_pairs: list[tuple[str, str]]    # ordered, >=1 endpoint held out
    train_nonces: list[str]
    heldout_nonces: list[str]
    train_names: list[str]
    heldout_names: list[str]
    train_name_pairs: list[tuple[str, str]]
    heldout_name_pairs: list[tuple[str, str]]     # unordered-key held out, endpoints train
    heldout_name_value_pairs: list[tuple[str, str]]  # both endpoints held-out names
    cities: list[str]
    train_templates: dict[str, list[str]] = field(default_factory=dict)   # stratum -> ids
    context_templates: dict[str, list[str]] = field(default_factory=dict) # stratum -> ids


def _ordered_pairs(values: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in values for b in values if a != b]


def _split_unordered(rng, values: list[str], n_hold: int):
    """Partition the unordered-pair space of `values`; return (train_ordered,
    heldout_ordered) with reverse directions kept together."""
    unordered = sorted({tuple(sorted(p)) for p in _ordered_pairs(values)})
    idx = rng.permutation(len(unordered))
    hold = {unordered[i] for i in idx[:n_hold]}
    train_o, hold_o = [], []
    for a, b in _ordered_pairs(values):
        (hold_o if tuple(sorted((a, b))) in hold else train_o).append((a, b))
    return train_o, hold_o


def build_pools(rng: np.random.Generator, *, colors: list[str], names: list[str],
                cities: list[str], nonces: list[str], holdout_cfg: dict,
                s_templates: list[str], s_context_templates: list[str],
                n_templates: list[str], n_context_templates: list[str]) -> Pools:
    colors = sorted(colors)
    names = sorted(names)
    cities = sorted(cities)
    nonces = sorted(nonces)

    def carve(values, n):
        perm = rng.permutation(len(values))
        held = [values[i] for i in perm[:n]]
        kept = [v for v in values if v not in held]
        return kept, held

    train_colors, heldout_colors = carve(colors, holdout_cfg["colors"])
    train_nonces, heldout_nonces = carve(nonces, holdout_cfg["nonces"])
    train_names, heldout_names = carve(names, holdout_cfg["names"])

    train_cp, heldout_cp = _split_unordered(
        rng, train_colors, holdout_cfg["color_transitions"])
    train_np_, heldout_np = _split_unordered(
        rng, train_names, holdout_cfg["name_transitions"])

    heldout_value_pairs = [
        (a, b) for a in colors for b in colors
        if a != b and (a in heldout_colors or b in heldout_colors)
    ]
    heldout_name_value_pairs = _ordered_pairs(heldout_names)

    return Pools(
        train_colors=train_colors, heldout_colors=heldout_colors,
        train_color_pairs=train_cp, heldout_color_pairs=heldout_cp,
        heldout_value_pairs=heldout_value_pairs,
        train_nonces=train_nonces, heldout_nonces=heldout_nonces,
        train_names=train_names, heldout_names=heldout_names,
        train_name_pairs=train_np_, heldout_name_pairs=heldout_np,
        heldout_name_value_pairs=heldout_name_value_pairs,
        cities=cities,
        train_templates={"S": s_templates, "N": n_templates},
        context_templates={"S": s_context_templates, "N": n_context_templates},
    )


def pool_sanity(pools: Pools) -> list[str]:
    """Structural disjointness checks on the pools themselves."""
    problems = []
    if set(pools.train_colors) & set(pools.heldout_colors):
        problems.append("train/heldout color overlap")
    if set(pools.train_names) & set(pools.heldout_names):
        problems.append("train/heldout name overlap")
    if set(pools.train_nonces) & set(pools.heldout_nonces):
        problems.append("train/heldout nonce overlap")
    tr_keys = {tuple(sorted(p)) for p in pools.train_color_pairs}
    ho_keys = {tuple(sorted(p)) for p in pools.heldout_color_pairs}
    if tr_keys & ho_keys:
        problems.append("color transition unordered-key overlap")
    tr_nk = {tuple(sorted(p)) for p in pools.train_name_pairs}
    ho_nk = {tuple(sorted(p)) for p in pools.heldout_name_pairs}
    if tr_nk & ho_nk:
        problems.append("name transition unordered-key overlap")
    for a, b in pools.train_color_pairs:
        if a in pools.heldout_colors or b in pools.heldout_colors:
            problems.append("held-out color inside train transition pool")
            break
    for tmpl_map in (pools.train_templates, pools.context_templates):
        pass
    for stratum in ("S", "N"):
        if set(pools.train_templates[stratum]) & set(pools.context_templates[stratum]):
            problems.append(f"{stratum}: context-holdout template also in train templates")
    return problems
