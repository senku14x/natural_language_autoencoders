import numpy as np

from ctf_data.splits import build_pools, pool_sanity


def _pools(seed=0):
    rng = np.random.default_rng(seed)
    colors = [f"c{i}" for i in range(20)]
    names = [f"N{i}" for i in range(30)]
    cities = [f"City{i}" for i in range(10)]
    nonces = [f"z{i}" for i in range(12)]
    return build_pools(
        rng, colors=colors, names=names, cities=cities, nonces=nonces,
        holdout_cfg={"colors": 4, "color_transitions": 10, "nonces": 3,
                     "names": 6, "name_transitions": 8},
        s_templates=["s1", "s2"], s_context_templates=["s4"],
        n_templates=["n1"], n_context_templates=["n4"])


def test_pool_sanity_clean():
    assert pool_sanity(_pools()) == []


def test_heldout_values_disjoint_from_train():
    p = _pools()
    assert not set(p.train_colors) & set(p.heldout_colors)
    assert not set(p.train_names) & set(p.heldout_names)
    assert not set(p.train_nonces) & set(p.heldout_nonces)
    assert len(p.heldout_colors) == 4


def test_heldout_transitions_symmetric_and_disjoint():
    p = _pools()
    held_keys = {tuple(sorted(x)) for x in p.heldout_color_pairs}
    train_keys = {tuple(sorted(x)) for x in p.train_color_pairs}
    assert not held_keys & train_keys
    # both directions of a held-out unordered pair are held out
    for a, b in list(held_keys):
        assert (a, b) in p.heldout_color_pairs and (b, a) in p.heldout_color_pairs
    assert len(held_keys) == 10


def test_heldout_value_pairs_touch_heldout_colors():
    p = _pools()
    for a, b in p.heldout_value_pairs:
        assert a in p.heldout_colors or b in p.heldout_colors


def test_train_pairs_never_touch_heldout_colors():
    p = _pools()
    for a, b in p.train_color_pairs:
        assert a not in p.heldout_colors and b not in p.heldout_colors


def test_deterministic_under_seed():
    p1, p2 = _pools(7), _pools(7)
    assert p1.heldout_colors == p2.heldout_colors
    assert p1.train_color_pairs == p2.train_color_pairs
    p3 = _pools(8)
    assert p1.heldout_colors != p3.heldout_colors
