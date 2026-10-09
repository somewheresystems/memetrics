import numpy as np
from rapidfuzz.distance import Levenshtein

from memetrics.phylo import delta_q, reference, simulate_taxa


def test_additive_tree_has_zero_delta():
    # ((a,b),(c,d)) with branch lengths -> additive metric
    D = np.array([[0, 3, 7, 8], [3, 0, 6, 7], [7, 6, 0, 3], [8, 7, 3, 0]], float)
    d, q = delta_q(D)
    assert abs(d) < 1e-9 and abs(q) < 1e-9


def test_references_ordered_tree_lt_network_lt_star():
    r = {k: reference(k, 16, 12, 0.5, reps=20, seed=3)[:, 0].mean() for k in ("tree", "network", "star")}
    assert r["tree"] < r["network"] < r["star"]


def test_simulated_tree_recovered_as_tree_like():
    s = simulate_taxa("tree", 16, 15, 0.4, seed=11)
    D = np.array([[Levenshtein.normalized_distance(a, b) for b in s] for a in s])
    star = simulate_taxa("star", 16, 15, 0.5, seed=11)
    Ds = np.array([[Levenshtein.normalized_distance(a, b) for b in star] for a in star])
    assert delta_q(D)[0] < delta_q(Ds)[0]
