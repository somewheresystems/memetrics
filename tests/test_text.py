import numpy as np
import pandas as pd

from memetrics import text


def test_normalize_and_sim():
    assert text.normalize("Hello, @bob.bsky.social!  See https://x.com/a  #Tag") == "hello <user> see <url> #tag"
    assert text.sim_edit("abc", "abc") == 1.0
    assert abs(text.sim_edit("abcd", "abce") - 0.75) < 1e-9
    assert abs(text.sim_word("a b c d", "a b c e") - 0.75) < 1e-9


def _planted(n_units=8, per=20, L=30, mu=0.01, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for u in range(n_units):
        base = rng.integers(0, 5000, L)
        for _ in range(per):
            t = base.copy()
            m = rng.random(L) < mu
            t[m] = rng.integers(0, 5000, m.sum())
            texts.append(" ".join(f"w{x}" for x in t))
            labels.append(u)
    return texts, np.array(labels)


def test_minhash_recovers_planted_units():
    texts, labels = _planted()
    got = text.minhash_units(texts, tau=0.3)
    # each planted unit maps to exactly one recovered unit and vice versa
    pairs = pd.crosstab(labels, got)
    assert (pairs.gt(0).sum(axis=1) == 1).all() and (pairs.gt(0).sum(axis=0) == 1).all()


def test_ashman_d_known_value():
    rng = np.random.default_rng(1)
    w, b = rng.normal(0.9, 0.05, 20000), rng.normal(0.3, 0.05, 20000)
    assert abs(text.ashman_d(w, b) - np.sqrt(2) * 0.6 / np.sqrt(2 * 0.0025)) < 0.3


def test_m1_units_vs_continuum():
    texts, labels = _planted()
    r = text.m1_test(pd.DataFrame({"text": texts, "unit": labels.astype(str)}), n_boot=200, silhouette_n=0)
    assert r.verdict == "units" and r.estimate > 2 and r.detail["blend_share"] < 0.1
    # a continuum: one random walk cut into arbitrary consecutive "units"
    rng = np.random.default_rng(2)
    cur = rng.integers(0, 5000, 30)
    walk = []
    for _ in range(160):
        cur = cur.copy()
        cur[rng.integers(30)] = rng.integers(5000)
        walk.append(" ".join(f"w{x}" for x in cur))
    lab = np.repeat(np.arange(8), 20).astype(str)
    r2 = text.m1_test(pd.DataFrame({"text": walk, "unit": lab}), n_boot=200, silhouette_n=0)
    # Known limitation (documented): arbitrary segments of a slow continuum pass the D/blend rule ...
    assert r2.estimate > 2
    # ... but the boundary-gap diagnostic exposes them, and threshold clustering does not cut a continuum.
    assert r2.detail["median_gap_ratio"] < 2 < r.detail["median_gap_ratio"]
    assert len(set(text.threshold_components(walk, 0.8))) == 1
