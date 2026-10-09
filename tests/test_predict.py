import numpy as np
import pandas as pd

from memetrics.predict import nested_compare, nonredundancy_verdict, prior_unit_features


def test_prior_unit_features_are_causal():
    d = pd.DataFrame({"u": ["a", "a", "a", "b"], "t": [0.0, 10.0, 20.0, 5.0], "v": [1.0, 3.0, 100.0, 7.0]})
    f = prior_unit_features(d, "u", "t", "v", lag=5, k=0)
    assert f.u_n.tolist() == [0, 1, 2, 0]
    assert f.u_fit.tolist() == [0.0, 1.0, 2.0, 0.0]  # never sees its own or later values


def _world(memetic, n=12000, seed=0):
    rng = np.random.default_rng(seed)
    unit = rng.integers(0, 300, n)
    t = np.sort(rng.uniform(0, 100, n))
    uf = rng.normal(0, 1.2 if memetic else 0, 300)
    x = rng.normal(0, 1, n)
    logit = -2 + 0.8 * x + uf[unit]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    d = pd.DataFrame({"unit": unit.astype(str), "ts": t, "x": x, "y": y})
    d = d.join(prior_unit_features(d, "unit", "ts", "y", k=5, prefix="m"))
    d["m_fit"] -= d.y.mean() * d.m_n / (d.m_n + 5)
    d["ca"] = np.log1p(d.m_n)
    return d


def test_nested_detects_memetic_and_null():
    for memetic, expect in ((True, "non-redundant"), (False, "no memetic gain")):
        d = _world(memetic)
        tr, te = d[d.ts < 60], d[d.ts >= 60]
        blocks = {"B": ["x"], "CA": ["ca"], "M": ["m_fit"]}
        r = nested_compare(tr, te, "y", blocks, [(["B"], ["M"]), (["B", "CA"], ["M"])], "unit", n_boot=200)
        assert nonredundancy_verdict(*r) == expect
