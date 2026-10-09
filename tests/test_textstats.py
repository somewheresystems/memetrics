import numpy as np

from memetrics.fitness import simulate_crossed
from memetrics.textstats import crossed_reml, reml_icc


def test_crossed_reml_recovers_components():
    d = simulate_crossed(n_content=600, n_author=800, n=6000, s2c=0.5, s2a=0.3, s2e=0.2, seed=4)
    r = crossed_reml(d.engagement.to_numpy(), np.ones((len(d), 1)), [d.unit.to_numpy(), d.author.to_numpy()])
    tot = sum(r["var"]) + r["sigma2"]
    assert np.allclose([r["var"][0] / tot, r["var"][1] / tot, r["sigma2"] / tot], [0.5, 0.3, 0.2], atol=0.07)


def test_reml_icc_verdict():
    d = simulate_crossed(n_content=300, n_author=400, n=3000, s2c=0.05, s2a=0.5, s2e=0.45, seed=5)
    r = reml_icc(d, y="engagement", content="unit", author="author", n_boot=20)
    assert r.verdict == "weak content fitness" and r.detail["author_icc"] > r.estimate


def test_reml_parametric_ci_covers_truth():
    d = simulate_crossed(n_content=300, n_author=400, n=3000, s2c=0.3, s2a=0.2, s2e=0.5, seed=5)
    r = reml_icc(d, y="engagement", content="unit", author="author", n_boot=40)
    lo, hi = r.detail["author_icc_ci"]
    assert r.ci_low < 0.3 < r.ci_high + 0.03 and lo - 0.03 < 0.2 < hi + 0.03
