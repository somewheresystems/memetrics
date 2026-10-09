import numpy as np

from memetrics.fitness import content_icc, crossed_components, oneway_icc, simulate_crossed


def test_crossed_components_recover_truth():
    est = np.array([[*crossed_components(d.engagement.values, d.unit.values, d.author.values).values()][:3]
                    for d in (simulate_crossed(s2c=0.6, s2a=0.2, s2e=0.2, seed=s) for s in range(5))]).mean(0)
    assert np.allclose(est, [0.6, 0.2, 0.2], atol=0.08)


def test_content_icc_verdicts():
    hi = content_icc(simulate_crossed(s2c=0.7, s2a=0.1, s2e=0.2, seed=1), n_boot=100)
    lo = content_icc(simulate_crossed(s2c=0.05, s2a=0.5, s2e=0.45, seed=2), n_boot=100)
    assert hi.verdict == "content dominates" and lo.verdict == "weak content fitness"
    assert lo.detail["icc_author"] > lo.estimate
    assert hi.ci_low < hi.estimate < hi.ci_high


def test_fixed_effects_removed():
    d = simulate_crossed(s2c=0.3, s2a=0.3, s2e=0.4, seed=3)
    rng = np.random.default_rng(0)
    d["era"] = rng.integers(0, 5, len(d)).astype(str)
    d["engagement"] += d.era.astype(int) * 2.0  # large fixed effect
    r = content_icc(d, fixed=["era"], n_boot=50)
    assert abs(r.estimate - 0.3) < 0.1


def test_oneway_icc():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(400), 5)
    y = rng.normal(0, 1, 400)[g] + rng.normal(0, 1, len(g))
    assert abs(oneway_icc(y, g) - 0.5) < 0.07
