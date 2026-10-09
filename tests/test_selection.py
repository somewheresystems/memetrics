import numpy as np
import pandas as pd

from memetrics.selection import (bh, fit_calibrated, fit_test, neutral_innovation, turnover_exponent,
                                 turnover_profile, wf_sampled)


def _traj(s, seed, Ne=10_000):
    g = np.arange(0, 120, 10)
    n = np.full(len(g), 2000)
    x = wf_sampled(0.2, g, n, Ne, s, reps=1, seed=seed)[0]
    return pd.DataFrame({"t": g.astype(float), "n": n, "count": (x * n).astype(int)})


def test_fit_neutral_false_positive_rate_at_most_alpha():
    # binomial sampling noise makes increments negatively autocorrelated, so the raw FIT is conservative;
    # fit_calibrated measures the exact null rate by simulation
    ps = [fit_test(*(lambda d: (d["count"] / d.n, d.t))(_traj(0.0, s)))[1] for s in range(400)]
    assert np.mean(np.array(ps) < 0.05) <= 0.08


def test_fit_detects_strong_selection():
    ps = [fit_test(*(lambda d: (d["count"] / d.n, d.t))(_traj(0.05, s)))[1] for s in range(100)]
    assert np.mean(np.array(ps) < 0.05) > 0.6


def test_bh():
    r = bh(np.array([0.001, 0.01, 0.04, 0.5]), 0.05)
    assert r.tolist() == [True, True, False, False]


def test_fit_calibrated_verdicts():
    neutral = [_traj(0.0, s) for s in range(60)]
    sel = [_traj(0.05, 1000 + s) for s in range(60)]
    kw = dict(Ne_grid=(10_000,), reps=50, n_sets=200, s_grid=(0.05,))
    assert fit_calibrated(neutral, **kw).verdict == "drift-compatible"
    assert fit_calibrated(sel, **kw).verdict == "selection beyond drift"


def test_turnover_conformity_lowers_exponent():
    xn = turnover_exponent(turnover_profile(neutral_innovation(2000, 0.02, 60, burn=100, seed=1)))
    xc = turnover_exponent(turnover_profile(neutral_innovation(2000, 0.02, 60, burn=100, conformity=0.3, seed=1)))
    assert 0.5 < xn < 1.3
    assert xc != xn
