import numpy as np
import pandas as pd

from memetrics import observation as ob

SMALL = ob.WorldParams(n_agents=800, n_units=12, steps=30)


def test_simulate_and_observe_hide_private_posts():
    log = ob.simulate("neutral", p_private=0.5, seed=1, params=SMALL)
    memes = log[log.unit.notna() & log.true_parent.notna()]
    assert 0.4 < memes.private.mean() < 0.6
    obs = ob.observe(log, "explicit", deletion=0.0)
    assert not obs.item_id.isin(log.item_id[log.private]).any()
    # explicit edges only point to observed posts; ancestor edges are never shorter than explicit ones
    assert obs.parent.dropna().isin(obs.item_id).all()
    anc = ob.observe(log, "ancestor", deletion=0.0)
    assert anc.parent.notna().sum() >= obs.parent.notna().sum()


def test_full_information_equals_truth_at_p0():
    log = ob.simulate("replicator", p_private=0.0, seed=2, params=SMALL)
    obs = ob.observe(log, "explicit", deletion=0.0)
    full = ob.observe(log, full=True)
    assert len(obs) == len(full)
    assert (obs.parent.fillna("") == full.parent.fillna("").to_numpy()).all()


def test_processes_differ_in_fidelity():
    f = {}
    for pr in ("replicator", "attraction"):
        items = ob.observe(ob.simulate(pr, 0.0, seed=3, params=SMALL), full=True)
        f[pr] = ob.evaluate(items, which=("M2",), n_boot=100)["M2"].estimate
    assert f["replicator"] > f["attraction"]


def test_thinning_lowers_observed_fidelity_on_ancestor_edges():
    log = ob.simulate("attraction", p_private=0.7, seed=4, params=SMALL)
    full = ob.evaluate(ob.observe(log, full=True), which=("M2",), n_boot=100)["M2"].detail["mean_f"]
    anc = ob.evaluate(ob.observe(log, "ancestor", deletion=0.0), which=("M2",), n_boot=100)["M2"].detail["mean_f"]
    assert anc < full


def test_confusion_and_breakdown_shapes():
    df = pd.DataFrame({"layer": "explicit", "p_private": [0, 0, 0.5, 0.5], "process": ["replicator", "attraction"] * 2,
                       "inferred": ["replicator", "attraction", "attraction", "attraction"]})
    c = ob.confusion(df)
    assert np.allclose(c.sum(axis=1), 1)
    assert ob.breakdown_fraction(df) == 0.5
