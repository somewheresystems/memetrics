import numpy as np
import pandas as pd

from memetrics import chains


def _words(rng, L=15):
    return [f"w{x}" for x in rng.integers(0, 3000, L)]


def test_infer_parents_most_similar_earlier():
    ts = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], utc=True)
    items = pd.DataFrame(
        {
            "item_id": list("abcd"),
            "ts": ts,
            "unit": "u",
            "text": ["the cat sat", "a dog ran far away", "the cat sat down", "the cat sat"],
        }
    )
    p = chains.infer_parents(items)
    assert p.tolist() == [None, "a", "a", "a"]  # d is identical to a (and to nothing later)


def test_convergence_slope_recovers_beta():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):
        x = rng.uniform(0, 0.5, 30)
        rows.append(
            pd.DataFrame({"unit": u, "d_parent": x, "delta": -0.4 * x + rng.normal(0, 0.02, 30) + rng.normal()})
        )
    e = pd.concat(rows)
    b, lo, hi = chains.convergence_slope(e, n_boot=500)
    assert abs(b + 0.4) < 0.02 and lo < -0.4 < hi


def _chain_world(kind, n_units=30, n_steps=40, seed=0):
    """Copying chains with exact copies (replication) or pulls toward an attractor (attraction)."""
    rng = np.random.default_rng(seed)
    items = []
    for u in range(n_units):
        attractor = _words(rng)
        cur = list(attractor)
        for j in range(len(cur)):  # start away from the attractor
            if rng.random() < 0.6:
                cur[j] = f"w{rng.integers(3000)}"
        prev = None
        for s in range(n_steps):
            nxt = list(cur)
            if kind == "attraction":
                for j in range(len(nxt)):
                    if rng.random() < 0.2:
                        nxt[j] = attractor[j]
                    if rng.random() < 0.2:
                        nxt[j] = f"w{rng.integers(3000)}"
            elif rng.random() < 0.05:
                nxt[rng.integers(len(nxt))] = f"w{rng.integers(3000)}"
            iid = f"{u}-{s}"
            items.append(
                {
                    "item_id": iid,
                    "unit": str(u),
                    "text": " ".join(nxt),
                    "parent": prev,
                    "ts": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=s),
                }
            )
            prev, cur = iid, nxt
    return pd.DataFrame(items)


def test_m2_attraction_and_the_regression_artifact():
    rep = _chain_world("replication", seed=2)
    att = _chain_world("attraction", seed=1)
    ra = chains.m2_test(chains.edges_from_items(att), chains.unit_medoids(att), n_boot=300, null_sims=30)
    assert ra.verdict == "attraction" and ra.detail["verdict_calibrated"] == "attraction"
    rr = chains.m2_test(chains.edges_from_items(rep), chains.unit_medoids(rep), n_boot=300, null_sims=30)
    # Pure copying along drifting chains: f is high, but the literal slope is negative because the medoid is
    # estimated from the same chain (regression artifact). The literal rule says "mixed"; the calibrated
    # replicator null recovers "replication".
    assert rr.estimate >= 0.9 and rr.detail["beta_ci"][1] < 0 and rr.verdict == "mixed"
    assert rr.detail["verdict_calibrated"] == "replication"


def test_replicator_null_brackets_pure_copying():
    rep = _chain_world("replication", seed=3)
    e = chains.edges_from_items(rep)
    r = chains.m2_test(e, chains.unit_medoids(rep), n_boot=200, null_sims=40)
    assert r.detail["verdict_calibrated"] == "replication"
