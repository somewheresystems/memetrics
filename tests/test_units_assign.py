import numpy as np
import pandas as pd

from memetrics.units import assign


def test_assign_minhash_and_phash():
    rng = np.random.default_rng(0)
    rows = []
    hashes = rng.integers(0, 2**63, 5, dtype=np.uint64)
    for u in range(4):
        base = rng.integers(0, 5000, 20)
        for k in range(8):
            t = base.copy()
            t[rng.integers(20)] = rng.integers(5000)
            rows.append({"text": " ".join(f"w{x}" for x in t), "truth": u, "phash": hashes[u] ^ np.uint64(k % 2)})
    rows.append({"text": "a lonely post", "truth": -1, "phash": hashes[4]})
    d = pd.DataFrame(rows).assign(ts=pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC"))
    out = assign(d, "minhash", tau=0.3)
    assert out.unit.iloc[-1] is None  # singleton below min_size
    assert pd.crosstab(out.truth[:-1], out.unit[:-1]).gt(0).sum(axis=1).eq(1).all()
    ph = assign(d.drop(columns="text"), "auto")
    assert ph.unit.notna().sum() == 32 and ph.unit.nunique() == 4
