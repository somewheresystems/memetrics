import numpy as np

from memetrics.units import discreteness, gaussian_surrogate, hamming64, identical_groups


def _clusters(k=8, n=60, d=32, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((k, d))
    x = np.repeat(c, n, 0) + spread * rng.standard_normal((k * n, d)) * np.sqrt(d)
    return x, np.repeat(np.arange(k), n)


def test_hamming():
    assert hamming64(np.uint64(0b1011), np.uint64(0)) == 3


def test_identical_groups_tolerates_small_flips_and_splits_captions():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2**63, 50, dtype=np.uint64)
    flip = base ^ np.uint64(0b101)  # 2 bits
    ph = np.concatenate([base, flip])
    g = identical_groups(ph, ts=np.arange(100))
    assert (g[50:] == np.arange(50)).all()
    # same hash, different captions -> different groups
    g2 = identical_groups(
        np.array([5, 5], dtype=np.uint64), text=["one does not simply walk", "brace yourselves winter is coming"]
    )
    assert g2[0] != g2[1]


def test_discreteness_separated_vs_gaussian():
    x, lab = _clusters(spread=0.05)
    r = discreteness(x, lab, n_boot=200)
    assert r.estimate > 2 and r.detail["blend_share"] < 0.1 and r.verdict == "units"
    # a cluster-free surrogate partitioned arbitrarily is a continuum
    z = gaussian_surrogate(x, dims=16)
    lab_z = (z[:, 0] > 0).astype(int) + 2 * (z[:, 1] > 0)
    rz = discreteness(z, lab_z, n_boot=200)
    assert rz.verdict.startswith("continuum")


def test_identical_groups_joins_earliest_when_rows_unsorted():
    # rows 0 and 1 are two representatives (5 bits apart); row 2 is within 4 bits of both. Row 1 was posted first,
    # so row 2 must join row 1 although row 0 has the lower row index
    ph = np.array([0b11111, 0, 0b111], dtype=np.uint64)
    g = identical_groups(ph, ts=np.array([5.0, 1.0, 9.0]))
    assert g.tolist() == [0, 1, 1]
