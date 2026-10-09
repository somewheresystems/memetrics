"""Simulation tests for memetrics.onchain: heritability recovery, the attention-chasing rule, families, matching."""
import numpy as np
import pandas as pd

from memetrics.io.onchain import activity_before, candles_frame, peak_to_date, to_items
from memetrics.onchain import (assign_parents, attention_features, build_alias_index, find_relatives,
                               heritability, heritability_verdict, match_catalogue, norm, strip_affixes)


def _families(n=800, b=0.5, attention=0.0, seed=1):
    """Offspring whose outcome inherits b·parent fitness, plus an optional attention term.

    The attention term is correlated with parent fitness: copiers time launches to hot parents.
    """
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    parent = rng.normal(5, 1, n)
    hot = 0.8 * (parent - 5) + rng.normal(0, 0.6, n)  # parent's current activity at copy time
    y = 3 + b * parent + attention * hot + rng.normal(0, 1, n)
    return pd.DataFrame({"item_id": [f"t{i}" for i in range(n)], "ts": ts, "engagement": y,
                         "parent_fitness": parent, "parent_hot": hot})


def test_recovers_parent_offspring_slope():
    d = _families(b=0.5)
    r = heritability(d, n_boot=300, n_perm=300, seed=0)
    assert r.ci_low < 0.5 < r.ci_high
    assert r.verdict == "heritable"


def test_neutral_not_heritable():
    d = _families(b=0.0)
    r = heritability(d, n_boot=300, n_perm=300, seed=0)
    assert r.verdict == "not heritable"


def test_attention_chasing_vs_inheritance():
    # Pure attention: the slope appears without control but vanishes once parent_hot is added.
    d = _families(b=0.0, attention=0.8, seed=2)
    plain = heritability(d, n_boot=300, n_perm=300, seed=0)
    adj = heritability(d, controls=["parent_hot"], n_boot=300, n_perm=300, seed=0)
    hot = heritability(d.assign(parent_fitness=d.parent_hot), n_boot=300, n_perm=300, seed=0)
    assert heritability_verdict(plain, adj, {"parent_hot": hot}) == "attention-chasing"
    # True inheritance survives the attention control.
    d = _families(b=0.6, attention=0.4, seed=3)
    plain = heritability(d, n_boot=300, n_perm=300, seed=0)
    adj = heritability(d, controls=["parent_hot"], n_boot=300, n_perm=300, seed=0)
    assert heritability_verdict(plain, adj, {}) == "memetic inheritance"


def test_relatives_and_parents():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    items = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "ts": [t, t + pd.Timedelta(hours=2), t + pd.Timedelta(days=3), t + pd.Timedelta(days=4)],
        "text": ["Chill Guy", "The Chill Guy coin", "Moo Deng", "chill guy"],
        "ticker": ["CHILL", "CHILLG", "MOO", "CG"],
        "phash": [np.uint64(0), np.uint64(1), np.uint64(2**63), np.nan],
    })
    rel = find_relatives(items, items)
    assert set(map(tuple, rel[rel.item_id == "d"][["rel_id"]].to_numpy().tolist())) == {("a",), ("b",)}
    assert rel[(rel.item_id == "b") & (rel.rel_id == "a")].via.item() == "both"
    fit = {"a": 4.0, "b": 6.0}
    par = assign_parents(rel, fitness=lambda p, o: fit.get(p, np.nan))
    row = par.set_index("item_id").loc["d"]
    assert row.parent == "b" and row.parent_fitness == 6.0
    assert par.set_index("item_id").loc["b"].n_siblings == 1  # a is only 2 h older


def test_matching_and_norm():
    assert norm("  $PEPE🐸 the Frog! ") == "pepe the frog"
    assert strip_affixes("the chill guy coin") == "chill guy"
    idx = build_alias_index(pd.DataFrame({"alias": ["chill guy", "moo deng"], "id": ["w:1", "w:2"]}))
    items = pd.DataFrame({"text": ["Chill Guy Token", "dog"], "ticker": ["CHILL", "DOG"], "topic": [None, "moo deng"]})
    assert match_catalogue(items, idx).tolist() == [["w:1"], ["w:2"]]


def test_attention_features_and_candles():
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    v = pd.Series(100.0, index=days)
    v.loc["2024-12-20":] = 10_000.0
    pop, spike = attention_features(v, pd.Timestamp("2024-12-28", tz="UTC"))
    assert abs(pop - np.log10(101)) < 0.01 and spike > 1.5
    c = [{"timestamp": int(pd.Timestamp(f"2024-01-0{i}", tz="UTC").timestamp() * 1000), "open": "1e-5",
          "high": "1e-4", "low": "1e-6", "close": str(c), "volume": str(vol)}
         for i, c, vol in [(1, 2e-5, 5e8), (2, 9e-5, 1e3), (3, 1e-5, 1e8)]]
    cf = candles_frame(c)
    # Day 2 is a thin spike (tiny volume), so the robust peak to date uses day 1 only.
    assert peak_to_date(cf, pd.Timestamp("2024-01-03", tz="UTC")) == 2e-5 * 1e9
    vol, last = activity_before(cf, pd.Timestamp("2024-01-03", tz="UTC"))
    assert last == 9e-5 * 1e9 and vol > 0


def test_to_items_hashes_author():
    d = pd.DataFrame({"mint": ["m1"], "created_at": ["2025-01-01T00:00:00Z"], "creator": ["W1"],
                      "name": ["Pepe"], "symbol": ["PEPE"], "img_phash": ["ff00ff00ff00ff00"]})
    it = to_items(d, salt="s", phash_col="img_phash")
    assert it.author.item() != "W1" and it.kind.item() == "token" and int(it.phash.item()) == 0xFF00FF00FF00FF00
