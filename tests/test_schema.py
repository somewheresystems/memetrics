import numpy as np
import pandas as pd
import pytest

from memetrics.schema import SchemaError, conform, read_items, validate, write_items


def _items(n=3):
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(n)],
                         "ts": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
                         "author": "a", "community": "c", "kind": "text", "parent": [None] + ["i0"] * (n - 1),
                         "engagement": np.arange(n, dtype=float)})


def test_validate_accepts_and_rejects():
    validate(_items())
    with pytest.raises(SchemaError, match="duplicates"):
        validate(pd.concat([_items(), _items()]))
    with pytest.raises(SchemaError, match="UTC"):
        validate(_items().assign(ts=pd.date_range("2024-01-01", periods=3)))
    with pytest.raises(SchemaError, match="unknown kind"):
        validate(_items().assign(kind="meme"))
    with pytest.raises(SchemaError, match="missing"):
        validate(_items().drop(columns="author"))


def test_conform_and_roundtrip(tmp_path):
    raw = _items().assign(ts=["2024-01-01T00:00:00Z"] * 3, item_id=[1, 2, 3], parent=[None, "1", "1"],
                          engagement=["1", "2", "x"], phash=["ff", None, 3])
    d = conform(raw)
    assert d.item_id.tolist() == ["1", "2", "3"] and d.parent.tolist() == [None, "1", "1"]
    assert np.isnan(d.engagement[2]) and d.phash[0] == np.uint64(255) and d.phash[1] is None
    back = read_items(write_items(d, tmp_path / "x.parquet"))
    assert back.phash[0] == np.uint64(255) and back.ts.dt.tz is not None and list(back.columns)[:5] == list(d)[:5]
