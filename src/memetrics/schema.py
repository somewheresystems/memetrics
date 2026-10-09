"""The items table: the one input every measure consumes.

| column | dtype | meaning |
|---|---|---|
| ``item_id`` | str | unique id |
| ``ts`` | datetime64[ns, UTC] | post time |
| ``author`` | str | salted hash, never a raw handle (:func:`memetrics.io.hash_author`) |
| ``community`` | str | subreddit, feed, language, platform, ... |
| ``kind`` | str | ``image``, ``text``, ``video`` or ``token`` |
| ``text`` | str, optional | caption, OCR or body |
| ``phash`` | uint64, optional | 64-bit perceptual hash (images) |
| ``unit`` | str, optional | meme unit (see ``memetrics units``) |
| ``parent`` | str, optional | ``item_id`` of an observed source (reply, quote, crosspost, remix) |
| ``engagement`` | float, optional | score, likes + reposts, market cap, ... |

Embeddings are stored separately as an ``.npy`` array whose rows follow the table's row order. Platform readers may
add extra columns (e.g. ``deleted``); measures ignore columns they do not use.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("item_id", "ts", "author", "community", "kind")
OPTIONAL = ("text", "phash", "unit", "parent", "engagement")
COLUMNS = REQUIRED + OPTIONAL
KINDS = frozenset({"image", "text", "video", "token"})


class SchemaError(ValueError):
    """The table does not satisfy the items-table schema."""


def require(df: pd.DataFrame, columns: Iterable[str], where: str = "") -> None:
    """Raise :class:`SchemaError` if any of ``columns`` is missing (a light check used by every measure)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        pre = f"{where}: " if where else ""
        raise SchemaError(f"{pre}missing column(s) {missing}; see memetrics.schema for the items table")


def validate(df: pd.DataFrame, require_columns: Iterable[str] = REQUIRED) -> pd.DataFrame:
    """Check the items-table schema and return ``df`` unchanged.

    Checks that ``require_columns`` exist, ``item_id`` is unique and non-null, ``ts`` is timezone-aware UTC,
    ``kind`` values are known, ``engagement`` is numeric, and ``parent`` ids (when present) are strings or null.
    """
    require(df, require_columns)
    errors = []
    if "item_id" in df:
        if df.item_id.isna().any():
            errors.append("item_id has nulls")
        if df.item_id.duplicated().any():
            errors.append(f"item_id has {int(df.item_id.duplicated().sum())} duplicates")
    if "ts" in df:
        if not isinstance(df.ts.dtype, pd.DatetimeTZDtype):
            errors.append(f"ts must be datetime64[ns, UTC], got {df.ts.dtype}")
        elif str(df.ts.dt.tz) != "UTC":
            errors.append(f"ts must be UTC, got {df.ts.dt.tz}")
    if "kind" in df:
        bad = set(df.kind.dropna().unique()) - KINDS
        if bad:
            errors.append(f"unknown kind(s) {sorted(bad)}; expected {sorted(KINDS)}")
    if "engagement" in df and not pd.api.types.is_numeric_dtype(df.engagement):
        errors.append(f"engagement must be numeric, got {df.engagement.dtype}")
    if "parent" in df and not df.parent.dropna().map(lambda x: isinstance(x, str)).all():
        errors.append("parent must hold item_id strings or nulls")
    if errors:
        raise SchemaError("; ".join(errors))
    return df


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a reader's output to the schema: UTC timestamps, string ids, float engagement, standard columns
    first (extra columns kept after them). Missing optional columns are not added."""
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    for c in ("item_id", "author", "community", "kind", "unit", "parent"):
        if c in d:
            d[c] = pd.Series([None if _null(x) else str(x) for x in d[c]], index=d.index, dtype=object)
    if "engagement" in d:
        d["engagement"] = pd.to_numeric(d["engagement"], errors="coerce").astype(float)
    if "phash" in d:
        d["phash"] = pd.Series([None if _null(h) else np.uint64(int(h, 16) if isinstance(h, str) else int(h))
                                for h in d["phash"]], index=d.index, dtype=object)
    cols = [c for c in COLUMNS if c in d] + [c for c in d.columns if c not in COLUMNS]
    return d[cols].reset_index(drop=True)


def _null(x) -> bool:
    return x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x))


def read_items(path: str | Path, validate_schema: bool = True) -> pd.DataFrame:
    """Read an items table from ``.parquet``, ``.jsonl``/``.json`` or ``.csv`` and conform it to the schema."""
    p = Path(path)
    suf = "".join(p.suffixes[-2:]) if p.suffix == ".gz" else p.suffix
    if suf == ".parquet":
        df = pd.read_parquet(p)
    elif suf in (".jsonl", ".jsonl.gz", ".json"):
        df = pd.read_json(p, lines=suf != ".json")
    elif suf in (".csv", ".csv.gz"):
        df = pd.read_csv(p)
    else:
        raise SchemaError(f"unsupported file type {p.suffix!r} (use parquet, jsonl or csv)")
    df = conform(df)
    return validate(df) if validate_schema else df


def write_items(df: pd.DataFrame, path: str | Path) -> Path:
    """Write an items table (parquet recommended; ``phash`` is stored as uint64 when present)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if "phash" in d:
        d["phash"] = pd.array([None if h is None else int(h) for h in d["phash"]], dtype="UInt64")
    if p.suffix == ".parquet":
        d.to_parquet(p, index=False)
    elif p.suffix == ".csv":
        d.to_csv(p, index=False)
    else:
        d.to_json(p, orient="records", lines=True, date_format="iso")
    return p
