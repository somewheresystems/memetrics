"""Reader for Pump.fun-style token data: launch records become the items table, and daily candles become fitness series.

Inputs are plain DataFrames and parsed JSON. This module makes no network calls, uses no RPC keys and contains no
paths. The fitness definitions follow the Pump.fun launch study that this toolkit was first applied to. Its "robust"
peak counts only days with ≥ $10k USD volume and ≥ 1% of supply traded, which removes single-trade spikes in thin
pools.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPLY = 1e9  # Pump.fun tokens have 1e9 whole tokens

from memetrics.io import hash_author


def to_items(launches: pd.DataFrame, salt: str, *, id_col="mint", ts_col="created_at", author_col="creator",
             name_col="name", ticker_col="symbol", phash_col=None, topic_col=None, engagement_col=None,
             community: str = "pumpfun") -> pd.DataFrame:
    """Map a launch table to the memetrics items table (`kind="token"`).

    The token-specific columns `ticker` and `topic` are kept. A pHash given as a hex string is parsed to uint64.
    """
    d = launches
    out = pd.DataFrame({
        "item_id": d[id_col].astype(str).to_numpy(),
        "ts": pd.to_datetime(d[ts_col], utc=True, format="ISO8601").to_numpy(),
        "author": [hash_author(a, salt) if isinstance(a, str) else None for a in d[author_col]],
        "community": community,
        "kind": "token",
        "text": d[name_col].to_numpy(),
        "ticker": d[ticker_col].to_numpy(),
    })
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    if phash_col:
        out["phash"] = [np.uint64(int(h, 16)) if isinstance(h, str) and h else np.nan for h in d[phash_col]]
        out["phash"] = out["phash"].astype(object)
    if topic_col:
        out["topic"] = d[topic_col].to_numpy()
    if engagement_col:
        out["engagement"] = d[engagement_col].to_numpy(float)
    return out


def candles_frame(candles: list[dict], supply: float = SUPPLY) -> pd.DataFrame:
    """Pump.fun daily candles (USD per whole token; volume in tokens) become day, mc_*, vol_usd and turnover."""
    cols = ["day", "mc_open", "mc_high", "mc_low", "mc_close", "vol_usd", "turnover"]
    if not candles:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(candles)
    out = pd.DataFrame({"day": pd.to_datetime(d["timestamp"].astype("int64"), unit="ms", utc=True)})
    for c in ("open", "high", "low", "close"):
        out[f"mc_{c}"] = d[c].astype(float) * supply
    vt = d["volume"].astype(float)
    out["vol_usd"] = vt * d["close"].astype(float)
    out["turnover"] = vt / supply
    return out.sort_values("day").reset_index(drop=True)[cols]


def qualifying(cf: pd.DataFrame, min_usd: float = 10_000, min_turnover: float = 0.01) -> pd.Series:
    """Robust peak day rule: USD volume ≥ `min_usd` and turnover ≥ `min_turnover` of supply."""
    return (cf["vol_usd"] >= min_usd) & (cf["turnover"] >= min_turnover)


def peak_close(cf: pd.DataFrame) -> float:
    """Maximum daily close market cap (unfiltered)."""
    return float(cf["mc_close"].max()) if len(cf) else np.nan


def peak_capped_high(cf: pd.DataFrame, k: float = 3.0) -> float:
    """Wick-capped peak: the maximum over days of min(high, k·max(open, close))."""
    if not len(cf):
        return np.nan
    return float(np.minimum(cf["mc_high"], k * np.maximum(cf["mc_open"], cf["mc_close"])).max())


def peak_to_date(cf: pd.DataFrame, before: pd.Timestamp) -> float:
    """Peak known before `before`'s UTC day.

    The peak is the maximum close over qualifying days if any exist, else over all prior days. NaN if there are no
    prior days.
    """
    prior = cf[cf["day"] < pd.Timestamp(before).tz_convert("UTC").normalize()]
    if prior.empty:
        return np.nan
    q = prior[qualifying(prior)]
    return float((q if not q.empty else prior)["mc_close"].max())


def activity_before(cf: pd.DataFrame, before: pd.Timestamp, days: int = 7) -> tuple[float, float]:
    """Return (USD volume over the `days` days before `before`'s UTC day, last close market cap before that day)."""
    day = pd.Timestamp(before).tz_convert("UTC").normalize()
    prior = cf[cf["day"] < day]
    if prior.empty:
        return 0.0, np.nan
    w = prior[prior["day"] >= day - pd.Timedelta(days=days)]
    return float(w["vol_usd"].sum()), float(prior["mc_close"].iloc[-1])
