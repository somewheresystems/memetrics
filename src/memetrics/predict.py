"""Non-redundancy by out-of-time prediction (N1): nested models, held-out delta log-likelihood and delta AUC.

Design: fit on earlier data, evaluate on later data. Feature blocks:
  B  rival baseline (author, timing, community, attention competition),
  CA cumulative advantage (prior frequency / prior total engagement of the unit),
  CB cultural-evolution content bias (generic content features with no unit identity),
  M  memetic replicator terms (heritable unit fitness from earlier instances, mutation distance, unit age).
Memetic terms are non-redundant if they improve held-out fit over B and over B+CA+CB (both CIs > 0).

Helpers here build causal unit features: ``prior_unit_features`` uses only instances dated at least ``lag`` before
each item. Heritable fitness is an empirical-Bayes-shrunk mean residual (shrinkage n / (n + k)), as in target
encoding with smoothing (Micci-Barreca 2001, SIGKDD Explor. 3(1):27-32, doi:10.1145/507533.507538).
CIs: cluster bootstrap over a grouping column (e.g. unit) of per-item differences in log-likelihood and in AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .result import Result


def prior_unit_features(
    df: pd.DataFrame,
    unit: str,
    ts: str,
    value: str | None = None,
    lag: float = 0.0,
    k: float = 5.0,
    prefix: str | None = None,
) -> pd.DataFrame:
    """For each row, statistics of earlier rows of the same unit with ``ts < row.ts - lag`` (strictly earlier;
    ``ts`` numeric, e.g. seconds):
    ``n`` (prior count), ``age`` (ts - first ts of the unit, available without lag since the first appearance is
    public), and if ``value`` is given ``fit`` = n/(n+k) * mean(value) (shrunk toward 0; pass residuals) and ``sum``.
    Rows with missing unit get n = 0, fit = 0, age = NaN."""
    p = prefix or unit
    out = pd.DataFrame(index=df.index, data={f"{p}_n": 0.0, f"{p}_fit": 0.0, f"{p}_sum": 0.0, f"{p}_age": np.nan})
    d = df[[unit, ts] + ([value] if value else [])].dropna(subset=[unit])
    for _, g in d.groupby(unit, sort=False):
        g = g.sort_values(ts)
        t = g[ts].to_numpy(float)
        n = np.searchsorted(t, t - lag, side="left")
        out.loc[g.index, f"{p}_n"] = n
        out.loc[g.index, f"{p}_age"] = t - t[0]
        if value:
            cs = np.concatenate([[0.0], np.cumsum(g[value].to_numpy(float))])
            s = cs[n]
            out.loc[g.index, f"{p}_sum"] = s
            out.loc[g.index, f"{p}_fit"] = s / (n + k) if k > 0 else np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return out


def _fit_predict(Xtr, ytr, Xte, model: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "logit":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    m.fit(Xtr, ytr)
    return np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def nested_compare(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: str,
    blocks: dict[str, list[str]],
    comparisons: list[tuple[list[str], list[str]]],
    cluster: str,
    model: str = "logit",
    n_boot: int = 2000,
    seed: int = 0,
    min_auc_gain: float = 0.0,
) -> list[Result]:
    """Held-out comparisons of nested logistic models.

    ``blocks``: name -> feature columns. ``comparisons``: (base block names, added block names). For each, returns a
    Result whose estimate is the mean per-item Delta log-likelihood (nats; added minus base), with cluster-bootstrap
    CI over ``cluster``; ``detail`` holds Delta AUC and its CI. verdict "adds" if both CIs exclude 0 on the positive
    side (and the AUC gain >= ``min_auc_gain``), else "no gain".
    """
    rng = np.random.default_rng(seed)
    ytr, yte = train[y].to_numpy(int), test[y].to_numpy(int)
    cl = pd.factorize(test[cluster].fillna("__none__"))[0]
    ncl = cl.max() + 1
    preds = {}

    def pred(names):
        key = tuple(sorted(names))
        if key not in preds:
            cols = [c for b in key for c in blocks[b]]
            preds[key] = _fit_predict(train[cols].to_numpy(float), ytr, test[cols].to_numpy(float), model, seed)
        return preds[key]

    out = []
    for base, add in comparisons:
        p0, p1 = pred(base), pred(base + add)
        ll0 = yte * np.log(p0) + (1 - yte) * np.log(1 - p0)
        ll1 = yte * np.log(p1) + (1 - yte) * np.log(1 - p1)
        dll = ll1 - ll0
        dauc = _auc(yte, p1) - _auc(yte, p0)
        bl, ba = [], []
        for _ in range(n_boot):
            w = rng.poisson(1.0, ncl)[cl]
            m = w > 0
            idx = np.repeat(np.flatnonzero(m), w[m])
            bl.append(dll[idx].mean())
            ba.append(_auc(yte[idx], p1[idx]) - _auc(yte[idx], p0[idx]))
        lo, hi = np.nanquantile(bl, [0.025, 0.975])
        alo, ahi = np.nanquantile(ba, [0.025, 0.975])
        adds = lo > 0 and alo > 0 and dauc >= min_auc_gain
        name = f"{'+'.join(base)} -> +{'+'.join(add)}"
        out.append(
            Result(
                name,
                float(dll.mean()),
                float(lo),
                float(hi),
                len(test),
                {
                    "delta_auc": float(dauc),
                    "delta_auc_ci": (float(alo), float(ahi)),
                    "auc_base": float(_auc(yte, p0)),
                    "auc_full": float(_auc(yte, p1)),
                    "model": model,
                },
                "adds" if adds else "no gain",
            )
        )
    return out


def nonredundancy_verdict(r_b_m: Result, r_key: Result) -> str:
    """Study rule: non-redundant if B -> B+M adds AND B+CA+CB -> B+CA+CB+M adds; "reproduced by rivals" if only the
    first; "no memetic gain" otherwise."""
    if r_b_m.verdict == "adds" and r_key.verdict == "adds":
        return "non-redundant"
    if r_b_m.verdict == "adds":
        return "reproduced by rivals"
    return "no memetic gain"
