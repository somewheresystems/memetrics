"""Memecoin families, known-meme matching and parent–offspring heritability of market fitness.

Works on the items table (:mod:`memetrics.schema`) with ``kind == "token"``:

| column | use here |
|---|---|
| `item_id` | token id (e.g. a mint address) |
| `ts` | launch time, UTC |
| `author` | salted creator hash |
| `text` | token name |
| `ticker` | *optional, token-specific*: ticker / symbol |
| `topic` | *optional*: a canonical subject label (e.g. from a local captioner) used as an extra match key |
| `phash` | *optional*: 64-bit perceptual hash of the token image |
| `engagement` | outcome, e.g. log10 peak market cap |
| `parent` | filled by `assign_parents` |

The estimators are:
- **Parent–offspring regression** (`heritability`). Galton (1886) regressed offspring on parent; Falconer &
  Mackay (1996, *Introduction to Quantitative Genetics*, ch. 10) treat the slope b_PO as the basis of heritability
  estimates. For culture the analogue is the regression of a copy's success on its model's success. The regression is
  OLS with HC3 errors (MacKinnon & White 1985), and the CI comes from a cluster (block) bootstrap over time blocks
  (Cameron, Gelbach & Miller 2008).
- **Shuffled-parent null**: parents are permuted within strata, a permutation test in the sense of Good (2005,
  *Permutation, Parametric and Bootstrap Tests of Hypotheses*). Keeping the strata fixed preserves timing, so
  common-wave shocks are not mistaken for inheritance.
- **Prior popularity versus recent spike** of a parent meme, from daily attention counts such as Wikipedia pageviews
  (Wikimedia REST API). Both are measured strictly before launch, separating long-run fitness from current attention.
  This is the distinction between content fitness and limited-attention dynamics drawn by Weng et al. (2012,
  *Sci. Rep.* 2:335).
- **Families**: earlier items whose normalised name or ticker is identical, or whose pHash is within a Hamming radius
  (Zauner 2010, pHash benchmark thesis; default radius 6).

No network calls and no project paths. Platform readers live in `memetrics.io.onchain`.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

import numpy as np
import pandas as pd

from .result import Result
from .units import hamming64

STRIP_PREFIX = ("the ", "$")
STRIP_SUFFIX = (" coin", " token", " on solana", " on sol", " sol")
CRYPTO_WORDS = re.compile(r"\b(coin|coins|token|tokens|crypto|solana|pump|memecoin|shitcoin|dogecoin|bitcoin|nft)\b")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍\U000E0000-\U000E007F]")


# ------------------------------------------------------------------ names and catalogue matching

def norm(s) -> str:
    """Normalise a name: NFKC, casefold, emoji and punctuation removed, whitespace collapsed."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _EMOJI.sub(" ", s)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def strip_affixes(n: str) -> str:
    """Remove promotional affixes ('the', '$', 'coin', 'token', 'on sol(ana)') from a normalised name."""
    prev = None
    while prev != n:
        prev = n
        for p in STRIP_PREFIX:
            if n.startswith(p):
                n = n[len(p):].strip()
        for x in STRIP_SUFFIX:
            if n.endswith(x):
                n = n[: -len(x)].strip()
    return n


def alias_ok(alias: str, dictionary: set[str], min_len: int = 4) -> bool:
    """Catalogue alias filter.

    An alias is kept only if it has at least `min_len` characters, contains no crypto word, and is not a single
    dictionary word. "dog" names a category, not a meme.
    """
    if len(alias) < min_len or CRYPTO_WORDS.search(alias):
        return False
    return not (" " not in alias and alias in dictionary)


def build_alias_index(aliases: pd.DataFrame) -> dict[str, list[str]]:
    """Build the alias index from a frame with columns `alias` (already normalised) and `id` (catalogue entry)."""
    idx: dict[str, list[str]] = {}
    for a, i in zip(aliases["alias"], aliases["id"]):
        idx.setdefault(a, []).append(i)
    return idx


def match_keys(name, ticker=None, topic=None, min_len: int = 4) -> set[str]:
    """Exact-match lookup keys for one item: the name, the name without affixes, the ticker and the topic."""
    keys = set()
    n = norm(name)
    if n:
        keys |= {n, strip_affixes(n)}
    for extra in (ticker, topic):
        e = norm(extra)
        if e:
            keys.add(e)
    return {k for k in keys if len(k) >= min_len}


def match_catalogue(items: pd.DataFrame, alias_index: dict[str, list[str]]) -> pd.Series:
    """Catalogue entries matched by exact alias equality, as one sorted list of ids per item.

    Substring matching is not used: precision is preferred over recall.
    """
    tick = items["ticker"] if "ticker" in items else pd.Series(None, index=items.index)
    topic = items["topic"] if "topic" in items else pd.Series(None, index=items.index)
    out = []
    for name, t, k in zip(items["text"], tick, topic):
        ids = set()
        for key in match_keys(name, t, k):
            ids.update(alias_index.get(key, ()))
        out.append(sorted(ids))
    return pd.Series(out, index=items.index)


def attention_features(views: pd.Series, t: pd.Timestamp, long=(365, 31), recent=(7, 1)) -> tuple[float, float]:
    """Prior popularity and recent spike of a parent meme, strictly before time t.

    `views` holds daily counts indexed by UTC midnight; missing days count as 0.
    - pop = log10(1 + mean over days [t-long[0], t-long[1]])
    - spike = log10(1 + mean over days [t-recent[0], t-recent[1]]) - pop

    If no counts exist before t (the meme did not exist yet), pop is 0.
    """
    day = pd.Timestamp(t).tz_convert("UTC").normalize().tz_localize(None)
    if getattr(views.index, "tz", None) is not None:
        views = views.tz_convert("UTC").tz_localize(None)

    def mean_between(a, b):
        idx = pd.date_range(day - pd.Timedelta(days=a), day - pd.Timedelta(days=b), freq="D")
        return float(views.reindex(idx, fill_value=0).mean())

    pop = float(np.log10(1 + mean_between(*long)))
    return pop, float(np.log10(1 + mean_between(*recent)) - pop)


# ------------------------------------------------------------------ families

def _hours(ts: pd.Series) -> np.ndarray:
    """UTC timestamps as float hours since the epoch (NaT -> NaN), independent of the datetime unit."""
    t = pd.to_datetime(pd.Series(ts), utc=True)
    return ((t - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(hours=1)).to_numpy(float)


def find_relatives(offspring: pd.DataFrame, pool: pd.DataFrame, max_hamming: int = 6,
                   min_ticker: int = 3) -> pd.DataFrame:
    """Find the earlier relatives of each offspring item in `pool`.

    A pool item is a relative if it has the same affix-stripped normalised name (`text`), the same normalised `ticker`
    (at least `min_ticker` characters), or a pHash within `max_hamming`.

    Returns a frame with columns `item_id`, `rel_id`, `via` (`name`, `image` or `both`) and `gap_h`, the hours from
    the relative's launch to the offspring's launch. Only relatives with `gap_h > 0` are returned.
    """
    pool = pool.reset_index(drop=True)
    kname = pool["text"].map(lambda s: strip_affixes(norm(s)))
    ktick = pool["ticker"].map(norm) if "ticker" in pool else pd.Series("", index=pool.index)
    by_name: dict[str, list[int]] = {}
    by_tick: dict[str, list[int]] = {}
    for i, (kn, kt) in enumerate(zip(kname, ktick)):
        if len(kn) >= 2:
            by_name.setdefault(kn, []).append(i)
        if len(kt) >= min_ticker:
            by_tick.setdefault(kt, []).append(i)
    has_ph = "phash" in pool and pool["phash"].notna().any()
    if has_ph:
        ph = pool[pool["phash"].notna()]
        ph_i, ph_v = ph.index.to_numpy(), ph["phash"].astype("uint64").to_numpy()
    ts, ids = _hours(pool["ts"]), pool["item_id"].to_numpy()
    off_ns = _hours(offspring["ts"])
    rows = []
    for r, t0 in zip(offspring.itertuples(index=False), off_ns):
        kn = strip_affixes(norm(r.text))
        kt = norm(getattr(r, "ticker", ""))
        via: dict[int, str] = {i: "name" for i in by_name.get(kn, [])}
        if len(kt) >= min_ticker:
            via.update({i: "name" for i in by_tick.get(kt, [])})
        p = getattr(r, "phash", None)
        if has_ph and p is not None and not (isinstance(p, float) and np.isnan(p)):
            for i in ph_i[hamming64(ph_v, int(p)) <= max_hamming]:
                via[i] = "both" if via.get(i) == "name" else "image"
        for i, v in via.items():
            if ids[i] == r.item_id:
                continue
            gap = t0 - ts[i]
            if gap > 0:
                rows.append((r.item_id, ids[i], v, float(gap)))
    return pd.DataFrame(rows, columns=["item_id", "rel_id", "via", "gap_h"])


def assign_parents(rel: pd.DataFrame, fitness: Callable[[str, str], float] | None = None,
                   min_gap_h: float = 24.0, rule: str = "fittest") -> pd.DataFrame:
    """Choose one parent per offspring from its relatives.

    Relatives launched less than `min_gap_h` hours earlier are *siblings*: both probably copy a common external
    source.

    - `rule="fittest"`: choose the candidate parent with the highest fitness known at the offspring's launch,
      `fitness(rel_id, item_id)`. That is the relative a copier would choose. Candidates whose fitness is NaN are
      dropped.
    - `rule="earliest"`: choose the oldest candidate.

    Returns one row per offspring with `parent`, `parent_fitness`, `parent_gap_h`, `n_relatives` and `n_siblings`.
    """
    out = []
    for oid, g in rel.groupby("item_id", sort=False):
        sib = g[g.gap_h < min_gap_h]
        cand = g[g.gap_h >= min_gap_h].copy()
        if fitness is not None and not cand.empty:
            cand["fit"] = [fitness(p, oid) for p in cand.rel_id]
            cand = cand[np.isfinite(cand.fit.astype(float))]
        else:
            cand["fit"] = np.nan
        row = {"item_id": oid, "n_relatives": len(g), "n_siblings": len(sib), "parent": None,
               "parent_fitness": np.nan, "parent_gap_h": np.nan, "parent_via": None}
        if not cand.empty:
            best = cand.loc[cand.fit.idxmax()] if rule == "fittest" else cand.loc[cand.gap_h.idxmax()]
            row.update(parent=best.rel_id, parent_fitness=float(best.fit), parent_gap_h=float(best.gap_h),
                       parent_via=best.via)
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ inference

def time_blocks(ts: pd.Series, freq: str = "W") -> np.ndarray:
    """Calendar blocks (default ISO weeks) used as bootstrap clusters."""
    t = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
    return t.dt.to_period(freq).astype(str).to_numpy()


def block_bootstrap(stat: Callable[[np.ndarray], float], blocks: np.ndarray, n_boot: int = 2000,
                    seed: int = 0):
    """Cluster bootstrap that resamples whole blocks with replacement (Cameron, Gelbach & Miller 2008).

    Returns (point, 2.5th percentile, 97.5th percentile, share of draws ≤ 0, draws).
    """
    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks)
    ub = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in ub}
    point = float(stat(np.arange(len(blocks))))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(ub, size=len(ub), replace=True)
        draws[i] = stat(np.concatenate([members[b] for b in pick]))
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), float((draws <= 0).mean()), draws


def ols_hc3(y, X):
    """OLS with an intercept. Returns the coefficients (intercept first) and HC3 standard errors (MacKinnon & White 1985)."""
    y = np.asarray(y, float)
    X1 = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    inv = np.linalg.pinv(X1.T @ X1)
    beta = inv @ X1.T @ y
    e = y - X1 @ beta
    h = np.einsum("ij,jk,ik->i", X1, inv, X1)
    meat = (X1 * (e / (1 - np.clip(h, 0, 0.999)))[:, None] ** 2).T @ X1
    return beta, np.sqrt(np.diag(inv @ meat @ inv))


def heritability(items: pd.DataFrame, parent_col: str = "parent_fitness", y_col: str = "engagement",
                 controls: list[str] | None = None, block_col: str | None = None, strata_col: str | None = None,
                 n_boot: int = 2000, n_perm: int = 2000, seed: int = 0, alpha: float = 0.05,
                 name: str = "b_PO") -> Result:
    """Parent–offspring regression slope b_PO of offspring `y_col` on `parent_col`, given `controls`.

    Model: y = a + b_PO·parent + γ·controls + ε (OLS; Galton 1886; Falconer & Mackay 1996).

    - The CI is a block bootstrap over `block_col`, by default ISO weeks of `ts`.
    - The null distribution permutes parent values within `strata_col` (by default the launch month), with the
      controls held fixed. This keeps timing and breaks only the parent–offspring pairing. `p_null` is the share of
      null slopes ≥ b_PO.
    - Verdict: `heritable` if the CI excludes 0 (b > 0) and p_null < alpha/2; otherwise `not heritable`.
    """
    d = items.dropna(subset=[y_col, parent_col] + (controls or []))
    y, par = d[y_col].to_numpy(float), d[parent_col].to_numpy(float)
    Z = d[controls].to_numpy(float) if controls else np.empty((len(d), 0))
    blocks = d[block_col].to_numpy() if block_col else time_blocks(d["ts"])
    strata = d[strata_col].to_numpy() if strata_col else time_blocks(d["ts"], "M")

    def slope(idx, p=par):
        return ols_hc3(y[idx], np.column_stack([p[idx], Z[idx]]))[0][1]

    b, lo, hi, p_boot, _ = block_bootstrap(slope, blocks, n_boot, seed)
    rng = np.random.default_rng(seed + 1)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    allidx = np.arange(len(y))
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = par.copy()
        for g in groups:
            perm[g] = par[rng.permutation(g)]
        null[i] = slope(allidx, perm)
    p_null = float((null >= b).mean())
    beta, se = ols_hc3(y, np.column_stack([par, Z]))
    verdict = "heritable" if (lo > 0 and p_null < alpha / 2) else "not heritable"
    return Result(name, float(b), lo, hi, int(len(y)),
                  {"p_boot_le0": p_boot, "null_q975": float(np.quantile(null, 0.975)), "p_null": p_null,
                   "se_hc3": float(se[1]), "controls": list(controls or []),
                   "coef": dict(zip(["intercept", parent_col] + list(controls or []), map(float, beta))),
                   "se": dict(zip(["intercept", parent_col] + list(controls or []), map(float, se)))},
                  verdict)


def heritability_verdict(plain: Result, attention_adjusted: Result, attention_terms: dict[str, Result]) -> str:
    """Combine the decision rule for inheritance versus attention-chasing.

    - `memetic inheritance`: b_PO passes, and so does b_PO with attention terms added.
    - `attention-chasing`: b_PO passes, the adjusted slope fails, and an attention term's CI excludes 0 (> 0).
    - `no heritability`: b_PO fails.
    - `ambiguous`: anything else.
    """
    if plain.verdict != "heritable":
        return "no heritability"
    if attention_adjusted.ci_low > 0:
        return "memetic inheritance"
    if any(r.ci_low > 0 for r in attention_terms.values()):
        return "attention-chasing"
    return "ambiguous"


def coef_ci(d: pd.DataFrame, y_col: str, terms: list[str], controls: list[str], blocks: np.ndarray,
            n_boot: int = 2000, seed: int = 0, logistic: bool = False) -> dict[str, Result]:
    """Block-bootstrap CIs for the coefficients of `terms` in a regression of `y_col` on terms + controls.

    The regression is OLS, or an unpenalised logit fit by Newton's method (statsmodels-free) when `logistic`.
    Verdict per term: `positive`, `negative` or `null`, by the sign of a CI that excludes 0.
    """
    cols = terms + controls
    X = d[cols].to_numpy(float)
    y = d[y_col].to_numpy(float)

    def fit(idx):
        if logistic:
            return _logit(y[idx], X[idx])
        return ols_hc3(y[idx], X[idx])[0]

    out = {}
    for j, t in enumerate(terms):
        b, lo, hi, p, _ = block_bootstrap(lambda idx: fit(idx)[1 + j], blocks, n_boot, seed + j)
        v = "positive" if lo > 0 else "negative" if hi < 0 else "null"
        out[t] = Result(t, b, lo, hi, len(y), {"p_le0": p, "model": "logit" if logistic else "ols"}, v)
    return out


def _logit(y, X, ridge: float = 1e-4, iters: int = 50):
    X1 = np.column_stack([np.ones(len(y)), X])
    sd = X1.std(0)
    sd[0] = 1
    sd[sd == 0] = 1
    Xs = X1 / sd
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xs @ b, -30, 30)))
        g = Xs.T @ (y - p) - ridge * b
        H = (Xs * (p * (1 - p))[:, None]).T @ Xs + ridge * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b / sd
