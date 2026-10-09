"""Text normalization, similarity, near-duplicate units and unit-discreteness statistics (M1 for text).

- ``normalize(t)``: NFKC, lowercase, URLs -> ``<url>``, @handles -> ``<user>``, punctuation except ' and # removed,
  whitespace collapsed.
- ``sim_edit(a, b) = 1 - Levenshtein_char / max(len)``: the primary similarity. ``sim_word``: the same on word
  tokens.
- ``minhash_units(texts, tau)``: connected components of word-5-shingle Jaccard >= tau (MinHash LSH plus an exact
  check of candidate pairs).
- ``unit_stats(items, labels)``: medoids, within/between similarity, Ashman's D per unit, blend flags.
- ``m1_test(items_table)``: the M1 decision on an items table (columns ``text``, ``unit``).

Ashman's D: Ashman, Bird & Zepf (1994). Silhouette: Rousseeuw (1987). Using D on own-medoid versus
nearest-other-medoid similarities, and the blend rule, are this package's operationalization (shared with
:mod:`memetrics.units`, which works on embeddings).

Similarity matrices use ``$MEMETRICS_WORKERS`` rapidfuzz threads (default -1 = all cores; set 1 inside process
pools).
"""

from __future__ import annotations

import os
import re
import unicodedata
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .result import Result
from .schema import require
from .units import ashman_d, m1_verdict

WORKERS = int(os.environ.get("MEMETRICS_WORKERS", "-1"))

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"(?<!\w)@[\w.\-]+")
_PUNCT = re.compile(r"[^\w\s'#<>]")
_WS = re.compile(r"\s+")


def normalize(t: str | None) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t).lower()
    t = _URL.sub(" <url> ", t)
    t = _USER.sub(" <user> ", t)
    t = t.replace("_", " ")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def sim_edit(a: str, b: str) -> float:
    """1 - Levenshtein(a, b) / max(|a|, |b|); 1.0 for two empty strings."""
    return Levenshtein.normalized_similarity(a, b)


def sim_word(a: str, b: str) -> float:
    """``sim_edit`` on word tokens."""
    return Levenshtein.normalized_similarity(a.split(), b.split())


def sim_matrix(a: list[str], b: list[str] | None = None, words: bool = False, workers: int | None = None) -> np.ndarray:
    """Pairwise similarity matrix (float32)."""
    if words:
        qa = [x.split() for x in a]
        qb = qa if b is None else [x.split() for x in b]
    else:
        qa, qb = a, (a if b is None else b)
    return process.cdist(
        qa,
        qb,
        scorer=Levenshtein.normalized_similarity,
        dtype=np.float32,
        workers=WORKERS if workers is None else workers,
    )


# ---------------------------------------------------------------- near-duplicate units (MinHash LSH)


def shingles(t: str, k: int = 5) -> set[int]:
    w = t.split()
    if len(w) < k:
        return {zlib.crc32(" ".join(w).encode())} if w else set()
    return {zlib.crc32(" ".join(w[i : i + k]).encode()) for i in range(len(w) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, x: int) -> int:
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.p))])


def minhash_units(
    texts: list[str], tau: float = 0.5, k: int = 5, n_perm: int = 128, bands: int = 64, seed: int = 20261008
) -> np.ndarray:
    """Cluster label per text: connected components of verified Jaccard(5-shingles) >= tau."""
    rng = np.random.default_rng(seed)
    P = (1 << 61) - 1
    a = rng.integers(1, P, n_perm, dtype=np.uint64)
    b = rng.integers(0, P, n_perm, dtype=np.uint64)
    sh = [shingles(t, k) for t in texts]
    rows = n_perm // bands
    uf = UnionFind(len(texts))
    sig = np.full((len(texts), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(sh):
        if not s:
            continue
        x = np.fromiter(s, dtype=np.uint64)
        # (a*x + b) mod 2^64 is a fine universal-ish hash for this purpose
        h = np.outer(x, a) + b
        sig[i] = h.min(axis=0)
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        block = sig[:, band * rows : (band + 1) * rows]
        for i in range(len(texts)):
            if sh[i]:
                buckets.setdefault(block[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) <= 60:  # verify every pair in small buckets
                pairs = ((x_, y_) for k_, x_ in enumerate(members) for y_ in members[k_ + 1 :])
            else:  # large buckets: star from the first member plus consecutive pairs
                pairs = list(zip([members[0]] * (len(members) - 1), members[1:])) + list(zip(members, members[1:]))
            for x_, y_ in pairs:
                if uf.find(x_) != uf.find(y_) and jaccard(sh[x_], sh[y_]) >= tau:
                    uf.union(x_, y_)
    return uf.labels()


def threshold_components(texts: list[str], tau: float, words: bool = False) -> np.ndarray:
    """Connected components of sim >= tau (exact all-pairs; use for <= ~20k texts)."""
    uf = UnionFind(len(texts))
    step = 2000
    for i in range(0, len(texts), step):
        S = sim_matrix(texts[i : i + step], texts, words=words)
        r, c = np.nonzero(S >= tau)
        for x, y in zip(r + i, c):
            if x < y:
                uf.union(int(x), int(y))
    return uf.labels()


# ---------------------------------------------------------------- M1 statistics


def medoid(
    items: list[str], weights: np.ndarray | None = None, words: bool = False, cap: int = 1500, seed: int = 20261008
) -> int:
    """Index of the item maximizing summed similarity to the others (sampled to `cap` columns)."""
    n = len(items)
    if n == 1:
        return 0
    rng = np.random.default_rng(seed)
    cols = np.arange(n) if n <= cap else rng.choice(n, cap, replace=False)
    S = sim_matrix(items, [items[j] for j in cols], words=words)
    score = S.sum(1)
    if weights is not None:
        score = score + 1e-9 * np.asarray(weights, float)  # tie-break by frequency
    return int(np.argmax(score))


@dataclass
class UnitResult:
    unit_ids: np.ndarray  # analysed unit labels
    D: np.ndarray  # Ashman's D per unit
    n_items: np.ndarray
    item_unit: np.ndarray  # per item: unit label
    w: np.ndarray  # per item: sim to own medoid
    b: np.ndarray  # per item: sim to nearest other medoid
    nearest: np.ndarray  # per item: nearest other unit
    blend: np.ndarray  # per item: blend flag
    medoids: dict

    def summary(self) -> dict:
        D = _d_for_median(self.D)
        D = D[np.isfinite(D)]
        return {
            "n_units": int(len(self.unit_ids)),
            "n_items": int(len(self.w)),
            "median_D": float(np.median(D)) if len(D) else np.nan,
            "pooled_D": _ashman64(self.w, self.b),
            "blend_share": float(self.blend.mean()) if len(self.blend) else np.nan,
        }


def _candidate_medoids(items: list[str], med_texts: list[str], k: int) -> np.ndarray:
    """Top-k candidate medoids per item by char 3-gram TF-IDF cosine (exact sim is then computed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, dtype=np.float32)
    M = vec.fit_transform(med_texts)
    X = vec.transform(items)
    k = min(k, len(med_texts))
    out = np.empty((len(items), k), dtype=np.int64)
    for i in range(0, len(items), 4096):
        S = (X[i : i + 4096] @ M.T).toarray()
        out[i : i + 4096] = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return out


def unit_stats(
    items: list[str],
    labels: np.ndarray,
    weights: np.ndarray | None = None,
    min_items: int = 5,
    words: bool = False,
    k_candidates: int = 30,
    exact_limit: int = 3000,
) -> UnitResult:
    """Within/between similarity, Ashman's D per unit, and blend flags.

    items: distinct variants; labels: unit label per item. Units with < min_items items still serve as
    'other units' for the between-similarity but are not analysed themselves.
    """
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    wts = np.ones(len(items)) if weights is None else np.asarray(weights, float)
    med_idx = np.array([g[medoid([items[j] for j in g], wts[g], words=words)] for g in groups])
    med_texts = [items[j] for j in med_idx]
    # similarity of every item to its own medoid
    w = np.array(
        [
            Levenshtein.normalized_similarity(
                *((items[i].split(), med_texts[inv[i]].split()) if words else (items[i], med_texts[inv[i]]))
            )
            for i in range(len(items))
        ]
    )
    # nearest other medoid
    if len(uniq) <= exact_limit:
        S = sim_matrix(items, med_texts, words=words)
        S[np.arange(len(items)), inv] = -1
        nearest = S.argmax(1)
        b = S[np.arange(len(items)), nearest]
    else:
        cand = _candidate_medoids(items, med_texts, k_candidates)
        nearest = np.empty(len(items), dtype=np.int64)
        b = np.empty(len(items))
        for i in range(len(items)):
            cs = [c for c in cand[i] if c != inv[i]]
            sims = [sim_word(items[i], med_texts[c]) if words else sim_edit(items[i], med_texts[c]) for c in cs]
            j = int(np.argmax(sims))
            nearest[i], b[i] = cs[j], sims[j]
    # blend threshold: 10th percentile of each unit's within-similarities
    q10 = np.array([np.quantile(w[g], 0.10) for g in groups])
    blend = b >= q10[nearest]
    analysed = [g for g in range(len(uniq)) if len(groups[g]) >= min_items]
    D = np.array([_ashman64(w[groups[g]], b[groups[g]]) for g in analysed])
    keep = np.isin(inv, analysed)
    return UnitResult(
        unit_ids=uniq[analysed],
        D=D,
        n_items=np.array([len(groups[g]) for g in analysed]),
        item_unit=labels[keep],
        w=w[keep],
        b=b[keep],
        nearest=uniq[nearest][keep],
        blend=blend[keep],
        medoids={uniq[g]: med_texts[g] for g in range(len(uniq))},
    )


def gap_ratio(
    items: list[str], labels: np.ndarray, nearest_unit: dict, cap: int = 300, words: bool = False, seed: int = 20261008
) -> dict:
    """Boundary-gap diagnostic (exploratory, not in the pre-registered rule).

    For unit u and its nearest unit v (by medoid): gap_u = (1 - max cross similarity between any item of u and any
    item of v) / median nearest-neighbour distance within u. A continuum cut into segments gives gap ~ 1 (the step
    across a boundary is a typical step); discrete units give gap >> 1. Ashman's D on medoid similarities cannot tell
    these apart when the segments are long. Returns {unit: gap}.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = {u: np.flatnonzero(labels == u) for u in np.unique(labels)}

    def samp(ix):
        return ix if len(ix) <= cap else rng.choice(ix, cap, replace=False)

    out = {}
    for u, v in nearest_unit.items():
        iu, iv = samp(idx[u]), samp(idx[v])
        if len(iu) < 2:
            continue
        tu = [items[i] for i in iu]
        W = sim_matrix(tu, words=words)
        np.fill_diagonal(W, -1)
        dnn = np.median(1 - W.max(1))
        cross = sim_matrix(tu, [items[i] for i in iv], words=words).max()
        out[u] = float((1 - cross) / dnn) if dnn > 0 else (np.inf if cross < 1 else 0.0)
    return out


def silhouette(
    items: list[str], labels: np.ndarray, n: int = 20000, words: bool = False, seed: int = 20261008
) -> float:
    """Mean silhouette on distance 1 - sim for a seeded random sample of items."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items)) if len(items) <= n else rng.choice(len(items), n, replace=False)
    sub = [items[i] for i in idx]
    lab = labels[idx]
    if len(np.unique(lab)) < 2:
        return np.nan
    Dm = 1.0 - sim_matrix(sub, words=words)
    np.fill_diagonal(Dm, 0.0)
    return float(silhouette_score(Dm, lab, metric="precomputed"))


def bootstrap_median(x: np.ndarray, B: int = 1000, seed: int = 20261008) -> tuple[float, float, float]:
    """(median, 2.5%, 97.5%) of the finite values of ``x``; non-finite values are dropped first."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = np.median(rng.choice(x, (B, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def m1_test(
    items: pd.DataFrame,
    min_items: int = 5,
    words: bool = False,
    weighted: bool = False,
    n_boot: int = 2000,
    silhouette_n: int = 5000,
    gap: bool = True,
    d_threshold: float = 2.0,
    blend_units: float = 0.10,
    blend_continuum: float = 0.25,
    seed: int = 20261008,
    name: str = "M1",
) -> Result:
    """M1 on an items table (columns ``text``, ``unit``). Items are deduplicated to distinct (unit, text) variants;
    ``weighted=True`` weights the blend share by mention counts instead.

    estimate = median per-unit Ashman's D over units with finite D (bootstrap CI over units); detail: blend share,
    pooled D, silhouette and the boundary-gap diagnostic. verdict = :func:`memetrics.units.m1_verdict`.
    """
    require(items, ["unit", "text"], "m1_test")
    d = items.dropna(subset=["unit"])
    v = d.groupby(["unit", "text"]).size().rename("n").reset_index()
    r = unit_stats(list(v.text), v.unit.to_numpy(), v.n.to_numpy(), min_items=min_items, words=words)
    med, lo, hi = bootstrap_median(_d_for_median(r.D), n_boot, seed)
    keep = v[v.unit.isin(r.unit_ids)]
    wts = keep.n.to_numpy(float) if weighted else np.ones(len(keep))
    blend = float(np.average(r.blend, weights=wts)) if len(wts) else np.nan
    sil = (
        silhouette(list(keep.text), keep.unit.to_numpy(), n=silhouette_n, words=words, seed=seed)
        if silhouette_n
        else np.nan
    )
    # nearest unit of each analysed unit = the most common nearest-other-unit among its items
    nn_unit = pd.Series(r.nearest).groupby(r.item_unit).agg(lambda x: x.value_counts().index[0]).to_dict()
    gaps = gap_ratio(list(v.text), v.unit.to_numpy(), nn_unit, words=words, seed=seed) if gap else {}
    detail = {
        "blend_share": blend,
        "pooled_D": _ashman64(r.w, r.b),
        "silhouette": sil,
        "median_gap_ratio": float(np.median(list(gaps.values()))) if gaps else np.nan,
        "D_quantiles": {q: float(np.quantile(r.D[np.isfinite(r.D)], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        if np.isfinite(r.D).any()
        else {},
        "share_units_D_gt_2": float((r.D > 2).mean()) if len(r.D) else np.nan,
        "D_hist": np.histogram(np.clip(r.D[np.isfinite(r.D)], 0, 10), bins=40, range=(0, 10))[0].tolist(),
        "n_units": int(len(r.unit_ids)),
        "mean_within": float(r.w.mean()),
        "mean_between": float(r.b.mean()),
        "thresholds": {"d_threshold": d_threshold, "blend_units": blend_units, "blend_continuum": blend_continuum},
    }
    return Result(
        name, med, lo, hi, int(len(keep)), detail, m1_verdict(med, blend, d_threshold, blend_units, blend_continuum)
    )


def _d_for_median(D: np.ndarray) -> np.ndarray:
    """Per-unit D for medians: zero-spread units (D = inf) count as very large (1e6), as in
    :func:`memetrics.units.discreteness`; NaN (undefined) units are dropped by :func:`bootstrap_median`."""
    D = np.asarray(D, float)
    return np.where(np.isposinf(D), 1e6, D)


def _ashman64(w, b) -> float:
    return ashman_d(np.asarray(w, float), np.asarray(b, float))
