"""Meme units: perceptual-hash grouping, clustering and unit-discreteness statistics (hypothesis M1).

Works on any embedding (image or text); :mod:`memetrics.text` applies the same statistics to edit similarity.

- ``w_i`` = cosine similarity of item i to the medoid of its own unit; ``b_i`` = cosine similarity to the medoid of
  the nearest other unit (the unit whose medoid is most similar to i).
- Ashman's D for unit u: ``D_u = sqrt(2) |mean(w) - mean(b)| / sqrt(var(w) + var(b))`` (Ashman, Bird & Zepf 1994,
  AJ 108:2348, doi:10.1086/117248; bimodality separation of two Gaussian components, D > 2 = clean separation).
- Blend: item i is a blend if ``b_i >= q10(v)``, the 10th percentile of the within-similarities of its nearest other
  unit v: i is at least as close to v as 90% of v's own members.
- Silhouette: Rousseeuw (1987), J. Comput. Appl. Math. 20:53-65, doi:10.1016/0377-0427(87)90125-7, cosine distance.

Default decision rule (:func:`m1_verdict`): "units" if median D > 2 and blend share < 0.10; "continuum" if median
D < 2 or blend share > 0.25; otherwise "intermediate".

No network calls; embeddings are computed elsewhere (e.g. ``memetrics.io`` image readers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .result import Result

# ---------------------------------------------------------------- perceptual hashes


def hamming64(a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
    """Hamming distance between 64-bit hashes (uint64 arrays or scalars)."""
    x = np.atleast_1d(np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).reshape(x.shape)


def identical_groups(
    phash: np.ndarray,
    ts: np.ndarray | None = None,
    max_hamming: int = 4,
    emb: np.ndarray | None = None,
    min_cos: float = 0.90,
    text: list[str] | None = None,
    min_text_sim: float = 0.8,
    min_tokens: int = 3,
) -> np.ndarray:
    """Greedy identical-content grouping in time order (study unit level U0).

    An item joins the group of the earliest representative with pHash Hamming <= ``max_hamming`` and, when given,
    embedding cosine >= ``min_cos`` and (when both items have >= ``min_tokens`` caption tokens) caption
    ``sim_edit`` >= ``min_text_sim``; otherwise it becomes a new representative. The caption check matters for image
    macros: an 8x8 pHash is dominated by the template background, so different captions on one template can collide.

    Uses multi-index hashing (the 64-bit hash is split into ``max_hamming + 1`` 16/13-bit bands; by pigeonhole any pair
    within ``max_hamming`` shares one band exactly), so it scales to millions of items.
    Returns an int array of group ids (the row index of each group's representative).
    """
    from rapidfuzz.distance import Levenshtein

    ph = np.asarray(phash, dtype=np.uint64)
    n = len(ph)
    order = np.argsort(ts, kind="stable") if ts is not None else np.arange(n)
    nb = max_hamming + 1
    widths = [64 // nb + (1 if i < 64 % nb else 0) for i in range(nb)]
    shifts = np.cumsum([0] + widths[:-1])
    tables: list[dict] = [dict() for _ in range(nb)]
    group = np.full(n, -1, dtype=np.int64)
    toks = [len(t.split()) if isinstance(t, str) else 0 for t in text] if text is not None else None
    for i in order:
        h = int(ph[i])
        keys = [(h >> int(s)) & ((1 << w) - 1) for s, w in zip(shifts, widths)]
        cand = set()
        for t, k in zip(tables, keys):
            cand.update(t.get(k, ()))
        best = -1
        for r in sorted(cand):  # representatives in insertion order ~ time order; first match = earliest
            if bin(h ^ int(ph[r])).count("1") > max_hamming:
                continue
            if emb is not None and float(emb[i] @ emb[r]) < min_cos:
                continue
            if (
                toks is not None
                and toks[i] >= min_tokens
                and toks[r] >= min_tokens
                and Levenshtein.normalized_similarity(text[i], text[r]) < min_text_sim
            ):
                continue
            best = r
            break
        if best < 0:
            group[i] = i
            for t, k in zip(tables, keys):
                t.setdefault(k, []).append(i)
        else:
            group[i] = best
    return group


# ---------------------------------------------------------------- clustering


def cluster_hdbscan(
    emb: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    pca_dims: int | None = 64,
    max_fit: int = 250_000,
    seed: int = 0,
    selection: str = "eom",
) -> np.ndarray:
    """HDBSCAN (Campello, Moulavi & Sander 2013, PAKDD, LNCS, pp. 160-172, doi:10.1007/978-3-642-37456-2_14) on
    (optionally PCA-reduced) unit-normalized embeddings.

    If there are more than ``max_fit`` items, HDBSCAN is fit on a seeded subsample and the remaining items are
    assigned to
    the cluster of their nearest clustered neighbour when within that cluster's 95th-percentile radius (else noise).
    ``selection``: HDBSCAN cluster-selection method, "eom" (excess of mass, default; tends to merge into a few large
    clusters) or "leaf" (finest stable clusters). Returns labels with -1 = noise.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    def fit(z):
        try:  # the hdbscan package (McInnes et al.) is much faster (Boruvka, parallel core distances)
            import hdbscan

            return hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=selection,
                core_dist_n_jobs=-1,
            ).fit_predict(z)
        except ImportError:
            from sklearn.cluster import HDBSCAN

            return HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=selection,
                copy=True,
            ).fit_predict(z)

    x = _unit(emb)
    rng = np.random.default_rng(seed)
    if pca_dims and x.shape[1] > pca_dims:
        fit_idx = rng.choice(len(x), min(len(x), 200_000), replace=False)
        x = PCA(pca_dims, random_state=seed).fit(x[fit_idx]).transform(x).astype(np.float32)
    if len(x) <= max_fit:
        return fit(x)
    sub = np.sort(rng.choice(len(x), max_fit, replace=False))
    lab_sub = fit(x[sub])
    labels = np.full(len(x), -1)
    labels[sub] = lab_sub
    core = sub[lab_sub >= 0]
    if len(core) == 0:
        return labels
    nn = NearestNeighbors(n_neighbors=1).fit(x[core])
    cents, radius = {}, {}
    for c in np.unique(lab_sub[lab_sub >= 0]):
        m = x[sub[lab_sub == c]]
        cents[c] = m.mean(0)
        radius[c] = np.quantile(np.linalg.norm(m - cents[c], axis=1), 0.95)
    rest = np.setdiff1d(np.arange(len(x)), sub)
    _, j = nn.kneighbors(x[rest])
    lab = labels[core[j[:, 0]]]
    dist = np.linalg.norm(x[rest] - np.stack([cents[c] for c in lab]), axis=1)
    ok = dist <= np.array([radius[c] for c in lab])
    labels[rest[ok]] = lab[ok]
    return labels


def knn(x: np.ndarray, k: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours (Euclidean, self included at position 0). Uses torch on a CUDA GPU when
    available, else scikit-learn."""
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else None
    except ImportError:
        dev = None
    if dev is None:
        from sklearn.neighbors import NearestNeighbors

        d, j = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
        return d, j
    X = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
    sq = (X * X).sum(1)
    D, J = [], []
    for a in range(0, len(X), chunk):
        d2 = sq[a : a + chunk, None] + sq[None, :] - 2 * X[a : a + chunk] @ X.T
        v, i = torch.topk(d2, k, dim=1, largest=False)
        D.append(v.clamp_min(0).sqrt().cpu())
        J.append(i.cpu())
    d, j = torch.cat(D).numpy().astype(np.float64), torch.cat(J).numpy()
    d[:, 0] = 0.0
    return d, j


def hdbscan_knn(
    x: np.ndarray, min_cluster_size: int = 10, min_samples: int = 5, k: int = 30, selection: str = "eom"
) -> np.ndarray:
    """HDBSCAN on the mutual-reachability graph restricted to each point's k nearest neighbours.

    Core distance = distance to the ``min_samples``-th neighbour (self excluded), as in the ``hdbscan`` package;
    edge weight = max(core_i, core_j, d_ij). The minimum spanning tree of the kNN graph (exact MST restricted to kNN
    edges; disconnected components joined at the top of the hierarchy) is condensed and clusters are selected with the
    reference ``hdbscan`` package's tree code (McInnes, Healy & Astels 2017, JOSS 2(11):205, doi:10.21105/joss.00205).
    Exact HDBSCAN uses the full mutual-reachability MST; restricting to kNN edges is the standard large-n approximation
    and changes only merges between points that are not mutual k-neighbours of anything in between.

    Requires the ``hdbscan`` package (``pip install memetrics[image]`` or ``[text]``)."""
    import scipy.sparse as sp
    from hdbscan._hdbscan_linkage import label
    from hdbscan._hdbscan_tree import compute_stability, condense_tree, get_clusters
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    n = len(x)
    d, j = knn(x, max(k, min_samples) + 1)
    core = d[:, min_samples]  # min_samples-th neighbour excluding self, as in the hdbscan package
    rows = np.repeat(np.arange(n), d.shape[1] - 1)
    cols = j[:, 1:].ravel()
    w = np.maximum(np.maximum(core[rows], core[cols]), d[:, 1:].ravel())
    w = np.maximum(w, 1e-12)
    G = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)  # weights are symmetric; keep an edge if either endpoint lists the other
    T = minimum_spanning_tree(G).tocoo()
    edges = np.column_stack([T.row, T.col, T.data]).astype(np.float64)
    ncomp, comp = connected_components(T, directed=False)
    if ncomp > 1:  # join components above every real edge
        top = edges[:, 2].max() * 2 if len(edges) else 1.0
        reps = [np.flatnonzero(comp == c)[0] for c in range(ncomp)]
        edges = np.vstack([edges, np.column_stack([reps[:-1], reps[1:], np.full(ncomp - 1, top)])])
    edges = edges[np.argsort(edges[:, 2], kind="stable")]
    slt = label(edges)
    ct = condense_tree(slt, min_cluster_size)
    labels, _, _ = get_clusters(ct, compute_stability(ct), selection, False, False, 0.0, 0)
    return labels


# ---------------------------------------------------------------- discreteness statistics


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def medoids(emb: np.ndarray, labels: np.ndarray, max_items: int = 2000, seed: int = 0) -> dict:
    """Medoid (max summed cosine to its unit) per label >= 0; units larger than ``max_items`` use a seeded subsample."""
    x = _unit(emb)
    rng = np.random.default_rng(seed)
    out = {}
    for c in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == c)
        if len(idx) > max_items:
            idx = rng.choice(idx, max_items, replace=False)
        s = x[idx] @ x[idx].T
        out[c] = idx[int(np.argmax(s.sum(1)))]
    return out


def ashman_d(w: np.ndarray, b: np.ndarray) -> float:
    """Ashman, Bird & Zepf (1994) bimodality separation of two samples:
    ``sqrt(2) |mean(w) - mean(b)| / sqrt(var(w) + var(b))``. Zero spread gives inf (NaN if the means are equal).
    Computed in the input dtype."""
    w, b = np.asarray(w), np.asarray(b)
    den = np.sqrt(np.var(w, ddof=1) + np.var(b, ddof=1)) if len(w) > 1 and len(b) > 1 else np.nan
    if not np.isfinite(den) or den == 0:
        return np.inf if abs(w.mean() - b.mean()) > 0 else np.nan
    return float(np.sqrt(2) * abs(w.mean() - b.mean()) / den)


def m1_verdict(
    median_D: float,
    blend_share: float,
    d_threshold: float = 2.0,
    blend_units: float = 0.10,
    blend_continuum: float = 0.25,
) -> str:
    """Default M1 rule: "units" if median D > ``d_threshold`` and blend share < ``blend_units``; "continuum" if
    median D < ``d_threshold`` or blend share > ``blend_continuum``; otherwise "intermediate"."""
    if median_D > d_threshold and blend_share < blend_units:
        return "units"
    if median_D < d_threshold or blend_share > blend_continuum:
        return "continuum"
    return "intermediate"


def unit_stats(emb: np.ndarray, labels: np.ndarray, min_unit: int = 10, seed: int = 0) -> pd.DataFrame:
    """Per-item w_i, b_i, nearest other unit and blend flag, plus per-unit D (column ``D_unit``).

    Items with label -1 (noise) get NaN. Units smaller than ``min_unit`` are kept as competitors (nearest-unit
    candidates) but their own D is NaN.
    """
    x = _unit(emb)
    labels = np.asarray(labels)
    med = medoids(x, labels, seed=seed)
    units = np.array(sorted(med))
    if len(units) < 2:
        raise ValueError("need at least two units")
    M = x[[med[u] for u in units]]
    pos = {u: k for k, u in enumerate(units)}
    out = pd.DataFrame({"label": labels, "w": np.nan, "b": np.nan, "nearest": -1, "blend": False, "D_unit": np.nan})
    idx = np.flatnonzero(labels >= 0)
    S = np.empty((len(idx), len(units)), dtype=np.float32)
    for a in range(0, len(idx), 50_000):
        S[a : a + 50_000] = x[idx[a : a + 50_000]] @ M.T
    own = np.array([pos[l] for l in labels[idx]])
    w = S[np.arange(len(idx)), own]
    S[np.arange(len(idx)), own] = -np.inf
    near = S.argmax(1)
    b = S[np.arange(len(idx)), near]
    out.loc[idx, "w"], out.loc[idx, "b"], out.loc[idx, "nearest"] = w, b, units[near]
    q10 = pd.Series(w).groupby(labels[idx]).quantile(0.10)
    out.loc[idx, "blend"] = b >= q10.reindex(units[near]).to_numpy()
    sizes = pd.Series(labels[idx]).value_counts()
    for u in units:
        if sizes.get(u, 0) >= min_unit:
            m = labels[idx] == u
            out.loc[idx[m], "D_unit"] = ashman_d(w[m], b[m])
    return out


def silhouette_cos(emb: np.ndarray, labels: np.ndarray, max_n: int = 20_000, seed: int = 0) -> float:
    """Mean silhouette (Rousseeuw 1987) on cosine distance over a seeded sample of clustered items."""
    from sklearn.metrics import silhouette_score

    m = np.flatnonzero(labels >= 0)
    rng = np.random.default_rng(seed)
    if len(m) > max_n:
        m = rng.choice(m, max_n, replace=False)
    return float(silhouette_score(_unit(emb)[m], labels[m], metric="cosine"))


def discreteness(
    emb: np.ndarray,
    labels: np.ndarray,
    min_unit: int = 10,
    d_threshold: float = 2.0,
    blend_units: float = 0.10,
    blend_continuum: float = 0.25,
    n_boot: int = 2000,
    seed: int = 0,
    name: str = "M1 discreteness",
    stats: pd.DataFrame | None = None,
    mask: np.ndarray | None = None,
) -> Result:
    """M1 test: median per-unit Ashman's D (bootstrap CI over units), blend share, silhouette, noise share.

    ``stats``: precomputed ``unit_stats`` output (reuse across subsets). ``mask``: summarize only these items (e.g.
    one genre); nearest-unit competitors still include every unit. Infinite D (zero spread) counts as very large.

    ``verdict``: :func:`m1_verdict` with the given thresholds. If more than half the items are noise, the verdict is
    suffixed " (clustered minority)".
    """
    labels = np.asarray(labels)
    st = unit_stats(emb, labels, min_unit=min_unit, seed=seed) if stats is None else stats
    if mask is not None:
        st = st[np.asarray(mask)]
        emb, labels = (emb[np.asarray(mask)] if emb is not None else None), labels[np.asarray(mask)]
    per = (
        st.dropna(subset=["D_unit"])
        .groupby("label")
        .agg(D=("D_unit", "first"), n=("w", "size"), blend=("blend", "mean"))
    )
    rng = np.random.default_rng(seed)
    Ds = per.D.replace(np.inf, 1e6).dropna().to_numpy()
    boots = [np.median(rng.choice(Ds, len(Ds))) for _ in range(n_boot)] if len(Ds) else [np.nan]
    clustered = st.label >= 0
    blend = float(st.loc[clustered, "blend"].mean())
    noise = float(1 - clustered.mean())
    med = float(np.median(Ds)) if len(Ds) else np.nan
    v = m1_verdict(med, blend, d_threshold, blend_units, blend_continuum)
    if noise > 0.5:
        v += " (clustered minority)"
    try:
        sil = silhouette_cos(emb, labels, seed=seed) if emb is not None else np.nan
    except ValueError:
        sil = np.nan
    w, b = st.loc[clustered, "w"].to_numpy(), st.loc[clustered, "b"].to_numpy()
    return Result(
        name,
        med,
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
        int(clustered.sum()),
        {
            "blend_share": blend,
            "noise_share": noise,
            "silhouette": sil,
            "n_units": int(len(per)),
            "pooled_D": ashman_d(w, b),
            "mean_D": float(np.mean(Ds)) if len(Ds) else np.nan,
        },
        v,
    )


def gaussian_surrogate(emb: np.ndarray, n: int | None = None, dims: int = 64, seed: int = 0) -> np.ndarray:
    """Cluster-free null: Gaussian draws with the mean and top-``dims`` PCA covariance of ``emb`` (unit-normalized)."""
    from sklearn.decomposition import PCA

    x = _unit(emb)
    rng = np.random.default_rng(seed)
    p = PCA(dims, random_state=seed).fit(x)
    z = rng.standard_normal((n or len(x), dims)) * np.sqrt(p.explained_variance_)
    return _unit(z @ p.components_ + p.mean_)


# ---------------------------------------------------------------- unit assignment on an items table


def assign(
    items: pd.DataFrame,
    method: str = "auto",
    emb: np.ndarray | None = None,
    tau: float = 0.5,
    min_size: int = 5,
    max_hamming: int = 4,
    min_cluster_size: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Fill the ``unit`` column of an items table; returns a copy.

    ``method``:
      - ``minhash``: near-duplicate text units, connected components of word-5-shingle Jaccard >= ``tau`` on
        normalized text (:func:`memetrics.text.minhash_units`);
      - ``phash``: identical-content groups, pHash Hamming <= ``max_hamming`` (:func:`identical_groups`), with the
        caption check when ``text`` is present;
      - ``hdbscan``: HDBSCAN clusters of ``emb`` (one row per item, in table order; :func:`cluster_hdbscan`);
      - ``auto``: ``hdbscan`` if ``emb`` is given, else ``minhash`` if most rows have text, else ``phash``.

    Groups with fewer than ``min_size`` items (and HDBSCAN noise) get ``unit = None``.
    """
    from .text import minhash_units, normalize

    d = items.copy()
    has_text = "text" in d and d.text.fillna("").str.strip().ne("").mean() > 0.5
    if method == "auto":
        method = "hdbscan" if emb is not None else "minhash" if has_text else "phash"
    lab = np.full(len(d), -1, dtype=np.int64)
    if method == "minhash":
        norm = d["text"].map(normalize) if "text" in d else pd.Series("", index=d.index)
        ok = np.flatnonzero(norm.ne("").to_numpy())
        lab[ok] = minhash_units(norm.iloc[ok].tolist(), tau=tau)
    elif method == "phash":
        if "phash" not in d:
            raise ValueError("method='phash' needs a phash column")
        ok = np.flatnonzero(d.phash.notna().to_numpy())
        caps = d.text.iloc[ok].tolist() if "text" in d else None
        lab[ok] = identical_groups(
            d.phash.iloc[ok].astype("uint64").to_numpy(), d.ts.iloc[ok].to_numpy(), max_hamming=max_hamming, text=caps
        )
    elif method == "hdbscan":
        if emb is None or len(emb) != len(d):
            raise ValueError("method='hdbscan' needs emb with one row per item")
        lab = np.asarray(cluster_hdbscan(emb, min_cluster_size=min_cluster_size, seed=seed), dtype=np.int64)
    else:
        raise ValueError("method must be auto, minhash, phash or hdbscan")
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    keep = (lab >= 0) & pd.Series(lab).map(sizes).fillna(0).ge(min_size).to_numpy()
    d["unit"] = pd.Series([f"{method[0]}{x}" if k else None for x, k in zip(lab, keep)], index=d.index, dtype=object)
    return d
