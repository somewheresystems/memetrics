"""Tree-likeness of variant sets (M5): delta score, Q-residual, and simulated tree / network / star references.

- delta score: Holland, Huber, Dress & Moulton (2002), "delta Plots: a tool for analyzing phylogenetic distance
  data", Mol. Biol. Evol. 19(12):2051-2059, doi:10.1093/oxfordjournals.molbev.a004030. For a quartet with the three
  pairwise sums m1 <= m2 <= m3, delta_q = (m3 - m2) / (m3 - m1) (0 when m3 = m1). 0 = perfectly tree-like.
- Q-residual: Gray, Bryant & Greenhill (2010), "On the shape and fabric of human history", Phil. Trans. R. Soc. B
  365:3923-3933, doi:10.1098/rstb.2010.0162. Here: (m3 - m2)^2 averaged over quartets, with distances first scaled
  to mean 1 so values are comparable across data sets (the scaling convention is ours; compare only with references
  computed by this module).
- Dataset values are means over all quartets (or a seeded random sample of quartets when there are too many).

References are simulated on token sequences (captions, phrases) with three generating processes, matched on number of
taxa, sequence length and mean pairwise distance (per-branch rate tuned by bisection):
  ``tree``       coalescent topology, token substitutions and deletions along branches;
  ``network``    the same, then each taxon recombines with another taxon (splice at a random point) with prob. 0.5;
  ``star``       every taxon mutated independently from one ancestor (a shared template, no shared descent).
"""
from __future__ import annotations

import itertools

import numpy as np

from .result import Result


def _quartets(n: int, max_q: int, rng) -> np.ndarray:
    total = n * (n - 1) * (n - 2) * (n - 3) // 24
    if total <= max_q:
        return np.array(list(itertools.combinations(range(n), 4)))
    q = np.array([np.sort(rng.choice(n, 4, replace=False)) for _ in range(max_q)])
    return q


def delta_q(D: np.ndarray, max_q: int = 20_000, seed: int = 0) -> tuple[float, float]:
    """(mean delta score, Q-residual) of a symmetric distance matrix."""
    D = np.asarray(D, float)
    n = len(D)
    if n < 4:
        return np.nan, np.nan
    m = D[np.triu_indices(n, 1)].mean()
    Ds = D / m if m > 0 else D
    q = _quartets(n, max_q, np.random.default_rng(seed))
    x, y, u, v = q.T
    s = np.sort(np.stack([Ds[x, y] + Ds[u, v], Ds[x, u] + Ds[y, v], Ds[x, v] + Ds[y, u]], 1), 1)
    den = s[:, 2] - s[:, 0]
    dl = np.where(den > 1e-12, (s[:, 2] - s[:, 1]) / np.where(den > 1e-12, den, 1), 0.0)
    return float(dl.mean()), float(((s[:, 2] - s[:, 1]) ** 2).mean())


def word_distance_matrix(texts: list[str]) -> np.ndarray:
    """1 - normalized token-level Levenshtein similarity (``sim_word``)."""
    from rapidfuzz.distance import Levenshtein

    toks = [t.split() for t in texts]
    n = len(toks)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = Levenshtein.normalized_distance(toks[i], toks[j])
    return D


# ---------------------------------------------------------------- simulation


def _coalescent(n: int, rng):
    """Random coalescent tree: list of (child, parent, branch_length); leaves 0..n-1, root = 2n-2."""
    nodes = list(range(n))
    t = 0.0
    height = {i: 0.0 for i in range(n)}
    edges = []
    nxt = n
    while len(nodes) > 1:
        k = len(nodes)
        t += rng.exponential(2.0 / (k * (k - 1)))
        a, b = rng.choice(len(nodes), 2, replace=False)
        ca, cb = nodes[a], nodes[b]
        height[nxt] = t
        edges += [(ca, nxt, t - height[ca]), (cb, nxt, t - height[cb])]
        nodes = [x for i, x in enumerate(nodes) if i not in (a, b)] + [nxt]
        nxt += 1
    return edges, nodes[0]


def _mutate(seq, p, vocab, rng, del_frac=0.2):
    out = []
    for tok in seq:
        if rng.random() < p:
            if rng.random() < del_frac and len(seq) > 2:
                continue
            out.append(int(rng.integers(vocab)))
        else:
            out.append(tok)
    return out or [int(rng.integers(vocab))]


def simulate_taxa(kind: str, n: int, length: int, rate: float, vocab: int = 2000, seed: int = 0) -> list[list[int]]:
    """Token sequences for n taxa under ``kind`` in {tree, network, star}. ``rate`` = substitution rate per unit
    branch length (tree/network; coalescent height ~ 2) or per-token change probability (star)."""
    rng = np.random.default_rng(seed)
    anc = list(rng.integers(vocab, size=length))
    if kind == "star":
        return [_mutate(anc, min(rate, 1.0), vocab, rng) for _ in range(n)]
    edges, root = _coalescent(n, rng)
    seqs = {root: anc}
    children = {}
    for c, p, bl in edges:
        children.setdefault(p, []).append((c, bl))
    stack = [root]
    while stack:
        p = stack.pop()
        for c, bl in children.get(p, []):
            seqs[c] = _mutate(seqs[p], 1 - np.exp(-rate * bl), vocab, rng)
            stack.append(c)
    leaves = [seqs[i] for i in range(n)]
    if kind == "network":
        orig = [list(s) for s in leaves]
        for i in range(n):
            if rng.random() < 0.5:
                j = int(rng.integers(n - 1))
                j += j >= i
                k = int(rng.integers(1, max(2, min(len(orig[i]), len(orig[j])))))
                leaves[i] = orig[i][:k] + orig[j][k:]
    return leaves


def _mean_dist(seqs) -> float:
    from rapidfuzz.distance import Levenshtein

    n = len(seqs)
    return float(np.mean([Levenshtein.normalized_distance(seqs[i], seqs[j]) for i in range(n) for j in range(i + 1, n)]))


def calibrate_rate(kind: str, n: int, length: int, target: float, seed: int = 0, iters: int = 14, pilot: int = 6) -> float:
    """Bisection on the rate so the expected mean pairwise distance matches ``target``."""
    lo, hi = 1e-4, 1.0 if kind == "star" else 20.0
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        md = np.mean([_mean_dist(simulate_taxa(kind, n, length, mid, seed=seed + k)) for k in range(pilot)])
        lo, hi = (mid, hi) if md < target else (lo, mid)
    return float(np.sqrt(lo * hi))


def reference(kind: str, n: int, length: int, target_mean_dist: float, reps: int = 100, seed: int = 0) -> np.ndarray:
    """(reps, 2) array of (delta, Q-residual) under a simulated ``kind`` matched to n, length, mean distance."""
    from rapidfuzz.distance import Levenshtein

    rate = calibrate_rate(kind, n, length, target_mean_dist, seed=seed)
    out = []
    for r in range(reps):
        s = simulate_taxa(kind, n, length, rate, seed=seed + 1000 + r)
        D = np.array([[Levenshtein.normalized_distance(a, b) for b in s] for a in s])
        out.append(delta_q(D, seed=r))
    return np.array(out)


def tree_likeness(D: np.ndarray, length: int, reps: int = 100, seed: int = 0, name: str = "M5 delta") -> Result:
    """Observed delta score of a variant set against matched tree / network / star references.

    verdict: "tree-like" if delta <= tree 95th pct; "reticulate" if >= network 5th pct and > tree 95th pct;
    "star-like" if within the star central 90% and > tree 95th pct; else "ambiguous"."""
    n = len(D)
    d_obs, q_obs = delta_q(D, seed=seed)
    target = float(D[np.triu_indices(n, 1)].mean())
    refs = {k: reference(k, n, length, target, reps, seed) for k in ("tree", "network", "star")}
    t95 = np.quantile(refs["tree"][:, 0], 0.95)
    n05 = np.quantile(refs["network"][:, 0], 0.05)
    s05, s95 = np.quantile(refs["star"][:, 0], [0.05, 0.95])
    if d_obs <= t95:
        v = "tree-like"
    elif d_obs >= n05:
        v = "reticulate" if not (s05 <= d_obs <= s95) else "reticulate or star"
    elif s05 <= d_obs <= s95:
        v = "star-like"
    else:
        v = "ambiguous"
    pct = {k: float((refs[k][:, 0] < d_obs).mean()) for k in refs}
    return Result(name, d_obs, np.nan, np.nan, n,
                  {"q_residual": q_obs, "mean_dist": target, "pct_in_ref": pct,
                   "ref_mean": {k: float(refs[k][:, 0].mean()) for k in refs}, "tree_q95": float(t95),
                   "network_q05": float(n05), "star_q05_q95": (float(s05), float(s95))}, v)
