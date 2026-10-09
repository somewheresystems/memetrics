"""Transmission chains: parent inference, per-step fidelity and the attraction (convergence) slope (M2).

Input is the items table (:mod:`memetrics.schema`) with columns ``item_id``, ``ts``, ``text``, ``unit`` and, for
explicit edges, ``parent``. Edges are (parent -> child) pairs of items in the same unit.

Estimators
----------
- Per-step fidelity f = sim(parent, child), default `memetrics.text.sim_edit` (1 - normalized Levenshtein).
  Copy fidelity as the replicator criterion follows Dawkins (1976) and Blackmore (1999); the operational
  version is ours.
- Convergence slope beta: OLS of Delta = d(child) - d(parent) on d(parent), with unit fixed effects, where
  d(x) = 1 - sim(x, medoid of the unit). Attraction (Sperber 1996; Claidiere, Scott-Phillips & Sperber 2014)
  predicts beta < 0 (steps move toward the attractor, more so from far away). CI: cluster bootstrap over units.
- Calibrated null: isotropic copying alone gives beta < 0 (in high dimension d(child)^2 ~ d(parent)^2 + s^2, so
  Delta ~ s^2 / 2d falls with d, and the medoid is estimated from the same data). `replicator_null` re-simulates
  texts along the observed topology with matched mutation and recomputes beta, which gives the null distribution.
- Directional asymmetry: among non-identical edges, the share whose child is closer to the medoid than its parent.

Default decision rule (thresholds are keyword arguments): replication if median f >= 0.9 and the 95% CI of beta
contains 0; attraction if beta is significantly < 0 and median f < 0.9; otherwise "mixed".
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .result import Result
from .schema import require
from .text import medoid, sim_edit

Sim = Callable[[str, str], float]


# ---------------------------------------------------------------- edges


def infer_parents(items: pd.DataFrame, sim: Sim = sim_edit, max_candidates: int = 2000) -> pd.Series:
    """E2 rule: each item's parent is the most similar *earlier* item of the same unit (ties -> most recent).

    This is a time-ordered guess, not an observed copy event: when many items copy one unobserved source, the
    inferred tree is a chain or star that never happened. Returns a Series of parent item_ids (None for the first
    item of each unit), indexed like ``items``.
    Distinct texts are compared once; `max_candidates` caps the number of distinct earlier texts compared
    (the most recent ones are kept).
    """
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    require(items, ["item_id", "ts", "text", "unit"], "infer_parents")
    out = pd.Series([None] * len(items), index=items.index, dtype=object)
    for _, g in items.dropna(subset=["unit"]).sort_values("ts", kind="stable").groupby("unit", sort=False):
        latest: dict[str, str] = {}  # text -> most recent item_id with that text
        order: list[str] = []  # distinct texts in order of last use
        for idx, text, iid in zip(g.index, g["text"], g["item_id"]):
            if text in latest:
                out[idx] = latest[text]
            elif order:
                cands = order[-max_candidates:]
                if sim is sim_edit:
                    scores = process.cdist([text], cands, scorer=Levenshtein.normalized_similarity)[0]
                else:
                    scores = np.array([sim(text, c) for c in cands])
                best = np.flatnonzero(scores == scores.max())[-1]  # ties -> most recently used text
                out[idx] = latest[cands[best]]
            if text in latest:
                order.remove(text)
            order.append(text)
            latest[text] = iid
    return out


def edges_from_items(items: pd.DataFrame, parent_col: str = "parent") -> pd.DataFrame:
    """Edge table (unit, parent_id, child_id, parent_text, child_text, dt) for items whose parent is present
    in the table and in the same unit."""
    require(items, ["item_id", "ts", "text", "unit", parent_col], "edges_from_items")
    it = items.set_index("item_id")
    c = items.dropna(subset=[parent_col, "unit"])
    c = c[c[parent_col].isin(it.index)]
    p = it.loc[c[parent_col]]
    same = p["unit"].to_numpy() == c["unit"].to_numpy()
    c, p = c[same], p[same]
    return pd.DataFrame(
        {
            "unit": c["unit"].to_numpy(),
            "parent_id": c[parent_col].to_numpy(),
            "child_id": c["item_id"].to_numpy(),
            "parent_text": p["text"].to_numpy(),
            "child_text": c["text"].to_numpy(),
            "dt": (c["ts"].to_numpy() - p["ts"].to_numpy()),
        }
    )


def unit_medoids(items: pd.DataFrame, cap: int = 1500, seed: int = 20261008) -> dict:
    """Medoid text per unit over distinct variants (ties broken by frequency), as in M1."""
    require(items, ["unit", "text"], "unit_medoids")
    out = {}
    for u, g in items.dropna(subset=["unit"]).groupby("unit"):
        vc = g["text"].value_counts()
        texts = list(vc.index)
        out[u] = texts[medoid(texts, vc.to_numpy(), cap=cap, seed=seed)]
    return out


# ---------------------------------------------------------------- statistics


def edge_stats(edges: pd.DataFrame, medoids: dict, sim: Sim = sim_edit) -> pd.DataFrame:
    """Adds f (fidelity), d_parent, d_child and delta to an edge table."""
    e = edges.copy()
    e["f"] = [sim(a, b) for a, b in zip(e.parent_text, e.child_text)]
    m = e["unit"].map(medoids)
    e["d_parent"] = [1 - sim(a, b) for a, b in zip(e.parent_text, m)]
    e["d_child"] = [1 - sim(a, b) for a, b in zip(e.child_text, m)]
    e["delta"] = e.d_child - e.d_parent
    return e


def _unit_moments(e: pd.DataFrame) -> pd.DataFrame:
    g = e.groupby("unit")
    xm = g.d_parent.transform("mean")
    ym = g.delta.transform("mean")
    t = pd.DataFrame({"unit": e.unit, "sxx": (e.d_parent - xm) ** 2, "sxy": (e.d_parent - xm) * (e.delta - ym)})
    return t.groupby("unit")[["sxx", "sxy"]].sum()


def convergence_slope(e: pd.DataFrame, n_boot: int = 2000, seed: int = 20261008) -> tuple[float, float, float]:
    """Within-unit (fixed-effects) OLS slope of delta on d_parent, with a cluster bootstrap over units."""
    mom = _unit_moments(e)
    mom = mom[mom.sxx > 0]
    if mom.empty:
        return (np.nan, np.nan, np.nan)
    sxx, sxy = mom.sxx.to_numpy(), mom.sxy.to_numpy()
    beta = sxy.sum() / sxx.sum()
    rng = np.random.default_rng(seed)
    k = len(sxx)
    w = rng.multinomial(k, np.full(k, 1 / k), size=n_boot)  # unit resampling weights
    bs = (w @ sxy) / np.maximum(w @ sxx, 1e-12)
    return float(beta), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def median_fidelity(e: pd.DataFrame, n_boot: int = 2000, seed: int = 20261008) -> tuple[float, float, float]:
    """Median f with a cluster bootstrap over units."""
    f = e.f.to_numpy()
    units = e.unit.to_numpy()
    uniq, inv = np.unique(units, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        bs[b] = np.median(np.concatenate([f[groups[i]] for i in pick]))
    return float(np.median(f)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def toward_share(e: pd.DataFrame) -> float:
    """Among non-identical edges, the share whose child is closer to the medoid than its parent."""
    m = e.f < 1
    return float((e.delta[m] < 0).mean()) if m.any() else np.nan


def m2_verdict(f_med: float, beta_lo: float, beta_hi: float, f_threshold: float = 0.9) -> str:
    """Literal M2 rule: "replication" if f >= ``f_threshold`` and the slope CI contains 0; "attraction" if the
    slope CI is below 0 and f < ``f_threshold``; else "mixed"."""
    conv = beta_hi < 0
    if f_med >= f_threshold and beta_lo <= 0 <= beta_hi:
        return "replication"
    if conv and f_med < f_threshold:
        return "attraction"
    return "mixed"


# ---------------------------------------------------------------- calibrated replicator null


def _mutate(words: list[str], k: int, vocab: list[str], rng: np.random.Generator, p_indel: float = 2 / 3) -> list[str]:
    """k word edits; each is an insertion or deletion with prob. p_indel (equally likely), else a substitution."""
    w = list(words)
    for _ in range(k):
        if rng.random() >= p_indel and w:
            op = 0
        else:
            op = (1 if rng.random() < 0.5 else 2) if len(w) > 1 else 2
        if op == 0:  # substitute
            w[rng.integers(len(w))] = vocab[rng.integers(len(vocab))]
        elif op == 1:  # delete
            del w[rng.integers(len(w))]
        else:  # insert
            w.insert(int(rng.integers(len(w) + 1)), vocab[rng.integers(len(vocab))])
    return w


def _resimulate(
    e: pd.DataFrame,
    roots: dict,
    p_mut: float,
    lam: float,
    vocab: list[str],
    rng: np.random.Generator,
    p_indel: float = 2 / 3,
) -> pd.DataFrame:
    """New child texts along the observed topology: identical with prob 1 - p_mut, else 1 + Poisson(lam) word
    edits. Edges must be sorted so that parents precede children."""
    text = dict(roots)
    child_text = []
    for pid, cid in zip(e.parent_id, e.child_id):
        pt = text.get(pid)
        if pt is None:
            pt = roots.get(pid, "")
        if rng.random() < p_mut:
            ct = " ".join(_mutate(pt.split(), 1 + rng.poisson(lam), vocab, rng, p_indel))
        else:
            ct = pt
        text[cid] = ct
        child_text.append(ct)
    out = e[["unit", "parent_id", "child_id"]].copy()
    out["parent_text"] = [text.get(p, roots.get(p, "")) for p in e.parent_id]
    out["child_text"] = child_text
    return out


def replicator_null(
    edges: pd.DataFrame, n_sims: int = 200, sim: Sim = sim_edit, seed: int = 20261008, n_boot: int = 0
) -> dict:
    """Null distribution of (beta, median f, toward share) under pure copying with matched mutation.

    Topology, edge counts and root texts are the observed ones. The share of non-identical edges matches the
    data; the edit mix matches it too (p_indel = share of non-identical edges that change the word count); the
    number of word edits per mutation (1 + Poisson(lam)) is tuned so the mean step size (1 - f among
    non-identical edges) matches the data. Medoids are re-estimated on each simulated set.
    """
    rng = np.random.default_rng(seed)
    e = edges.copy()
    f_obs = np.array([sim(a, b) for a, b in zip(e.parent_text, e.child_text)])
    p_mut = float((f_obs < 1).mean())
    step_obs = float((1 - f_obs[f_obs < 1]).mean()) if (f_obs < 1).any() else 0.0
    nonid = f_obs < 1
    lens_differ = np.array([len(a.split()) != len(b.split()) for a, b in zip(e.parent_text, e.child_text)])
    p_indel = float(lens_differ[nonid].mean()) if nonid.any() else 0.5
    vocab = sorted({w for t in pd.concat([e.parent_text, e.child_text]) for w in t.split()}) or ["x"]
    # topological order: a child whose id is also a parent must come after its own parent
    child_set = set(e.child_id)
    roots = {p: t for p, t in zip(e.parent_id, e.parent_text) if p not in child_set}
    depth: dict = {r: 0 for r in roots}
    par = dict(zip(e.child_id, e.parent_id))

    def dep(x):
        seen = []
        while x not in depth and x in par and len(seen) < 100000:
            seen.append(x)
            x = par[x]
        d = depth.get(x, 0)
        for y in reversed(seen):
            d += 1
            depth[y] = d
        return depth.get(seen[0], d) if seen else d

    e["_depth"] = [dep(c) for c in e.child_id]
    e = e.sort_values("_depth", kind="stable").drop(columns="_depth")

    def mean_step(lam: float, n: int = 400) -> float:
        r = np.random.default_rng(seed + 1)
        ts = e.parent_text.sample(min(n, len(e)), random_state=int(r.integers(1 << 31)), replace=True)
        steps = [1 - sim(t, " ".join(_mutate(t.split(), 1 + r.poisson(lam), vocab, r, p_indel))) for t in ts]
        return float(np.mean(steps))

    lam = 0.0
    if step_obs > 0:
        lo, hi = 0.0, 50.0
        if mean_step(0.0) < step_obs:
            for _ in range(14):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if mean_step(mid) < step_obs else (lo, mid)
            lam = (lo + hi) / 2
    betas, fmeds, tows = [], [], []
    for _ in range(n_sims):
        s = _resimulate(e, roots, p_mut, lam, vocab, rng, p_indel)
        items = pd.concat(
            [
                pd.DataFrame({"unit": s.unit, "text": s.parent_text}),
                pd.DataFrame({"unit": s.unit, "text": s.child_text}),
            ]
        )
        meds = {
            u: (lambda vc: list(vc.index)[medoid(list(vc.index), vc.to_numpy(), cap=400)])(g.text.value_counts())
            for u, g in items.groupby("unit")
        }
        st = edge_stats(s, meds, sim)
        b, _, _ = convergence_slope(st, n_boot=1)
        betas.append(b)
        fmeds.append(float(np.median(st.f)))
        tows.append(toward_share(st))
    return {
        "beta": np.array(betas),
        "f_median": np.array(fmeds),
        "toward": np.array(tows),
        "p_mut": p_mut,
        "lam": lam,
        "step_obs": step_obs,
        "p_indel": p_indel,
    }


# ---------------------------------------------------------------- the M2 test


def m2_test(
    edges: pd.DataFrame,
    medoids: dict,
    sim: Sim = sim_edit,
    f_threshold: float = 0.9,
    n_boot: int = 2000,
    null_sims: int = 0,
    seed: int = 20261008,
    name: str = "M2",
) -> Result:
    """Fidelity vs attraction on an edge table.

    estimate = median per-step fidelity f; detail holds beta (convergence slope) with CI, the share of identical
    copies, the toward-medoid share and, if `null_sims` > 0, the calibrated replicator null and calibrated verdict.
    verdict = literal rule (:func:`m2_verdict`). ``detail['verdict_m2a']`` applies the fidelity threshold to
    derivatives only (f < 1), so verbatim copies (reposts) cannot carry the fidelity criterion on their own.
    """
    st = edge_stats(edges, medoids, sim)
    f_med, f_lo, f_hi = median_fidelity(st, n_boot, seed)
    beta, b_lo, b_hi = convergence_slope(st, n_boot, seed)
    der = st[st.f < 1]
    if len(der):
        fd, fd_lo, fd_hi = median_fidelity(der, n_boot, seed)
    else:
        fd = fd_lo = fd_hi = np.nan
    detail = {
        "beta": beta,
        "beta_ci": [b_lo, b_hi],
        "mean_f": float(st.f.mean()),
        "f_median_derivatives": fd,
        "f_median_derivatives_ci": [fd_lo, fd_hi],
        "n_derivatives": int(len(der)),
        "verdict_m2a": ("high fidelity" if fd >= f_threshold else "low fidelity") if np.isfinite(fd) else None,
        "share_identical": float((st.f >= 1).mean()),
        "toward_share": toward_share(st),
        "n_units": int(st.unit.nunique()),
        "f_threshold": f_threshold,
    }
    if null_sims:
        nl = replicator_null(edges, null_sims, sim, seed)
        q_lo, q_hi = np.nanquantile(nl["beta"], [0.025, 0.975])
        detail.update(
            {
                "null_beta_q025": float(q_lo),
                "null_beta_q975": float(q_hi),
                "null_beta_median": float(np.nanmedian(nl["beta"])),
                "null_toward_median": float(np.nanmedian(nl["toward"])),
                "null_p_mut": nl["p_mut"],
                "null_lam": nl["lam"],
            }
        )
        if beta < q_lo and f_med < f_threshold:
            cal = "attraction"
        elif f_med >= f_threshold and q_lo <= beta <= q_hi:
            cal = "replication"
        else:
            cal = "mixed"
        detail["verdict_calibrated"] = cal
    return Result(name, f_med, f_lo, f_hi, len(st), detail, m2_verdict(f_med, b_lo, b_hi, f_threshold))


# ---------------------------------------------------------------- lineage depth and Price decomposition


def depths(items: pd.DataFrame, parent_col: str = "parent") -> pd.DataFrame:
    """Root id and depth of every item along parent links (roots: no parent in the table)."""
    par = dict(zip(items.item_id, items[parent_col]))
    ids = set(items.item_id)
    root, depth = {}, {}
    for i in items.item_id:
        path = []
        x = i
        while x in ids and x not in depth and isinstance(par.get(x), str) and par[x] in ids and len(path) < 100000:
            path.append(x)
            x = par[x]
        if x not in depth:
            root[x], depth[x] = x, 0
        r, d = root[x], depth[x]
        for y in reversed(path):
            d += 1
            root[y], depth[y] = r, d
    return pd.DataFrame({"item_id": list(depth), "root": [root[k] for k in depth], "depth": list(depth.values())})


def depth_profile(items: pd.DataFrame, sim: Sim = sim_edit, max_depth: int = 10, parent_col: str = "parent") -> dict:
    """Distance to the chain root by depth (Adamic et al. 2016 report growth with cascade depth).

    Returns mean distance per depth and the saturation ratio s_late / s_early, where s_early is the OLS slope of
    distance on depth over depths 0-2 and s_late over depths 3..max_depth. Ratio near 1: steady divergence (copying
    with drift). Ratio near 0: saturation (bounded transformation, attraction)."""
    d = depths(items, parent_col).merge(items[["item_id", "text"]], on="item_id")
    rt = dict(zip(items.item_id, items.text))
    d = d[d.depth <= max_depth]
    d["dist"] = [1 - sim(t, rt[r]) for t, r in zip(d.text, d.root)]
    prof = d.groupby("depth").dist.agg(["mean", "count"])

    def slope(lo, hi):
        x = d[(d.depth >= lo) & (d.depth <= hi)]
        return float(np.polyfit(x.depth, x.dist, 1)[0]) if x.depth.nunique() >= 2 else np.nan

    se, sl = slope(0, 2), slope(3, max_depth)
    return {
        "profile": prof["mean"].to_dict(),
        "counts": prof["count"].to_dict(),
        "slope_early": se,
        "slope_late": sl,
        "saturation_ratio": sl / se if se and np.isfinite(se) and se > 0 else np.nan,
    }


def price_decomposition(
    items: pd.DataFrame, medoids: dict, sim: Sim = sim_edit, parent_col: str = "parent", min_items: int = 10
) -> pd.DataFrame:
    """Price equation for the trait z = 1 - sim(x, unit medoid), per unit (Price 1970; applied to cultural
    attraction vs selection by Claidiere, Scott-Phillips & Sperber 2014).

    Every item is a potential parent with fitness w = number of observed children. With mean-child trait
    change dz (0 for childless items): w_bar * dZ = Cov(w, z) + E(w dz). Returns per unit the selection term
    Cov(w, z) / w_bar and the transmission term E(w dz) / w_bar."""
    d = items.dropna(subset=["unit"]).copy()
    d["z"] = [1 - sim(t, medoids[u]) for t, u in zip(d.text, d.unit)]
    zc = d[d[parent_col].isin(set(d.item_id))].groupby(parent_col).z.agg(["mean", "count"])
    d["w"] = d.item_id.map(zc["count"]).fillna(0.0)
    d["dz"] = (d.item_id.map(zc["mean"]) - d.z).fillna(0.0)
    rows = []
    for u, g in d.groupby("unit"):
        if len(g) < min_items or g.w.sum() == 0:
            continue
        wb = g.w.mean()
        sel = float(((g.w - wb) * (g.z - g.z.mean())).mean() / wb)
        tra = float((g.w * g.dz).mean() / wb)
        rows.append({"unit": u, "n": len(g), "selection": sel, "transmission": tra})
    return pd.DataFrame(rows)
