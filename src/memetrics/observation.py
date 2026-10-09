"""Observation-model simulator: what can public social-media data tell apart?

A ground-truth world (agents on a follower graph, a feed, likes, reshares with copying) runs under one of four
transmission processes. An observation layer then hides what a researcher cannot see: private shares (DMs, group
chats), impressions (exposure), deletions and the ranking itself. The same metrics as for real data (M1-M5, N1)
are computed on the observed items table and on the full ground truth. The pre-registered decision rules are
applied to both.

Processes
---------
- ``replicator``: high-fidelity copying (per-token mutation 0.01). Variants carry heritable fitness that scales
  reshare and like probabilities (log-normal unit effect, perturbed on mutation): Dawkins (1976), Blackmore (1999).
- ``attraction``: lower fidelity (mutation 0.08) and each copy is pulled toward a unit attractor (prob. 0.15 per
  token). No heritable fitness: Sperber (1996), Claidiere, Scott-Phillips & Sperber (2014).
- ``neutral``: unbiased random copying with mutation 0.01 (Bentley, Hahn & Shennan 2004).
- ``contagion``: complex contagion: an agent reshares a unit only after exposure from >= 2 distinct followees
  (Centola & Macy 2007), with no content effect.

Observation layers
------------------
- ``p_private``: each reshare is private with this probability. It is delivered to a few followers' inboxes,
  never appears publicly, and its likes are invisible. Its public descendants keep their text, but their source
  is hidden.
- ``recommender``: True ranks followee posts by log(1 + likes) plus noise and adds a few slots from a global
  trending pool (a toy engagement ranker). False gives a chronological followee feed. The ranker changes the world itself: who sees what.
- exposure censoring: impressions are never part of the observed table.
- ``deletion``: each public post is deleted with this probability, doubled in the bottom engagement quartile.
- ``edge_mode``: how observed parents are formed. ``explicit``: the true parent if it is observed, else no edge
  (as a quote of a deleted or private post). ``ancestor``: the nearest observed ancestor (thinned chains).
  ``inferred``: the most similar earlier observed instance (time-ordered guess).

Quick use
---------
>>> from memetrics import observation as ob
>>> log = ob.simulate("attraction", p_private=0.5, seed=1)
>>> obs = ob.observe(log, edge_mode="explicit")
>>> res = ob.evaluate(obs)            # dict of Result for M1..M5, N1
>>> ob.composite(res)                 # inferred process label
>>> ob.distinguishability(p_private=0.5, reps=10)   # confusion matrix: can this design tell processes apart?

All randomness flows from ``seed``. Pure numpy/pandas; no network access.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import chains, fitness, phylo, predict, selection, text
from .result import Result

PROCESSES = ("replicator", "attraction", "neutral", "contagion")


@dataclass(frozen=True)
class WorldParams:
    """Defaults are the pre-registered ones (notes/PREREG-text.md section 10)."""
    n_agents: int = 3000
    mean_out_degree: int = 20
    indegree_exponent: float = 2.5
    n_units: int = 40
    text_len: int = 15
    vocab: int = 2000
    steps: int = 60
    seed_window: int = 10           # units are seeded in the first `seed_window` steps
    seeds_per_unit: int = 6
    activity: float = 0.5           # probability an agent is active in a step
    background_rate: float = 0.15   # probability an active agent writes a non-meme post
    feed_k: int = 10
    recent_window: int = 3          # feeds draw from posts of the last `recent_window` steps
    trending_pool: int = 30         # size of the global trending pool (recommender on)
    trending_slots: int = 3         # trending posts added to each feed (recommender on)
    rank_weight: float = 1.0        # weight of log(1 + likes) against Gumbel(0, 1) noise in the ranker
    inbox_size: int = 3             # recipients of a private share
    like_base: float = 0.08
    share_base: float = 0.035
    contagion_share: float = 0.12
    contagion_threshold: int = 2
    mu_high_fidelity: float = 0.01
    mu_attraction: float = 0.08
    attraction_alpha: float = 0.15
    fitness_sigma_unit: float = 0.8
    fitness_sigma_mut: float = 0.2


# =============================================================== ground truth


def _follow_graph(P: WorldParams, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each agent follows `mean_out_degree` others chosen with probability proportional to a Pareto popularity
    weight, which gives a heavy-tailed in-degree."""
    n = P.n_agents
    pop = rng.pareto(P.indegree_exponent - 1, n) + 1
    pop = pop / pop.sum()
    following = []
    for i in range(n):
        k = max(1, rng.poisson(P.mean_out_degree))
        f = rng.choice(n, size=min(k, n - 1), replace=False, p=pop)
        following.append(f[f != i])
    followers: list[list[int]] = [[] for _ in range(n)]
    for i, fs in enumerate(following):
        for j in fs:
            followers[j].append(i)
    return following, [np.array(x, dtype=np.int64) for x in followers]


def simulate(process: str, p_private: float = 0.0, recommender: bool = True, seed: int = 20261008,
             params: WorldParams | None = None) -> pd.DataFrame:
    """Run one ground-truth world. Returns the full log (one row per post, incl. private and background posts)
    with true parents, private flags, likes (public/private, with like times) and true fitness."""
    if process not in PROCESSES:
        raise ValueError(f"process must be one of {PROCESSES}")
    P = params or WorldParams()
    rng = np.random.default_rng(seed)
    following, followers = _follow_graph(P, rng)
    n = P.n_agents
    L = P.text_len
    seeds = rng.integers(0, P.vocab, (P.n_units, L))
    attractors = seeds.copy()
    flip = rng.random((P.n_units, L)) < 0.5
    attractors[flip] = rng.integers(0, P.vocab, flip.sum())
    unit_fit = np.exp(rng.normal(0, P.fitness_sigma_unit, P.n_units)) if process == "replicator" else np.ones(P.n_units)

    # post store (python lists; converted at the end)
    author, unit, step_of, parent, private, fit, toks = [], [], [], [], [], [], []
    likes_pub, likes_priv, like_times = [], [], []
    recent: dict[int, list[int]] = {}          # author -> recent public post ids
    inbox: list[list[int]] = [[] for _ in range(n)]
    adopted = [set() for _ in range(n)]
    exposure: list[dict] = [dict() for _ in range(n)]   # unit -> set of authors (contagion)

    def new_post(a, u, t, par, priv, f, tk):
        pid = len(author)
        author.append(a); unit.append(u); step_of.append(t); parent.append(par); private.append(priv)
        fit.append(f); toks.append(tk); likes_pub.append(0); likes_priv.append(0); like_times.append([])
        if not priv:
            recent.setdefault(a, []).append(pid)
        return pid

    seed_steps = rng.integers(0, P.seed_window, P.n_units)
    by_step_seed: dict[int, list[int]] = {}
    for u, s in enumerate(seed_steps):
        by_step_seed.setdefault(int(s), []).append(u)

    for t in range(P.steps):
        # seeding: a unit appears at several followees of one agent (so complex contagion can start)
        for u in by_step_seed.get(t, []):
            hub = int(rng.integers(n))
            cands = following[hub] if len(following[hub]) >= P.seeds_per_unit else rng.choice(n, P.seeds_per_unit, replace=False)
            for a in rng.choice(cands, min(P.seeds_per_unit, len(cands)), replace=False):
                a = int(a)
                new_post(a, u, t, -1, False, float(unit_fit[u]), seeds[u].copy())
                adopted[a].add(u)
        # prune recent lists
        for a in list(recent):
            recent[a] = [p for p in recent[a] if step_of[p] >= t - P.recent_window]
            if not recent[a]:
                del recent[a]
        trending = []
        if recommender:
            pool = [p for ps in recent.values() for p in ps]
            if pool:
                sc = P.rank_weight * np.log1p(np.array([likes_pub[p] for p in pool], float)) \
                    + rng.gumbel(0, 1, len(pool))
                trending = [pool[i] for i in np.argsort(-sc)[:P.trending_pool]]
        active = np.flatnonzero(rng.random(n) < P.activity)
        new_inbox: list[list[int]] = [[] for _ in range(n)]
        for a in active:
            a = int(a)
            cand = [p for f in following[a] for p in recent.get(int(f), ())]
            if len(cand) > P.feed_k:
                if recommender:   # engagement-ranked followee posts
                    sc = P.rank_weight * np.log1p(np.array([likes_pub[p] for p in cand], float)) \
                        + rng.gumbel(0, 1, len(cand))
                else:             # chronological
                    sc = np.array([step_of[p] for p in cand], float) + rng.random(len(cand))
                cand = [cand[i] for i in np.argsort(-sc)[:P.feed_k]]
            if recommender and trending:  # a few slots of globally trending posts
                extra = [trending[i] for i in rng.choice(len(trending), min(P.trending_slots, len(trending)),
                                                          replace=False)]
                cand = cand + [p for p in extra if p not in cand]
            seen = inbox[a] + cand            # private inbox items are always seen first
            shared = False
            for p in seen:
                if author[p] == a:
                    continue
                u = unit[p]
                w = fit[p] if process == "replicator" else 1.0
                if rng.random() < min(1.0, P.like_base * w):
                    if private[p]:
                        likes_priv[p] += 1
                    else:
                        likes_pub[p] += 1
                    like_times[p].append(t)
                if u < 0:
                    continue
                if process == "contagion":
                    ex = exposure[a].setdefault(u, set())
                    ex.add(author[p])
                if shared or u in adopted[a]:
                    continue
                if process == "contagion":
                    if len(ex) < P.contagion_threshold or rng.random() >= P.contagion_share:
                        continue
                elif rng.random() >= min(1.0, P.share_base * w):
                    continue
                # reshare with copying
                tk = toks[p].copy()
                if process == "attraction":
                    pull = rng.random(L) < P.attraction_alpha
                    tk[pull] = attractors[u][pull]
                    mu = P.mu_attraction
                else:
                    mu = P.mu_high_fidelity
                m = rng.random(L) < mu
                f_new = fit[p]
                if m.any():
                    tk[m] = rng.integers(0, P.vocab, m.sum())
                    if process == "replicator":
                        f_new = fit[p] * float(np.exp(rng.normal(0, P.fitness_sigma_mut)))
                priv = bool(rng.random() < p_private)
                pid = new_post(a, u, t, p, priv, f_new, tk)
                adopted[a].add(u)
                shared = True
                if priv and len(followers[a]):
                    for r in rng.choice(followers[a], min(P.inbox_size, len(followers[a])), replace=False):
                        new_inbox[int(r)].append(pid)
            if rng.random() < P.background_rate:
                new_post(a, -1, t, -1, False, 1.0, rng.integers(0, P.vocab, L))
        inbox = new_inbox

    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    jitter = rng.random(len(author))
    ts = t0 + pd.to_timedelta(np.array(step_of) + jitter, unit="D")
    return pd.DataFrame({
        "item_id": [f"p{i}" for i in range(len(author))], "ts": ts, "step": step_of,
        "author": [f"a{a}" for a in author], "author_followers": [len(followers[a]) for a in author],
        "unit": [f"u{u}" if u >= 0 else None for u in unit],
        "text": [" ".join(f"w{x}" for x in tk) for tk in toks],
        "true_parent": [f"p{p}" if p >= 0 else None for p in parent],
        "private": private, "likes_public": likes_pub, "likes_private": likes_priv,
        "like_times": like_times, "fitness": fit, "process": process, "p_private": p_private,
        "recommender": recommender, "community": "sim", "kind": "text"})


# =============================================================== observation


def observe(log: pd.DataFrame, edge_mode: str = "explicit", deletion: float = 0.05, horizon: int = 5,
            full: bool = False, seed: int = 20261008) -> pd.DataFrame:
    """Turn a ground-truth log into the items table a researcher would see.

    full=True returns the ground truth instead: all posts incl. private ones, true parents, and engagement =
    public + private likes. Otherwise private posts and deleted posts are dropped, engagement = public likes
    within `horizon` steps, and `parent` follows `edge_mode` ('explicit', 'ancestor', 'inferred').
    """
    d = log.copy()
    lt = d.like_times.to_numpy()
    st = d.step.to_numpy()
    within = np.array([sum(1 for x in l if x <= s + horizon) for l, s in zip(lt, st)])
    if full:
        d["engagement"] = within.astype(float)  # all likes (public + private) within the horizon
        d["parent"] = d.true_parent
        d["observed"] = True
        return d.drop(columns=["like_times"])
    tot = (d.likes_public + d.likes_private).to_numpy()
    pub_share = np.where(tot > 0, d.likes_public / np.maximum(tot, 1), 0)
    d["engagement"] = np.round(within * pub_share).astype(float)
    rng = np.random.default_rng(seed)
    q25 = np.quantile(d.engagement[~d.private], 0.25) if (~d.private).any() else 0
    pdel = np.where(d.engagement <= q25, 2 * deletion, deletion)
    deleted = rng.random(len(d)) < pdel
    vis = (~d.private.to_numpy()) & (~deleted)
    d["observed"] = vis
    ids = d.item_id.to_numpy()
    par = dict(zip(ids, d.true_parent))
    visible = set(ids[vis])
    if edge_mode == "explicit":
        d["parent"] = [p if (p in visible) else None for p in d.true_parent]
    elif edge_mode == "ancestor":
        out = []
        for p in d.true_parent:
            while p is not None and p not in visible:
                p = par.get(p)
            out.append(p)
        d["parent"] = out
    elif edge_mode == "inferred":
        d["parent"] = None
    else:
        raise ValueError("edge_mode must be explicit, ancestor or inferred")
    o = d[vis].drop(columns=["like_times"]).reset_index(drop=True)
    if edge_mode == "inferred":
        o["parent"] = chains.infer_parents(o)
    return o


# =============================================================== metrics


def _trajectories(items: pd.DataFrame, min_n: int = 60, kmax: int = 12) -> list:
    out = []
    for _, g in items.dropna(subset=["unit"]).groupby("unit"):
        if len(g) < min_n:
            continue
        top = g.text.value_counts().index[0]
        k = min(kmax, len(g) // 30)
        if k < 5:
            continue
        t = (g.ts - g.ts.min()).dt.total_seconds().to_numpy() / 86400
        out.append(selection.equal_count_bins(t, (g.text == top).to_numpy(), k))
    return out


def _m3_frame(items: pd.DataFrame) -> pd.DataFrame:
    """Identical-variant postings by >= 2 distinct authors, with repost order and leave-one-out author baseline."""
    d = items.copy()
    d["y"] = np.log1p(d.engagement)
    a = d.groupby("author").y.agg(["sum", "count"])
    s, c = d.author.map(a["sum"]), d.author.map(a["count"])
    d["author_base"] = np.where(c > 1, (s - d.y) / np.maximum(c - 1, 1), 0.0)
    d["author_new"] = (c <= 1).astype(float)
    m = d.dropna(subset=["unit"]).copy()
    m["content"] = m.unit + "|" + m.text
    na = m.groupby("content").author.transform("nunique")
    m = m[na >= 2].sort_values("ts")
    m["log_order"] = np.log(m.groupby("content").cumcount() + 1.0)
    m["author_new"] = m.author_new.astype(float)
    return m


def _n1(items: pd.DataFrame, n_boot: int = 300, seed: int = 20261008) -> Result | None:
    d = items.sort_values("ts").reset_index(drop=True)
    d["y"] = np.log1p(d.engagement)
    # author prior mean (expanding, strictly earlier)
    g = d.groupby("author").y
    d["author_prior_n"] = g.cumcount()
    d["author_prior_mean"] = (g.cumsum() - d.y) / d.author_prior_n.replace(0, np.nan)
    d["author_prior_mean"] = d.author_prior_mean.fillna(d.y.mean())
    d["log_followers"] = np.log1p(d.author_followers)
    d["t"] = d.step
    m = d.dropna(subset=["unit"]).copy()
    gu = m.groupby("unit")
    m["prior_n"] = gu.cumcount()
    m["log_prior_n"] = np.log1p(m.prior_n)
    m["log_prior_eng"] = np.log1p(gu.engagement.cumsum() - m.engagement)
    k = 5.0
    prior_sum = gu.y.cumsum() - m.y
    m["unit_fitness"] = (prior_sum + k * d.y.mean()) / (m.prior_n + k)
    first = gu.ts.transform("min")
    m["unit_age"] = (m.ts - first).dt.total_seconds() / 3600
    pt = dict(zip(d.item_id, d.text))
    m["mut_dist"] = [1 - text.sim_edit(t, pt[p]) if isinstance(p, str) and p in pt else 0.0
                     for t, p in zip(m.text, m.parent)]
    m = m[m.prior_n >= 1]
    if len(m) < 200:
        return None
    cut = m.t.quantile(0.6)
    tr, te = m[m.t < cut].copy(), m[m.t >= cut].copy()
    thr = tr.engagement.quantile(0.9)
    tr["hit"] = (tr.engagement > thr).astype(int) if (tr.engagement > thr).any() else (tr.engagement >= thr).astype(int)
    te["hit"] = (te.engagement > thr).astype(int)
    if tr.hit.nunique() < 2 or te.hit.nunique() < 2:
        return None
    blocks = {"B": ["log_followers", "author_prior_mean", "author_prior_n", "t"],
              "CA": ["log_prior_n", "log_prior_eng"], "M": ["unit_fitness", "unit_age", "mut_dist"]}
    r = predict.nested_compare(tr, te, "hit", blocks, [(["B", "CA"], ["M"])], cluster="unit",
                               n_boot=n_boot, seed=seed)
    return r[0]


def evaluate(items: pd.DataFrame, null_sims: int = 0, n_boot: int = 300, m4_null: int = 200,
             seed: int = 20261008, which: tuple = ("M1", "M2", "M3", "M4", "M5", "N1")) -> dict:
    """All metrics on an items table (columns of the common data model plus `author_followers` for N1).
    Returns {name: Result}; metrics whose data requirements are not met are omitted."""
    out: dict = {}
    memes = items.dropna(subset=["unit"])
    if "M1" in which and memes.unit.nunique() >= 2:
        out["M1"] = text.m1_test(memes, n_boot=n_boot, silhouette_n=500, seed=seed)
    if "M2" in which:
        e = chains.edges_from_items(memes)
        if len(e) >= 30:
            meds = chains.unit_medoids(memes, cap=400, seed=seed)
            out["M2"] = chains.m2_test(e, meds, n_boot=n_boot, null_sims=null_sims, seed=seed)
            dp = chains.depth_profile(memes)
            pr = chains.price_decomposition(memes, meds)
            share = float((pr.transmission.abs() > pr.selection.abs()).mean()) if len(pr) else np.nan
            out["M6"] = Result("M6 Price (share |transmission| > |selection|)", share, np.nan, np.nan, len(pr),
                               {"median_selection": float(pr.selection.median()) if len(pr) else np.nan,
                                "median_transmission": float(pr.transmission.median()) if len(pr) else np.nan,
                                "depth_saturation_ratio": dp["saturation_ratio"],
                                "depth_slope_early": dp["slope_early"], "depth_slope_late": dp["slope_late"]}, None)
    if "M3" in which:
        m = _m3_frame(items)
        if m.content.nunique() >= 10 and m.author.nunique() >= 10:
            try:
                out["M3"] = fitness.content_icc(m, y="y", content="content", author="author",
                                                fixed=["log_order", "author_base", "author_new"],
                                                n_boot=n_boot, seed=seed)
            except Exception as ex:  # noqa: BLE001
                out["M3_error"] = str(ex)
    if "M4" in which:
        tr = _trajectories(memes)
        if len(tr) >= 3:
            out["M4"] = selection.fit_calibrated(tr, Ne_grid=(100, 1000, 10000), reps=m4_null, n_sets=500,
                                                 s_grid=(), seed=seed)
    if "M5" in which:
        ds = []
        for _, g in memes.groupby("unit"):
            v = list(g.text.value_counts().index[:30])
            if len(v) >= 8:
                ds.append(phylo.delta_q(phylo.word_distance_matrix(v), max_q=3000, seed=seed)[0])
        if ds:
            med, lo, hi = text.bootstrap_median(np.array(ds), n_boot, seed)
            out["M5"] = Result("M5-delta", med, lo, hi, len(ds), {"mean_delta": float(np.mean(ds))}, None)
    if "N1" in which and "author_followers" in items:
        r = _n1(items, n_boot=n_boot, seed=seed)
        if r is not None:
            out["N1"] = r
    return out


def composite(res: dict, calibrated: bool = False, content_strong: float = 0.5) -> str:
    """Pre-registered composite rule mapping metric verdicts to an inferred process."""
    m2 = res.get("M2")
    if m2 is None:
        return "undetermined"
    v = m2.detail.get("verdict_calibrated", m2.verdict) if calibrated else m2.verdict
    if v == "attraction":
        return "attraction"
    if v != "replication":
        return "undetermined"
    m3, m4 = res.get("M3"), res.get("M4")
    c_icc = m3.estimate if m3 is not None else np.nan
    a_icc = m3.detail["icc_author"] if m3 is not None else np.nan
    if (np.isfinite(c_icc) and c_icc >= content_strong) or (m4 is not None and str(m4.verdict).startswith("selection")):
        return "replicator"
    if np.isfinite(a_icc) and np.isfinite(c_icc) and a_icc > c_icc:
        return "contagion"
    return "neutral"


# =============================================================== sweeps


def _flatten(res: dict) -> dict:
    row = {}
    for k, r in res.items():
        if not isinstance(r, Result):
            continue
        row[f"{k}_est"] = r.estimate
        row[f"{k}_lo"] = r.ci_low
        row[f"{k}_hi"] = r.ci_high
        row[f"{k}_n"] = r.n
        row[f"{k}_verdict"] = r.verdict
        if k == "M2":
            row["M2_verdict_m2a"] = r.detail.get("verdict_m2a")
        for dk, dv in r.detail.items():
            if isinstance(dv, (int, float, str, bool, np.floating, np.integer)) or dv is None:
                row[f"{k}_{dk}"] = dv
            elif isinstance(dv, (list, tuple)) and len(dv) == 2 and all(isinstance(x, (int, float, np.floating)) for x in dv):
                row[f"{k}_{dk}_lo"], row[f"{k}_{dk}_hi"] = dv
    return row


def run_one(process: str, p_private: float, recommender: bool, rep: int, base_seed: int = 20261008,
            params: WorldParams | None = None, edge_modes=("explicit", "ancestor", "inferred"),
            null_sims: int = 0, n_boot: int = 200, m4_null: int = 100, deletion: float = 0.05) -> list[dict]:
    """One world: ground truth plus each edge mode's observed metrics. Returns tidy rows."""
    seed = base_seed + rep
    log = simulate(process, p_private, recommender, seed, params)
    rows = []
    layers = [("truth", observe(log, full=True))] + [(m, observe(log, m, deletion=deletion, seed=seed)) for m in edge_modes]
    for layer, items in layers:
        res = evaluate(items, null_sims=null_sims, n_boot=n_boot, m4_null=m4_null, seed=seed)
        row = {"process": process, "p_private": p_private, "recommender": recommender, "rep": rep, "layer": layer,
               "n_items": len(items), "n_meme_items": int(items.unit.notna().sum()),
               "share_private": float(log.private.mean()), "inferred": composite(res),
               "inferred_calibrated": composite(res, calibrated=True) if null_sims else None}
        row.update(_flatten(res))
        rows.append(row)
    return rows


def sweep(processes=PROCESSES, p_grid=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9), recommender=(True, False),
          reps: int = 30, workers: int = 1, **kw) -> pd.DataFrame:
    """Full factorial sweep; `workers` > 1 uses a process pool. Extra keywords go to `run_one`."""
    jobs = [(pr, p, r, i) for pr in processes for p in p_grid for r in recommender for i in range(reps)]
    if workers <= 1:
        rows = [x for j in jobs for x in run_one(*j, **kw)]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(run_one, *j, **kw) for j in jobs]
            rows = [x for f in futs for x in f.result()]
    return pd.DataFrame(rows)


def confusion(df: pd.DataFrame, layer: str = "explicit", col: str = "inferred") -> pd.DataFrame:
    """Row-normalized confusion matrix (true process x inferred label) per p_private."""
    d = df[df.layer == layer]
    return (d.groupby(["p_private", "process"])[col].value_counts(normalize=True)
            .rename("share").reset_index()
            .pivot_table(index=["p_private", "process"], columns=col, values="share", fill_value=0.0))


def breakdown_fraction(df: pd.DataFrame, a: str = "replicator", b: str = "attraction", layer: str = "explicit",
                       col: str = "inferred", threshold: float = 0.8) -> float:
    """Smallest p_private at which the balanced accuracy for telling `a` from `b` falls below `threshold`.
    Accuracy for each process is the share of its worlds labelled with its own name; nan if never below."""
    d = df[(df.layer == layer) & df.process.isin([a, b])]
    for p, g in d.groupby("p_private"):
        acc = np.mean([(g[g.process == x][col] == x).mean() for x in (a, b)])
        if acc < threshold:
            return float(p)
    return float("nan")


def distinguishability(p_private: float, reps: int = 10, recommender: bool = True,
                       params: WorldParams | None = None, edge_mode: str = "explicit", workers: int = 1,
                       **kw) -> pd.DataFrame:
    """For a planned study: simulate each process at the given private-share fraction and return the confusion
    matrix of the composite decision rule on the observed layer. Use `params` to match your data's scale
    (number of units, agents, steps)."""
    df = sweep(PROCESSES, (p_private,), (recommender,), reps, workers, params=params, edge_modes=(edge_mode,), **kw)
    return confusion(df, layer=edge_mode)


def main(argv=None):  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Can public data tell replication from attraction or drift?")
    ap.add_argument("--p-private", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-recommender", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    print(distinguishability(a.p_private, a.reps, not a.no_recommender, workers=a.workers).round(2).to_string())


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["WorldParams", "PROCESSES", "simulate", "observe", "evaluate", "composite", "run_one", "sweep",
           "confusion", "breakdown_fraction", "distinguishability", "asdict", "replace"]
