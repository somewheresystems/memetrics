"""Scorecard: every memetic measure on one items table, with literal and calibrated readings.

``run(items)`` computes

- M1 units: discreteness of units (text edit similarity; embeddings too when given);
- M2 fidelity versus attraction, on all transmission edges and on derivatives only (verbatim copies excluded);
- M3 identical-content ICC versus author ICC, with fixed effects and a shuffled-content null;
- M4 selection versus drift: Frequency Increment Test with a Wright-Fisher null and a binning sweep, plus the
  neutral-model turnover test;
- M5 tree-likeness: delta scores against simulated tree, network and star references;
- N1 non-redundancy: nested out-of-time prediction (rival baseline, cumulative advantage, content bias, memetic terms);

and returns a JSON-safe dict. Each measure reports two readings side by side:

- **literal**: the default decision rule applied to the raw estimate, with the thresholds in :class:`Rules`;
- **calibrated**: the estimate compared with a simulated null of the same size and structure (replicator null for
  M2, Wright-Fisher null for the FIT, neutral-model simulations for turnover, simulated trees and networks for M5,
  shuffled content for M3, the boundary-gap diagnostic and a Gaussian surrogate for M1).

Every card carries the "what this data cannot see" panel, and optionally a distinguishability run of the
observation simulator (:func:`memetrics.observation.distinguishability`) at the dataset's size and an assumed
private-share fraction. Render with :func:`memetrics.report.write`.

Text measures run on ``memetrics.text.normalize``-d text. Engagement enters as ``log1p(max(engagement, 0))``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import __version__, chains, fitness, phylo, predict, selection, text, units
from .result import Result, jsonable
from .schema import validate


@dataclass(frozen=True)
class Rules:
    """Decision-rule thresholds (defaults: the pre-registered rules of the accompanying study)."""

    d_threshold: float = 2.0  # M1: median Ashman's D above which units are discrete
    blend_units: float = 0.10  # M1: blend share below which units are discrete
    blend_continuum: float = 0.25  # M1: blend share above which variation is a continuum
    f_threshold: float = 0.9  # M2: per-step fidelity for replication
    strong: float = 0.5  # M3: content ICC at or above which content dominates
    weak: float = 0.2  # M3: content ICC below which content fitness is weak
    fdr: float = 0.05  # M4: Benjamini-Hochberg false discovery rate
    min_auc_gain: float = 0.0  # N1: minimum held-out AUC gain


@dataclass(frozen=True)
class Effort:
    """Resampling and simulation sizes. ``Effort.fast()`` for a quick look, the defaults for a report."""

    n_boot: int = 1000
    null_sims: int = 100  # M2 replicator-null simulations
    m3_perm: int = 200  # M3 shuffled-content replicates
    fit_reps: int = 100  # M4 Wright-Fisher replicates per trajectory and N_e
    fit_sets: int = 500
    turnover_runs: int = 50
    m5_units: int = 10
    m5_reps: int = 50
    max_edges: int = 20_000

    @classmethod
    def fast(cls) -> Effort:
        return cls(n_boot=200, null_sims=20, m3_perm=50, fit_reps=30, fit_sets=200, turnover_runs=15, m5_units=3,
                   m5_reps=15, max_edges=5000)


BLIND_SPOTS = [
    ("Exposure (impressions)",
     "Engagement counts adoptions without a denominator. Content fitness (M3) and memetic prediction terms (N1) "
     "mix how appealing an item is with how often it was shown."),
    ("Private and cross-platform sharing",
     "DMs, group chats and screenshots moved between platforms leave no public trace. Chains are thinned, so "
     "fidelity (M2) spans unobserved steps, inferred parents are guesses, and tree-likeness (M5) is distorted by "
     "missing links."),
    ("Deletion",
     "Deleted and removed posts are missing, and not at random: low-engagement posts disappear more often. This "
     "inflates engagement and breaks chains."),
    ("Offline transmission",
     "Speech, print and TV spread is invisible. A variant can reappear with no observed parent, which looks like "
     "independent invention."),
    ("Algorithmic ranking",
     "The feed decides who sees what. Frequency change (M4) and engagement mix people's choices with the "
     "ranker's, and a ranker that rewards engagement can manufacture apparent selection."),
]


# ================================================================== helpers


def _log_eng(e: pd.Series) -> np.ndarray:
    return np.log1p(pd.to_numeric(e, errors="coerce").clip(lower=0).fillna(0).to_numpy(float))


def _days(ts: pd.Series) -> np.ndarray:
    return ((ts - ts.min()).dt.total_seconds() / 86400).to_numpy(float)


def _fmt(r: Result | None) -> str:
    if r is None or r.estimate is None or not np.isfinite(r.estimate):
        return "n/a"
    ci = f" [{r.ci_low:.3g}, {r.ci_high:.3g}]" if np.isfinite([r.ci_low, r.ci_high]).all() else ""
    return f"{r.estimate:.3g}{ci}"


def _row(label: str, r: Result | None, literal: str | None, calibrated: str | None, estimate: str | None = None,
         **extra) -> dict:
    return {"label": label, "estimate": estimate if estimate is not None else _fmt(r), "literal": literal,
            "calibrated": calibrated, "n": None if r is None else r.n, "result": None if r is None else r.to_json(),
            **extra}


def _measure(mid: str, title: str, claim: str, rule: str, calibration: str) -> dict:
    return {"id": mid, "title": title, "claim": claim, "rule": rule, "calibration": calibration, "status": "ok",
            "reason": None, "rows": [], "notes": [], "seconds": None}


def _skip(m: dict, reason: str) -> dict:
    m.update(status="skipped", reason=reason)
    return m


# ================================================================== summary


def summarize(items: pd.DataFrame) -> dict:
    d = items
    n = len(d)
    has = {c: c in d and d[c].notna().any() for c in ("text", "phash", "unit", "parent", "engagement")}
    resolvable = int(d.parent.isin(set(d.item_id)).sum()) if has["parent"] else 0
    unit_sizes = d.unit.value_counts() if has["unit"] else pd.Series(dtype=int)
    return {"n_items": n, "n_authors": int(d.author.nunique()), "n_communities": int(d.community.nunique()),
            "kinds": d.kind.value_counts().to_dict(), "start": str(d.ts.min()), "end": str(d.ts.max()),
            "columns": {k: bool(v) for k, v in has.items()}, "n_units": int(len(unit_sizes)),
            "share_in_units": float(d.unit.notna().mean()) if has["unit"] else 0.0,
            "median_unit_size": float(unit_sizes.median()) if len(unit_sizes) else None,
            "share_with_parent_in_table": resolvable / n if n else 0.0,
            "share_deleted_flag": float(d.deleted.mean()) if "deleted" in d else None}


# ================================================================== M1


def m1(d: pd.DataFrame, emb: np.ndarray | None, rules: Rules, eff: Effort, seed: int) -> dict:
    m = _measure("M1", "Units", "Cultural variants cluster into discrete, bounded units (replicators).",
                 f"units if median Ashman's D > {rules.d_threshold} and blend share < {rules.blend_units}; "
                 f"continuum if D < {rules.d_threshold} or blends > {rules.blend_continuum}; else intermediate",
                 "Text: boundary-gap ratio (a continuum cut into segments gives < 2, discrete units > 2). "
                 "Embeddings: D compared with a cluster-free Gaussian surrogate of the same covariance.")
    memes = d[d.unit.notna() & d.text.ne("")] if "text" in d else d.iloc[:0]
    if memes.unit.nunique() >= 2:
        r = text.m1_test(memes, n_boot=eff.n_boot, silhouette_n=2000, d_threshold=rules.d_threshold,
                         blend_units=rules.blend_units, blend_continuum=rules.blend_continuum, seed=seed,
                         name="M1 text")
        gap = r.detail.get("median_gap_ratio", np.nan)
        cal = None if not np.isfinite(gap) else ("discrete boundaries" if gap >= 2 else "continuum-like boundaries")
        m["rows"].append(_row("text (edit similarity)", r, r.verdict, cal, gap_ratio=gap))
    if emb is not None:
        lab = pd.factorize(d.unit)[0]
        if len(np.unique(lab[lab >= 0])) >= 2:
            r = units.discreteness(emb, lab, min_unit=5, n_boot=eff.n_boot, d_threshold=rules.d_threshold,
                                   blend_units=rules.blend_units, blend_continuum=rules.blend_continuum, seed=seed,
                                   name="M1 embedding")
            z = units.gaussian_surrogate(emb, dims=min(64, emb.shape[1]), seed=seed)
            rz = units.discreteness(z, lab, min_unit=5, n_boot=eff.n_boot, seed=seed, name="M1 surrogate")
            cal = "above cluster-free surrogate" if r.ci_low > rz.ci_high else "not above cluster-free surrogate"
            m["rows"].append(_row("embedding (cosine)", r, r.verdict, cal, surrogate_D=rz.estimate))
    if not m["rows"]:
        return _skip(m, "needs at least two units with text (run `memetrics units`) or embeddings")
    return m


# ================================================================== M2


def _edges(memes: pd.DataFrame, max_edges: int, seed: int) -> tuple[pd.DataFrame, str]:
    if "parent" in memes and memes.parent.isin(set(memes.item_id)).sum() >= 30:
        e, src = chains.edges_from_items(memes), "observed parent links"
    else:
        e = chains.edges_from_items(memes.assign(parent=chains.infer_parents(memes)))
        src = "inferred parents (most similar earlier item: a guess, not an observed copy)"
    if len(e) > max_edges:  # sample whole units so chains stay intact
        rng = np.random.default_rng(seed)
        sizes = e.unit.value_counts().sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        keep = sizes.index[sizes.cumsum() <= max_edges]
        e = e[e.unit.isin(keep)].reset_index(drop=True)
        src += f"; {len(keep)} units sampled to <= {max_edges} edges"
    return e, src


def m2(d: pd.DataFrame, rules: Rules, eff: Effort, seed: int, min_edges: int = 30) -> dict:
    m = _measure("M2", "Fidelity versus attraction",
                 "Replication: high per-step fidelity and no directional drift. Attraction: convergence toward "
                 "attractors with lower fidelity.",
                 f"replication if median fidelity >= {rules.f_threshold} and the convergence-slope CI contains 0; "
                 f"attraction if the slope CI is below 0 and fidelity < {rules.f_threshold}; else mixed",
                 "Texts re-simulated by pure copying along the observed topology with matched mutation; the observed "
                 "slope is compared with that null (copying alone produces a negative slope).")
    memes = d[d.unit.notna() & d.text.ne("")]
    if memes.unit.nunique() < 1:
        return _skip(m, "needs units with text")
    e, src = _edges(memes, eff.max_edges, seed)
    m["notes"].append(f"Edges: {src}.")
    if len(e) < min_edges:
        return _skip(m, f"only {len(e)} parent-child edges within units (need {min_edges})")
    meds = chains.unit_medoids(memes, seed=seed)
    r = chains.m2_test(e, meds, f_threshold=rules.f_threshold, n_boot=eff.n_boot, null_sims=eff.null_sims,
                       seed=seed, name="M2 all edges")
    m["rows"].append(_row("all edges", r, r.verdict, r.detail.get("verdict_calibrated"),
                          share_verbatim=r.detail["share_identical"]))
    f = np.array([text.sim_edit(a, b) for a, b in zip(e.parent_text, e.child_text)])
    der = e[f < 1].reset_index(drop=True)
    m["notes"].append(f"Verbatim copies: {int((f >= 1).sum())} of {len(e)} edges ({(f >= 1).mean():.0%}).")
    if len(der) >= min_edges:
        rd = chains.m2_test(der, meds, f_threshold=rules.f_threshold, n_boot=eff.n_boot, null_sims=eff.null_sims,
                            seed=seed, name="M2 derivatives")
        m["rows"].append(_row("derivatives only (f < 1)", rd, rd.verdict, rd.detail.get("verdict_calibrated")))
    else:
        m["notes"].append(f"Derivatives only: {len(der)} edges, too few to test.")
    return m


# ================================================================== M3


def _content_keys(d: pd.DataFrame) -> pd.Series:
    key = pd.Series(None, index=d.index, dtype=object)
    if "phash" in d and d.phash.notna().any():
        ok = d.phash.notna() & d.kind.eq("image")
        if ok.any():
            sub = d[ok]
            g = units.identical_groups(sub.phash.astype("uint64").to_numpy(), sub.ts.to_numpy(),
                                       text=sub.text.tolist() if "text" in sub else None)
            key[ok] = ["p" + str(sub.item_id.iloc[i]) for i in g]
    if "text" in d:
        ok = key.isna() & d.text.str.split().str.len().ge(3)
        key[ok] = "t" + d.text[ok]
    return key


def m3(d: pd.DataFrame, rules: Rules, eff: Effort, seed: int) -> dict:
    m = _measure("M3", "Content-intrinsic fitness",
                 "Spread depends mainly on the meme: identical content posted independently by different authors "
                 "gets similar engagement.",
                 f"content dominates if content ICC >= {rules.strong}; weak content fitness if < {rules.weak}; "
                 "else intermediate (compare with the author ICC)",
                 "Content labels shuffled within community; the observed content ICC is compared with the shuffled "
                 "97.5th percentile.")
    if "engagement" not in d or d.engagement.isna().all():
        return _skip(m, "needs an engagement column")
    x = d.assign(content=_content_keys(d), y=_log_eng(d.engagement))
    a = x.groupby("author").y.agg(["sum", "count"])
    s, c = x.author.map(a["sum"]), x.author.map(a["count"])
    x["author_base"] = np.where(c > 1, (s - x.y) / np.maximum(c - 1, 1), 0.0)
    x["author_new"] = (c <= 1).astype(float)
    x = x[x.content.notna()]
    x = x[x.groupby("content").author.transform("nunique") >= 2].sort_values("ts")
    if x.content.nunique() < 10 or x.author.nunique() < 10:
        return _skip(m, "needs >= 10 identical contents each posted by >= 2 authors")
    x["log_order"] = np.log(x.groupby("content").cumcount() + 1.0)
    x["how"] = (x.ts.dt.dayofweek * 24 + x.ts.dt.hour).astype(str)
    x["era"] = x.ts.dt.strftime("%Y-%m")
    fixed = [f for f in ("community", "how", "era") if x[f].nunique() > 1] + ["log_order", "author_base",
                                                                            "author_new"]
    r = fitness.content_icc(x, y="y", content="content", author="author", fixed=fixed, n_boot=eff.n_boot,
                            seed=seed, strong=rules.strong, weak=rules.weak, name="M3 content ICC")
    resid = fitness.residualize(x, "y", fixed)
    rng = np.random.default_rng(seed)
    cc, aa = pd.factorize(x.content)[0], pd.factorize(x.author)[0]
    groups = [np.flatnonzero(x.community.to_numpy() == g) for g in x.community.unique()]
    null = []
    for _ in range(eff.m3_perm):
        p = cc.copy()
        for g in groups:
            p[g] = cc[rng.permutation(g)]
        null.append(fitness.crossed_components(resid, p, aa)["icc_content"])
    q975 = float(np.nanquantile(null, 0.975))
    cal = "above shuffled-content null" if r.estimate > q975 else "within shuffled-content null"
    m["rows"].append(_row("content ICC", r, r.verdict, cal, null_q975=q975))
    ra = Result("M3 author ICC", r.detail["icc_author"], *r.detail["icc_author_ci"], r.n)
    lo, hi = r.detail["diff_ci"]
    comp = "content > author" if lo > 0 else "author > content" if hi < 0 else "not separable"
    m["rows"].append(_row("author ICC (same model)", ra, None, comp))
    m["notes"].append(f"Fixed effects: {', '.join(fixed)}. {r.detail['n_contents']} contents, "
                      f"{r.detail['n_authors']} authors.")
    return m


# ================================================================== M4


def _trajectories(memes: pd.DataFrame, k: int, min_n: int = 60) -> list[pd.DataFrame]:
    out = []
    for _, g in memes.groupby("unit"):
        kk = min(k, len(g) // 10)
        if len(g) < min_n or kk < 5:
            continue
        top = g.text.value_counts().index[0]
        b = selection.equal_count_bins(_days(g.ts), (g.text == top).to_numpy(), kk)
        if ((b.x > 0) & (b.x < 1)).sum() >= 3:
            out.append(b)
    return out


def m4(d: pd.DataFrame, rules: Rules, eff: Effort, seed: int, Ne_grid=(100, 1000, 10_000)) -> dict:
    m = _measure("M4", "Selection versus drift",
                 "Variant frequencies within units change by selection beyond neutral drift.",
                 f"FIT: share of trajectories rejecting neutrality at FDR {rules.fdr}, against the nominal 5% "
                 "of the raw test; turnover: no fixed rule (the exponent is reported)",
                 "FIT: the rejection share compared with Wright-Fisher neutral trajectories matched on start "
                 "frequency, bin times and sample sizes, for each N_e; repeated for 6, 9 and 12 bins. Turnover: "
                 "exponent compared with neutral-model simulations matched on innovation rate and sample sizes.")
    memes = d[d.unit.notna() & d.text.ne("")]
    main = _trajectories(memes, 12)
    if len(main) >= 3:
        spacing = np.concatenate([np.diff(b.t.to_numpy()) for b in main])
        gen_unit = float(min(1.0, np.quantile(spacing[spacing > 0], 0.05))) if (spacing > 0).any() else 1.0
        m["notes"].append(f"Generation length for the Wright-Fisher null: {gen_unit:.3g} days.")
        for k in (12, 9, 6):
            tr = main if k == 12 else _trajectories(memes, k)
            if len(tr) < 3:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = selection.fit_calibrated(tr, Ne_grid=Ne_grid, gen_unit=gen_unit, reps=eff.fit_reps,
                                             n_sets=eff.fit_sets, fdr=rules.fdr, s_grid=(0.05,) if k == 12 else (),
                                             seed=seed, name=f"M4 FIT ({k} bins)")
            raw = r.detail["raw_share_p05"]
            lit = "exceeds nominal 5%" if raw > 0.05 else "within nominal 5%"
            m["rows"].append(_row(f"FIT, up to {k} equal-count bins", r, lit, r.verdict,
                                  estimate=f"BH share {r.estimate:.3g}; raw p<0.05 share {raw:.3g}"))
    else:
        m["notes"].append("FIT: fewer than 3 units with >= 60 items and a varying top variant.")
    var = (memes.unit + "|" + memes.text).to_numpy()
    order = np.argsort(memes.ts.to_numpy(), kind="stable")
    T = int(min(20, len(memes) // 100))
    if T >= 8:
        steps = [var[order[a:b]] for a, b in zip(*(lambda e: (e[:-1], e[1:]))(np.linspace(0, len(var), T + 1)
                                                                              .astype(int)))]
        sizes = [len(s) for s in steps]
        share = selection.new_variant_share(steps)
        mu = selection.calibrate_mu(share, sizes, seed=seed)
        r = selection.turnover_test(steps, mu=mu, sample_sizes=sizes, runs=eff.turnover_runs, seed=seed,
                                    name="M4 turnover")
        if np.isfinite(r.estimate):
            m["rows"].append(_row("neutral turnover (top-y lists)", r, None, r.verdict,
                                  estimate=f"x = {r.estimate:.3g} (neutral 95%: {r.ci_low:.3g} to {r.ci_high:.3g})"))
        else:
            m["notes"].append("Turnover: too few distinct variants per step for top-y lists.")
    else:
        m["notes"].append("Turnover: fewer than 800 meme items.")
    if not m["rows"]:
        return _skip(m, "not enough units with long frequency trajectories")
    return m


# ================================================================== M5


def m5(d: pd.DataFrame, eff: Effort, seed: int, max_variants: int = 30) -> dict:
    m = _measure("M5", "Tree-like descent",
                 "Replicator lineages are tree-like enough for phylogenetics.",
                 "no fixed threshold: the delta score (0 = perfectly tree-like) is reported",
                 "Each unit's delta compared with token sequences simulated on coalescent trees, on reticulate "
                 "networks and as independent mutations of one template (star), matched on number of variants, "
                 "length and mean distance.")
    memes = d[d.unit.notna() & d.text.ne("")]
    cand = []
    for u, g in memes.groupby("unit"):
        v = list(g.text.value_counts().index[:max_variants])
        if len(v) >= 8:
            cand.append((len(g), u, v))
    if not cand:
        return _skip(m, "needs units with >= 8 distinct variants")
    cand.sort(key=lambda c: (-c[0], str(c[1])))
    res = []
    for i, (_, u, v) in enumerate(cand[: eff.m5_units]):
        L = int(np.median([len(t.split()) for t in v]))
        res.append(phylo.tree_likeness(phylo.word_distance_matrix(v), length=max(L, 2), reps=eff.m5_reps,
                                       seed=seed + i, name=f"M5 {u}"))
    deltas = np.array([r.estimate for r in res])
    med, lo, hi = text.bootstrap_median(deltas, eff.n_boot, seed)
    verdicts = pd.Series([r.verdict for r in res]).value_counts()
    summary = Result("M5 median delta", med, lo, hi, len(res),
                     {"verdicts": verdicts.to_dict(), "units": [r.to_json() for r in res]})
    cal = ", ".join(f"{k} {v}/{len(res)}" for k, v in verdicts.items())
    m["rows"].append(_row(f"{len(res)} largest units", summary, None, cal))
    return m


# ================================================================== N1


def _n1_frame(d: pd.DataFrame, lag_hours: float) -> pd.DataFrame:
    x = d.sort_values("ts", kind="stable").reset_index(drop=True)
    x["t_s"] = (x.ts - x.ts.min()).dt.total_seconds()
    x["y"] = _log_eng(x.engagement)
    lag = lag_hours * 3600
    a = predict.prior_unit_features(x, "author", "t_s", "y", lag=lag, k=0, prefix="author")
    c = predict.prior_unit_features(x, "community", "t_s", "y", lag=lag, k=0, prefix="comm")
    x["author_prior_n"] = np.log1p(x.groupby("author").cumcount())
    x["author_prior_mean"] = a.author_fit
    x["community_prior_mean"] = c.comm_fit
    how = (x.ts.dt.dayofweek * 24 + x.ts.dt.hour).to_numpy() * 2 * np.pi / 168
    x["how_sin"], x["how_cos"] = np.sin(how), np.cos(how)
    x["t_days"] = x.t_s / 86400
    t = x.t_s.to_numpy()
    x["competition"] = np.log1p(np.arange(len(t)) - np.searchsorted(t, t - 86400))
    raw = x.text.fillna("") if "text" in x else pd.Series("", index=x.index)
    x["len_chars"] = np.log1p(raw.str.len())
    x["has_url"] = raw.str.contains("<url>|http", regex=True).astype(float)
    x["n_tags"] = raw.str.count("#").astype(float)
    x["is_image"] = x.kind.eq("image").astype(float)
    m = x[x.unit.notna()].copy()
    u = predict.prior_unit_features(m, "unit", "t_s", "y", lag=lag, k=5, prefix="unit")
    m = m.join(u)
    m["unit_prior_n"] = np.log1p(m.groupby("unit").cumcount())
    m["unit_prior_eng"] = np.log1p(m.unit_sum.clip(lower=0))
    m["unit_age_days"] = m.unit_age / 86400
    pt = dict(zip(x.item_id, raw))
    has_par = m.parent.isin(pt) if "parent" in m else pd.Series(False, index=m.index)
    m["has_parent"] = has_par.astype(float)
    m["mut_dist"] = [1 - text.sim_edit(t_, pt[p]) if h else 0.0
                     for t_, p, h in zip(raw[m.index], m.get("parent", pd.Series(None, index=m.index)), has_par)]
    return m


def n1(d: pd.DataFrame, rules: Rules, eff: Effort, seed: int, lag_hours: float = 48.0,
       train_share: float = 0.6) -> dict:
    m = _measure("N1", "Non-redundancy (prediction)",
                 "Memetic replicator terms predict later spread beyond the best rival baseline.",
                 "non-redundant if B -> B+M and B+CA+CB -> B+CA+CB+M both improve held-out log-likelihood and AUC "
                 "(CIs above 0); reproduced by rivals if only the first does",
                 "No simulated null: the reverse nesting (M -> M+B) shows how much the rival baseline adds beyond "
                 "the memetic terms.")
    if "engagement" not in d or d.engagement.isna().all():
        return _skip(m, "needs an engagement column")
    x = _n1_frame(d, lag_hours)
    if len(x) < 500 or x.unit.nunique() < 10:
        return _skip(m, "needs >= 500 items in >= 10 units")
    cut = x.ts.quantile(train_share)
    tr, te = x[x.ts < cut].copy(), x[x.ts >= cut].copy()
    thr = tr.engagement.quantile(0.9)
    tr["hit"], te["hit"] = (tr.engagement > thr).astype(int), (te.engagement > thr).astype(int)
    if tr.hit.nunique() < 2 or te.hit.nunique() < 2:
        return _skip(m, "engagement has too little variation for a top-decile outcome")
    mean_tr = float(tr.y.mean())  # centre unit fitness on the training mean (no look-ahead)
    for f in (tr, te):
        f["unit_fit"] = f.unit_fit - mean_tr * f.unit_n / (f.unit_n + 5)
    blocks = {"B": ["author_prior_n", "author_prior_mean", "community_prior_mean", "how_sin", "how_cos", "t_days",
                    "competition"],
              "CA": ["unit_prior_n", "unit_prior_eng"], "CB": ["len_chars", "has_url", "n_tags", "is_image"],
              "M": ["unit_fit", "unit_age_days", "mut_dist", "has_parent"]}
    comps = [(["B"], ["M"]), (["B", "CA", "CB"], ["M"]), (["M"], ["B"]), (["B"], ["CA", "CB"])]
    rs = predict.nested_compare(tr, te, "hit", blocks, comps, cluster="unit", n_boot=eff.n_boot, seed=seed,
                                min_auc_gain=rules.min_auc_gain)
    verdict = predict.nonredundancy_verdict(rs[0], rs[1])
    labels = ["rival baseline B -> +M", "B + cumulative advantage + content bias -> +M", "reverse: M -> +B",
              "context: B -> +CA+CB"]
    for lab, r in zip(labels, rs):
        est = f"dLL {r.estimate:.3g} [{r.ci_low:.3g}, {r.ci_high:.3g}]; dAUC {r.detail['delta_auc']:.3g}"
        m["rows"].append(_row(lab, r, r.verdict, None, estimate=est))
    m["rows"].append(_row("non-redundancy rule (rows 1 and 2)", None, verdict, None, estimate="—"))
    m["notes"].append(f"Outcome: top-decile engagement (> {thr:.3g}). Train before {cut:%Y-%m-%d %H:%M} UTC, "
                      "test after. Unit and "
                      f"author histories lag {lag_hours:g} h so engagement still accruing is not used.")
    return m


# ================================================================== distinguishability


def distinguishability(n_items: int, p_private: float, reps: int = 3, recommender: bool = True,
                       edge_mode: str = "explicit", seed: int = 20261008, workers: int = 1, fast: bool = True) -> dict:
    """Simulate each ground-truth process at a world size giving about ``n_items`` observed items and the assumed
    private-share fraction; return the confusion matrix of the composite decision rule and per-process accuracy."""
    from . import observation as ob

    params = ob.params_for_size(n_items)
    kw = dict(n_boot=100, m4_null=30, null_sims=10) if fast else dict(n_boot=200, m4_null=100, null_sims=30)
    t0 = time.time()
    df = ob.sweep(ob.PROCESSES, (p_private,), (recommender,), reps, workers, params=params, edge_modes=(edge_mode,),
                  base_seed=seed, **kw)
    out = {"n_items_target": n_items, "n_agents": params.n_agents, "p_private": p_private, "reps": reps,
           "recommender": recommender, "edge_mode": edge_mode, "seconds": round(time.time() - t0, 1)}
    for col in ("inferred", "inferred_calibrated"):
        sub = df[df.layer == edge_mode]
        cm = (sub.groupby("process")[col].value_counts(normalize=True).unstack(fill_value=0.0)
              .reindex(index=list(ob.PROCESSES)))
        acc = {p: float(cm.loc[p].get(p, 0.0)) for p in ob.PROCESSES}
        out[col] = {"confusion": {p: {k: float(v) for k, v in cm.loc[p].items()} for p in ob.PROCESSES},
                    "accuracy": acc, "balanced_accuracy": float(np.mean(list(acc.values())))}
    out["mean_observed_items"] = float(df[df.layer == edge_mode].n_items.mean())
    return out


# ================================================================== run


def run(items: pd.DataFrame, emb: np.ndarray | None = None, *, rules: Rules = Rules(), effort: Effort = Effort(),
        seed: int = 20261008, which=("M1", "M2", "M3", "M4", "M5", "N1"), lag_hours: float = 48.0,
        p_private: float | None = None, distinguish_reps: int = 3, workers: int = 1, progress=None) -> dict:
    """Run the scorecard on an items table. Units must be assigned (``memetrics.units.assign``).

    ``p_private``: if given, also run the observation simulator at this dataset's size and that assumed
    private-share fraction (``distinguish_reps`` worlds per process). ``progress``: optional callable(str).
    """
    validate(items)
    if "unit" not in items or items.unit.notna().sum() == 0:
        raise ValueError("no units assigned: run `memetrics units` (or memetrics.units.assign) first")
    if emb is not None and len(emb) != len(items):
        raise ValueError(f"emb has {len(emb)} rows for {len(items)} items")
    d = items.reset_index(drop=True).copy()
    d["text"] = d.text.map(text.normalize) if "text" in d else ""
    say = progress or (lambda s: None)
    fns = {"M1": lambda: m1(d, emb, rules, effort, seed), "M2": lambda: m2(d, rules, effort, seed),
           "M3": lambda: m3(d, rules, effort, seed), "M4": lambda: m4(d, rules, effort, seed),
           "M5": lambda: m5(d, effort, seed), "N1": lambda: n1(d, rules, effort, seed, lag_hours)}
    measures = []
    for k in which:
        say(f"{k} ...")
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                res = fns[k]()
            except Exception as ex:  # a failing measure must not sink the card
                res = _skip(_measure(k, k, "", "", ""), f"error: {type(ex).__name__}: {ex}")
        res["seconds"] = round(time.time() - t0, 1)
        measures.append(res)
        say(f"{k} {res['status']} ({res['seconds']} s)")
    card = {"memetrics_version": __version__, "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "seed": seed, "rules": asdict(rules), "effort": asdict(effort), "dataset": summarize(items),
            "measures": measures, "blind_spots": [{"what": w, "consequence": c} for w, c in BLIND_SPOTS],
            "distinguishability": None}
    if p_private is not None:
        say(f"distinguishability at p_private={p_private} ...")
        card["distinguishability"] = distinguishability(len(items), p_private, distinguish_reps, seed=seed,
                                                        workers=workers)
    return jsonable(card)


__all__ = ["Rules", "Effort", "BLIND_SPOTS", "run", "summarize", "distinguishability", "m1", "m2", "m3", "m4", "m5",
           "n1"]
