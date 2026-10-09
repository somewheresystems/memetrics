"""Fixed simulated inputs whose estimator outputs are pinned in ``golden.json``.

The goldens were generated from the research code before packaging. Any refactor must reproduce them exactly
(up to floating-point noise): the published numbers come from these estimators. Regenerate only for a deliberate,
documented change in estimator behaviour (``python tests/regression/make_golden.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def plain(x):
    """Recursively convert results to JSON-compatible python values."""
    if hasattr(x, "to_dict") and not isinstance(x, pd.DataFrame | pd.Series):
        x = x.to_dict()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [plain(v) for v in x.tolist()]
    if isinstance(x, pd.Series):
        return [plain(v) for v in x.tolist()]
    if isinstance(x, pd.DataFrame):
        return {c: plain(x[c]) for c in x.columns}
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        x = float(x)
        return "nan" if math.isnan(x) else ("inf" if x == math.inf else ("-inf" if x == -math.inf else x))
    return x


def _clusters(k=6, n=40, d=24, spread=0.08, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((k, d))
    x = np.repeat(c, n, 0) + spread * rng.standard_normal((k * n, d)) * np.sqrt(d)
    return x, np.repeat(np.arange(k), n)


def _planted(n_units=6, per=15, L=25, mu=0.03, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for u in range(n_units):
        base = rng.integers(0, 5000, L)
        for _ in range(per):
            t = base.copy()
            m = rng.random(L) < mu
            t[m] = rng.integers(0, 5000, m.sum())
            texts.append(" ".join(f"w{x}" for x in t))
            labels.append(u)
    return texts, np.array(labels)


def _chain_world(kind, n_units=12, n_steps=25, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for u in range(n_units):
        attractor = [f"w{x}" for x in rng.integers(0, 3000, 15)]
        cur = list(attractor)
        for j in range(len(cur)):
            if rng.random() < 0.6:
                cur[j] = f"w{rng.integers(3000)}"
        prev = None
        for s in range(n_steps):
            nxt = list(cur)
            if kind == "attraction":
                for j in range(len(nxt)):
                    if rng.random() < 0.2:
                        nxt[j] = attractor[j]
                    if rng.random() < 0.2:
                        nxt[j] = f"w{rng.integers(3000)}"
            elif rng.random() < 0.05:
                nxt[rng.integers(len(nxt))] = f"w{rng.integers(3000)}"
            iid = f"{u}-{s}"
            items.append({"item_id": iid, "unit": str(u), "text": " ".join(nxt), "parent": prev,
                          "ts": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=s)})
            prev, cur = iid, nxt
    return pd.DataFrame(items)


# ---------------------------------------------------------------- cases


def case_units():
    from memetrics import units

    x, lab = _clusters()
    r = units.discreteness(x, lab, n_boot=200, seed=3)
    z = units.gaussian_surrogate(x, dims=8, seed=1)
    rz = units.discreteness(z, (z[:, 0] > 0).astype(int) + 2 * (z[:, 1] > 0), n_boot=100, seed=1)
    rng = np.random.default_rng(5)
    base = rng.integers(0, 2**63, 40, dtype=np.uint64)
    ph = np.concatenate([base, base ^ np.uint64(0b111), base ^ np.uint64(0xFF)])
    g = units.identical_groups(ph, ts=rng.permutation(len(ph)))
    st = units.unit_stats(x, lab, min_unit=5, seed=2)
    return {"discreteness": plain(r), "surrogate": plain(rz), "groups": plain(g),
            "w_sum": float(st.w.sum()), "b_sum": float(st.b.sum()), "blend": int(st.blend.sum()),
            "hamming": plain(units.hamming64(ph[:5], ph[40:45])),
            "medoids": plain(units.medoids(x, lab, seed=4))}


def case_text():
    from memetrics import text

    texts, labels = _planted()
    r = text.m1_test(pd.DataFrame({"text": texts, "unit": labels.astype(str)}), n_boot=200, silhouette_n=60, seed=7)
    rw = text.m1_test(pd.DataFrame({"text": texts, "unit": labels.astype(str)}), n_boot=100, silhouette_n=0,
                      weighted=True, words=True, gap=False)
    return {"m1": plain(r), "m1_words": plain(rw), "minhash": plain(text.minhash_units(texts, tau=0.3)),
            "threshold": plain(text.threshold_components(texts[:30], 0.6)),
            "norm": text.normalize("Hello, @bob.bsky.social!  See https://x.com/a  #Tag_x"),
            "boot_median": plain(text.bootstrap_median(np.arange(20.0), 200))}


def case_chains():
    from memetrics import chains

    att = _chain_world("attraction", seed=1)
    rep = _chain_world("replication", seed=2)
    out = {}
    for name, w in (("att", att), ("rep", rep)):
        e = chains.edges_from_items(w)
        meds = chains.unit_medoids(w)
        out[name] = plain(chains.m2_test(e, meds, n_boot=200, null_sims=8, seed=11))
        out[name + "_depth"] = plain(chains.depth_profile(w))
        out[name + "_price"] = plain(chains.price_decomposition(w, meds))
    w = rep.drop(columns="parent").sample(frac=1.0, random_state=0)
    out["inferred"] = plain(chains.infer_parents(w).sort_index())
    return out


def case_fitness():
    from memetrics import fitness, textstats

    d = fitness.simulate_crossed(n_content=400, n_author=600, n=4000, s2c=0.4, s2a=0.25, s2e=0.35, seed=9)
    rng = np.random.default_rng(0)
    d["era"] = rng.integers(0, 4, len(d)).astype(str)
    d["order"] = rng.random(len(d))
    d["engagement"] += d.era.astype(int) * 0.5 + d.order
    r = fitness.content_icc(d, fixed=["era", "order"], n_boot=60, seed=4)
    r0 = fitness.content_icc(d, n_boot=30)
    small = d.iloc[:1200]
    reml = textstats.reml_icc(small, y="engagement", content="unit", author="author", fixed=["era"], n_boot=8, seed=2)
    return {"icc": plain(r), "icc_nofixed": plain(r0), "reml": plain(reml),
            "oneway": fitness.oneway_icc(d.engagement.to_numpy(), d.unit.to_numpy())}


def case_selection():
    from memetrics import selection

    def traj(s, seed):
        g = np.arange(0, 100, 10)
        n = np.full(len(g), 1500)
        x = selection.wf_sampled(0.25, g, n, 5000, s, reps=1, seed=seed)[0]
        return pd.DataFrame({"t": g.astype(float), "n": n, "count": (x * n).astype(int)})

    trajs = [traj(0.02 if i % 3 == 0 else 0.0, i) for i in range(18)]
    r = selection.fit_calibrated(trajs, Ne_grid=(1000, 5000), reps=25, n_sets=80, s_grid=(0.05,), seed=3)
    steps = selection.neutral_innovation(600, 0.03, 25, burn=40, conformity=0.2, seed=2)
    rt = selection.turnover_test(steps, mu=0.03, N=600, ys=(5, 10, 15, 20), runs=8, seed=1)
    mu = selection.calibrate_mu(0.05, [300] * 10, N=600, grid=np.geomspace(1e-3, 0.3, 6), seed=4)
    b = selection.equal_count_bins(np.linspace(0, 9, 50), np.arange(50) % 3 == 0, 5)
    return {"fit": plain(r), "turnover": plain(rt), "mu": mu, "bins": plain(b),
            "bh": plain(selection.bh(np.array([0.001, 0.02, 0.03, np.nan, 0.5]))),
            "nvs": selection.new_variant_share(steps)}


def case_phylo():
    from memetrics import phylo

    s = phylo.simulate_taxa("network", 10, 10, 0.5, seed=3)
    texts = [" ".join(f"w{t}" for t in x) for x in s]
    D = phylo.word_distance_matrix(texts)
    r = phylo.tree_likeness(D, length=10, reps=6, seed=5)
    return {"tree_likeness": plain(r), "dq": plain(phylo.delta_q(D, max_q=50, seed=1))}


def case_predict():
    from memetrics import predict

    rng = np.random.default_rng(0)
    n = 3000
    unit = rng.integers(0, 120, n)
    t = np.sort(rng.uniform(0, 100, n))
    uf = rng.normal(0, 1.0, 120)
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 0.8 * x + uf[unit])))).astype(int)
    d = pd.DataFrame({"unit": unit.astype(str), "ts": t, "x": x, "y": y})
    d = d.join(predict.prior_unit_features(d, "unit", "ts", "y", lag=1.0, k=5, prefix="m"))
    d["ca"] = np.log1p(d.m_n)
    tr, te = d[d.ts < 60], d[d.ts >= 60]
    blocks = {"B": ["x"], "CA": ["ca"], "M": ["m_fit", "m_age"]}
    res = predict.nested_compare(tr, te, "y", blocks, [(["B"], ["M"]), (["B", "CA"], ["M"])], "unit", n_boot=80,
                                 seed=2)
    hgb = predict.nested_compare(tr, te, "y", blocks, [(["B"], ["M"])], "unit", model="hgb", n_boot=20, seed=2)
    return {"logit": plain(res), "hgb": plain(hgb), "verdict": predict.nonredundancy_verdict(*res),
            "features": plain(d[["m_n", "m_fit", "m_sum"]].sum())}


def case_observation():
    from memetrics import observation as ob

    small = ob.WorldParams(n_agents=500, n_units=10, steps=25)
    out = {}
    for proc in ob.PROCESSES:
        log = ob.simulate(proc, p_private=0.3, seed=5, params=small)
        obs = ob.observe(log, "ancestor", seed=5)
        res = ob.evaluate(obs, null_sims=4, n_boot=40, m4_null=15, seed=5)
        out[proc] = {"n_log": len(log), "n_obs": len(obs), "eng": float(obs.engagement.sum()),
                     "res": plain({k: v for k, v in res.items()}), "composite": ob.composite(res),
                     "composite_cal": ob.composite(res, calibrated=True)}
    inf = ob.observe(ob.simulate("neutral", 0.5, seed=6, params=small), "inferred", seed=6)
    out["inferred_parents"] = int(inf.parent.notna().sum())
    return out


def case_onchain():
    from memetrics import onchain

    rng = np.random.default_rng(1)
    n = 400
    ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 200, n), unit="D")
    parent = rng.normal(5, 1, n)
    hot = 0.8 * (parent - 5) + rng.normal(0, 0.6, n)
    y = 3 + 0.3 * parent + 0.5 * hot + rng.normal(0, 1, n)
    d = pd.DataFrame({"item_id": [f"t{i}" for i in range(n)], "ts": ts, "engagement": y,
                      "parent_fitness": parent, "parent_hot": hot, "yb": (y > 5).astype(float)})
    r = onchain.heritability(d, n_boot=100, n_perm=100, seed=3)
    ra = onchain.heritability(d, controls=["parent_hot"], n_boot=100, n_perm=100, seed=3)
    cc = onchain.coef_ci(d, "yb", ["parent_fitness"], ["parent_hot"], onchain.time_blocks(d.ts), n_boot=50,
                         logistic=True)
    return {"plain": plain(r), "adjusted": plain(ra), "logit": plain(cc), "norm": onchain.norm(" $PEPE🐸 the Frog! ")}


CASES = {f.__name__.removeprefix("case_"): f for f in
         (case_units, case_text, case_chains, case_fitness, case_selection, case_phylo, case_predict,
          case_observation, case_onchain)}
