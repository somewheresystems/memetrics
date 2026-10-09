"""Selection versus drift (M4): Frequency Increment Test, Wright-Fisher null calibration, neutral turnover.

- FIT: Feder, Kryazhimskiy & Plotkin (2014), "Identifying signatures of selection in genetic time series", Genetics
  196(2):509-522, doi:10.1534/genetics.113.158220; applied to language by Newberry, Ahern, Clark & Plotkin (2017),
  "Detecting evolutionary forces in language change", Nature 551:223-226, doi:10.1038/nature24455.
  Rescaled increments ``Y_i = (x_i - x_{i-1}) / sqrt(2 x_{i-1} (1 - x_{i-1}) (t_i - t_{i-1}))`` are ~ N(0, 1/N_e) iid
  under neutrality (diffusion approximation), so a two-sided one-sample t-test of mean(Y) = 0 tests for selection.
- Null calibration: neutral Wright-Fisher trajectories with the observed initial frequency, bin times and per-bin
  sample sizes (binomial sampling), for several N_e; the rejection share of the observed set is compared with the null
  distribution of rejection shares (Benjamini-Hochberg 1995 FDR).
- Turnover: Bentley, Lipo, Herzog & Hahn (2007), "Regular rates of popular culture change reflect random copying",
  Evol. Hum. Behav. 28(3):151-158, doi:10.1016/j.evolhumbehav.2006.10.002; Acerbi & Bentley (2014), "Biases in cultural
  transmission shape the turnover of popular traits", Evol. Hum. Behav. 35(3):228-236,
  doi:10.1016/j.evolhumbehav.2014.02.003. z(y) = mean number of new entries per step in the top-y list; fit
  z = a y^x. The neutral reference x comes from simulation of the neutral model with innovation, matched on N, mu and
  sample size, instead of a fixed published constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .result import Result

# ---------------------------------------------------------------- FIT


def fit_test(x: np.ndarray, t: np.ndarray) -> tuple[float, float, int]:
    """Frequency Increment Test on one trajectory. ``x`` frequencies (strictly inside (0, 1) are used), ``t`` times.
    Returns (t statistic, two-sided p, number of increments). NaN p if fewer than 2 increments."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    keep = (x > 0) & (x < 1)
    x, t = x[keep], t[keep]
    if len(x) < 3:
        return np.nan, np.nan, max(len(x) - 1, 0)
    dt = np.diff(t)
    y = np.diff(x) / np.sqrt(2 * x[:-1] * (1 - x[:-1]) * dt)
    if np.allclose(y.std(ddof=1), 0):
        return np.nan, np.nan, len(y)
    ts, p = stats.ttest_1samp(y, 0.0)
    return float(ts), float(p), len(y)


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini & Hochberg (1995, JRSS B 57:289-300, doi:10.1111/j.2517-6161.1995.tb02031.x) rejections at FDR q (NaNs never rejected)."""
    p = np.asarray(p, float)
    ok = np.flatnonzero(~np.isnan(p))
    rej = np.zeros(len(p), bool)
    if len(ok) == 0:
        return rej
    o = ok[np.argsort(p[ok])]
    m = len(ok)
    passed = p[o] <= q * np.arange(1, m + 1) / m
    if passed.any():
        rej[o[: np.max(np.flatnonzero(passed)) + 1]] = True
    return rej


def equal_count_bins(ts: np.ndarray, is_unit: np.ndarray, k: int) -> pd.DataFrame:
    """Split a time-ordered pool into k bins with equal total counts; return per-bin midpoint time, n and unit count.
    ``ts`` times of all pool items in the unit's active span, ``is_unit`` boolean membership."""
    o = np.argsort(ts, kind="stable")
    ts, is_unit = np.asarray(ts)[o], np.asarray(is_unit)[o]
    edges = np.linspace(0, len(ts), k + 1).astype(int)
    rows = [(float(np.mean(ts[a:b])), b - a, int(is_unit[a:b].sum())) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return pd.DataFrame(rows, columns=["t", "n", "count"]).assign(x=lambda d: d["count"] / d["n"])


# ---------------------------------------------------------------- Wright-Fisher


def wf_sampled(x0: float, gens: np.ndarray, n_sample: np.ndarray, Ne: float, s: float = 0.0,
               reps: int = 200, seed: int = 0) -> np.ndarray:
    """Wright-Fisher trajectories (haploid, selection coefficient s per generation) started at x0, sampled
    binomially with ``n_sample[i]`` at generation ``gens[i]`` (sorted, gens[0] = 0). Returns (reps, len(gens))
    sample frequencies."""
    rng = np.random.default_rng(seed)
    gens = np.asarray(np.round(gens), int)
    N = int(max(Ne, 2))
    p = np.full(reps, float(x0))
    out = np.empty((reps, len(gens)))
    g = 0
    for i, G in enumerate(gens):
        while g < G:
            ps = p * (1 + s) / (p * (1 + s) + (1 - p))
            p = rng.binomial(N, ps) / N
            g += 1
        out[:, i] = rng.binomial(int(n_sample[i]), p) / n_sample[i]
    return out


def fit_calibrated(trajs: list[pd.DataFrame], Ne_grid=(1_000, 10_000, 100_000), gen_unit: float = 1.0,
                   reps: int = 200, n_sets: int = 1000, q: float = 0.05, s_grid=(0.01, 0.05, 0.1),
                   seed: int = 0, name: str = "M4 FIT") -> Result:
    """FIT on observed trajectories with a matched Wright-Fisher null.

    ``trajs``: list of per-bin tables with columns t (time), n (pool sample size), count (unit count). Time is
    converted to generations by dividing by ``gen_unit``. For each trajectory and N_e, ``reps`` neutral replicates
    are simulated with the observed x0, bin times and sample sizes; the null distribution of the BH rejection share
    is built from ``n_sets`` random draws of one replicate per trajectory.

    verdict: "selection beyond drift" if the observed BH share exceeds the null 97.5th percentile for every N_e;
    "drift-compatible" if inside the central 95% for some N_e; else "ambiguous".
    """
    rng = np.random.default_rng(seed)
    obs_p = []
    for d in trajs:
        obs_p.append(fit_test(d["count"] / d["n"], d.t / gen_unit)[1])
    obs_p = np.array(obs_p)
    valid = ~np.isnan(obs_p)
    obs_share = float(bh(obs_p[valid], q).mean()) if valid.any() else np.nan
    raw_share = float((obs_p[valid] < 0.05).mean()) if valid.any() else np.nan
    null = {}
    power = {}
    for Ne in Ne_grid:
        P = np.full((len(trajs), reps), np.nan)
        for i, d in enumerate(trajs):
            if not valid[i]:
                continue
            g = np.round((d.t.to_numpy() - d.t.iloc[0]) / gen_unit)
            x0 = float(np.clip(d["count"].iloc[0] / d.n.iloc[0], 1 / (2 * d.n.iloc[0]), 1 - 1 / (2 * d.n.iloc[0])))
            sim = wf_sampled(x0, g, d.n.to_numpy(), Ne, 0.0, reps, seed + i)
            P[i] = [fit_test(sim[r], g)[1] for r in range(reps)]
        Pv = P[valid]
        shares = np.array([bh(Pv[np.arange(len(Pv)), rng.integers(0, reps, len(Pv))], q).mean() for _ in range(n_sets)])
        null[Ne] = {"fpr_raw": float(np.nanmean(Pv < 0.05)), "bh_share_q025": float(np.quantile(shares, 0.025)),
                    "bh_share_q975": float(np.quantile(shares, 0.975)), "bh_share_mean": float(shares.mean())}
        for s in s_grid:
            pw = []
            for i, d in enumerate(trajs):
                if not valid[i]:
                    continue
                g = np.round((d.t.to_numpy() - d.t.iloc[0]) / gen_unit)
                x0 = float(np.clip(d["count"].iloc[0] / d.n.iloc[0], 0.01, 0.99))
                sim = wf_sampled(x0, g, d.n.to_numpy(), Ne, s, 20, seed + 7919 + i)
                pw += [fit_test(sim[r], g)[1] for r in range(20)]
            power[(Ne, s)] = float(np.nanmean(np.array(pw) < 0.05))
    above = all(obs_share > null[N]["bh_share_q975"] for N in Ne_grid)
    inside = any(null[N]["bh_share_q025"] <= obs_share <= null[N]["bh_share_q975"] for N in Ne_grid)
    verdict = "selection beyond drift" if above else "drift-compatible" if inside else "ambiguous"
    return Result(name, obs_share, np.nan, np.nan, int(valid.sum()),
                  {"raw_share_p05": raw_share, "null": null, "power_p05": {f"Ne={k[0]},s={k[1]}": v for k, v in power.items()},
                   "p_values": obs_p.tolist()}, verdict)


# ---------------------------------------------------------------- neutral model with innovation, turnover


def neutral_innovation(N: int, mu: float, steps: int, burn: int = 200, conformity: float = 0.0,
                       fitness_sd: float = 0.0, seed: int = 0) -> list[np.ndarray]:
    """Random-copying model with innovation (Bentley et al. 2004/2007): each step, N new items copy a variant of a
    random item from the previous step (probability 1 - mu) or invent a new variant (mu). ``conformity`` > 0 biases
    copying toward common variants (weights ~ count^(1+conformity)); ``fitness_sd`` > 0 gives each variant a
    heritable log-normal fitness (content bias). Returns per-step arrays of variant ids (after burn-in)."""
    rng = np.random.default_rng(seed)
    pop = np.arange(N)
    nxt = N
    fit: dict[int, float] = {}
    out = []
    for step in range(burn + steps):
        ids, cnt = np.unique(pop, return_counts=True)
        w = cnt.astype(float) ** (1 + conformity)
        if fitness_sd > 0:
            for v in ids:
                if v not in fit:
                    fit[v] = float(np.exp(rng.normal(0, fitness_sd)))
            w = w * np.array([fit[v] for v in ids])
        new = rng.choice(ids, N, p=w / w.sum())
        inv = rng.random(N) < mu
        k = int(inv.sum())
        new[inv] = np.arange(nxt, nxt + k)
        nxt += k
        pop = new
        if step >= burn:
            out.append(pop.copy())
    return out


def turnover_profile(steps: list[np.ndarray], ys=tuple(range(5, 55, 5)), seed: int = 0) -> pd.DataFrame:
    """z(y): mean number of variants entering the top-y list per step. Ties in count are broken by a seeded random
    order. Breaking ties by id would be biased: simulated ids grow with time (ties would favour old variants),
    observed ids do not."""
    rng = np.random.default_rng(seed)
    tops = []
    for s in steps:
        ids, cnt = np.unique(s, return_counts=True)
        o = np.lexsort((rng.permutation(len(ids)), -cnt))
        tops.append(ids[o])
    rows = []
    for y in ys:
        z = [len(set(b[:y]) - set(a[:y])) for a, b in zip(tops[:-1], tops[1:]) if len(a) >= y and len(b) >= y]
        rows.append((y, float(np.mean(z)) if z else np.nan))
    return pd.DataFrame(rows, columns=["y", "z"])


def turnover_exponent(prof: pd.DataFrame) -> float:
    """Slope x of log z on log y (OLS), over y with z > 0."""
    d = prof[(prof.z > 0) & prof.z.notna()]
    if len(d) < 3:
        return np.nan
    return float(np.polyfit(np.log(d.y), np.log(d.z), 1)[0])


def turnover_test(steps: list[np.ndarray], mu: float, N: int = 10_000, sample_sizes: list[int] | None = None,
                  ys=tuple(range(5, 55, 5)), runs: int = 200, gens_per_step: int = 1, seed: int = 0,
                  name: str = "M4 turnover") -> Result:
    """Compare the observed turnover exponent with neutral simulations matched on mu and (optionally) per-step sample
    sizes (the neutral population of size N is subsampled to the observed sizes before ranking).

    verdict: "neutral-compatible" inside the simulated 95% interval; "conformity/popularity bias" below it;
    "novelty/anti-conformity bias" above it. ``ci_low``/``ci_high`` hold the simulated neutral 95% interval of x."""
    rng = np.random.default_rng(seed)
    x_obs = turnover_exponent(turnover_profile(steps, ys, seed=seed))
    T = len(steps)
    sims = []
    for r in range(runs):
        sim = neutral_innovation(N, mu, T * gens_per_step, seed=seed + r)
        sizes = sample_sizes if sample_sizes is not None else [N * gens_per_step] * T
        sim = subsample_steps(sim, sizes, gens_per_step, rng)
        sims.append(turnover_exponent(turnover_profile(sim, ys, seed=seed + r)))
    sims = np.array(sims)
    lo, hi = np.nanquantile(sims, [0.025, 0.975])
    v = "neutral-compatible" if lo <= x_obs <= hi else ("conformity/popularity bias" if x_obs < lo else "novelty/anti-conformity bias")
    return Result(name, float(x_obs), float(lo), float(hi), T, {"null_mean": float(np.nanmean(sims)), "mu": mu, "N": N}, v)


def new_variant_share(steps: list[np.ndarray]) -> float:
    """Mean share of items per step whose variant had not been seen in any earlier step (first step excluded)."""
    seen: set = set()
    shares = []
    for i, s in enumerate(steps):
        if i > 0 and len(s):
            shares.append(float(np.mean([v not in seen for v in s])))
        seen.update(s.tolist())
    return float(np.mean(shares)) if shares else np.nan


def subsample_steps(steps: list[np.ndarray], sizes, gens_per_step: int = 1, rng=None) -> list[np.ndarray]:
    """Group consecutive generations into observation steps and draw ``sizes[i]`` items without replacement."""
    rng = rng or np.random.default_rng(0)
    out = []
    for i, n in enumerate(sizes):
        pool = np.concatenate(steps[i * gens_per_step:(i + 1) * gens_per_step])
        out.append(rng.choice(pool, min(int(n), len(pool)), replace=False))
    return out


def calibrate_mu(target_share: float, sizes, N: int = 10_000, gens_per_step: int = 1, grid=None, seed: int = 0) -> float:
    """Innovation rate mu whose subsampled new-variant share best matches ``target_share`` (grid search, log scale)."""
    grid = grid if grid is not None else np.geomspace(1e-4, 0.5, 18)
    rng = np.random.default_rng(seed)
    best, err = grid[0], np.inf
    for mu in grid:
        sim = neutral_innovation(N, mu, len(sizes) * gens_per_step, burn=100, seed=seed)
        e = abs(new_variant_share(subsample_steps(sim, sizes, gens_per_step, rng)) - target_share)
        if e < err:
            best, err = mu, e
    return float(best)
