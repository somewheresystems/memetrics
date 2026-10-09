"""REML crossed random-effects variance components (M3), the estimator pre-registered for the text analyses.

`fitness.py` (image workstream) estimates the same crossed model by pairwise moment U-statistics (Gao & Owen
2017), which is fast and is the package default. This module adds the REML fit that notes/PREREG-text.md names
as primary for the text data. It uses the penalized least-squares formulation of lme4 (Bates, Maechler, Bolker & Walker 2015,
J. Stat. Softw. 67(1), doi:10.18637/jss.v067.i01), with a sparse LU for the log-determinant. Reporting both
estimators gives an estimator-robustness check.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from scipy.sparse.linalg import splu

from .fitness import oneway_icc
from .result import Result



def _indicator(codes: np.ndarray, n_levels: int) -> sparse.csc_matrix:
    n = len(codes)
    return sparse.csc_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, n_levels))


def crossed_reml(y: np.ndarray, X: np.ndarray, groups: list[np.ndarray], x0: np.ndarray | None = None) -> dict:
    """REML fit of y = X b + sum_k Z_k u_k + e, u_k ~ N(0, s2_k I), e ~ N(0, s2 I), for crossed factors.

    Penalized least squares with relative covariance factors lambda_k = sqrt(s2_k / s2) (Bates et al. 2015);
    the log-determinant comes from a sparse LU of the joint system. Returns dict with 'sigma2' (residual),
    'var' (list of s2_k), 'beta', 'converged'.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, p = X.shape
    Zs, qs = [], []
    for g in groups:
        codes, _ = pd.factorize(pd.Series(g), sort=False)
        q = int(codes.max()) + 1
        Zs.append(_indicator(codes, q))
        qs.append(q)
    Z = sparse.hstack(Zs).tocsc()
    ZtZ = (Z.T @ Z).tocsc()
    ZtX = Z.T @ X
    XtX = X.T @ X
    Zty = Z.T @ y
    Xty = X.T @ y
    q = Z.shape[1]
    Xs = sparse.csc_matrix(X)

    def solve(log_lam):
        lam = np.repeat(np.exp(log_lam), qs)
        L = sparse.diags(lam)
        A = (L @ ZtZ @ L + sparse.identity(q)).tocsc()
        B = sparse.csc_matrix(L @ ZtX)
        M = sparse.bmat([[A, B], [B.T, sparse.csc_matrix(XtX)]]).tocsc()
        lu = splu(M, permc_spec="MMD_AT_PLUS_A")
        sol = lu.solve(np.concatenate([lam * Zty, Xty]))
        u, b = sol[:q], sol[q:]
        r = y - Xs @ b - Z @ (lam * u)
        r2 = float(r @ r + u @ u)
        logdet = float(np.sum(np.log(np.abs(lu.U.diagonal()))) + np.sum(np.log(np.abs(lu.L.diagonal()))))
        return logdet, r2, b

    def obj(log_lam):
        logdet, r2, _ = solve(log_lam)
        return logdet + (n - p) * (1 + math.log(2 * math.pi * r2 / (n - p)))

    best = None
    starts = [np.asarray(x0, float)] if x0 is not None else [np.full(len(groups), s_) for s_ in (-1.0, 0.0)]
    for start in starts:
        res = optimize.minimize(obj, start, method="L-BFGS-B", bounds=[(-9, 5)] * len(groups))
        if best is None or res.fun < best.fun:
            best = res
    logdet, r2, b = solve(best.x)
    s2 = r2 / (n - p)
    return {"sigma2": s2, "var": list(np.exp(2 * best.x) * s2), "beta": b, "converged": bool(best.success),
            "n_levels": qs, "log_lambda": best.x}


def _design(df: pd.DataFrame, fixed: list[str]) -> np.ndarray:
    if not fixed:
        return np.ones((len(df), 1))
    X = pd.get_dummies(df[fixed], drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    return np.column_stack([np.ones(len(df)), X.to_numpy(float)])


def reml_icc(df: pd.DataFrame, y: str = "y", content: str = "content", author: str = "author",
             fixed: list[str] | None = None, n_boot: int = 200, seed: int = 20261008,
             strong: float = 0.5, weak: float = 0.2, name: str = "M3 content ICC (REML)",
             bootstrap: str = "parametric") -> Result:
    """Content-intrinsic fitness: content ICC = s2_content / (s2_content + s2_author + s2_e) from a crossed
    random-effects REML fit with fixed effects `fixed` (categorical columns are dummy-coded, numeric kept).

    CI: parametric bootstrap from the fitted model (default). `bootstrap="groups"` resamples content groups, which
    duplicates authors' rows and biases the author component upward; it is kept only for comparison. verdict (STUDY.md defaults):
    'content dominates' if ICC >= strong, 'weak content fitness' if < weak, else 'intermediate'.
    detail: author ICC, variance components, the unadjusted one-way content ICC, and P(content ICC > author ICC).
    """
    fixed = fixed or []
    d = df.reset_index(drop=True)

    def fit(dd, x0=None):
        X = _design(dd, fixed)
        r = crossed_reml(dd[y].to_numpy(float), X, [dd[content].to_numpy(), dd[author].to_numpy()], x0=x0)
        sc, sa = r["var"]
        tot = sc + sa + r["sigma2"]
        return sc / tot, sa / tot, r

    ic, ia, r = fit(d)
    rng = np.random.default_rng(seed)
    bc, ba = [], []
    if bootstrap == "parametric":
        # simulate from the fitted model and refit (valid for crossed designs; resampling content groups would
        # duplicate authors' rows and inflate the author component)
        X = _design(d, fixed)
        mu = X @ r["beta"]
        cc = pd.factorize(d[content])[0]
        aa = pd.factorize(d[author])[0]
        sc_, sa_ = np.sqrt(r["var"][0]), np.sqrt(r["var"][1])
        se_ = np.sqrt(r["sigma2"])
        yy = d[y].to_numpy(float)
        for _ in range(n_boot):
            ys = mu + rng.normal(0, sc_, cc.max() + 1)[cc] + rng.normal(0, sa_, aa.max() + 1)[aa] \
                + rng.normal(0, se_, len(d))
            try:
                rr = crossed_reml(ys, X, [cc, aa], x0=r["log_lambda"])
            except Exception:  # noqa: BLE001
                continue
            tot = sum(rr["var"]) + rr["sigma2"]
            bc.append(rr["var"][0] / tot)
            ba.append(rr["var"][1] / tot)
        del yy
    else:
        groups = d.groupby(content).indices
        keys = list(groups)
        for _ in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            parts = []
            for j, k in enumerate(pick):
                part = d.iloc[groups[keys[k]]].copy()
                part[content] = f"b{j}"
                parts.append(part)
            try:
                c_, a_, _ = fit(pd.concat(parts, ignore_index=True), x0=r["log_lambda"])  # warm start
                bc.append(c_)
                ba.append(a_)
            except Exception:  # noqa: BLE001 - singular resamples are skipped and counted
                continue
    bc, ba = np.array(bc), np.array(ba)
    lo, hi = (np.quantile(bc, [0.025, 0.975]) if len(bc) else (np.nan, np.nan))
    verdict = "content dominates" if ic >= strong else ("weak content fitness" if ic < weak else "intermediate")
    detail = {"author_icc": ia, "author_icc_ci": list(np.quantile(ba, [0.025, 0.975])) if len(ba) else None,
              "var_content": r["var"][0], "var_author": r["var"][1], "var_resid": r["sigma2"],
              "oneway_content_icc": oneway_icc(d[y].to_numpy(), d[content].to_numpy()),
              "p_content_gt_author": float((bc > ba).mean()) if len(bc) else np.nan,
              "n_content": int(d[content].nunique()), "n_author": int(d[author].nunique()),
              "n_boot_ok": int(len(bc)), "converged": r["converged"], "thresholds": [weak, strong]}
    return Result(name, float(ic), float(lo), float(hi), len(d), detail, verdict)


