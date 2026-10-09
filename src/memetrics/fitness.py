"""Content-intrinsic fitness: identical-content intraclass correlation with crossed content and author effects (M3).

Model: ``y = X beta + a_content + a_author + e`` with independent random effects (variances s2_c, s2_a, s2_e).

Estimator. Fixed effects are removed by OLS (sparse dummies), then the variance components of the residuals r are
estimated by pairwise U-statistics for a crossed design (method of moments; Gao & Owen 2017, "Efficient moment
calculations for variance components in large unbalanced crossed random effects models", Electron. J. Statist.
11, doi:10.1214/17-EJS1236):

- W_c = mean of (r_i - r_j)^2 / 2 over pairs with the same content and different authors  = s2_a + s2_e
- W_a = mean over pairs with the same author and different content                       = s2_c + s2_e
- T   = mean over pairs sharing neither                                                   = s2_c + s2_a + s2_e

so s2_c = T - W_c, s2_a = T - W_a, s2_e = W_c + W_a - T. Pair sums are computed in O(n) from group sums
(sum_{i<j} (r_i - r_j)^2 = n sum r^2 - (sum r)^2). The OLS step treats fixed effects as known (slightly optimistic).

Confidence intervals: pigeonhole bootstrap for crossed data (Owen 2007, "The pigeonhole bootstrap", Ann. Appl. Stat.
1(2), doi:10.1214/07-AOAS122, which resamples rows and columns independently) in the reweighting form of Owen &
Eckles (2012, "Bootstrapping data arrays of arbitrary order", Ann. Appl. Stat. 6(3):895-927, doi:10.1214/12-AOAS547):
independent Poisson(1) weights for every content and every author, item weight = product.

Default decision rule (:func:`m3_verdict`): content ICC >= 0.5 "content dominates"; < 0.2 "weak content fitness";
otherwise "intermediate".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .result import Result
from .schema import require


def m3_verdict(icc: float, strong: float = 0.5, weak: float = 0.2) -> str:
    """Default M3 rule on the content ICC."""
    return "content dominates" if icc >= strong else "weak content fitness" if icc < weak else "intermediate"


def residualize(df: pd.DataFrame, y: str, fixed: list[str]) -> np.ndarray:
    """OLS residuals of ``df[y]`` on categorical fixed effects ``fixed`` (each column treated as a factor;
    numeric columns are entered linearly if their dtype is float)."""
    yv = df[y].to_numpy(float)
    blocks = [sp.csr_matrix(np.ones((len(df), 1)))]
    for f in fixed:
        col = df[f]
        if pd.api.types.is_float_dtype(col):
            v = col.to_numpy(float)
            blocks.append(sp.csr_matrix(((v - v.mean()) / (v.std() or 1))[:, None]))
        else:
            codes, _ = pd.factorize(col.astype(str))
            blocks.append(sp.csr_matrix((np.ones(len(df)), (np.arange(len(df)), codes)))[:, 1:])
    X = sp.hstack(blocks).tocsr()
    beta = lsqr(X, yv, atol=1e-10, btol=1e-10, iter_lim=10_000)[0]
    return yv - X @ beta


def _pair_sums(r, w, g):
    """Weighted sum over pairs within groups g of w_i w_j (r_i - r_j)^2 / 2, and the weighted pair count."""
    W = np.bincount(g, w)
    S1 = np.bincount(g, w * r)
    S2 = np.bincount(g, w * r * r)
    W2 = np.bincount(g, w * w)
    # sum_{i<j} w_i w_j (r_i-r_j)^2 = W*S2 - S1^2 ; ordered pairs i!=j weight sum = W^2 - W2
    return float(np.sum(W * S2 - S1**2) / 2.0), float(np.sum(W**2 - W2) / 2.0)


def crossed_components(
    r: np.ndarray, content: np.ndarray, author: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """Moment estimates (s2_c, s2_a, s2_e) for residuals ``r`` with crossed factors ``content`` and ``author``."""
    r = np.asarray(r, float)
    c = pd.factorize(pd.Series(content))[0]
    a = pd.factorize(pd.Series(author))[0]
    ca = pd.factorize(pd.Series(c).astype(str) + "|" + pd.Series(a).astype(str))[0]
    w = np.ones_like(r) if weights is None else np.asarray(weights, float)
    all_ss, all_n = _pair_sums(r, w, np.zeros(len(r), int))
    c_ss, c_n = _pair_sums(r, w, c)
    a_ss, a_n = _pair_sums(r, w, a)
    ca_ss, ca_n = _pair_sums(r, w, ca)

    def ratio(ss, n):
        return ss / n if n > 1e-9 else np.nan  # NaN when the design has no such pairs (not identified)

    Wc = ratio(c_ss - ca_ss, c_n - ca_n)  # same content, different author
    Wa = ratio(a_ss - ca_ss, a_n - ca_n)  # same author, different content
    T = ratio(all_ss - c_ss - a_ss + ca_ss, all_n - c_n - a_n + ca_n)
    s2c, s2a, s2e = T - Wc, T - Wa, Wc + Wa - T
    tot = s2c + s2a + s2e
    return {
        "s2_content": s2c,
        "s2_author": s2a,
        "s2_resid": s2e,
        "icc_content": s2c / tot,
        "icc_author": s2a / tot,
        "pairs_content": c_n - ca_n,
        "pairs_author": a_n - ca_n,
    }


def content_icc(
    df: pd.DataFrame,
    y: str = "engagement",
    content: str = "unit",
    author: str = "author",
    fixed: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    strong: float = 0.5,
    weak: float = 0.2,
    name: str = "M3 content ICC",
) -> Result:
    """Identical-content natural experiment (M3): content ICC vs author ICC.

    ``df`` is an items table (one row per posting). ``content`` identifies identical content (e.g. a pHash group);
    rows whose content appears once still inform the author and residual components. ``fixed`` lists fixed-effect
    columns (e.g. community x era, hour-of-week, repost order). Returns a Result for the content ICC with
    ``detail`` holding the author ICC, their difference, and bootstrap CIs for each.
    """
    require(df, [y, content, author] + list(fixed or []), "content_icc")
    rng = np.random.default_rng(seed)
    r = residualize(df, y, fixed or []) if fixed else df[y].to_numpy(float) - df[y].mean()
    cc = pd.factorize(df[content])[0]
    aa = pd.factorize(df[author])[0]
    est = crossed_components(r, cc, aa)
    boots = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, cc.max() + 1)[cc] * rng.poisson(1.0, aa.max() + 1)[aa]
        m = w > 0
        boots.append(crossed_components(r[m], cc[m], aa[m], w[m]))
    B = pd.DataFrame(boots)
    B["diff"] = B.icc_content - B.icc_author
    q = lambda col: (float(B[col].quantile(0.025)), float(B[col].quantile(0.975)))  # noqa: E731
    ic = est["icc_content"]
    verdict = m3_verdict(ic, strong, weak)
    lo, hi = q("icc_content")
    region = lambda v: 2 if v >= strong else 0 if v < weak else 1  # noqa: E731
    detail = dict(
        est,
        icc_author_ci=q("icc_author"),
        diff=ic - est["icc_author"],
        diff_ci=q("diff"),
        ci_in_one_region=region(lo) == region(hi),
        n_contents=int(cc.max() + 1),
        n_authors=int(aa.max() + 1),
    )
    return Result(name, float(ic), lo, hi, len(df), detail, verdict)


def oneway_icc(y: np.ndarray, group: np.ndarray) -> float:
    """Unadjusted one-way ANOVA ICC(1): ICC(1,1) of Shrout & Fleiss (1979, Psychol. Bull. 86(2):420-428,
    doi:10.1037/0033-2909.86.2.420), defined there for equal group sizes. Unequal groups use the average group size
    n0 = (N - sum n_i^2 / N) / (k - 1) in place of k, the standard ANOVA estimator of the one-way random-effects
    model (Donner 1986, Int. Stat. Rev. 54(1):67-82, doi:10.2307/1403259)."""
    d = pd.DataFrame({"y": y, "g": group})
    g = d.groupby("g").y
    k, n = g.ngroups, len(d)
    ni = g.size().to_numpy()
    msb = (ni * (g.mean().to_numpy() - d.y.mean()) ** 2).sum() / (k - 1)
    msw = ((d.y - g.transform("mean")) ** 2).sum() / (n - k)
    n0 = (n - (ni**2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def simulate_crossed(
    n_content: int = 2000,
    n_author: int = 3000,
    n: int = 20000,
    s2c: float = 0.5,
    s2a: float = 0.3,
    s2e: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulated postings with crossed content and author effects (for tests and power analysis). Content and
    author popularity are heavy-tailed (Zipf-like), as with reposts."""
    rng = np.random.default_rng(seed)
    pc = 1 / np.arange(1, n_content + 1) ** 0.8
    pa = 1 / np.arange(1, n_author + 1) ** 0.6
    c = rng.choice(n_content, n, p=pc / pc.sum())
    a = rng.choice(n_author, n, p=pa / pa.sum())
    y = (
        rng.normal(0, np.sqrt(s2c), n_content)[c]
        + rng.normal(0, np.sqrt(s2a), n_author)[a]
        + rng.normal(0, np.sqrt(s2e), n)
    )
    return pd.DataFrame({"unit": c.astype(str), "author": a.astype(str), "engagement": y})
