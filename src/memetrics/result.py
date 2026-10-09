"""The common return type of every test in memetrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class Result:
    """One test outcome.

    ``estimate`` with a 95% interval (``ci_low``, ``ci_high``; NaN when the test has none), the sample size ``n``,
    free-form ``detail`` and ``verdict``: the default decision rule applied with the thresholds passed to the test
    (``None`` when the measure has no rule).
    """

    name: str
    estimate: float
    ci_low: float = math.nan
    ci_high: float = math.nan
    n: int = 0
    detail: dict = field(default_factory=dict)
    verdict: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> dict:
        """``to_dict`` with numpy scalars, arrays, tuples and non-finite floats made JSON-safe."""
        return jsonable(self.to_dict())

    def __str__(self) -> str:
        ci = f" [{self.ci_low:.3g}, {self.ci_high:.3g}]" if np.isfinite([self.ci_low, self.ci_high]).all() else ""
        v = f" -> {self.verdict}" if self.verdict else ""
        return f"{self.name}: {self.estimate:.3g}{ci} (n={self.n}){v}"


def jsonable(x):
    """Recursively convert results, numpy and pandas values to JSON-compatible python (NaN and inf -> None)."""
    if isinstance(x, Result):
        return x.to_json()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if hasattr(x, "to_dict") and hasattr(x, "columns"):  # DataFrame
        return jsonable(x.to_dict(orient="records"))
    if hasattr(x, "tolist") and hasattr(x, "index"):  # Series
        return jsonable(x.tolist())
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        return float(x) if math.isfinite(x) else None
    return x
