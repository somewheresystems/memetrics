"""Common return type for every test in memetrics (see toolkit/DESIGN.md)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Result:
    """name, point estimate, 95% CI, sample size, free-form detail, and the default decision-rule verdict."""
    name: str
    estimate: float
    ci_low: float = math.nan
    ci_high: float = math.nan
    n: int = 0
    detail: dict = field(default_factory=dict)
    verdict: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
