"""Pinned estimator outputs: refactors must not change the numbers (see tests/regression/cases.py)."""

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent / "regression"))
from cases import CASES, environment  # noqa: E402

GOLDEN = json.loads((pathlib.Path(__file__).parent / "regression" / "golden.json").read_text())
# Outputs depend on numpy/scipy/scikit-learn/pandas algorithms (sampling, solvers), so the pins are enforced only in
# the environment they were generated in (the locked environment on Python >= 3.12). Elsewhere they are skipped.
SAME_ENV = environment() == GOLDEN["_env"]


def _close(a, b, path="", rtol=1e-7, atol=1e-10):
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in b:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and not isinstance(a, str):
        assert a is not None and math.isclose(a, b, rel_tol=rtol, abs_tol=atol), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.skipif(not SAME_ENV, reason=f"goldens pinned under {GOLDEN['_env']}, running {environment()}")
@pytest.mark.parametrize("name", sorted(CASES))
def test_pinned(name):
    _close(json.loads(json.dumps(CASES[name]())), GOLDEN[name], name)
