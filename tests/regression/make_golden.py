"""Regenerate ``golden.json`` (only for a deliberate, documented estimator change)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cases import CASES  # noqa: E402

out = {name: f() for name, f in CASES.items()}
path = pathlib.Path(__file__).with_name("golden.json")
path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(f"wrote {path} ({len(out)} cases)")
