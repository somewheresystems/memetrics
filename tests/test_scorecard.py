import pandas as pd
import pytest

from memetrics import observation as ob
from memetrics import report, schema, scorecard


@pytest.fixture(scope="module")
def items():
    log = ob.simulate("attraction", 0.2, seed=3, params=ob.WorldParams(n_agents=400))
    obs = ob.observe(log, "explicit", seed=3)
    return schema.conform(obs[["item_id", "ts", "author", "community", "kind", "text", "unit", "parent", "engagement"]])


def test_scorecard_runs_every_measure_and_always_has_blind_spots(items, tmp_path):
    card = scorecard.run(items, effort=scorecard.Effort.fast())
    ms = {m["id"]: m for m in card["measures"]}
    assert list(ms) == ["M1", "M2", "M3", "M4", "M5", "N1"]
    assert ms["M1"]["status"] == "ok" and ms["M2"]["status"] == "ok"
    labels = [r["label"] for r in ms["M2"]["rows"]]
    assert labels == ["all edges", "derivatives only (f < 1)"]
    assert all(r["calibrated"] in ("replication", "attraction", "mixed") for r in ms["M2"]["rows"])
    assert len(card["blind_spots"]) == 5
    paths = report.write(card, tmp_path)
    md, html = paths["md"].read_text(), paths["html"].read_text()
    for text in (md, html):
        assert "What this data cannot see" in text and "Algorithmic ranking" in text and "Offline" in text


def test_scorecard_skips_cleanly_without_engagement_or_units(items):
    card = scorecard.run(items.drop(columns="engagement"), effort=scorecard.Effort.fast(), which=("M3", "N1"))
    assert all(m["status"] == "skipped" and "engagement" in m["reason"] for m in card["measures"])
    with pytest.raises(ValueError, match="units"):
        scorecard.run(items.assign(unit=None))


def test_rules_change_literal_verdict(items):
    strict = scorecard.run(items, effort=scorecard.Effort.fast(), which=("M1",), rules=scorecard.Rules(d_threshold=1e9))
    assert strict["measures"][0]["rows"][0]["literal"] == "continuum"


def test_distinguishability_shape():
    r = scorecard.distinguishability(1200, 0.3, reps=1, seed=1)
    assert set(r["inferred"]["accuracy"]) == set(ob.PROCESSES)
    assert 0 <= r["inferred_calibrated"]["balanced_accuracy"] <= 1
    assert isinstance(pd.DataFrame(r["inferred"]["confusion"]), pd.DataFrame)
