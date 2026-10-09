import gzip
import json

import pytest

from memetrics.cli import main
from memetrics.schema import read_items


def test_simulate_units_scorecard(tmp_path, capsys):
    sim = tmp_path / "sim.parquet"
    main(["simulate", "--process", "neutral", "--agents", "300", "--seed", "2", "--out", str(sim)])
    main(["units", str(sim), "--method", "minhash", "--out", str(tmp_path / "u.parquet")])
    assert read_items(tmp_path / "u.parquet").unit.notna().any()
    main(
        [
            "scorecard",
            str(tmp_path / "u.parquet"),
            "--out",
            str(tmp_path / "r"),
            "--fast",
            "--only",
            "M1",
            "M2",
            "--rule",
            "f_threshold=0.95",
        ]
    )
    card = json.loads((tmp_path / "r" / "scorecard.json").read_text())
    assert card["rules"]["f_threshold"] == 0.95 and [m["id"] for m in card["measures"]] == ["M1", "M2"]
    assert "M2" in capsys.readouterr().out


def test_ingest_requires_salt_and_hashes(tmp_path, monkeypatch):
    p = tmp_path / "posts.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for i in range(3):
            fh.write(
                json.dumps(
                    {
                        "id": f"x{i}",
                        "author": "someone",
                        "created_utc": 1_700_000_000 + i,
                        "subreddit": "memes",
                        "score": i,
                        "title": "hello there",
                    }
                )
                + "\n"
            )
    monkeypatch.delenv("MEMETRICS_SALT", raising=False)
    with pytest.raises(SystemExit, match="salt"):
        main(["ingest", "reddit", str(p), "--out", str(tmp_path / "i.parquet")])
    main(["ingest", "reddit", str(p), "--salt", "s3cret", "--out", str(tmp_path / "i.parquet")])
    d = read_items(tmp_path / "i.parquet")
    assert len(d) == 3 and "someone" not in set(d.author) and d.engagement.tolist() == [0.0, 1.0, 2.0]
