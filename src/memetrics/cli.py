"""``memetrics`` command line: ingest, units, scorecard, simulate, distinguishability."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .schema import read_items, write_items


def _salt(a) -> str:
    salt = a.salt or os.environ.get("MEMETRICS_SALT")
    if not salt:
        sys.exit("error: author hashing needs a private salt: pass --salt or set MEMETRICS_SALT")
    return salt


def _done(df: pd.DataFrame, out: str) -> None:
    p = write_items(df, out)
    print(f"wrote {len(df):,} items to {p}")


# ------------------------------------------------------------------ ingest


def cmd_ingest(a) -> None:
    if a.platform == "reddit":
        from .io import reddit

        if a.fetch:
            if not (a.subreddit and a.after and a.before):
                sys.exit("error: --fetch needs --subreddit, --after and --before")
            posts = reddit.fetch_posts(a.subreddit, a.after, a.before, a.cache)
        else:
            if not a.inputs:
                sys.exit("error: give dump files (JSONL, .gz or .zst) or use --fetch")
            posts = (x for f in a.inputs for x in reddit.read_jsonl(f))
        _done(reddit.to_items(posts, salt=_salt(a)), a.out)
    elif a.platform == "bluesky":
        from .io import bluesky

        if not a.root:
            sys.exit("error: --root is required (the collector's output directory)")
        if a.collect:
            import asyncio

            print(f"collecting Jetstream events into {a.root}; Ctrl-C to stop")
            asyncio.run(bluesky.collect(Path(a.root), (bluesky.DEFAULT_HOST, *bluesky.FALLBACK_HOSTS)))
            return
        ev = bluesky.iter_events(Path(a.root), a.start, a.end)
        _done(bluesky.to_items(ev, horizon_h=a.horizon_h, parent_from=a.parent_from, salt=_salt(a)), a.out)
    elif a.platform == "memetracker":
        from .io import memetracker

        if len(a.inputs) != 1:
            sys.exit("error: give one MemeTracker cluster file (clust-*.txt.gz)")
        _, phrases, mentions = memetracker.read_clusters(a.inputs[0], a.min_cluster_mentions)
        _done(memetracker.mentions_to_items(phrases, mentions, salt=_salt(a)), a.out)


# ------------------------------------------------------------------ units


def cmd_units(a) -> None:
    from . import units

    items = read_items(a.items)
    emb = np.load(a.emb) if a.emb else None
    out = units.assign(items, a.method, emb=emb, tau=a.tau, min_size=a.min_size, max_hamming=a.max_hamming,
                       seed=a.seed)
    sizes = out.unit.value_counts()
    print(f"{len(sizes):,} units with >= {a.min_size} items cover {out.unit.notna().mean():.0%} of items")
    _done(out, a.out or a.items)


# ------------------------------------------------------------------ scorecard


def _rules(pairs: list[str]):
    from .scorecard import Rules

    kw = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if k not in Rules.__dataclass_fields__:
            sys.exit(f"error: unknown rule {k!r}; choose from {', '.join(Rules.__dataclass_fields__)}")
        kw[k] = float(v)
    return Rules(**kw)


def cmd_scorecard(a) -> None:
    from . import report, scorecard

    items = read_items(a.items)
    emb = np.load(a.emb) if a.emb else None
    if a.assign_units or "unit" not in items or items.unit.isna().all():
        from . import units

        print("assigning units (method auto)")
        items = units.assign(items, emb=emb)
    card = scorecard.run(items, emb, rules=_rules(a.rule), effort=scorecard.Effort.fast() if a.fast
                         else scorecard.Effort(), seed=a.seed, which=tuple(a.only or ("M1", "M2", "M3", "M4", "M5",
                                                                                        "N1")),
                         lag_hours=a.lag_hours, p_private=a.p_private, distinguish_reps=a.reps, workers=a.workers,
                         progress=lambda s: print(s, flush=True))
    paths = report.write(card, a.out)
    for m in card["measures"]:
        verdicts = "; ".join(f"{r['label']}: {r['literal'] or '-'} / {r['calibrated'] or '-'}" for r in m["rows"])
        print(f"{m['id']:3} {verdicts if m['status'] == 'ok' else 'not run: ' + m['reason']}")
    print("wrote " + ", ".join(str(p) for p in paths.values()))


# ------------------------------------------------------------------ simulate / distinguishability


def cmd_simulate(a) -> None:
    from . import observation as ob
    from .io import hash_author

    params = ob.params_for_size(a.n_items) if a.n_items else ob.WorldParams(n_agents=a.agents)
    log = ob.simulate(a.process, a.p_private, recommender=not a.no_recommender, seed=a.seed, params=params)
    obs = ob.observe(log, a.edge_mode, deletion=a.deletion, seed=a.seed)
    cols = ["item_id", "ts", "author", "community", "kind", "text", "unit", "parent", "engagement",
            "author_followers"]
    obs = obs[cols].assign(author=[hash_author(x, "memetrics-simulation") for x in obs.author])
    if a.truth:
        write_items(log.drop(columns=["like_times"]), a.truth)
        print(f"wrote ground truth ({len(log):,} posts incl. private) to {a.truth}")
    _done(obs, a.out)


def cmd_distinguishability(a) -> None:
    from .scorecard import distinguishability

    r = distinguishability(a.n_items, a.p_private, a.reps, recommender=not a.no_recommender, edge_mode=a.edge_mode,
                           seed=a.seed, workers=a.workers, fast=not a.full)
    for col, title in (("inferred", "literal rules"), ("inferred_calibrated", "calibrated rules")):
        cm = pd.DataFrame(r[col]["confusion"]).T.fillna(0.0)
        print(f"\n{title}: balanced accuracy {r[col]['balanced_accuracy']:.0%} "
              f"(rows: true process, columns: inferred)")
        print((cm * 100).round(0).astype(int).to_string())
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
        print(f"\nwrote {a.json}")


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="memetrics", description="Measure memetic claims on social-media data.")
    ap.add_argument("--version", action="version", version=f"memetrics {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("ingest", help="turn platform data into an items table")
    g.add_argument("platform", choices=["reddit", "bluesky", "memetracker"])
    g.add_argument("inputs", nargs="*", help="input files (reddit: JSONL dumps; memetracker: cluster file)")
    g.add_argument("--out", default="items.parquet")
    g.add_argument("--salt", help="private salt for author hashes (default $MEMETRICS_SALT)")
    g.add_argument("--fetch", action="store_true", help="reddit: page the Arctic Shift API (cached, rate-limited)")
    g.add_argument("--subreddit")
    g.add_argument("--after", help="ISO date, inclusive")
    g.add_argument("--before", help="ISO date, exclusive")
    g.add_argument("--cache", default="arctic-shift-cache", help="reddit --fetch: page cache directory")
    g.add_argument("--root", help="bluesky: collector directory")
    g.add_argument("--collect", action="store_true", help="bluesky: run the Jetstream collector into --root")
    g.add_argument("--start", help="bluesky: first day YYYY-MM-DD")
    g.add_argument("--end", help="bluesky: last day YYYY-MM-DD")
    g.add_argument("--horizon-h", type=float, default=48.0, help="bluesky: engagement window in hours")
    g.add_argument("--parent-from", choices=["quote", "reply"], default="quote")
    g.add_argument("--min-cluster-mentions", type=int, default=0, help="memetracker: drop smaller clusters")
    g.set_defaults(func=cmd_ingest)

    g = sub.add_parser("units", help="assign meme units to an items table")
    g.add_argument("items")
    g.add_argument("--out", help="output table (default: overwrite the input)")
    g.add_argument("--method", choices=["auto", "minhash", "phash", "hdbscan"], default="auto")
    g.add_argument("--emb", help=".npy embeddings, one row per item (for hdbscan)")
    g.add_argument("--tau", type=float, default=0.5, help="minhash Jaccard threshold")
    g.add_argument("--min-size", type=int, default=5)
    g.add_argument("--max-hamming", type=int, default=4, help="phash Hamming radius")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_units)

    g = sub.add_parser("scorecard", help="run M1-M5 and N1 and write a report")
    g.add_argument("items")
    g.add_argument("--out", default="report")
    g.add_argument("--emb", help=".npy embeddings, one row per item")
    g.add_argument("--assign-units", action="store_true", help="(re)assign units before scoring")
    g.add_argument("--only", nargs="+", choices=["M1", "M2", "M3", "M4", "M5", "N1"])
    g.add_argument("--fast", action="store_true", help="fewer bootstrap and simulation replicates")
    g.add_argument("--rule", action="append", metavar="KEY=VALUE", help="override a decision threshold")
    g.add_argument("--lag-hours", type=float, default=48.0, help="N1: history lag while engagement accrues")
    g.add_argument("--p-private", type=float, help="also simulate distinguishability at this private-share fraction")
    g.add_argument("--reps", type=int, default=3, help="simulated worlds per process for --p-private")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--seed", type=int, default=20261008)
    g.set_defaults(func=cmd_scorecard)

    g = sub.add_parser("simulate", help="simulate a world and write the observed items table")
    g.add_argument("--process", choices=["replicator", "attraction", "neutral", "contagion"], required=True)
    g.add_argument("--p-private", type=float, default=0.0)
    g.add_argument("--edge-mode", choices=["explicit", "ancestor", "inferred"], default="explicit")
    g.add_argument("--deletion", type=float, default=0.05)
    g.add_argument("--no-recommender", action="store_true", help="chronological feeds")
    g.add_argument("--agents", type=int, default=3000)
    g.add_argument("--n-items", type=int, help="size the world for about this many observed items")
    g.add_argument("--seed", type=int, default=20261008)
    g.add_argument("--out", default="simulated.parquet")
    g.add_argument("--truth", help="also write the full ground-truth log here")
    g.set_defaults(func=cmd_simulate)

    g = sub.add_parser("distinguishability", help="can a dataset this size tell the processes apart?")
    g.add_argument("--n-items", type=int, required=True)
    g.add_argument("--p-private", type=float, required=True)
    g.add_argument("--reps", type=int, default=3)
    g.add_argument("--edge-mode", choices=["explicit", "ancestor", "inferred"], default="explicit")
    g.add_argument("--no-recommender", action="store_true")
    g.add_argument("--full", action="store_true", help="more replicates inside each evaluation")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--seed", type=int, default=20261008)
    g.add_argument("--json", help="write the result as JSON")
    g.set_defaults(func=cmd_distinguishability)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN reductions on sparse simulated cells
    a.func(a)


if __name__ == "__main__":
    main()
