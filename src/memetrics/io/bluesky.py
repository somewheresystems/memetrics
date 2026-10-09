"""Bluesky Jetstream collector and reader.

Collector: subscribes (read-only, no login) to a public Jetstream instance for posts, likes,
reposts and follows, and writes compact JSON lines to hourly gzip files

    <root>/YYYY-MM-DD/HH.jsonl.gz

Each line is one event: {"t": time_us, "k": kind, "did": ..., "c": collection, "op": op,
"rk": rkey, "r": record-or-null}. Kinds: "commit", "account", "identity".
Resume: the last processed time_us is checkpointed to <root>/cursor; on (re)connect we ask
Jetstream to replay from cursor - 5 s (Jetstream replays by cursor; duplicates are dropped
downstream by (did, collection, rkey, op)).

Reader: `iter_events(root, start, end)` streams events; `parse_uri` splits at:// URIs.

Run:  uv run --python 3.13 --with websockets python -m memetrics.io.bluesky collect --root DIR

`to_items(events)` turns the event stream into the common items table (toolkit/DESIGN.md), with
`engagement` = likes + reposts + quotes received within a horizon (default 48 h), and `parent` = the quoted
post (or replied-to post if `parent_from='reply'`).
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

COLLECTIONS = (
    "app.bsky.feed.post",
    "app.bsky.feed.like",
    "app.bsky.feed.repost",
    "app.bsky.graph.follow",
)
DEFAULT_HOST = "wss://jetstream2.us-east.bsky.network/subscribe"
FALLBACK_HOSTS = (
    "wss://jetstream1.us-east.bsky.network/subscribe",
    "wss://jetstream1.us-west.bsky.network/subscribe",
    "wss://jetstream2.us-west.bsky.network/subscribe",
)


def compact(ev: dict) -> dict | None:
    """Reduce a Jetstream event to the fields we keep."""
    kind = ev.get("kind")
    t = ev.get("time_us")
    did = ev.get("did")
    if kind == "commit":
        c = ev.get("commit") or {}
        rec = c.get("record")
        if rec is not None:
            rec = {k: v for k, v in rec.items() if k != "$type"}
        return {"t": t, "k": "commit", "did": did, "c": c.get("collection"),
                "op": c.get("operation"), "rk": c.get("rkey"), "r": rec}
    if kind == "account":
        a = ev.get("account") or {}
        return {"t": t, "k": "account", "did": did, "active": a.get("active"),
                "status": a.get("status")}
    if kind == "identity":
        return {"t": t, "k": "identity", "did": did}
    return None


def hour_path(root: Path, t_us: int) -> Path:
    dt = datetime.fromtimestamp(t_us / 1e6, tz=timezone.utc)
    return root / dt.strftime("%Y-%m-%d") / f"{dt:%H}.jsonl.gz"


class HourlyWriter:
    def __init__(self, root: Path):
        self.root = root
        self.path: Path | None = None
        self.fh = None

    def write(self, t_us: int, line: str) -> None:
        p = hour_path(self.root, t_us)
        if p != self.path:
            self.close()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.fh = gzip.open(p, "at", compresslevel=6)  # multi-member gzip on append
            self.path = p
        self.fh.write(line)
        self.fh.write("\n")

    def flush(self) -> None:
        if self.fh:
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None
            self.path = None


def read_cursor(root: Path) -> int | None:
    f = root / "cursor"
    try:
        return int(f.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def write_cursor(root: Path, t_us: int) -> None:
    tmp = root / "cursor.tmp"
    tmp.write_text(str(t_us))
    os.replace(tmp, root / "cursor")


async def collect(root: Path, hosts: tuple[str, ...], zstd_dict: str | None = None) -> None:  # pragma: no cover
    """Run the collector. With `zstd_dict` (the Jetstream v1 dictionary file), asks for `compress=true` and
    decompresses frames, which cuts bandwidth roughly in half on constrained links."""
    import websockets

    dec = None
    if zstd_dict:
        import zstandard
        dec = zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(open(zstd_dict, "rb").read()))

    root.mkdir(parents=True, exist_ok=True)
    writer = HourlyWriter(root)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    host_i, n, last_t, last_flush = 0, 0, read_cursor(root), time.time()
    log = root / "collector.log"

    def note(msg: str) -> None:
        with open(log, "a") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")

    while not stop.is_set():
        qs = "&".join(f"wantedCollections={c}" for c in COLLECTIONS) + ("&compress=true" if dec else "")
        url = f"{hosts[host_i % len(hosts)]}?{qs}"
        if last_t:
            url += f"&cursor={last_t - 5_000_000}"
        try:
            note(f"connect {url[:80]} cursor={last_t}")
            async with websockets.connect(url, max_size=2**22, ping_interval=None,
                                          open_timeout=30, max_queue=4096) as ws:
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=120)
                    except asyncio.TimeoutError:
                        raise ConnectionError("no message in 120 s")
                    if dec is not None and isinstance(msg, (bytes, bytearray)):
                        msg = dec.decompress(msg, max_output_size=2**24)
                    ev = json.loads(msg)
                    rec = compact(ev)
                    if rec is None or rec["t"] is None:
                        continue
                    writer.write(rec["t"], json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
                    last_t = rec["t"]
                    n += 1
                    if time.time() - last_flush > 10:
                        writer.flush()
                        write_cursor(root, last_t)
                        last_flush = time.time()
                    if n % 1_000_000 == 0:
                        note(f"events={n} cursor={last_t}")
        except Exception as e:  # reconnect on any failure, rotate host
            writer.flush()
            if last_t:
                write_cursor(root, last_t)
            note(f"error {type(e).__name__}: {e}; reconnecting")
            host_i += 1
            await asyncio.sleep(5)
    writer.close()
    if last_t:
        write_cursor(root, last_t)
    note("stopped")


# ---------------------------------------------------------------- reader

def iter_files(root: Path, start: str | None = None, end: str | None = None) -> Iterator[Path]:
    """Hourly files in time order; start/end are 'YYYY-MM-DD' (inclusive)."""
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if (start and d.name < start) or (end and d.name > end):
            continue
        yield from sorted(d.glob("*.jsonl.gz"))


def iter_events(root: Path, start: str | None = None, end: str | None = None) -> Iterator[dict]:
    """Stream events; tolerates a truncated final gzip member (live file)."""
    for f in iter_files(root, start, end):
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except (EOFError, gzip.BadGzipFile, OSError):
            continue


def parse_uri(uri: str) -> tuple[str, str, str] | None:
    """at://did/collection/rkey -> (did, collection, rkey)."""
    if not uri or not uri.startswith("at://"):
        return None
    parts = uri[5:].split("/")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def post_uri(did: str, rkey: str) -> str:
    return f"at://{did}/app.bsky.feed.post/{rkey}"


def quoted_uri(record: dict) -> str | None:
    """URI of the post quoted by a post record (record or recordWithMedia embeds)."""
    emb = (record or {}).get("embed") or {}
    t = emb.get("$type", "")
    if t == "app.bsky.embed.record":
        return (emb.get("record") or {}).get("uri")
    if t == "app.bsky.embed.recordWithMedia":
        return ((emb.get("record") or {}).get("record") or {}).get("uri")
    return None


def to_items(events, horizon_h: float = 48.0, parent_from: str = "quote", salt: str | None = None):
    """Build the items table from Jetstream events (an iterable of compact dicts, e.g. `iter_events`).

    Only post creations become items. Engagement counts like/repost creations and quote posts whose subject is
    the item and whose time is within `horizon_h` hours of the item. Posts deleted in the stream are flagged in
    column `deleted`. Authors are salted hashes. Returns a pandas DataFrame.
    """
    import pandas as pd

    from . import hash_author

    posts, eng, deleted = {}, {}, set()
    horizon = horizon_h * 3600e6
    for e in events:
        if e.get("k") != "commit":
            continue
        c, op = e.get("c"), e.get("op")
        if c == "app.bsky.feed.post":
            uri = post_uri(e["did"], e["rk"])
            if op == "delete":
                deleted.add(uri)
                continue
            if op != "create":
                continue
            r = e.get("r") or {}
            q = quoted_uri(r)
            reply = ((r.get("reply") or {}).get("parent") or {}).get("uri")
            posts[uri] = (e["t"], e["did"], r.get("text", ""), q, reply, ",".join(r.get("langs") or []))
            if q:
                eng.setdefault(q, []).append(e["t"])
        elif c in ("app.bsky.feed.like", "app.bsky.feed.repost") and op == "create":
            subj = ((e.get("r") or {}).get("subject") or {}).get("uri")
            if subj:
                eng.setdefault(subj, []).append(e["t"])
    rows = []
    for uri, (t, did, text, q, reply, langs) in posts.items():
        n = sum(1 for x in eng.get(uri, ()) if t <= x <= t + horizon)
        rows.append({"item_id": uri, "ts": pd.Timestamp(t, unit="us", tz="UTC"), "author": hash_author(did, salt),
                     "community": langs or "und", "kind": "text", "text": text,
                     "parent": q if parent_from == "quote" else reply, "quote_of": q, "reply_to": reply,
                     "engagement": float(n), "deleted": uri in deleted})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- public AppView (read-only, no login)

APPVIEW = "https://public.api.bsky.app/xrpc"


def _get(method: str, params: list[tuple[str, str]], retries: int = 6) -> dict | None:  # pragma: no cover - network
    import urllib.error
    import urllib.parse
    import urllib.request

    url = f"{APPVIEW}/{method}?{urllib.parse.urlencode(params)}"
    delay = 2.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "memetrics-research/0.1 (read-only)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            if e.code == 429:
                reset = e.headers.get("ratelimit-reset")
                wait = max(float(reset) - time.time(), 5) if reset else 60
                time.sleep(min(wait, 300))
                continue
        except Exception:  # noqa: BLE001 - network errors: back off and retry
            pass
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return None


def get_posts(uris: list[str]) -> dict:  # pragma: no cover - network
    """uri -> post view (dict) for posts that still exist; missing uris are deleted or hidden. Batches of 25."""
    out = {}
    for i in range(0, len(uris), 25):
        r = _get("app.bsky.feed.getPosts", [("uris", u) for u in uris[i:i + 25]])
        for p in (r or {}).get("posts", []):
            out[p["uri"]] = p
        time.sleep(0.1)
    return out


def get_profiles(dids: list[str]) -> dict:  # pragma: no cover - network
    """did -> {followersCount, followsCount, postsCount}. Batches of 25."""
    out = {}
    for i in range(0, len(dids), 25):
        r = _get("app.bsky.actor.getProfiles", [("actors", d) for d in dids[i:i + 25]])
        for p in (r or {}).get("profiles", []):
            out[p["did"]] = {k: p.get(k) for k in ("followersCount", "followsCount", "postsCount")}
        time.sleep(0.1)
    return out


def get_relationship(actor: str, other: str) -> dict | None:  # pragma: no cover - network
    """Whether `actor` follows `other` (and vice versa), via app.bsky.graph.getRelationships."""
    r = _get("app.bsky.graph.getRelationships", [("actor", actor), ("others", other)])
    rel = ((r or {}).get("relationships") or [None])[0]
    if not rel or rel.get("notFound"):
        return None
    return {"actor_follows": "following" in rel, "followed_by": "followedBy" in rel}


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--root", required=True)
    c.add_argument("--zstd-dict", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        asyncio.run(collect(Path(a.root), (DEFAULT_HOST,) + FALLBACK_HOSTS, a.zstd_dict))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
