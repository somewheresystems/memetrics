"""Reddit reader: Arctic Shift / Pushshift-style submission objects -> the items table.

Offline use: :func:`read_jsonl` streams a JSONL dump (``.jsonl``, ``.gz`` or ``.zst``) and :func:`to_items` turns
submissions into the items table. Images must be downloaded and hashed separately; pass their pHash and caption via
``image_info`` keyed by submission id.

Network use (optional): :func:`fetch_posts` pages through the public Arctic Shift archive API
(https://arctic-shift.photon-reddit.com), read-only and without a Reddit account. It waits between requests,
honours the ``x-ratelimit-*`` headers and HTTP 429, and caches every page to disk so a rerun makes no requests.
Respect Arctic Shift's and Reddit's terms: research use, no redistribution of raw content, and no attempt to
identify users (authors are salted hashes in the items table).

Columns produced: item_id, ts, author (salted hash), community, kind, text, phash, parent, engagement; plus
``num_comments``, ``removed`` and ``image_url`` (candidate image URLs, resized variants first, space-separated),
which measures ignore.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from ..schema import conform
from . import USER_AGENT, hash_author

ARCTIC_SHIFT = "https://arctic-shift.photon-reddit.com/api"

_IMG_EXT = (".jpg", ".jpeg", ".png")
_IMGUR_I = re.compile(r"^https?://i\.imgur\.com/([A-Za-z0-9]{5,8})\.[a-zA-Z]+$")
_IMGUR_PAGE = re.compile(r"^https?://(?:m\.)?imgur\.com/([A-Za-z0-9]{5,8})(?:\.[a-z]+)?/?$")


def _preview(x: dict, minw: int = 640) -> str | None:
    try:
        im = x["preview"]["images"][0]
    except (KeyError, IndexError, TypeError):
        return None
    rs = [r for r in (im.get("resolutions") or []) if "fit=crop" not in r.get("url", "")]
    big = [r for r in rs if r.get("width", 0) >= minw]
    r = big[0] if big else (rs[-1] if rs else None)
    return r["url"].replace("&amp;", "&") if r else None


def image_candidates(x: dict) -> tuple[str | None, str]:
    """(space-separated candidate URLs, rule) for a static-image submission, or (None, reason).
    Resized variants come first (Reddit preview >= 640 px, Imgur ``l`` thumbnail), originals as fallback.
    NSFW, galleries and animated media are skipped."""
    if x.get("is_self"):
        return None, "self"
    if x.get("over_18"):
        return None, "nsfw"
    u = (x.get("url") or "").split("?")[0]
    if x.get("is_gallery") or "/gallery/" in u:
        return None, "gallery"
    ul = u.lower()
    pv = _preview(x)
    if "i.redd.it/" in ul and ul.endswith(_IMG_EXT):
        return " ".join(c for c in (pv, u) if c), "direct"
    m = _IMGUR_I.match(u) or _IMGUR_PAGE.match(u)
    if m and (ul.endswith(_IMG_EXT) or _IMGUR_PAGE.match(u)):
        return f"https://i.imgur.com/{m.group(1)}l.jpg https://i.imgur.com/{m.group(1)}.jpg", "imgur"
    if ul.endswith((".gif", ".gifv", ".mp4")) or "v.redd.it" in ul or "gfycat" in ul:
        return None, "animated"
    return (pv, "preview") if pv else (None, "other_host")


def to_items(posts: Iterable[dict], salt: str | None = None, image_info: dict | None = None) -> pd.DataFrame:
    """Submissions -> items table. ``image_info``: {submission id: {"phash": int, "text": caption}}."""
    image_info = image_info or {}
    rows = []
    for x in posts:
        a = x.get("author")
        url, _ = image_candidates(x)
        info = image_info.get(x["id"], {})
        rows.append(
            {
                "item_id": x["id"],
                "ts": pd.to_datetime(int(x["created_utc"]), unit="s", utc=True),
                "author": None if a in (None, "[deleted]") else hash_author(a, salt),
                "community": x.get("subreddit"),
                "kind": "image" if url else "text",
                "text": info.get("text", x.get("title")),
                "phash": info.get("phash"),
                "parent": (x.get("crosspost_parent") or "").removeprefix("t3_") or None,
                "engagement": x.get("score"),
                "num_comments": x.get("num_comments"),
                "removed": x.get("removed_by_category") is not None,
                "image_url": url,
            }
        )
    return conform(pd.DataFrame(rows))


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream objects from a JSONL file (plain, ``.gz`` or ``.zst``; zstd needs the ``zstandard`` package)."""
    path = Path(path)
    if path.suffix == ".zst":
        import zstandard

        raw = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(open(path, "rb"))
        fh = io.TextIOWrapper(raw, encoding="utf-8")
    elif path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = open(path, encoding="utf-8")
    with fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _get(url: str, min_interval: float, state: dict, retries: int = 8) -> dict:  # pragma: no cover - network
    delay = 2.0
    for _ in range(retries):
        wait = state.get("next", 0.0) - time.time()
        if wait > 0:
            time.sleep(wait)
        state["next"] = time.time() + min_interval
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                remaining, reset = r.headers.get("x-ratelimit-remaining"), r.headers.get("x-ratelimit-reset")
                body = json.loads(r.read())
            if remaining is not None and reset is not None and float(remaining) < 2:
                state["next"] = max(state["next"], time.time() + float(reset) + 1)
            if body.get("error"):
                raise RuntimeError(body["error"])
            return body
        except urllib.error.HTTPError as e:
            if e.code == 429:
                state["next"] = time.time() + float(e.headers.get("x-ratelimit-reset") or 30) + 1
                continue
            if e.code in (400, 404):
                raise
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            pass
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"Arctic Shift request failed after {retries} attempts: {url}")


def fetch_posts(
    subreddit: str, after: str, before: str, cache_dir: str | Path, min_interval: float = 1.0
) -> Iterator[dict]:  # pragma: no cover - network
    """Submissions of ``subreddit`` created in [after, before) (ISO dates), oldest first, from the Arctic Shift API.

    Pages of 100 are cached as ``<cache_dir>/<subreddit>/<after-epoch>.json``; cached pages are never re-requested.
    At most one request per ``min_interval`` seconds; on HTTP 429 or a nearly exhausted ``x-ratelimit-remaining`` the
    client pauses until ``x-ratelimit-reset``.
    """
    cache = Path(cache_dir) / subreddit
    cache.mkdir(parents=True, exist_ok=True)
    t = int(pd.Timestamp(after, tz="UTC").timestamp())
    end = int(pd.Timestamp(before, tz="UTC").timestamp())
    state: dict = {}
    while t < end:
        page = cache / f"{t}.json"
        if page.exists():
            data = json.loads(page.read_text())
        else:
            q = urllib.parse.urlencode({"subreddit": subreddit, "after": t, "before": end, "limit": 100, "sort": "asc"})
            data = _get(f"{ARCTIC_SHIFT}/posts/search?{q}", min_interval, state).get("data") or []
            page.write_text(json.dumps(data))
        if not data:
            break
        yield from data
        last = int(max(x["created_utc"] for x in data))
        t = last + 1 if last >= t else end
