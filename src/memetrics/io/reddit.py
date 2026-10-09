"""Reddit reader: Arctic Shift / Pushshift-style submission objects -> the common items table.

No network calls. Input is an iterable of submission dicts (e.g. lines of an Arctic Shift JSONL dump or API
responses saved to disk). Images must be downloaded and hashed separately; pass their pHash / caption via
``image_info`` keyed by submission id.

Columns produced (toolkit/DESIGN.md): item_id, ts, author (salted hash), community, kind, text, phash, parent,
engagement; plus reddit-specific helpers ``num_comments``, ``removed`` and ``image_url`` (resized-variant-first
candidate URLs, space-separated) that downstream metric code ignores.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from . import hash_author

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
        rows.append({
            "item_id": x["id"], "ts": pd.to_datetime(int(x["created_utc"]), unit="s", utc=True),
            "author": None if a in (None, "[deleted]") else hash_author(a, salt),
            "community": x.get("subreddit"), "kind": "image" if url else "text",
            "text": info.get("text", x.get("title")), "phash": info.get("phash"),
            "parent": (x.get("crosspost_parent") or "").removeprefix("t3_") or None,
            "engagement": x.get("score"), "num_comments": x.get("num_comments"),
            "removed": x.get("removed_by_category") is not None, "image_url": url})
    return pd.DataFrame(rows)
