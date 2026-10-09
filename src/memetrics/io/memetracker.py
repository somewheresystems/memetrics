"""Reader for the MemeTracker phrase-cluster and raw quote files (Leskovec, Backstrom & Kleinberg 2009).

Data: https://snap.stanford.edu/data/memetracker9.html (raw `quotes_YYYY-MM.txt.gz`) and the phrase-cluster file
`clust-qt08080902w3mfq5.txt.gz` from the MemeTracker project.

Cluster file format::

    <ClSz> <TotFq> <Root> <ClusterId>
        <QtFq> <Urls> <QtStr> <QuoteId>
            <Tm> <Fq> <UrlTy> <Url>

Raw quote file: blocks of lines `P url`, `T time`, `Q phrase` (0+), `L link` (0+), separated by blank lines.

`read_clusters` yields three DataFrames (clusters, phrases, mentions). `mentions_to_items` turns mentions into the
items table: one item per (phrase, url); author = hashed site domain; community = 'blog' or
'news'; unit = cluster id. `iter_documents` streams the raw files (for hyperlink edges).
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from urllib.parse import urlsplit

import pandas as pd

from ..schema import conform
from . import hash_author


def read_clusters(path: str, min_cluster_mentions: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse the cluster file. Returns (clusters, phrases, mentions) DataFrames."""
    cl, ph = [], []
    m_pid, m_ts, m_fq, m_ty, m_url = [], [], [], [], []
    cid = pid = None
    keep = True
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith("format") or line.lstrip().startswith("<"):
                continue
            if line.startswith("\t\t"):
                if keep and pid is not None:
                    parts = line[2:].rstrip("\n").split("\t")
                    if len(parts) >= 4:
                        m_pid.append(pid)
                        m_ts.append(parts[0])
                        m_fq.append(int(parts[1]))
                        m_ty.append(parts[2])
                        m_url.append(parts[3])
            elif line.startswith("\t"):
                parts = line[1:].rstrip("\n").split("\t")
                if len(parts) >= 4 and keep:
                    pid = int(parts[3])
                    ph.append((pid, cid, parts[2], int(parts[0]), int(parts[1])))
            else:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    cid = int(parts[3])
                    tot = int(parts[1])
                    keep = tot >= min_cluster_mentions
                    if keep:
                        cl.append((cid, int(parts[0]), tot, parts[2]))
    clusters = pd.DataFrame(cl, columns=["cluster_id", "n_phrases", "total_freq", "root"])
    phrases = pd.DataFrame(ph, columns=["phrase_id", "cluster_id", "text", "freq", "n_urls"])
    mentions = pd.DataFrame(
        {"phrase_id": m_pid, "ts": pd.to_datetime(m_ts, utc=True), "freq": m_fq, "url_type": m_ty, "url": m_url}
    )
    return clusters, phrases, mentions


def domain(url: str) -> str:
    try:
        h = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def mentions_to_items(phrases: pd.DataFrame, mentions: pd.DataFrame, salt: str | None = None) -> pd.DataFrame:
    """Common items table: one row per (phrase, url) mention."""
    m = mentions.merge(phrases[["phrase_id", "cluster_id", "text"]], on="phrase_id")
    doms = m.url.map(domain)
    hashed = {d: hash_author(d, salt) for d in doms.unique()}
    return conform(
        pd.DataFrame(
            {
                "item_id": m.phrase_id.astype(str) + "|" + m.url,
                "ts": m.ts,
                "author": doms.map(hashed),
                "community": m.url_type.map({"B": "blog", "M": "news"}).fillna("other"),
                "kind": "text",
                "text": m.text,
                "unit": m.cluster_id.astype(str),
                "parent": None,
                "engagement": m.freq.astype(float),
                "url": m.url,
                "phrase_id": m.phrase_id,
            }
        )
    )


def iter_documents(path: str) -> Iterator[dict]:
    """Stream documents {url, ts, quotes: [...], links: [...]} from a raw quotes file."""
    doc: dict = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                if doc.get("url"):
                    yield doc
                doc = {}
                continue
            tag, _, val = line.rstrip("\n").partition("\t")
            if tag == "P":
                if doc.get("url"):
                    yield doc
                doc = {"url": val, "ts": None, "quotes": [], "links": []}
            elif tag == "T":
                doc["ts"] = val
            elif tag == "Q":
                doc.setdefault("quotes", []).append(val)
            elif tag == "L":
                doc.setdefault("links", []).append(val)
    if doc.get("url"):
        yield doc
