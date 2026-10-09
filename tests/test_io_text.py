import gzip

from memetrics.io import bluesky, hash_author, memetracker


def test_hash_author_salted():
    assert hash_author("x", "s1") != hash_author("x", "s2") and len(hash_author("x", "s")) == 16


def test_memetracker_cluster_parser(tmp_path):
    p = tmp_path / "c.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(
            "format:\n<ClSz>\t<TotFq>\t<Root>\t<ClusterId>\n\n"
            "2\t5\tlipstick on a pig\t7\n\t3\t2\tlipstick on a pig\t70\n"
            "\t\t2008-09-09 22:35:24\t2\tM\thttp://a.com/x\n\t\t2008-09-10 01:00:00\t1\tB\thttp://www.b.org/y\n"
            "\t2\t1\tput lipstick on a pig\t71\n\t\t2008-09-10 02:00:00\t2\tB\thttp://c.net/z\n"
        )
    cl, ph, me = memetracker.read_clusters(str(p))
    assert len(cl) == 1 and len(ph) == 2 and len(me) == 3
    items = memetracker.mentions_to_items(ph, me, salt="t")
    assert set(items.unit) == {"7"} and items.community.tolist() == ["news", "blog", "blog"]


def test_bluesky_to_items_counts_engagement_within_horizon():
    did, t0 = "did:plc:a", 1_700_000_000_000_000
    uri = bluesky.post_uri(did, "r1")
    ev = [
        {
            "k": "commit",
            "did": did,
            "c": "app.bsky.feed.post",
            "op": "create",
            "rk": "r1",
            "t": t0,
            "r": {"text": "hello world", "langs": ["en"]},
        },
        {
            "k": "commit",
            "did": "did:plc:b",
            "c": "app.bsky.feed.like",
            "op": "create",
            "rk": "l1",
            "t": t0 + 10,
            "r": {"subject": {"uri": uri}},
        },
        {
            "k": "commit",
            "did": "did:plc:c",
            "c": "app.bsky.feed.post",
            "op": "create",
            "rk": "q1",
            "t": t0 + 20,
            "r": {"text": "quote", "embed": {"$type": "app.bsky.embed.record", "record": {"uri": uri}}},
        },
        {
            "k": "commit",
            "did": "did:plc:d",
            "c": "app.bsky.feed.repost",
            "op": "create",
            "rk": "x",
            "t": t0 + 49 * 3600e6,
            "r": {"subject": {"uri": uri}},
        },
    ]
    it = bluesky.to_items(ev, salt="s").set_index("item_id")
    assert it.loc[uri, "engagement"] == 2  # like + quote; the repost is after 48 h
    assert it.loc[bluesky.post_uri("did:plc:c", "q1"), "parent"] == uri
