from memetrics.io.reddit import image_candidates, to_items


def test_candidates_prefer_resized_and_skip_nsfw():
    x = {"id": "a", "url": "https://i.redd.it/abc.png", "created_utc": 0,
         "preview": {"images": [{"resolutions": [
             {"width": 320, "url": "https://preview.redd.it/abc.png?width=320&amp;crop=smart&amp;s=1"},
             {"width": 640, "url": "https://preview.redd.it/abc.png?width=640&amp;crop=smart&amp;s=2"}]}]}}
    u, rule = image_candidates(x)
    assert rule == "direct" and u.split()[0].endswith("s=2") and u.split()[1] == "https://i.redd.it/abc.png"
    assert image_candidates({**x, "over_18": True})[0] is None
    assert image_candidates({"url": "https://imgur.com/AbCdE12"})[0].startswith("https://i.imgur.com/AbCdE12l.jpg")


def test_to_items_hashes_authors():
    d = to_items([{"id": "a", "author": "someone", "created_utc": 0, "subreddit": "memes", "score": 5}], salt="s")
    assert d.author[0] != "someone" and len(d.author[0]) == 16 and d.engagement[0] == 5
