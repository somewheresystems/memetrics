# Adding a platform reader

A reader turns platform data into the [items table](data-model.md). Measures never see platform fields.

1. Create `src/memetrics/io/<platform>.py`. Start the module docstring with what the input is, which columns it
   produces, and, if it makes network requests, the endpoint, its rate limits and the terms of service.
2. Write `to_items(records, salt=None, ...) -> pd.DataFrame`:
   - one row per public post; `item_id` unique;
   - `author = hash_author(handle, salt)`, never the raw handle;
   - `ts` timezone-aware UTC; `kind` one of `image`, `text`, `video`, `token`;
   - `parent` only for observed copy events (quote, reply, crosspost, remix), as an `item_id` in the same
     namespace;
   - `engagement` as a float, with the counting window documented (e.g. likes within 48 h);
   - return `memetrics.schema.conform(df)`.
3. Keep file parsing and network access apart. Fetchers are opt-in functions that:
   - send `memetrics.io.USER_AGENT`;
   - wait between requests and honour `429` and rate-limit headers;
   - cache every response on disk so reruns make no requests;
   - never log in, post, vote or message.
   See `memetrics.io.reddit.fetch_posts`.
4. Add a test with a small hand-written fixture (no real user data) that checks hashing, timestamps, parent links
   and engagement.
5. Wire it into `memetrics ingest` in `src/memetrics/cli.py` and document it in the README.

Existing readers:

| module | input | notes |
|---|---|---|
| `io.reddit` | Arctic Shift or Pushshift submission JSONL; optional API fetcher | `parent` = crosspost parent |
| `io.bluesky` | Jetstream events from the bundled collector | engagement = likes + reposts + quotes in a window; `parent` = quoted (or replied-to) post |
| `io.memetracker` | MemeTracker phrase-cluster file | one item per (phrase, url); author = hashed site domain; `unit` = phrase cluster |
| `io.onchain` | Pump.fun-style launch table and daily candles | `kind = token`; fitness series from candles |
