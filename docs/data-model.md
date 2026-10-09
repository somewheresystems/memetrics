# Data model: the items table

Every measure consumes one table, a pandas DataFrame (on disk: parquet, JSONL or CSV), one row per public post.

| column | dtype | meaning |
|---|---|---|
| `item_id` | str | unique id |
| `ts` | datetime64[ns, UTC] | post time |
| `author` | str | salted hash, never a raw handle |
| `community` | str | subreddit, feed, language, platform |
| `kind` | str | `image`, `text`, `video` or `token` |
| `text` | str, optional | caption, OCR or body |
| `phash` | uint64, optional | 64-bit perceptual hash (images) |
| `unit` | str, optional | meme unit, filled by `memetrics units` |
| `parent` | str, optional | `item_id` of an observed source: reply, quote, crosspost, remix |
| `engagement` | float, optional | score, likes plus reposts, market cap |

Embeddings live in a separate `.npy` array whose rows follow the table's row order (`memetrics.embed` computes them
locally).

```python
from memetrics import read_items, validate, write_items
from memetrics.schema import conform

items = conform(my_frame)      # coerce types and column order
validate(items)                # raises memetrics.SchemaError with every problem found
write_items(items, "items.parquet")
items = read_items("items.parquet")
```

Readers may add platform columns (`deleted`, `quote_of`, `num_comments`, ...). Measures ignore columns they do not
use.

## Authors are hashed

`memetrics.io.hash_author(handle, salt)` is a salted SHA-256 truncated to 16 hex characters. Keep the salt private
and fixed for a study: anyone with the salt and a list of handles can recompute the hashes. `memetrics ingest`
refuses to run without `--salt` or `$MEMETRICS_SALT`.

## What a parent link means

`parent` records an observed copy event: a quote, a reply, a crosspost, an explicit remix link. Where no such
links exist, M2 infers parents as the most similar earlier item in the same unit. That is a time-ordered guess,
and the scorecard labels it as one.
