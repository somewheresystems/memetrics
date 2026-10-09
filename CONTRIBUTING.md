# Contributing

Thanks for helping. memetrics is a measurement toolkit, so the bar is: every estimator states what it estimates,
cites where it comes from, is tested on simulated data with a known answer, and never changes its numbers silently.

## Setup

```bash
git clone https://github.com/somewheresystems/memetrics && cd memetrics
uv sync                      # Python 3.11+; development uses 3.13
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Ground rules

- **No network calls outside `memetrics/io/`.** Readers that fetch data must be opt-in, read-only, identify
  themselves with `memetrics.io.USER_AGENT`, honour rate-limit headers, cache what they fetch, and document the
  platform's terms in their module docstring.
- **No remote model APIs.** Embeddings and any model inference run locally with open weights.
- **No real user data in the repository.** Fixtures and examples are simulated (`memetrics simulate`). Authors are
  always salted hashes (`memetrics.io.hash_author`).
- **Seeds.** Every stochastic function takes `seed` and draws all randomness from it.
- **Decision rules.** Thresholds are keyword arguments with the published defaults; verdict strings are part of the
  API.
- **Estimator changes.** `tests/test_regression.py` pins outputs on fixed simulated data. If your change alters
  them, that is an estimator change: explain it in the pull request and in `CHANGELOG.md`, then regenerate with
  `uv run python tests/regression/make_golden.py`. Refactors must leave the goldens untouched.
- **Tests test our logic.** Check that an estimator recovers a known parameter on simulated data, or that a rule
  separates simulated processes. There is no need to test numpy or pandas.
- Heavy dependencies (torch, open_clip, sentence-transformers, websockets) belong in extras and are imported
  inside functions. Tests that need them use `pytest.importorskip`.

## Adding a platform reader

See [docs/readers.md](docs/readers.md). In short: write `memetrics/io/<platform>.py` with a `to_items(...)` that
returns `memetrics.schema.conform(df)`, add a fixture-based test, and wire it into `memetrics ingest`.

## Adding a measure

1. Put the estimator in its own module, with a docstring that states the estimand, the estimator, the default
   rule and the sources.
2. Return `memetrics.Result`.
3. Add a simulation test that the estimator recovers a known value, and a regression case in
   `tests/regression/cases.py`.
4. If it belongs in the scorecard, add it to `memetrics/scorecard.py` with both a literal and a calibrated reading,
   and add a methods page under `docs/methods/`.
