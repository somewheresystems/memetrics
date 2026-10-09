# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First public release, packaged from the estimators used in the accompanying paper.

### Added

- Measures: M1 units (`units`, `text`), M2 fidelity versus attraction (`chains`), M3 content-intrinsic fitness
  (`fitness`, `textstats`), M4 selection versus drift (`selection`), M5 tree-likeness (`phylo`), N1 non-redundancy
  (`predict`), memecoin heritability (`onchain`).
- Observation-model simulator (`observation`): replicator, attraction, neutral and contagion worlds, thinned by
  private sharing, ranked by a toy recommender, with exposure censored and posts deleted.
- `memetrics.schema`: the items table, with validation and parquet/JSONL/CSV I/O.
- `memetrics.scorecard` and `memetrics.report`: all measures on one table, literal and calibrated readings, the
  "what this data cannot see" panel and an optional distinguishability run; JSON, Markdown and HTML output.
- `memetrics` command line: `ingest`, `units`, `scorecard`, `simulate`, `distinguishability`.
- Readers for Reddit (Arctic Shift dumps and an opt-in, cached, rate-limited API fetcher), Bluesky (Jetstream
  collector and reader), MemeTracker and Pump.fun-style token data.
- `memetrics.embed`: local OpenCLIP, sentence-transformers and pHash helpers (optional extras).
- Regression tests that pin estimator outputs on fixed simulated data.

### Changed (relative to the research code)

- Threshold keyword names are consistent: `d_threshold`, `blend_units`, `blend_continuum` (M1), `strong`, `weak`
  (M3), `fdr` (FIT). The M1 result's `detail["thresholds"]` uses the same names.
- `textstats.reml_icc` column defaults match `fitness.content_icc` (`engagement`, `unit`, `author`).
- One `ashman_d` for embeddings and text. With zero spread and equal means it returns NaN (the embedding version
  returned inf); all other inputs give identical values.
- `selection.fit_calibrated` warns when bins are closer than one generation (see `docs/methods/m4-selection.md`).
- `io.hash_author` warns when called without a salt; `memetrics ingest` refuses to run without one.
