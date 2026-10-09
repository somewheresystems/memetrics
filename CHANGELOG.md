# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09

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
- `io.hash_author` warns when called without a salt; `memetrics ingest` refuses to run without one.

### Fixed (found by code review after the paper's results; the research code is fixed the same way)

The regression goldens were regenerated only for the cases whose estimator behaviour these fixes change
(`selection`: new `testable` and `power_testable` keys, values unchanged; `observation`: N1, M4 and, through the
deletion fix, every observed-layer metric). All other cases still match the goldens generated from the research
code before packaging.

- `selection.fit_calibrated`: the Wright-Fisher null and the power simulations were run at whole generations
  `round((t - t0) / gen_unit)` and tested on those rounded times, so bins closer than one generation gave null
  increments with `dt = 0` and NaN p-values, which counted as non-rejections. `wf_sampled` now accepts fractional
  generations (drift variance `dt x(1 - x) / N_e` per interval), null and power replicates are tested on the observed
  times, and `fpr_raw`, the null BH shares and `power_p05` are computed among replicates with a defined p-value
  (new `testable` and `power_testable`). Power now starts from the same clipped frequency as the null
  (`[1/(2n), 1 - 1/(2n)]` instead of `[0.01, 0.99]`). `fit_test` returns NaN when two sampling times coincide.
- `units.identical_groups`: with unsorted input an item joined the matching group with the lowest row index instead
  of the earliest representative. Candidates are now tried in time order.
- `observation._n1` (simulator N1): author and unit engagement histories included likes given after the focal post,
  and the shrinkage target was the mean over training and test periods. Histories now use only earlier posts whose
  engagement window has closed, and the target is a training-period mean.
- `observation.observe`: deletion was doubled for every post at or below the 25th engagement percentile, which with
  ties at zero likes was often far more than a quarter of posts. Exactly the bottom quartile (ties broken at random)
  is now doubled.
- Zero-spread units in M1: `units.ashman_d` returns NaN (not inf) when a sample has fewer than two values, and
  `text.m1_test` counts zero-spread units (D = inf) as very large in the median, as `units.discreteness` does,
  instead of dropping them.
- `chains.replicator_null`: the medoid of each simulated set was estimated with a fixed column-sample seed whatever
  the `seed` argument; it now follows `seed`. Only units with more than 400 distinct texts are affected.
- Docstrings: the pigeonhole bootstrap's product Poisson weights are attributed to Owen & Eckles (2012); the
  unequal-group-size ICC(1) adjustment to the standard one-way ANOVA estimator (Donner 1986); the Q-residual's
  mean-1 distance scaling to Gray, Bryant & Greenhill (2010). Statistical references are listed in
  `docs/references.md`.

