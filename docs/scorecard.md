# Reading a scorecard

```bash
memetrics scorecard items.parquet --out report/            # writes scorecard.json, report.md, report.html
memetrics scorecard items.parquet --out report/ --fast     # fewer replicates, for a first look
memetrics scorecard items.parquet --out report/ --p-private 0.3 --reps 5   # add a distinguishability run
memetrics scorecard items.parquet --rule f_threshold=0.95 --rule strong=0.4
```

Each measure reports one or more rows with two readings.

**Literal** applies the default decision rule to the raw estimate: for example "replication if median fidelity
>= 0.9 and the convergence slope's 95% CI contains 0". These are the rules of the study the package was built for.
Override them with `--rule KEY=VALUE` (keys: `d_threshold`, `blend_units`, `blend_continuum`, `f_threshold`,
`strong`, `weak`, `fdr`, `min_auc_gain`).

**Calibrated** compares the estimate with a simulated null of the same size and structure:

| measure | null |
|---|---|
| M1 | text: boundary-gap ratio (segments of a continuum give < 2); embeddings: a cluster-free Gaussian surrogate |
| M2 | pure copying re-simulated along the observed topology with matched mutation |
| M3 | content labels shuffled within community |
| M4 FIT | Wright-Fisher neutral trajectories matched on start frequency, bin times and sample sizes |
| M4 turnover | neutral model with innovation, matched on innovation rate and sample sizes |
| M5 | token sequences simulated on trees, reticulate networks and stars, matched on size and distance |
| N1 | none; the reverse nesting shows what the rival baseline adds beyond memetic terms |

The two readings can disagree. The standard example is M2: pure copying along a drifting chain produces a negative
convergence slope, because the unit's medoid is estimated from the same chain. The literal rule then says "mixed"
for data that the copying null reproduces exactly.

## What this data cannot see

Every report ends with this panel. Public data shows adoption, not exposure; it misses private and cross-platform
sharing, deleted posts, offline spread and the ranking that decided who saw what. The panel states, measure by
measure, which estimates each gap biases. With `--p-private P` the report also simulates replicator, attraction,
neutral and contagion worlds at the dataset's size with a share `P` of private reshares, applies the same rules to
the observed tables, and prints the confusion matrix: how often each true process would be labelled correctly from
data like yours. See [the observation simulator](methods/observation.md).

## JSON

`scorecard.json` holds `dataset` (counts and coverage), `rules`, `effort`, `measures` (each with `rows`, and each
row with the full `Result` under `result`), `blind_spots` and `distinguishability`. Non-finite numbers are `null`.
