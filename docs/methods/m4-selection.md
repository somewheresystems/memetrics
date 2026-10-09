# M4 Selection versus drift

**Claim.** Within a unit, variant frequencies change because some variants out-replicate others, beyond what
random copying produces. Neutral random-copying models reproduce many popularity distributions and turnover rates
of cultural traits (Bentley, Hahn & Shennan 2004; Hahn & Bentley 2003; Bentley et al. 2007), and transmission biases
leave distinctive turnover signatures (Acerbi & Bentley 2014; O'Dwyer & Kandler 2017; Crema, Kandler & Shennan 2016;
Kandler & Shennan 2013).

## What memetrics computes

**Frequency Increment Test** (Feder, Kryazhimskiy & Plotkin 2014; applied to language by Newberry et al. 2017).
For each unit, the share of its top variant is measured in equal-count time bins. Rescaled increments
`(x_i - x_{i-1}) / sqrt(2 x_{i-1} (1 - x_{i-1}) (t_i - t_{i-1}))` are approximately iid normal with mean zero under
neutrality, so a one-sample t-test of the mean tests for selection. Benjamini-Hochberg controls the false discovery
rate across units.

**Neutral turnover.** Items are split into time steps; `z(y)` is the mean number of new entries per step in the
top-`y` list of variants, and the exponent `x` of `z = a y^x` is compared with the neutral model with innovation.

## Default rule

The study rule reports the share of trajectories that reject neutrality at FDR 0.05 and compares it with the
false-positive rate expected under drift. The literal reading in the scorecard compares the raw p < 0.05 share
with the nominal 5%. Keyword: `fdr`.

## Calibrated reading

- **FIT.** For every trajectory and each effective population size `N_e` (100, 1,000 and 10,000 generations' worth
  by default), neutral Wright-Fisher trajectories are simulated with the observed start frequency, bin times and
  per-bin sample sizes (binomial sampling). The observed rejection share is compared with the null distribution of
  rejection shares. "selection beyond drift" requires exceeding the null 97.5th percentile for every `N_e`;
  "drift-compatible" means inside the central 95% for some `N_e`. Binomial sampling noise makes raw increments
  negatively autocorrelated, so the raw FIT is conservative and the simulated null matters. Results depend on
  binning (Karjus et al. 2020), so the scorecard repeats the test with 6, 9 and 12 bins.
- **Turnover.** The innovation rate `mu` is tuned so that simulated new-variant shares match the data; neutral
  simulations subsampled to the observed step sizes give the null interval of `x`. Below it suggests conformity
  or popularity bias; above it, novelty or anti-conformity bias.

## Bins closer than one generation

Up to 0.1.0 the null was simulated at whole generations, `round((t - t0) / gen_unit)`, and tested on those rounded
times. Bins that rounded to the same generation gave a null increment with `dt = 0` and a NaN p-value, NaN counted
as a non-rejection, and the null rejection share and power were deflated. Now `wf_sampled` accepts fractional
generations (an interval of `dt` generations gets drift variance `dt x(1 - x) / N_e`), the null and power
replicates are tested on exactly the observed times, and rates are computed among replicates with a defined
p-value (`testable` and `power_testable` report that share). The scorecard still sets `gen_unit` to the 5th
percentile of observed bin spacing (at most one day).

## Limits

- A ranker that rewards engagement creates frequency-dependent amplification that the FIT reads as selection.
- Private and cross-platform copies are missing from the frequencies; deleted posts are missing non-randomly.
- The generation length is an assumption; the `N_e` grid spans it.

## API

`selection.fit_test`, `selection.fit_calibrated`, `selection.wf_sampled`, `selection.equal_count_bins`,
`selection.neutral_innovation`, `selection.turnover_profile`, `selection.turnover_test`, `selection.calibrate_mu`.

## Sources

Feder, Kryazhimskiy & Plotkin 2014; Newberry et al. 2017; Karjus et al. 2020; Bentley, Hahn & Shennan 2004; Hahn &
Bentley 2003; Bentley et al. 2007; Acerbi & Bentley 2014; O'Dwyer & Kandler 2017; Crema, Kandler & Shennan 2016;
Kandler & Shennan 2013. See [references](../references.md).
