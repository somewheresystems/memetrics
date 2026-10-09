# M1 Units

**Claim.** Cultural variants come in discrete, bounded units, as replicators would (Dawkins 1976; Blackmore 1999).
The alternative is a continuum of variants with no natural boundaries, which is what reconstructive transmission
toward attractors would leave behind (Sperber 1996; Claidière, Scott-Phillips & Sperber 2014). Internet-meme
scholarship treats memes as families of items sharing characteristics (Shifman 2013, 2014), which leaves open how
sharp the family boundaries are.

## What memetrics computes

Units come from `memetrics units`: near-duplicate text components (MinHash on word 5-shingles, Jaccard >= `tau`),
identical-content image groups (pHash Hamming <= 4, with a caption check), or HDBSCAN clusters of embeddings.
Then, for every item *i* in unit *u*:

- `w_i`: similarity to the medoid of its own unit;
- `b_i`: similarity to the medoid of the nearest other unit;
- per unit, Ashman's D between the `w` and `b` samples, `sqrt(2) |mean(w) - mean(b)| / sqrt(var(w) + var(b))`;
- blend: *i* is a blend if `b_i` reaches the 10th percentile of the within-similarities of that nearest unit, so
  *i* is as close to the other unit as 90% of its own members.

Text uses normalized edit similarity (`text.m1_test`); embeddings use cosine (`units.discreteness`). The estimate is
the median D over units, with a bootstrap CI over units. Silhouette and pooled D are in `detail`.

## Default rule

"units" if median D > 2 and blend share < 10%; "continuum" if median D < 2 or blend share > 25%; otherwise
"intermediate". Keywords: `d_threshold`, `blend_units`, `blend_continuum`.

## Calibrated reading

D compares each item with medoids, so it cannot tell discrete units from a slow continuum cut into long segments:
both give large D. Two checks address this.

- **Text, boundary-gap ratio.** For a unit and its nearest neighbour, the smallest cross-boundary distance divided
  by the median nearest-neighbour distance inside the unit. A continuum cut into segments gives a ratio near 1;
  discrete units give ratios well above 1. The scorecard reads >= 2 as discrete boundaries. In the test suite, a
  random walk cut into eight segments passes the D rule but fails the gap ratio.
- **Embeddings, Gaussian surrogate.** The same labels scored on Gaussian draws with the data's mean and leading
  covariance. Units count as discrete only if their D CI lies above the surrogate's.

## Limits

- Units are defined by a clustering choice (`tau`, HDBSCAN parameters). Report results across choices.
- Text M1 drops units with infinite D (zero spread) before taking the median; embedding M1 counts them as very
  large. Both are rare with deduplicated variants, but the two summaries are not interchangeable.
- Missing links do not affect M1 directly, but deleted and private variants are missing from the units.

## API

`memetrics.units.assign`, `units.discreteness`, `units.gaussian_surrogate`, `text.m1_test`, `text.minhash_units`,
`text.gap_ratio`.

## Sources

Dawkins 1976; Blackmore 1999; Sperber 1996; Claidière, Scott-Phillips & Sperber 2014; Shifman 2013; Shifman 2014.
See [references](../references.md).
