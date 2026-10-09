# M5 Tree-like descent

**Claim.** If units are replicators with lineages, variant sets within a unit should be tree-like enough for
phylogenetic methods, as has been argued for some cultural datasets (Tehrani 2013; Howe & Windram 2011). Blending
and borrowing make histories reticulate (Collard, Shennan & Tehrani 2006; Gray, Bryant & Greenhill 2010).

## What memetrics computes

For each of the largest units (default 10) with at least 8 distinct variants, the 30 most frequent variants form a
word-level edit-distance matrix. For every quartet with pairwise sums `m1 <= m2 <= m3`, the delta score is
`(m3 - m2) / (m3 - m1)` (Holland et al. 2002); the dataset value is the mean over quartets (0 = perfectly
tree-like). The Q-residual (Gray, Bryant & Greenhill 2010) is reported in `detail`, computed on distances scaled to
mean 1; compare it only with references computed by this package.

## Default rule

None: the study reports where the observed values fall against simulated references.

## Calibrated reading

Token sequences are simulated three ways, matched on number of variants, sequence length and mean pairwise
distance (the mutation rate is tuned by bisection):

- **tree**: substitutions and deletions along a coalescent tree;
- **network**: the same, then half the taxa recombine with another taxon;
- **star**: independent mutations of one template (a shared format without shared descent).

Per unit: "tree-like" if delta <= the tree 95th percentile; "reticulate" if >= the network 5th percentile;
"star-like" if inside the star central 90%; otherwise "ambiguous". The scorecard reports the count of each verdict.

## Limits

Missing variants (private, deleted, offline) and unobserved intermediate copies distort delta in both directions.
Distances between short texts are coarse.

## API

`phylo.delta_q`, `phylo.word_distance_matrix`, `phylo.simulate_taxa`, `phylo.reference`, `phylo.tree_likeness`.

## Sources

Holland et al. 2002; Gray, Bryant & Greenhill 2010; Tehrani 2013; Howe & Windram 2011; Collard, Shennan & Tehrani
2006. See [references](../references.md).
