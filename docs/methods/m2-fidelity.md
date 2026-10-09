# M2 Fidelity versus attraction

**Claim.** Replicators are copied with high fidelity and do not drift in a preferred direction (Dawkins 1976;
Blackmore 1999). Cultural attraction predicts lower-fidelity, reconstructive transmission that converges on
attractors whatever the starting variant (Sperber 1996; Claidière & Sperber 2007; Claidière, Scott-Phillips &
Sperber 2014; Morin 2016). Serial-reproduction and transmission-chain experiments document such convergence in the
lab (Bartlett 1995; Kashima 2000; Mesoudi & Whiten 2008; Kirby, Cornish & Smith 2008), and studies of quotations
and online information find systematic transformation along chains (Lerique & Roth 2018; Adamic et al. 2016;
Simmons, Adamic & Adar 2011).

## What memetrics computes

Edges are parent-child pairs within a unit: observed links (quote, reply, crosspost, remix) or, if there are none,
inferred parents (the most similar earlier item; a guess). On each edge:

- fidelity `f = sim(parent, child)` (1 - normalized Levenshtein);
- `d(x) = 1 - sim(x, unit medoid)` and the step `delta = d(child) - d(parent)`;
- the convergence slope `beta`: within-unit OLS of `delta` on `d(parent)`, CI by cluster bootstrap over units.
  Attraction predicts `beta < 0`.

The scorecard runs this twice: on all edges, and on derivatives only (`f < 1`). Verbatim reposts dominate many
platforms and can carry the fidelity criterion on their own, so the derivative-only row asks whether the copies that
change anything change faithfully.

## Default rule

"replication" if median `f` >= 0.9 and the CI of `beta` contains 0; "attraction" if the CI of `beta` is below 0 and
median `f` < 0.9; otherwise "mixed". Keyword: `f_threshold`.

## Calibrated reading

Pure copying produces a negative slope. In high dimensions a random step from a point far from the centre tends to
move it inward, and the medoid is estimated from the same data. The replicator null (`chains.replicator_null`)
keeps the observed topology and root texts, re-simulates every child by copying with the observed share of changed
edges, the observed mix of insertions/deletions versus substitutions, and a number of word edits tuned to the
observed mean step size, then recomputes `beta` and the medoids. "attraction" requires `beta` below the null's 2.5th
percentile with `f` < 0.9; "replication" requires `f` >= 0.9 with `beta` inside the null's central 95%.

## Limits

- Thinned chains: an observed edge may span several unobserved copies (private shares, deletions), which lowers
  `f` and can mimic attraction. The observation simulator measures how much (`edge_mode="ancestor"`).
- Inferred parents form chains or stars that never happened when many items copy one unobserved source.
- `detail` also reports the toward-medoid share, depth profiles and a Price-equation split of change into
  selection and transmission (`chains.price_decomposition`; Claidière, Scott-Phillips & Sperber 2014).

## API

`chains.edges_from_items`, `chains.infer_parents`, `chains.unit_medoids`, `chains.m2_test`,
`chains.replicator_null`, `chains.depth_profile`, `chains.price_decomposition`.

## Sources

Dawkins 1976; Blackmore 1999; Sperber 1996; Claidière & Sperber 2007; Claidière, Scott-Phillips & Sperber 2014;
Morin 2016; Bartlett 1995; Kashima 2000; Mesoudi & Whiten 2008; Kirby, Cornish & Smith 2008; Lerique & Roth 2018;
Adamic et al. 2016; Simmons, Adamic & Adar 2011. See [references](../references.md).
