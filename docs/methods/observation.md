# Observation-model simulator

Public social-media data shows adoption, not exposure. It misses private and cross-platform sharing, deletions,
offline spread and the ranking that decided who saw what. `memetrics.observation` asks how much that matters: it
runs a ground-truth world under a known transmission process, hides what a researcher cannot see, computes the same
measures on the observed table and on the full truth, and applies the same decision rules to both.

## Ground-truth processes

Agents sit on a follower graph with heavy-tailed in-degree. Each step, active agents read a feed, like posts and
sometimes reshare them with copying. Units are seeded at several followees of one agent so that complex contagion
can start.

| process | copying | spread |
|---|---|---|
| `replicator` | high fidelity (per-token mutation 0.01) | heritable variant fitness scales like and reshare probabilities (Dawkins 1976; Blackmore 1999) |
| `attraction` | lower fidelity (0.08), each copy pulled toward a unit attractor (prob. 0.15 per token) | no heritable fitness (Sperber 1996; Claidière, Scott-Phillips & Sperber 2014) |
| `neutral` | high fidelity | unbiased random copying (Bentley, Hahn & Shennan 2004) |
| `contagion` | high fidelity | reshare only after exposure from >= 2 distinct followees, no content effect (Centola & Macy 2007) |

## Observation layers

- `p_private`: each reshare is private with this probability; it reaches a few followers' inboxes, never appears
  publicly, and its likes are invisible. Public descendants keep their text but lose their visible source.
- `recommender`: an engagement-ranked feed with a global trending pool, or a chronological feed. The ranker changes
  the world itself.
- Exposure: impressions are never in the observed table.
- `deletion`: each public post is deleted with this probability, doubled in the bottom engagement quartile.
- `edge_mode`: `explicit` keeps a parent link only if the parent is visible (as with a quote of a deleted post);
  `ancestor` links to the nearest visible ancestor (thinned chains); `inferred` guesses the most similar earlier
  visible item.

## Use

```bash
memetrics simulate --process attraction --p-private 0.5 --out world.parquet --truth truth.parquet
memetrics distinguishability --n-items 20000 --p-private 0.3 --reps 5 --workers 4
```

```python
from memetrics import observation as ob

log = ob.simulate("attraction", p_private=0.5, seed=1)
obs = ob.observe(log, edge_mode="explicit")
res = ob.evaluate(obs, null_sims=50)       # {"M1": Result, "M2": ..., "N1": ...}
ob.composite(res, calibrated=True)         # inferred process label
ob.sweep(p_grid=(0, 0.3, 0.6, 0.9), reps=10, workers=8)   # tidy table for confusion() and breakdown_fraction()
```

`params_for_size(n_items)` scales the world so the observed table has about `n_items` rows. The composite rule
labels a world "attraction" from M2; "replicator" if M2 says replication and either the content ICC dominates or the
FIT finds selection; "contagion" if the author ICC exceeds the content ICC; otherwise "neutral".

## Limits

The worlds are toys. The simulator shows whether a design can separate processes that differ in the ways built into
it, and how each metric is biased by thinning, ranking and deletion. It does not show that real platforms work this
way. The simulator's own N1 (`observation._n1`) builds unit and author histories from engagement that accrues
after the focal post; the scorecard's N1 lags histories instead.

## Sources

Dawkins 1976; Blackmore 1999; Sperber 1996; Claidière, Scott-Phillips & Sperber 2014; Bentley, Hahn & Shennan
2004; Centola & Macy 2007. See [references](../references.md).
