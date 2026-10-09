# memetrics

Measure the distinctive claims of memetics (discrete units, high-fidelity copying, content-intrinsic fitness,
selection beyond drift, tree-like descent, predictive non-redundancy) on social-media data, next to what rival
frameworks predict and what the data cannot see.

Paper: [Is Memetics a Field?](https://somewhere.systems/research/memetics) (working paper v1, October 2026).

## What it measures and why

Dawkins (1976) coined "meme" for a unit of cultural transmission, and Dennett (1995) and Blackmore (1999) developed
memetics into a programme in which cultural units are replicators whose differential copying explains culture.
Critics argue that human transmission is reconstructive rather than replicative (Sperber 2000; Atran 2001), that
the units are ill-defined, and that the programme produced little empirical work (Edmonds 2005; see the debate in
Aunger 2000 and between Blackmore 2009 and Wimsatt 2009). Several neighbouring fields explain the same phenomena in
other terms: cultural evolution and dual inheritance (Cavalli-Sforza & Feldman 1981; Boyd & Richerson 1985),
cultural attraction (Sperber 1996; Claidière, Scott-Phillips & Sperber 2014), network diffusion under limited
attention (Leskovec, Backstrom & Kleinberg 2009; Weng et al. 2012), neutral random copying (Bentley, Hahn & Shennan
2004), and the study of internet memes as a genre (Shifman 2013). Online platforms are where memetics should do
best: copying can be lossless, templates are discrete, and remixes are public and timestamped. memetrics turns each
distinctive memetic claim into a test with a pre-stated decision rule, runs it on an items table built from public
posts, calibrates it against simulated nulls, and states which rival frameworks predict the same outcome. It does
not presuppose an answer.

## Install

Python 3.11 or newer.

```bash
uv pip install "memetrics @ git+https://github.com/somewheresystems/memetrics"
# extras: [image] OpenCLIP + pHash, [text] sentence-transformers, [bluesky] collector, [onchain] pHash, [all]
uv pip install "memetrics[bluesky] @ git+https://github.com/somewheresystems/memetrics"
```

or from a clone: `git clone https://github.com/somewheresystems/memetrics && cd memetrics && uv sync`.

The core needs only numpy, pandas, pyarrow, scipy, scikit-learn and rapidfuzz, and runs on a laptop CPU. Nothing
calls a remote model API; the optional embedding helpers run open weights locally.

## Five-minute quickstart

The repository ships a small synthetic dataset, `examples/synthetic_items.parquet`: about 4,100 observed posts from
a simulated world in which memes spread as replicators, 30% of reshares were private, and an engagement-ranked feed
decided what people saw. It contains no real user data. It was made with

```bash
memetrics simulate --process replicator --p-private 0.3 --n-items 3000 --seed 1 --out examples/synthetic_items.parquet
```

Score it:

```bash
memetrics scorecard examples/synthetic_items.parquet --out report/ --fast
open report/report.html          # also report/report.md and report/scorecard.json
```

```
M1  text (edit similarity): units / discrete boundaries
M2  all edges: replication / replication; derivatives only (f < 1): mixed / replication
M3  content ICC: intermediate / above shuffled-content null; author ICC (same model): - / content > author
M4  FIT, up to 12 equal-count bins: within nominal 5% / drift-compatible; ...; neutral turnover (top-y lists): - / novelty/anti-conformity bias
M5  3 largest units: - / tree-like 3/3
N1  ...; non-redundancy rule (rows 1 and 2): no memetic gain / -
```

(Progress lines omitted.) Each row shows the literal decision rule, then the calibrated reading (the estimate
against a simulated null). On this replicator world the scorecard recovers discrete units, faithful copying,
tree-like descent and a content effect above the shuffled null (content ICC about 0.3, below the 0.5 "content
dominates" threshold). It does not detect selection with the FIT or a gain from memetic prediction terms at this
size. That is the kind of gap `memetrics distinguishability` quantifies.

Then try the rest of the pipeline:

```bash
# re-derive units from text alone (MinHash near-duplicates) instead of using the simulator's labels
memetrics units examples/synthetic_items.parquet --method minhash --out items_units.parquet

# a world where variants converge on attractors instead
memetrics simulate --process attraction --p-private 0.5 --out attraction.parquet
memetrics scorecard attraction.parquet --out report-attraction/ --fast

# could a dataset of 20,000 posts with 30% private reshares tell the four processes apart?
memetrics distinguishability --n-items 20000 --p-private 0.3 --reps 3 --workers 4
```

The same from Python:

```python
from memetrics import read_items, report, scorecard

items = read_items("examples/synthetic_items.parquet")
card = scorecard.run(items, effort=scorecard.Effort.fast(), p_private=0.3)
report.write(card, "report/")
```

## The measures

| claim (memetics) | measure | default rule | sources |
|---|---|---|---|
| **M1** Variants form discrete units | median Ashman's D of own- versus nearest-other-unit similarity; blend share; boundary-gap ratio | units if D > 2 and blends < 10%; continuum if D < 2 or blends > 25% | Dawkins 1976; Sperber 1996; Shifman 2013 |
| **M2** Copying is high-fidelity, without directional drift | per-step fidelity *f*; convergence slope toward the unit medoid; all edges and derivatives only | replication if median *f* >= 0.9 and slope CI contains 0; attraction if slope < 0 and *f* < 0.9 | Blackmore 1999; Claidière, Scott-Phillips & Sperber 2014; Lerique & Roth 2018 |
| **M3** Spread depends mainly on the meme | ICC of log engagement across identical content posted by different authors, crossed with author, with fixed effects | content dominates if ICC >= 0.5; weak if < 0.2; compare with author ICC | Dennett 1995; Weng et al. 2012; Tan, Lee & Pang 2014; Salganik, Dodds & Watts 2006 |
| **M4** Variant frequencies change by selection | Frequency Increment Test with a Wright-Fisher null and a binning sweep; neutral-model turnover | share rejecting neutrality at FDR 0.05 versus the simulated false-positive share | Feder, Kryazhimskiy & Plotkin 2014; Newberry et al. 2017; Bentley et al. 2007; Karjus et al. 2020 |
| **M5** Lineages are tree-like | delta score and Q-residual of variant distances against simulated trees, networks and stars | none: report where the value falls | Holland et al. 2002; Gray, Bryant & Greenhill 2010; Tehrani 2013 |
| **N1** Memetic terms are non-redundant | held-out Δlog-likelihood and ΔAUC of nested out-of-time models | non-redundant if memetic terms add beyond the rival baseline and beyond cumulative advantage plus content bias | Cheng et al. 2014; Martin et al. 2016; Bakshy et al. 2011 |

Full definitions, calibration and limits: [docs/methods](docs/methods). Every threshold is a keyword argument
(`--rule KEY=VALUE` on the command line). Memecoin families and parent-offspring heritability are in
`memetrics.onchain` ([docs](docs/methods/memecoins.md)).

## What this data cannot see

Public posts and public reactions are a thin slice of transmission. We do not observe:

- **exposure**: who saw an item. Engagement measures adoption without a denominator, so content fitness is
  confounded with how often the ranker showed the content;
- **private and cross-platform sharing**: DMs, group chats, screenshots moved between platforms. Chains are thinned,
  so fidelity is computed across unobserved steps, and tree-likeness is distorted by missing links;
- **deletion**: removed posts are missing, and not at random;
- **offline transmission**: speech, print and broadcast;
- **algorithmic ranking**: the feed decides who sees what, and an engagement ranker can manufacture apparent
  selection.

Every scorecard prints this panel. The observation-model simulator (`memetrics simulate`, `memetrics
distinguishability`, [docs](docs/methods/observation.md)) makes it quantitative: it runs replicator, attraction,
neutral and contagion worlds, hides a chosen share of reshares as private, ranks feeds with a toy recommender,
censors exposure and deletes posts, then shows how each metric is biased and whether the decision rules still
tell the processes apart at your dataset's size.

## Platform readers

```bash
export MEMETRICS_SALT=...   # private; authors are always stored as salted hashes
memetrics ingest reddit dump.jsonl.zst --out reddit.parquet
memetrics ingest reddit --fetch --subreddit memes --after 2024-01-01 --before 2024-02-01 --out reddit.parquet
memetrics ingest bluesky --collect --root jetstream/          # needs memetrics[bluesky]; Ctrl-C to stop
memetrics ingest bluesky --root jetstream/ --start 2026-01-01 --end 2026-01-07 --out bluesky.parquet
memetrics ingest memetracker clust-qt08080902w3mfq5.txt.gz --out memetracker.parquet
```

Network access happens only in `memetrics.io`, only when asked (`--fetch`, `--collect`), read-only and without a
login. Fetchers identify themselves, pause between requests, honour rate-limit headers and HTTP 429, and cache
responses so reruns make no requests. You are responsible for following each platform's terms of service; publish
aggregates, not raw posts.

**Adding a platform**: write `memetrics/io/<platform>.py` with a `to_items(...)` that returns
`memetrics.schema.conform(df)` with the [items-table columns](docs/data-model.md), hash authors with
`memetrics.io.hash_author`, keep network code opt-in, add a fixture test, and wire it into `memetrics ingest`. Details
in [docs/readers.md](docs/readers.md).

## How to cite

memetrics was built for, and every test in it was run in, the working paper
[*Is Memetics a Field? A Pre-Registered Test of Its Distinctive Claims on Its Home Turf*](https://somewhere.systems/research/memetics)
(Bennington 2026). If you use memetrics, please cite it:

```bibtex
@unpublished{bennington2026memetics,
  author = {Bennington, Justin},
  title  = {Is Memetics a Field? {A} Pre-Registered Test of Its Distinctive Claims on Its Home Turf},
  year   = {2026},
  note   = {Working paper v1, Somewhere Systems. https://somewhere.systems/research/memetics.
            Software: memetrics, https://github.com/somewheresystems/memetrics}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Estimator outputs are pinned by regression tests on simulated data; changes
that move them must be explained in the changelog.

## License

MIT. See [LICENSE](LICENSE).
