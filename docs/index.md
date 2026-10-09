# memetrics documentation

- [Data model: the items table](data-model.md)
- [Reading a scorecard](scorecard.md)
- [Adding a platform reader](readers.md)
- Methods
  - [M1 Units](methods/m1-units.md)
  - [M2 Fidelity versus attraction](methods/m2-fidelity.md)
  - [M3 Content-intrinsic fitness](methods/m3-content-fitness.md)
  - [M4 Selection versus drift](methods/m4-selection.md)
  - [M5 Tree-like descent](methods/m5-tree-likeness.md)
  - [N1 Non-redundancy by prediction](methods/n1-nonredundancy.md)
  - [Observation-model simulator](methods/observation.md)
  - [Memecoin heritability](methods/memecoins.md)
- [References](references.md)

Every measure returns a `memetrics.Result(name, estimate, ci_low, ci_high, n, detail, verdict)`. The `verdict` applies
the default decision rule; thresholds are keyword arguments. Every stochastic function takes `seed`.

Statistical building blocks (Ashman's D, silhouettes, HDBSCAN, moment and REML variance components, the pigeonhole
bootstrap, Benjamini-Hochberg) are cited in the docstrings of the modules that implement them. The methods pages
cite the substantive literature.
