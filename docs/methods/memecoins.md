# Memecoin heritability

Memecoins attach a market price to a meme, which makes fitness explicit. Launch platforms are also heavily
manipulated (Mongardini & Mei 2025; Szwajcok et al. 2026), so any fitness signal must survive controls for timing
and attention.

## What memetrics computes

- **Families** (`onchain.find_relatives`, `assign_parents`): earlier tokens with the same normalized name or ticker,
  or an image pHash within Hamming distance 6. Relatives launched less than 24 h earlier are treated as siblings
  copying a common source; the parent is the fittest older relative known at launch (or the earliest).
- **Parent-offspring regression** (`onchain.heritability`): OLS slope of offspring fitness on parent fitness, with
  optional controls, a block bootstrap over weeks and a null that permutes parents within launch months.
- **Attention versus inheritance** (`heritability_verdict`): if the slope vanishes once the parent's current
  attention is controlled for and an attention term is positive, the pattern is attention-chasing rather than
  inheritance (compare limited-attention dynamics, Weng et al. 2012).
- **Known-meme matching** (`match_catalogue`): exact alias matches against a catalogue of internet memes, preferring
  precision to recall.

## Default rule

"heritable" if the bootstrap CI is above 0 and the permutation p < alpha/2 (keyword `alpha`); the combined verdict is
"memetic inheritance", "attention-chasing", "no heritability" or "ambiguous".

## Sources

Mongardini & Mei 2025; Szwajcok et al. 2026; Weng et al. 2012. See [references](../references.md).
