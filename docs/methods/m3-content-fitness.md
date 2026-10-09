# M3 Content-intrinsic fitness

**Claim.** From the meme's-eye view, spread depends mainly on the meme (Dawkins 1976; Dennett 1995; Blackmore
1999). Rival accounts locate much of the variance elsewhere: in who posts, where and when (Bakshy et al. 2011;
Lakkaraju, McAuley & Leskovec 2013), in limited attention and competition (Weng et al. 2012; Gleeson et al. 2014),
and in social influence that makes success unpredictable (Salganik, Dodds & Watts 2006). Content effects do exist:
wording matters when topic and author are held fixed (Tan, Lee & Pang 2014), and content biases shape which items
are retold (Berger & Milkman 2012; Acerbi & Tehrani 2018; Stubbersfield, Tehrani & Flynn 2015).

## What memetrics computes

A natural experiment: identical content (pHash Hamming <= 4 for images, identical normalized text otherwise)
posted independently by at least two different authors. On `y = log1p(max(engagement, 0))`:

`y = fixed effects + a_content + a_author + e`, with crossed random effects.

Fixed effects: community, hour of week, era (month), log repost order (a novelty penalty), the author's
leave-one-out mean and a new-author flag. The variance components are estimated by pairwise moment U-statistics
for crossed designs (`fitness.content_icc`, the default) or by REML (`textstats.reml_icc`, an
estimator-robustness check). The content ICC is `s2_content / (s2_content + s2_author + s2_resid)`; the author ICC
is reported from the same fit, with a bootstrap CI for their difference.

## Default rule

"content dominates" if the content ICC >= 0.5; "weak content fitness" if < 0.2; otherwise "intermediate".
Keywords: `strong`, `weak`.

## Calibrated reading

Content labels are shuffled within community and the ICC recomputed. The content ICC counts as real only above the
shuffled 97.5th percentile. The scorecard also reports whether the content ICC exceeds the author ICC.

## Limits

- **Exposure is not observed.** Engagement counts adoptions without a denominator, so a ranker that showed some
  copies more often inflates the content ICC as well as the author ICC. Bluesky's public likes and reposts help
  only partly.
- Identical content posted by different authors is not a random sample of content.
- The CI treats the fixed effects as known, which is slightly optimistic.

## API

`fitness.content_icc`, `fitness.crossed_components`, `fitness.residualize`, `fitness.simulate_crossed`,
`textstats.reml_icc`.

## Sources

Dawkins 1976; Dennett 1995; Blackmore 1999; Bakshy et al. 2011; Lakkaraju, McAuley & Leskovec 2013; Weng et al.
2012; Gleeson et al. 2014; Salganik, Dodds & Watts 2006; Tan, Lee & Pang 2014; Berger & Milkman 2012; Acerbi &
Tehrani 2018; Stubbersfield, Tehrani & Flynn 2015. See [references](../references.md).
