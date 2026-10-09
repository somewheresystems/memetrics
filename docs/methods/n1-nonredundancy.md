# N1 Non-redundancy by prediction

**Claim.** If memetics adds something its rivals lack, memetic replicator terms should improve out-of-time
prediction of spread beyond the best rival baseline. Cascade-prediction work shows how much author, network,
timing and early-adoption features already explain, and where the limits of prediction lie (Bakshy et al. 2011;
Cheng et al. 2014; Martin et al. 2016; Goel et al. 2016).

## What memetrics computes

Items in units are sorted by time; the first 60% train, the rest test. The outcome is top-decile engagement
(threshold from the training data). Nested logistic models (standardized features) are compared on held-out data:

| block | features |
|---|---|
| B rival baseline | author's prior post count and prior mean engagement, community prior mean, hour of week, time, items posted in the previous 24 h (attention competition) |
| CA cumulative advantage | the unit's prior count and prior engagement |
| CB content bias | length, URL, hashtags, image flag: content features without unit identity |
| M memetic terms | heritable unit fitness (shrunken mean engagement of earlier instances), unit age, mutation distance from the parent, has-parent flag |

Unit and author histories use only items older than a lag (default 48 h) so engagement that is still accruing is
not used. Each comparison reports the mean per-item held-out difference in log-likelihood and the AUC difference,
with cluster-bootstrap CIs over units.

## Default rule

"non-redundant" if B -> B+M and B+CA+CB -> B+CA+CB+M both add (both CIs above 0); "reproduced by rivals" if only the
first does; otherwise "no memetic gain". Keyword: `min_auc_gain`.

## Calibrated reading

There is no simulated null. The reverse nesting (M -> M+B) shows how much the rival baseline adds beyond the
memetic terms.

## Limits

Unit fitness estimated from engagement inherits every exposure and ranking confound of M3. Prediction gains are
evidence of non-redundancy only if they survive the cumulative-advantage and content-bias terms.

## API

`predict.prior_unit_features`, `predict.nested_compare`, `predict.nonredundancy_verdict`.

## Sources

Cheng et al. 2014; Martin et al. 2016; Bakshy et al. 2011; Goel et al. 2016. See [references](../references.md).
