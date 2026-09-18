# ACV data audit

All six training workbooks and the supplied test workbook were opened. Each has
one sheet. Training sheets are `Sheet1`; the test sheet is
`故障案例3-0620_20210624`. No existing ACV code was found.

| Case | Rows | Columns | Train | Faulty car | Typical interval |
|---|---:|---:|---|---|---|
| 01 | 6,999 | 67 | A:0620 | 01 | 30 s |
| 02 | 9,187 | 67 | A:0620 | 02 | 30 s |
| 03 | 8,310 | 67 | A:0619 | 03 | 30 s |
| 04 | 22,262 | 483 | B:0208 | 01 | 10 s |
| 05 | 6,972 | 67 | C:0407 | 04 | 30 s |
| 06 | 3,263 | 67 | C:0408 | 06 | 30 s |
| Test | 9,082 | 67 | A:0620 | Unpublished | 30 s |

The 48 training car examples contain six faults and 42 normal cars (1:7).
Faulty ID frequencies are 01 twice and 02, 03, 04, 06 once each. Car IDs are
therefore excluded as features. Filenames and sheet names are excluded too.

Training timestamps are ordered, with no invalid timestamps, exact duplicates,
or duplicate timestamps. Gaps occur: not every interval is the nominal interval.
Case 04 has 22,213 ten-second intervals, plus irregular shorter and longer gaps.
This differs from the Info Kit's general statement of 30-second sampling.
No resampling or frequency assumptions are used by the baseline.

Raw missing-cell counts are zero because missingness often appears as strings
`None` and `Invalid`, not empty cells. Mixed numeric/text temperature columns
have object dtype. Indoor invalid counts per car are 677 in case 01, 805 in
case 02, zero in case 03, 674 in case 05, and 448 in case 06. In case 04,
cars 01–03 have 413 invalid readings, car 04 has 420, and **cars 05–08 have
22,262 invalid readings each (all missing)**. Numeric-typed columns contain no
infinities; numeric coercion also masks infinities in mixed columns during cleaning.

Case 04 has 242 constant columns, versus 2 in cases 01, 02, 05, 06 and 24 in
case 03, including metadata. Schema width is not an indication of usable signal.
Detailed pressure and status channels are omitted from the common MVP features.
Ambient sensor readings in model C may also be invalid and are optional.

Cabin ranges are usually around 17–34 degrees Celsius. Zero values occur for the
fault-labelled car in cases 01–03 and may be sensor artefacts; they are not used
as a direct fault rule. Broad bounds preserve zeros and robust quantiles limit
their effect. No semantic meaning for zero is supplied by the documentation.
Temperature drift varies with operating case: case 06 cabin means rise roughly
1.2–1.6 degrees between halves, whereas case 02 generally cools. Peer comparison
helps remove shared operating effects. Case 06's faulty car is consistently
warmer; not every case has such a strong signature. See `eda_features.svg`.

`eda.json` records per-file shape, sheet, dtypes, timestamps, leading sampling
interval counts, missingness, duplicates, constant-channel counts, and per-car
range, median, standard deviation, and first/second-half means.
`training_features.csv` records all features and labels for inspection. Neither
file is required for inference. Test EDA is performed only after model fitting;
it is never used to fit, tune or select the baseline.

The test timestamps are ordered and unique, with no invalid timestamps, and
8,888 intervals of 30 seconds. Each car has 1,068 nonnumeric indoor readings.
Its numeric dtype channels have no infinities. The same cleaning and feature
implementation successfully processes this case without any test-specific changes.

Leave-one-train-out validation ranks the true car 1, 1, 1, 3, 2, 1 for cases
01–06, giving mean rank-decay 0.9375. Cases 04 and 05 show the limits of this
small cross-fleet dataset. There is no independent test score without test labels.
