# Working subset

*Rebuilt 2026-08-04T22:01:31 by `data_io.build_subset`. Regenerated on every rebuild - do not edit by hand.*

Seeded random sample of **30 of 898 stores** (seed 123), keeping every series they carry, all categories.

| | corpus | subset |
|---|---|---|
| series | 50,000 | 1,816 |
| stores | 898 | 30 |
| categories | 32 | 28 |
| products | 865 | 401 |
| train rows | 4,500,000 | 163,440 |
| units/day | 0.9986 | 0.9831 |
| censored days | 44.3% | 43.9% |

The subset is **3.6% of the corpus** at ~60 series per store. Matching the corpus on
units/day and censored-day share is what makes it representative: nothing is selected on sales
volume, so no scope restriction has to be declared.

A further **12,712 rows** of shipped eval data are held back for the final evaluation and are
not read by any earlier stage.

Categories present: 0, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31

Store IDs: 13, 22, 46, 154, 161, 191, 192, 215, 218, 223, 238, 244, 292, 305, 367, 396, 397, 456, 516, 593, 665, 698, 708, 714, 726, 736, 764, 790, 793, 814
