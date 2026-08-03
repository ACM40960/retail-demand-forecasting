# Working subset

*Rebuilt 2026-08-04T00:31:14 by `data_io.build_subset`. Regenerated on every rebuild - do not edit by hand.*

Seeded random sample of **30 of 898 stores** (seed 0), keeping every series they carry, all categories.

| | corpus | subset |
|---|---|---|
| series | 50,000 | 1,522 |
| stores | 898 | 30 |
| categories | 32 | 26 |
| products | 865 | 372 |
| train rows | 4,500,000 | 136,980 |
| units/day | 0.9986 | 1.0472 |
| censored days | 44.3% | 43.8% |

The subset is **3.0% of the corpus** at ~50 series per store. Matching the corpus on units/day and censored-day share is what makes it representative: nothing is selected on sales volume, so no scope restriction has to be declared.

A further **10,654 rows** of shipped eval data are held back for the final evaluation and are not read by any earlier stage.

Categories present: 0, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31

Store IDs: 2, 14, 30, 35, 65, 153, 235, 246, 268, 351, 443, 445, 481, 496, 535, 554, 559, 570, 597, 644, 655, 686, 714, 726, 739, 766, 803, 830, 857, 894
