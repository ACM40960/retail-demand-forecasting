# Working subset

*Rebuilt 2026-08-09T17:24:59 by `data_io.build_subset`. Regenerated on every rebuild - do not edit by hand.*

Seeded random sample of **100 of 898 stores** (seed 123), keeping every series they carry, all categories.

| | corpus | subset |
|---|---|---|
| series | 50,000 | 5,601 |
| stores | 898 | 100 |
| categories | 32 | 30 |
| products | 865 | 557 |
| train rows | 4,500,000 | 504,090 |
| units/day | 0.9986 | 0.997 |
| censored days | 44.3% | 43.8% |

The subset is **11.2% of the corpus** at ~56 series per store. Matching the corpus on
units/day and censored-day share is what makes it representative: nothing is selected on sales
volume, so no scope restriction has to be declared.

A further **39,207 rows** of shipped eval data are held back for the final evaluation and are
not read by any earlier stage.

Categories present: 0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31

Store IDs: 10, 13, 19, 26, 36, 54, 56, 76, 78, 80, 91, 101, 106, 113, 119, 129, 142, 147, 176, 181, 187, 191, 192, 225, 234, 235, 255, 284, 293, 294, 299, 300, 307, 312, 318, 323, 327, 337, 343, 358, 362, 365, 368, 397, 398, 410, 413, 416, 423, 440, 447, 463, 466, 472, 487, 504, 515, 520, 522, 524, 526, 527, 539, 542, 547, 562, 587, 588, 597, 606, 611, 618, 622, 627, 628, 651, 669, 672, 677, 679, 685, 688, 706, 724, 725, 747, 752, 758, 776, 787, 806, 809, 826, 846, 850, 855, 862, 873, 892, 897
