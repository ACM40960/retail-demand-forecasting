"""Limitations: the handful worth putting on a poster, then the fuller list.

Held as a constant rather than parsed out of a notebook at runtime, because this is fixed narrative
text and not a data artifact. Keep it in step with README section 7 by hand.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import page_config  # noqa: E402

page_config("Limitations")

st.markdown("**A proof of concept on a dataset that records sales, not stock.**")
st.markdown(
    "- Demand during a stockout is fundamentally unobservable - no test can close that gap.\n"
    "- The chronic-stockout headline result rests on 1,375 rows from one seeded subset draw.\n"
    "- The cost ratio is an assumption, swept rather than known - real prices would collapse the "
    "sweep to one number.\n"
    "- Everything degrades with distance from the calibration window - the test-week figures are "
    "the honest ones, not the more optimistic validation figures."
)

FULL_LIMITATIONS = [
    ("Demand during a stockout is unobservable",
     "the load-bearing one; validation only on full-shelf days; no test can close that gap"),
    ("`recovered_demand` is a floor, not a measurement", "closer than raw sales, errs short if it errs"),
    ("Nothing corrects the recovered forecast's centring",
     "runs ~8.5% high; conformal only widens; q* over-orders ~0.15 quantile (~10% excess cost)"),
    ("Headline recovery band is 1,375 rows",
     "smallest cell, single seeded subset (11.2% of corpus)"),
    ("Recovery's validation is a worst case",
     "held-out days blanked across all 16 hours, harsher than typical censored days"),
    ("Stage-1 history features are themselves censored",
     "lag7/roll7 biased downward, same direction as the floor"),
    ("Raw TFT's tuning table was never saved", "encoder length borrowed/reconstructed"),
    ("Test-week coverage rests on 7 days",
     "day-block bootstrap over 7 blocks is coarse vs. validation's 14"),
    ("One demand regime scores against recovered_demand", "reported as a bound, not chosen as neutral"),
    ("Cost ratio is an assumption", "swept across nine values only"),
    ("Waste is arithmetic, never observed", "spoilage not recorded in dataset"),
    ("Newsvendor is single-period", "shelf life unknown, no carryover"),
    ("Weather enters as realised, not forecast",
     "flatters absolute accuracy; relative comparisons hold"),
    ("TFT's ~4% accuracy edge favours it",
     "early-stops on scored window; tree stops on held-back training days"),
    ("SARIMA fitted on 30 sampled series", "reference point, not like-for-like"),
    ("Ordering results carry no error bars", "only the calibration layer is quantified"),
    ("Units are normalised", "no saving quotable in currency"),
]

with st.expander(f"Full list ({len(FULL_LIMITATIONS)} items, from the notebook's fuller table)"):
    for title, effect in FULL_LIMITATIONS:
        st.markdown(f"**{title}** - {effect}")

st.divider()
st.caption(
    "Future work: stock-on-hand and delivery data would turn the recovery assumption into a "
    "direct measurement; recorded spoilage would make waste observed rather than arithmetic."
)
