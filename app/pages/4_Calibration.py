"""Calibration Results - poster card 4 (are the bands honest?), plus the content posterplan.md
cut for space: the full 4-arm x 2-band x 2-window table (16 rows, only the headline arm made it
onto the poster).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config  # noqa: E402

page_config("Calibration: are the bands honest?")

st.markdown(
    "A forecast quantile band is only useful if it means what it says - an \"80% band\" should "
    "contain the truth 80% of the time. Split conformal prediction widens each band by one fitted "
    "offset so that it does, checked here against what actually happened, not assumed."
)

cal = load_csv(config.REPORTS_DIR / "conclusion_calibration.csv")
headline = cal[(cal["arm"] == "tft_recovered") & (cal["band"] == "80%")]
val_row = headline[headline["window"] == "validation"].iloc[0]
test_row = headline[headline["window"] == "test"].iloc[0]

col1, col2 = st.columns(2)
col1.metric("80% band coverage - validation weeks",
            f"{val_row['after']:.1%}", delta=f"from {val_row['before']:.1%} uncorrected")
col2.metric("80% band coverage - sealed test week",
            f"{test_row['after']:.1%}", delta=f"from {test_row['before']:.1%} uncorrected")
st.caption(
    f"Test-week coverage ({test_row['after']:.1%}) sits inside its own bootstrap confidence "
    f"interval ({test_row['ci_low']:.1%}-{test_row['ci_high']:.1%}) for the 80% target - the "
    "correction transfers to genuinely unseen weeks, not just the window it was fitted on."
)

st.divider()
st.subheader("Why two bands, not one?")
st.markdown(
    "**Each band is calibrated separately because coverage error isn't the same shape everywhere "
    "in the distribution.** The 80% band corrects the inner pair (q10/q90); the 95% band corrects "
    "the outer pair (q0.025/q0.975) - two different offsets, each fitted to the miscoverage its "
    "own pair actually has. A single offset tuned to fix the 80% band would not correctly widen "
    "the 95% band's much wider tails."
)
st.markdown(
    "**Ordering needs both, together, because the order percentile isn't fixed at 10/50/90.** "
    "The newsvendor rule can ask for any quantile from 0 to 1, depending on the cost ratio. For "
    "q\\* between 0.10 and 0.90 - most of the cost ratios tested - the order quantity "
    "**interpolates** between the corrected 80%-band points. For a very stockout-averse policy "
    "(q\\* above 0.90), it instead **interpolates toward the corrected 95%-band's q0.975** - a real "
    "calibrated point out in the tail - rather than linearly extrapolating past q90 using only the "
    "much shorter 80%-band segment, which would guess the tail's shape instead of measuring it."
)

with st.expander("Full calibration table - every arm, both bands, both windows"):
    show = cal.copy()
    for c in ("claims", "before", "after", "ci_low", "ci_high", "offset", "drift"):
        show[c] = (show[c] * 100).round(1)
    show = show.rename(columns={"claims": "target %", "before": "before %", "after": "after %",
                                "ci_low": "ci low %", "ci_high": "ci high %",
                                "offset": "offset %", "drift": "drift/day %",
                                "nominal_in_ci": "target inside CI"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(
        "16 rows: 4 arms (TFT/XGBoost x raw/recovered) x 2 bands (80%/95%) x 2 windows "
        "(validation/sealed test week). The poster kept only the headline row above; this is "
        "the rest of it."
    )
