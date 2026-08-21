"""Final Ordering Decision: the cost sweep and the matched-availability result.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config  # noqa: E402

page_config("Final Ordering Decision")

st.markdown(
    "**Rule:** order = F⁻¹(q\\*), where q\\* = c_u / (c_u + c_o) - the newsvendor fractile that "
    "minimises expected cost when a missed sale costs c_u and a wasted unit costs c_o. Read off "
    "the calibrated, recovered TFT forecast."
)

ci = load_csv(config.REPORTS_DIR / "cost_savings_ci.csv").set_index("window")
col1, col2 = st.columns(2)
col1.metric("Cheaper than the naive rule - validation weeks",
            f"{ci.loc['validation', 'point']:.1f}%",
            help=f"95% CI: {ci.loc['validation', 'ci_low']:.1f}-"
                 f"{ci.loc['validation', 'ci_high']:.1f}% ({int(ci.loc['validation', 'n_days'])} "
                 "day blocks)")
col2.metric("Cheaper than the naive rule - sealed test week",
            f"{ci.loc['test', 'point']:.1f}%",
            help=f"95% CI: {ci.loc['test', 'ci_low']:.1f}-{ci.loc['test', 'ci_high']:.1f}% "
                 f"({int(ci.loc['test', 'n_days'])} day blocks)")
st.caption(
    "Headline operating point c_u/c_o = 4 (q\\* = 0.80), recovered TFT. Naive = \"order what sold "
    "the same day last week.\" Smaller on the sealed test week than on validation, but still real "
    "at every cost assumption tested (see Choosing an Operating Point)."
)

st.image(str(config.PLOTS_DIR / "cost_sweep.png"), width=700,
        caption="Cost saved vs. the naive rule across all nine cost ratios tested, validation "
                "weeks and the sealed test week.")

st.divider()
st.subheader("Does it hold at matched availability?")
st.markdown(
    "An arm that simply orders more always stocks out less and wastes more, so waste alone can't "
    "separate \"forecasts better\" from \"orders harder.\" Holding every arm to the **same** share "
    "of demand met (95%) removes that: same availability, so waste is what the forecast bought."
)
raw_vs_rec = load_csv(config.REPORTS_DIR / "conclusion_raw_vs_recovered.csv")
st.dataframe(raw_vs_rec, hide_index=True, use_container_width=True)
st.image(str(config.PLOTS_DIR / "waste_comparison.png"), width=700,
        caption="Waste at 95% demand met, raw vs. recovered, both model families, both windows.")
st.caption(
    "At 95% demand met, the recovered arm wastes less than the raw arm in both model families - "
    "a small edge on validation, a much larger one on the sealed test week, where raw sales' "
    "under-forecasting bias costs the most."
)
