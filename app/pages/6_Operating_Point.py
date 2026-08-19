"""Choosing an Operating Point - poster card 6/7: three quantile picks against the naive rule on
three metrics at once (posterplan.md sec 7).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config, plots, poster  # noqa: E402

page_config("Choosing an Operating Point")

st.markdown(
    "Three quantiles, three priorities - pick based on what the business is optimizing for, not "
    "on which number looks smallest. All three read the **same** calibrated, recovered TFT "
    "forecast on the validation weeks; only the order quantile changes."
)

qmap = load_csv(config.REPORTS_DIR / "conclusion_quantile_map.csv").set_index("order at")

# Reconstructed the same way poster.fig_quantile_anchors does (src/utils/poster.py:352-368): the
# naive-rule row isn't in this CSV, but every row already reports its own waste relative to it,
# so it's recovered algebraically from the one row where that column is exactly 0 by construction.
neutral = qmap.loc["q0.513"]
naive = {"ran out %": 41.7, "demand met %": 79.9,
        "waste % of demand": neutral["waste % of demand"] / (1 - neutral["waste vs naive %"] / 100)}

rows = [{"scenario": "naive", **naive}]
for label, q in poster._QUANTILE_PICKS:   # same 3 anchors the poster names - never re-typed
    r = qmap.loc[q]
    rows.append({"scenario": label, "ran out %": r["ran out %"],
                "demand met %": r["demand met %"], "waste % of demand": r["waste % of demand"]})
table = pd.DataFrame(rows).set_index("scenario")

st.pyplot(plots.plot_operating_points(table))
st.caption(
    "Order quantiles behind each scenario: waste-focused q0.461, balanced q0.513, "
    "stockout-focused q0.900 - all three read the same forecast; only the order quantile changes."
)
with st.expander("Table"):
    st.dataframe(table.style.format("{:.1f}"), use_container_width=True)

st.markdown(
    "- **Stockout-focused (q0.90):** cuts empty-shelf days by over 7x, at a real waste cost - a "
    "choice, not a free win.\n"
    "- **Balanced (q0.513):** doesn't waste more than the naive rule, while cutting empty days by "
    "7 points and lifting demand met by 8.6 points. Needs no cost assumption at all.\n"
    "- **Waste-focused (q0.461):** the lowest quantile that still matches the naive rule's "
    "stockout rate - 19% less waste than the naive rule, before any trade-off begins."
)
st.caption(
    "Important nuance: the sweep's absolute lowest-waste quantile (q0.33) stocks out 60% of the "
    "time - worse than the naive rule. \"Lowest waste\" and \"best for waste\" are different "
    "claims; q0.461 is the honest answer to the second one."
)
