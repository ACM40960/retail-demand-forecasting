"""Recovery Results: how the demand a stockout hides is reconstructed, and how well.

Carries what the poster had no room for - all four censoring bands rather than the headline one, and
the stockout-timing figure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config  # noqa: E402

page_config("Recovery: recovering the demand a stockout hides")

st.markdown(
    "**Stage 1** learns the hourly in-stock demand rate with a **LightGBM regressor trained on "
    "Tweedie loss** - the shipped choice out of six candidates (two other LightGBM losses, "
    "XGBoost, a Poisson GLM, and a no-model control) scored on held-out full-shelf days; see "
    "*which recovery model was chosen* below for the full comparison. **Stage 2** then substitutes "
    "that hourly prediction into stocked-out hours only - floored at what was actually sold - and "
    "sums the day back up, leaving every hour the shelf was never empty untouched."
)
st.markdown(
    "**What counts as a stockout, and how it's calculated:** each product-day carries a 24-value "
    "hourly in-stock flag (`hours_stock_status`, 1 = shelf recorded empty that hour). "
    "`stock_hour6_22_cnt` counts how many of the 16 trading hours (06:00-22:00) were flagged empty, "
    "and the day is marked **censored** the moment that count is above zero - one empty hour is "
    "enough to mark the *whole day*, not just that hour. That's why 43.8% of product-days come out "
    "censored (Introduction page) even though, as the timing check below shows, most outages don't "
    "run the full 16 hours."
)

recovery = load_csv(config.REPORTS_DIR / "conclusion_recovery.csv").set_index("what")["value"]
by_band = load_csv(config.REPORTS_DIR / "conclusion_by_band.csv").rename(
    columns={"Unnamed: 0": "censoring band"})
by_band_test = load_csv(config.REPORTS_DIR / "conclusion_by_band_test.csv").rename(
    columns={"Unnamed: 0": "censoring band"})
demand_shift = load_csv(config.REPORTS_DIR / "conclusion_demand_shift.csv").set_index(
    "period")["avg_sale_amount_per_clean_day"]

col1, col2 = st.columns(2)
col1.metric("Recovered-demand model WAPE", recovery["day WAPE on held-out full-shelf days"],
            help="Scored on held-out full-shelf days, where recorded sales ARE true demand.")
col2.metric("vs. no-model control", recovery["no-model control (series_hour_mean)"],
            delta=f"{recovery['better than the control by']} better",
            help="A per-series-hourly-average baseline with no learned structure at all.")
st.caption(
    "Confirms the recovered-demand model captures real demand structure, not noise - it isn't "
    "just smoothing towards an average."
)

st.image(str(config.PLOTS_DIR / "recovery_bias_and_example.png"), width=700,
        caption="Recorded sales vs. recovered demand for the single most-censored series - the "
                "shaded days are where the shelf went empty and the till under-counted demand.")

st.divider()
st.subheader("Does recovery change the forecast where it should?")
st.caption(
    "The 4 bands below group **products** (series) by how often **that product's own** days were "
    "censored, over its training-period history - 0-25% / 25-50% / 50-75% / ≥75% of *its* days. "
    "The percentage is always a per-product day-share, never a share of all products or all rows."
)
chronic = by_band[by_band["censoring band"] == ">=75%"].iloc[0]
st.markdown(
    f"**Chronic-stockout products** (their own days censored ≥75% of the time, "
    f"n={int(chronic['n_scored']):,} scored rows): WAPE improves {chronic['tft WAPE change %']:.1f}% "
    f"(TFT) / {chronic['xgb WAPE change %']:.1f}% (XGBoost). Bias (WPE) is corrected from roughly "
    f"{chronic['tft WPE raw'] * 100:.0f}% to near zero "
    f"({chronic['tft WPE recovered'] * 100:+.1f}%)."
)
st.caption(
    "Pooled accuracy barely moves across the other bands - that's the pooled metric's fault, not "
    "the recovery layer's: most product-days rarely stock out, so there's little bias to correct."
)

with st.expander("See all 4 censoring bands"):
    st.dataframe(by_band, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Validation looks mixed - the sealed test week doesn't. Why?")
all_val = by_band[by_band["censoring band"] == "ALL"].iloc[0]
all_test = by_band_test[by_band_test["censoring band"] == "ALL"].iloc[0]
lift = 100 * (demand_shift["test"] / demand_shift["training"] - 1)
st.markdown(
    f"On validation, recovery's **pooled** WAPE change is essentially flat "
    f"({all_val['tft WAPE change %']:+.2f}% TFT / {all_val['xgb WAPE change %']:+.2f}% XGBoost) - "
    f"it only clearly helps the chronic band above. Repeating the identical by-band check on the "
    f"**sealed test week** improves in every band, pooling to "
    f"**{all_test['tft WAPE change %']:.1f}% (TFT) / {all_test['xgb WAPE change %']:.1f}% (XGBoost)**."
)
st.caption(
    f"Why: raw sales under-forecast (negative WPE) in every band that week, not just the chronic "
    f"one - the test week's realised demand ran {lift:+.1f}% above the training window the models "
    f"were fitted on, rising monotonically training → validation → calibration → test, so "
    f"it isn't a one-window fluke. Raw's under-forecast bias, tuned on the lower-demand training "
    f"window, gets more expensive everywhere once demand shifts up - not just where stockouts were "
    f"already chronic."
)
with st.expander("See all 4 censoring bands, test week"):
    st.dataframe(by_band_test, hide_index=True, use_container_width=True)

with st.expander("When do stockouts happen, and does the shelf ever stock back the same day?"):
    st.image(str(config.PLOTS_DIR / "stockout_timing.png"), width=500,
            caption="First hour of the trading day (06:00-22:00) a stockout is recorded.")
    st.caption(
        "13.4% of stockout days are already under way at opening (06:00) - yesterday's stock never "
        "got replenished overnight - and the rest build through the day, peaking in the mid-to-late "
        "afternoon."
    )

    recovery_by_start = load_csv(config.REPORTS_DIR / "stockout_recovery_by_start.csv")
    st.dataframe(recovery_by_start, hide_index=True, use_container_width=True)
    st.caption(
        "Once a shelf goes empty, it stays empty: only **12.6%** of stockout days see it stocked "
        "back before close, and when it does, it's almost always exactly one restock - 94.2% of "
        "stockout days have a single unbroken empty stretch, not several short flickers. Outages "
        "already under way at opening recover more often (28.4%) than ones that start later in the "
        "day (10.2%) - a morning gap is more likely to be *yesterday's stock arriving late* than a "
        "shelf that runs out and stays out. This is why the recovery model carries a predicted rate "
        "forward for the rest of the day, rather than assuming a same-day return to stock."
    )

with st.expander("Which recovery model was chosen, and why"):
    model_comp = load_csv(config.REPORTS_DIR / "recovery_model_comparison.csv").rename(
        columns={"Unnamed: 0": "model"})
    st.dataframe(model_comp, hide_index=True, use_container_width=True)
    st.caption(
        "lightgbm_tweedie wins on WAPE and MAE among models scored honestly on held-out full-shelf "
        "days; series_hour_mean (no learned structure) is the control that shows the model is "
        "capturing real signal, not just its bias-correction multiplier."
    )
