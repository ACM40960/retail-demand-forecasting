"""One store, recommended for all its products - the practical finale.

Illustrative only (n=1): the 100-store aggregate on the other pages is the evidence; this is what
that evidence looks like for one real, typical store's real products on the sealed test week. No
recompute of anything expensive - reads the already-calibrated test-week forecast and the
already-recovered daily data, then applies the same vectorized newsvendor arithmetic
(`orders.critical_fractile` / `order_quantity` / `naive_orders` / `simulate`) already used
everywhere else in this project, filtered down to one store.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import page_config  # noqa: E402
from src import orders  # noqa: E402
from src.utils import data_io, plots, store_view  # noqa: E402

page_config("One Store, Recommended for All Its Products")

st.markdown(
    "Everything on the earlier pages is a 100-store aggregate. This page is the opposite zoom: "
    "**one real store**, its real products, the sealed test week (2024-06-26 to 2024-07-02) - "
    "what the model would actually have told it to order, day by day."
)
st.info(
    "**Illustrative only (n=1).** Whichever store is showing below is not new evidence - the "
    "100-store aggregate on the other pages is the actual result. FreshRetailNet-50K also has no "
    "product-name mapping anywhere, so products below are shown as \"Product #id\", not named "
    "items.",
    icon="ℹ️",
)


@st.cache_data
def _load_recovered_daily():
    return data_io.load("recovered")


@st.cache_data
def _load_test_forecast():
    return orders.load_forecast(period="test", family="tft", target="recovered")


daily = _load_recovered_daily()
test_df, qcols = _load_test_forecast()
train_recovered = daily[daily["split"] == "train"]

stats = store_view.store_stats(daily)
default_store = store_view.pick_store(stats)
available_stores = sorted(set(stats["store_id"]) & set(test_df["store_id"].unique()))

default_idx = available_stores.index(default_store) if default_store in available_stores else 0
store_id = st.selectbox(
    "Store to walk through:", available_stores,
    index=default_idx,
    format_func=lambda s: f"Store #{s}" + ("  (default: closest-to-median store - see EDA)"
                                            if s == default_store else ""),
    help="Defaults to the store closest to the subset median on censored-day share (see the EDA "
        "page for the full ranking). Pick any other store to compare it against that default.",
)
store_row = stats[stats["store_id"] == store_id].iloc[0]

st.subheader(f"Store #{store_id}")
if store_id == default_store:
    st.caption(
        f"Picked as the closest-to-median store in the 100-store subset: "
        f"{store_row['censored_share']:.1%} of its product-days go out of stock (subset median "
        f"{stats['censored_share'].median():.1%}), carrying {int(store_row['n_products'])} "
        f"products (subset median {int(stats['n_products'].median())})."
    )
else:
    pct_rank = 100 * (stats["censored_share"] < store_row["censored_share"]).mean()
    st.caption(
        f"You've switched away from the default pick (store #{default_store}, the closest-to-"
        f"median store). Store #{store_id} is censored on {store_row['censored_share']:.1%} of "
        f"its product-days - a higher stockout rate than {pct_rank:.0f}% of the 100 stores - and "
        f"carries {int(store_row['n_products'])} products. See the EDA page to rank every store."
    )

ratio = st.select_slider(
    "If a stockout costs this many times more than a wasted item:",
    options=orders.DEFAULT_RATIOS, value=orders.HEADLINE_RATIO,
    format_func=lambda v: f"{v:g}x",
)
q_star = orders.critical_fractile(ratio, 1.0)

store_slice = test_df[test_df["store_id"] == store_id].copy()
store_slice["model_order_quantity"] = orders.order_quantity(store_slice, q_star, qcols)
store_slice["naive_order_quantity"] = orders.naive_orders(store_slice, train_recovered=train_recovered)

result = orders.run(forecast_df=store_slice, qcols=qcols, period="test", regime="observed",
                    c_u=ratio, c_o=1.0, save=False, verbose=False)
per_day, kpi_result = result["per_day"], result["kpi"]
kpi = kpi_result["headline"]

waste_pct = 100 * per_day["model_waste"].sum() / per_day["sale_amount"].sum()

st.markdown(f"#### This store, at q\\* = {q_star:.2f}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stockout days", f"{kpi['stockout_pct']:.1f}%")
c2.metric("Demand met", f"{kpi['demand_met_pct']:.1f}%")
c3.metric("Cheaper than naive", f"{kpi['cost_vs_naive_pct']:.1f}%")
c4.metric("Waste", f"{waste_pct:.1f}%",
         help="Model's wasted units as a share of this store's total demand, full-shelf days only.")
st.caption(
    f"{kpi_result['n_scored']} of this store's product-days had a full shelf this week and could "
    "be honestly scored against what actually sold."
)

st.markdown("#### Whole store, test week")
st.pyplot(plots.plot_store_week(store_slice, store_id))
st.caption(
    "Grey shading = share of this store's products stocked out that day (darker = more of the "
    "store was out) - with dozens of products, some are stocked out most days, so this is a "
    "share, not a yes/no flag."
)

st.divider()
st.subheader("Three of its products")

store_daily = daily[daily["store_id"] == store_id]
picks = store_view.pick_products(store_daily, per_day)

CARDS = [("chronic", "Most chronic stockouts"), ("typical", "A typical product"),
        ("lowest_waste", "Its lowest-waste product")]

for key, label in CARDS:
    pid = picks[key]
    category = int(store_daily.loc[store_daily["product_id"] == pid, "first_category_id"].iloc[0])
    st.markdown(f"##### {label} - Product #{pid} (category {category})")

    week = store_slice[store_slice["product_id"] == pid]
    st.pyplot(plots.plot_store_product_week(week, store_id, pid))

    scored = per_day[per_day["product_id"] == pid]
    n_obs = len(scored)
    if n_obs == 0:
        st.caption("No full-shelf days this week - nothing to score under the observed-demand regime.")
    else:
        stockout_pct = 100 * (scored["model_shortfall"] > 0).mean()
        waste = scored["model_waste"].sum()
        naive_cost, model_cost = scored["naive_cost"].sum(), scored["model_cost"].sum()
        cost_vs_naive = 100 * (1 - model_cost / naive_cost) if naive_cost else float("nan")
        st.caption(
            f"{n_obs} of 7 days scored (full-shelf only) - {stockout_pct:.0f}% stockout days, "
            f"{waste:.1f} units wasted, {cost_vs_naive:+.0f}% cost vs. naive at this ratio."
        )

st.divider()
st.markdown(
    "**Reminder:** this is one store, illustrative only. The real evidence is the 100-store "
    "aggregate - **34.7% cheaper than the naive rule on the validation weeks, 28.0% cheaper on "
    "the sealed test week** (see Final Ordering Decision)."
)
