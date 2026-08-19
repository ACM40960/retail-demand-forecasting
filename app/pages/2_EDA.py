"""EDA - how much the censoring problem varies store to store and product to product: the
population underneath every scored number on the LATER pages (Recovery onward), and the evidence
behind which store the One Store page (app/pages/10_One_Store.py) walks through by default.
Placed right after Pipeline, before any modelling page, so a reader sees how uneven the problem
actually is before seeing how it's addressed.

Reads the already-recovered daily parquet only. The `store_view.*_stats` rollups are cheap pandas
groupbys (counts and means), not a model and not a recompute of anything upstream - same pattern
app/pages/10_One_Store.py already uses for `store_stats`/`pick_store`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import page_config  # noqa: E402
from src.utils import data_io, plots, store_view  # noqa: E402

page_config("EDA: how stockout varies store to store and product to product")

st.markdown(
    "Every other page scores **series** (store x product) or pools everything into one number. "
    "This page is the population underneath those numbers: how much a store's stockout rate "
    "varies from store to store, how much a product's varies from product to product - across "
    "the full 100-store, 5,601-series subset, over its **full history** (train+eval, not just "
    "one scored week). Pick a direction and a store or product below rather than reading a fixed "
    "list - both charts respond."
)


@st.cache_data
def _load_recovered_daily():
    return data_io.load("recovered")


daily = _load_recovered_daily()
stores = store_view.store_stats(daily)
products = store_view.product_stats(daily)
picked_store = store_view.pick_store(stores)
picked_censored = stores.set_index("store_id").loc[picked_store, "censored_share"]

# ------------------------------------------------------------------------------- per store
st.divider()
st.subheader("Per store: how much does the stockout rate vary store to store?")

col1, col2, col3 = st.columns(3)
col1.metric("Highest censored-day share", f"{stores['censored_share'].max():.1%}")
col2.metric("Subset median", f"{stores['censored_share'].median():.1%}")
col3.metric("Lowest censored-day share", f"{stores['censored_share'].min():.1%}")

st.pyplot(plots.plot_censoring_distribution(
    stores["censored_share"], unit="store", highlight_value=picked_censored,
    highlight_label=f"One Store pick (#{picked_store})"))
st.caption(
    "Every store carries dozens of products, so no store is ever fully stocked - the spread here "
    "is in HOW OFTEN. The blue line is the store the One Store page walks through by default; see "
    "*choosing the One Store view* below for why it sits where it does."
)

st.markdown("###### Rank the stores")
c1, c2 = st.columns([1, 2])
with c1:
    store_dir = st.radio("Show stores with the:", ["highest", "lowest"], horizontal=True,
                        format_func=lambda d: f"{d} stockout rate", key="store_dir")
    store_n = st.slider("How many to show", 5, 30, 15, key="store_n")
with c2:
    store_lookup = st.selectbox(
        "Or look up one store directly:", ["(none)"] + sorted(stores["store_id"].tolist()),
        format_func=lambda v: "Choose a store..." if v == "(none)" else f"Store #{v}",
        key="store_lookup")

st.pyplot(plots.plot_ranked_bar(stores, "store_id", "censored_share", n=store_n,
                                direction=store_dir, highlight_id=picked_store,
                                xlabel="share of days censored (%)"))

if store_lookup != "(none)":
    row = stores.set_index("store_id").loc[store_lookup]
    pct_rank = 100 * (stores["censored_share"] < row["censored_share"]).mean()
    st.info(
        f"**Store #{store_lookup}**: censored on {row['censored_share']:.1%} of its "
        f"product-days, carrying {int(row['n_products'])} products - a higher stockout rate than "
        f"{pct_rank:.0f}% of the 100 stores."
    )

with st.expander("Full store table"):
    st.dataframe(stores.sort_values("censored_share", ascending=False).reset_index(drop=True),
                hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------- per product
st.divider()
st.subheader("Per product, pooled across every store that carries it: how much does it vary "
             "product to product?")

min_stores = st.slider(
    "Only include products carried by at least this many stores (below that, one small store's "
    "supply habits can swing the number on very few days):",
    1, 20, 5, key="min_stores")
eligible = products[products["n_stores"] >= min_stores]
st.caption(f"{len(eligible):,} of {len(products):,} products carried by >= {min_stores} stores.")

col1, col2, col3 = st.columns(3)
col1.metric("Highest censored-day share", f"{eligible['censored_share'].max():.1%}")
col2.metric("Median (eligible set)", f"{eligible['censored_share'].median():.1%}")
col3.metric("Lowest censored-day share", f"{eligible['censored_share'].min():.1%}")

st.markdown("###### Rank the products")
c1, c2 = st.columns([1, 2])
with c1:
    product_dir = st.radio("Show products with the:", ["highest", "lowest"], horizontal=True,
                          format_func=lambda d: f"{d} stockout rate", key="product_dir")
    product_n = st.slider("How many to show", 5, 30, 15, key="product_n")
with c2:
    product_lookup = st.selectbox(
        "Or look up one product directly:",
        ["(none)"] + sorted(eligible["product_id"].tolist()),
        format_func=lambda v: "Choose a product..." if v == "(none)" else f"Product #{v}",
        key="product_lookup")

st.pyplot(plots.plot_ranked_bar(eligible, "product_id", "censored_share", n=product_n,
                                direction=product_dir,
                                xlabel="share of days censored (%), pooled across stores"))
st.caption(
    "Pooled across stores on purpose: this asks whether a product itself tends to sell out "
    "wherever it's stocked, as distinct from one store simply managing it badly."
)

if product_lookup != "(none)":
    row = eligible.set_index("product_id").loc[product_lookup]
    pct_rank = 100 * (eligible["censored_share"] < row["censored_share"]).mean()
    st.info(
        f"**Product #{product_lookup}**: censored on {row['censored_share']:.1%} of its days, "
        f"pooled across the {int(row['n_stores'])} stores that carry it - a higher stockout rate "
        f"than {pct_rank:.0f}% of the {len(eligible):,} eligible products."
    )

with st.expander("Full product table (unfiltered, all 557 products)"):
    st.dataframe(products.sort_values("censored_share", ascending=False).reset_index(drop=True),
                hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------- choosing One Store
st.divider()
st.subheader("Choosing the One Store view from this")
st.markdown(
    f"**Store #{picked_store}** is what the **One Store** page (last in the sidebar) walks "
    f"through in full by default - picked as the store closest to the subset **median** on "
    f"censored-day share "
    f"({picked_censored:.1%} vs a subset median of {stores['censored_share'].median():.1%}), "
    "rather than a hand-picked extreme in either direction. A dramatic pick would make recovery "
    "look more impressive than it typically is; a flattering one would hide the problem this "
    "project exists to address - the median is the honest **typical case**. Use the ranking and "
    "lookup tools above to compare it against any other store, and the One Store page itself now "
    "lets you switch to a different store directly."
)
st.caption(
    "Uses `store_view.pick_store` - the exact function the One Store page calls, so this claim "
    "and that page's default pick can never drift apart."
)
