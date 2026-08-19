"""Reviewer dashboard - entry page (Objective + Data). Every page in this app reads only files
already committed under outputs/ and data/processed/ - nothing here re-fits a model or re-runs
recovery/forecast/conformal. See the sidebar for the rest of the project.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import page_config  # noqa: E402
from src.utils import config  # noqa: E402

page_config("Perishable Demand Forecasting and Zero-waste Inventory Engine")

st.markdown(
    "Fresh-food stores order against a forecast. A stockout means recorded sales are not the same "
    "as true demand, so a model trained on raw sales learns to under-order exactly the products "
    "that keep selling out. This project **recovers the demand a stockout hides**, forecasts it as "
    "an honest range, and turns that range into an order quantity that beats the naive rule most "
    "stores already follow."
)
st.markdown(
    "**Success criterion:** cheaper than that naive rule - *\"order what sold last week\"* - at "
    "every plausible cost assumption."
)

st.divider()
summary = config.SUBSET_SUMMARY.read_text(encoding="utf-8")
main_text, _, id_lists = summary.partition("Categories present:")
st.markdown(main_text)
st.caption(
    "**Products are IDs, not names.** FreshRetailNet-50K carries no product description a person "
    "would recognise - \"Product #4213\" is as specific as it gets, and category is likewise a "
    "numeric ID (`first_category_id` etc.), not a label like \"dairy\" or \"leafy greens\". Every "
    "page below that mentions a product means one of these anonymised IDs."
)
if id_lists:
    with st.expander("Full store & category ID list (for reproducibility)"):
        st.markdown("Categories present:" + id_lists)
