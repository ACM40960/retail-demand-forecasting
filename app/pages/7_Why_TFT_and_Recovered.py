"""Why TFT and Recovered Data - poster's own two-number framing: a large data-layer effect and a
small, honestly-reported model-architecture effect. Deliberately light - this is the one page
where the poster's own instinct ("don't oversell a ~4% edge") argues against adding more detail.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config  # noqa: E402

page_config("Why TFT and Recovered Data")

st.markdown("**The data layer did the work; the model architecture was a confirmation, not the finding.**")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Recovered over raw")
    st.write(
        "Raw sales are truncated by stockouts - training on them teaches a model that a stockout "
        "day had low demand. Recovery removes that systematic downward bias exactly where it "
        "occurs (chronic-stockout products), verified against held-out truth on the Recovery page."
    )
with col2:
    st.markdown("#### TFT over XGBoost - but only barely")
    acc = load_csv(config.REPORTS_DIR / "conclusion_accuracy.csv").rename(
        columns={"Unnamed: 0": "arm"}).set_index("arm")
    tft_pb = acc.loc["tft_recovered", "pinball(avg)"]
    xgb_pb = acc.loc["xgb_recovered", "pinball(avg)"]
    edge = 100 * (xgb_pb - tft_pb) / xgb_pb
    st.metric("TFT pinball loss (recovered)", f"{tft_pb:.4f}", help="Lower is better")
    st.metric("XGBoost pinball loss (recovered)", f"{xgb_pb:.4f}", delta=f"{edge:.1f}% worse than TFT")
    st.write(
        "A real but modest edge, reported honestly rather than oversold - and the comparison "
        "structurally favours TFT, since it early-stops on the window it's scored on."
    )

st.divider()
st.caption(
    "Why run both models at all: to prove the recovery result isn't an artefact of one "
    "architecture. Two unrelated model families agreeing on direction and magnitude is more "
    "load-bearing than either model's individual accuracy."
)

st.caption(
    "Curious what's actually inside each model, and what every hyperparameter does? See "
    "**Model Architecture & Tuning** in the sidebar - every model in the pipeline, every setting "
    "explained, and each grid search's winner read live off its saved tuning table."
)
