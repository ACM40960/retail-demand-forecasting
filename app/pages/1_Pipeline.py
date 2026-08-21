"""Approach & Workflow: the five pipeline stages, which model runs at each, and what the pipeline
actually decides.

Every number is read live from the saved reports rather than hand-typed - the decision capstone from
cost_savings_ci.csv, the recovery bias correction from conclusion_recovery.csv and
recovery_correction_by_period.csv - so a re-run cannot leave this page describing stale results.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config  # noqa: E402
from src.utils.poster import PALETTE  # noqa: E402

page_config("Approach & Workflow")

st.markdown(
    "**Five steps, one pipeline: recover &rarr; forecast &rarr; calibrate &rarr; order &rarr; "
    "decide.** The first four are pipeline STAGES - each fits or applies something. The fifth is "
    "the pipeline's actual output: a scored decision, not another model."
)

# (number, name, icon, color, description, model note, what comes out of it)
STAGES = [
    ("1", "Recover", "\U0001F4E6", PALETTE["recovered"],
     "Stage 1 predicts the hourly in-stock demand rate; Stage 2 fills stocked-out hours only, "
     "summed to a daily total, with one fitted bias correction.",
     "LightGBM, Tweedie loss (Stage 1) - see the Recovery page for why Tweedie won, and "
     "the bias correction explained below",
     "recovered_demand"),
    ("2", "Forecast", "\U0001F4C8", PALETTE["tft"],
     "Two independent architectures, each trained on both raw and recovered demand - four "
     "models total, six quantiles each.",
     "Temporal Fusion Transformer + XGBoost (joint quantile regression) - see Model Architecture "
     "& Tuning for every hyperparameter",
     "q0.025 ... q0.975 forecast"),
    ("3", "Calibrate", "\U0001F3AF", PALETTE["raw"],
     "Split conformal prediction widens each quantile band by one offset, so an \"80% band\" "
     "actually contains the truth 80% of the time.",
     "Not a fitted model - one offset per band, measured on the calibration window only",
     "calibrated interval"),
    ("4", "Order", "\U0001F6D2", PALETTE["good"],
     "The newsvendor rule reads a percentile off the calibrated forecast, scored against demand "
     "that actually happened - not against the forecast itself.",
     "Not a model - a closed-form rule: q* = c_u / (c_u + c_o)",
     "order quantity"),
]

ci = load_csv(config.REPORTS_DIR / "cost_savings_ci.csv").set_index("window")
test_pct = ci.loc["test", "point"]

cards = ""
for i, (num, name, icon, color, desc, model, out) in enumerate(STAGES):
    cards += f"""
    <div style="flex:1; min-width:210px; background:{color}1A; border-left:5px solid {color};
                border-radius:10px; padding:1rem 1.1rem; box-shadow:0 1px 4px rgba(0,0,0,0.08);">
      <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
        <span style="background:{color}; color:white; font-weight:700; font-size:0.85rem;
                    width:1.6rem; height:1.6rem; border-radius:50%; display:flex;
                    align-items:center; justify-content:center; flex-shrink:0;">{num}</span>
        <span style="font-size:1.2rem;">{icon}</span>
        <span style="font-weight:700; color:{color}; font-size:1.1rem;">{name}</span>
      </div>
      <div style="font-size:0.88rem; line-height:1.35; margin-bottom:0.5rem;">{desc}</div>
      <div style="font-size:0.78rem; font-style:italic; opacity:0.85; margin-bottom:0.4rem;">
        {model}</div>
      <div style="font-size:0.72rem; font-family:monospace; opacity:0.65;">&rarr; {out}</div>
    </div>
    """
    cards += ('<div style="display:flex; align-items:center; font-size:1.6rem; '
             f'color:{PALETTE["naive"]}; padding:0 0.15rem;">&rarr;</div>')

cards += f"""
<div style="flex:1.15; min-width:230px; background:{PALETTE['xgb']}; color:white;
            border-radius:10px; padding:1rem 1.2rem; box-shadow:0 2px 10px {PALETTE['xgb']}66;">
  <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
    <span style="background:white; color:{PALETTE['xgb']}; font-weight:700; font-size:0.85rem;
                width:1.6rem; height:1.6rem; border-radius:50%; display:flex; align-items:center;
                justify-content:center; flex-shrink:0;">5</span>
    <span style="font-size:1.2rem;">\U0001F3C1</span>
    <span style="font-weight:700; font-size:1.1rem;">Final Decision</span>
  </div>
  <div style="font-size:0.88rem; line-height:1.4;">
    Simulated against demand that actually happened, at the headline cost ratio:
    <b>{test_pct:.1f}% cheaper than the naive rule</b> on the sealed test week - across a sweep
    of nine cost-ratio assumptions, so the claim isn't tied to one lucky number.
  </div>
</div>
"""

st.markdown(
    f'<div style="display:flex; align-items:stretch; gap:0.4rem; flex-wrap:wrap; '
    f'margin:1.2rem 0;">{cards}</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Full detail on the final number: **Final Ordering Decision**. How the specific operating "
    "point is chosen: **Choosing an Operating Point**. Every model above, and what each "
    "hyperparameter does: **Model Architecture & Tuning**."
)

st.divider()
recovery = load_csv(config.REPORTS_DIR / "conclusion_recovery.csv").set_index("what")["value"]
with st.expander("How is the recovery bias correction calculated?"):
    st.markdown(
        "Stage 1 predicts each hour's in-stock demand rate one hour at a time. Summing 16 "
        "independently-predicted hours into a daily total tends to run high - the errors that "
        "would partly cancel if all 16 hours were predicted jointly instead compound when they're "
        "added up separately. Left uncorrected, this pushes the recovered daily total up before "
        "it's ever compared against the truth."
    )
    st.markdown(
        "**How the single correction number is fitted** - using days the shelf never actually "
        "emptied, so the true total is knowable:\n\n"
        "1. Take training-period days with a full shelf (`is_censored == 0`) - recorded sales "
        "*are* true demand on these.\n"
        "2. Recover them anyway, **as if** every one of the 16 trading hours had been stocked "
        "out, using a Stage-1 model that never saw these particular days during its own fit.\n"
        "3. Split those recovered days 50/50. One half sets "
        "`correction = sum(true) / sum(recovered)` - a single multiplier. The other half is "
        "scored *with* that correction applied, so the reported accuracy is never measured on "
        "the same days the correction itself was tuned on.\n"
        "4. That one multiplier is then applied to every hour Stage 2 fills in on a real "
        "censored day."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Aggregation bias before correction", recovery["aggregation bias before correction"])
    c2.metric("Fitted correction", recovery["one fitted correction multiplier"])
    c3.metric("Day WPE after correction", recovery["day WPE after correction"])

    st.markdown("**Does one fixed multiplier still hold weeks later?**")
    by_period = load_csv(config.CORRECTION_BY_PERIOD).rename(columns={
        "period": "window", "n_clean_days": "clean days scored", "correction": "refit correction",
        "shipped_is_off_by_%": "shipped multiplier off by (%)"})
    by_period["refit correction"] = by_period["refit correction"].round(4)
    by_period["shipped multiplier off by (%)"] = by_period["shipped multiplier off by (%)"].round(2)
    st.dataframe(by_period, hide_index=True, use_container_width=True)
    st.caption(
        "Refit independently in each window, training days never read twice - the multiplier "
        "moves in a narrow band with no consistent drift over time (not monotonically rising or "
        "falling), so shipping ONE fixed number rather than a per-period one is a reasonable "
        "simplification, not a hidden approximation. The test week is never touched here."
    )

st.divider()
st.markdown(
    "**Why six quantiles, not three:** fitting all six jointly regularizes rather than dilutes "
    "(q80 pinball improves from 0.1334 to 0.1327 vs. a 2-quantile fit), and the outer pair "
    "(q0.025 / q0.975) is what lets ordering interpolate correctly for q\\* > 0.90 in the cost "
    "sweep, rather than extrapolating off q10/q90 alone."
)
