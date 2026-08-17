"""Model Architecture & Tuning - every model in the pipeline (baselines, Recovery Stage 1, and
the two forecasters), what each hyperparameter actually controls, and the tuning grid/winner for
each searched model - read live from the saved tuning tables, never hand-typed, so a re-run that
changes the winner can't leave this page describing a config nobody shipped.

Promoted out of a "Why TFT and Recovered" expander into its own page: that page's whole editorial
stance is "don't oversell a ~4% edge" and stays deliberately light, which argued against growing
a 6-paragraph technical appendix inside it rather than for it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import load_csv, page_config  # noqa: E402
from src.utils import config  # noqa: E402

page_config("Model Architecture & Tuning")

st.markdown(
    "Every model this project fits, what it's for, and how it was configured - baselines "
    "(untuned on purpose, the bar to clear), Recovery's Stage 1 (a 6-way model-SELECTION "
    "comparison, not a hyperparameter grid), and the two forecasters (each its own grid search). "
    "Hyperparameter tables spell out what each setting actually does, not just the value it was "
    "left at."
)

tab_base, tab_recovery, tab_tft, tab_xgb = st.tabs(
    ["Baselines", "Recovery (Stage 1)", "Forecast: TFT", "Forecast: XGBoost"])

# ------------------------------------------------------------------------------------ baselines
with tab_base:
    st.markdown(
        "Three models the recovery and forecasting stages have to beat - deliberately **not** "
        "optimised, since the point is a reasonable status-quo effort, not the best possible one. "
        "Tuning the bar you're trying to clear would be self-defeating."
    )
    st.dataframe(pd.DataFrame([
        {"model": "seasonal_naive", "what it is": "A rule, not a fit - predicts `lag7` (the same "
         "weekday, one week back), falling back to the trailing 7-day mean if that's missing.",
         "hyperparameters": "none"},
        {"model": "sarima", "what it is": "SARIMAX(1,0,1)(1,0,0,7), fixed order (not searched), "
         "fit independently on each of a 30-series sample. Falls back to a flat mean-of-last-7 "
         "forecast on a series it can't converge on.",
         "hyperparameters": "order fixed, see below"},
        {"model": "xgboost_quantile (baseline)", "what it is": "Three SEPARATE XGBoost models "
         "(one each for q10/q50/q90) - not the single joint-quantile model the tuned XGBoost "
         "forecaster uses (Forecast: XGBoost tab). Same library, different job.",
         "hyperparameters": "fixed, see below"},
    ]), hide_index=True, use_container_width=True)

    with st.expander("What SARIMA's (1,0,1)(1,0,0,7) actually means"):
        st.markdown(
            "SARIMA(p,d,q)(P,D,Q,s) combines two of the same three ideas at two time scales:\n\n"
            "- **p (=1), the AR order:** how many of the immediately preceding days feed into "
            "today's prediction.\n"
            "- **d (=0), differencing:** how many times the series is trend-removed before "
            "fitting; 0 means it's fit on the level, not a trend-adjusted version.\n"
            "- **q (=1), the MA order:** how many recent forecast ERRORS (not values) feed into "
            "today's prediction - lets the model correct for a run of recent misses.\n"
            "- **(P, D, Q, s) = (1, 0, 0, 7):** the same AR/differencing/MA idea again, but at a "
            "7-day lag - the model also looks at the same weekday from recent weeks, which is "
            "where weekly seasonality (weekend vs weekday demand) enters.\n\n"
            "None of these six numbers were searched; they're a fixed, reasonable classical "
            "specification, which is exactly why SARIMA is a reference point in this project, "
            "not a like-for-like competitor to the tuned models."
        )

    with st.expander("The untuned XGBoost baseline's fixed settings"):
        st.dataframe(pd.DataFrame([
            {"setting": "n_estimators", "value": "300", "controls": "how many trees are added in "
             "sequence - more trees means more capacity to fit, and more time to train."},
            {"setting": "learning_rate", "value": "0.05", "controls": "how much each new tree is "
             "allowed to correct the previous trees' errors - smaller steps are more careful but "
             "need more trees to get anywhere."},
            {"setting": "max_depth", "value": "6", "controls": "how many splits deep one tree can "
             "go - deeper trees capture more feature interactions, at higher overfitting risk."},
            {"setting": "subsample / colsample_bytree", "value": "0.8 / 0.8", "controls": "the "
             "fraction of rows / features randomly offered to each tree - adds randomness so "
             "trees don't all latch onto the same quirks (a form of regularisation called "
             "bagging)."},
        ]), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- recovery, stage 1
with tab_recovery:
    st.markdown(
        "Stage 1 predicts the hourly in-stock demand rate that Stage 2 substitutes into stocked-"
        "out hours. Six candidate model FAMILIES are compared head to head on held-out full-shelf "
        "days - a model-**selection** question (which family and which loss function fits this "
        "data best), not a hyperparameter grid within one family. Full scored comparison table is "
        "on the Recovery page; this tab is about what makes the six candidates different."
    )
    st.dataframe(pd.DataFrame([
        {"candidate": "lightgbm_tweedie (shipped)", "family": "LightGBM",
         "what's different": "Tweedie loss"},
        {"candidate": "lightgbm_poisson", "family": "LightGBM", "what's different": "Poisson loss"},
        {"candidate": "lightgbm_l2", "family": "LightGBM",
         "what's different": "squared-error (L2) loss"},
        {"candidate": "xgboost", "family": "XGBoost", "what's different": "squared-error loss "
         "(XGBoost's default) - the cross-library check on the winning LightGBM config"},
        {"candidate": "poisson_glm", "family": "linear (Poisson GLM)",
         "what's different": "no trees at all - checks whether a tree model is even necessary"},
        {"candidate": "series_hour_mean", "family": "no model",
         "what's different": "each (store, product, hour)'s own historical average, falling back "
         "to (product, hour) then the global hour mean - the no-learning control"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("###### Why the loss function is the main thing being compared")
    st.markdown(
        "The three LightGBM variants share every setting except the loss, which is the point: "
        "most hours in a day sell nothing at all, so which loss function a model is trained "
        "to minimise matters more here than almost any other choice.\n\n"
        "- **Tweedie** (winner): assumes a mix of exact zeros plus a continuous positive tail - a "
        "close match for per-hour sales, where most hours are zero and the rest are a real "
        "amount.\n"
        "- **Poisson**: assumes integer count data with variance that grows with the mean - "
        "reasonable for sales counts, but doesn't model the exact-zero mass as well as Tweedie "
        "here.\n"
        "- **L2 / squared-error**: fits the conditional mean directly, no assumption about the "
        "distribution's shape - simplest, but most easily dragged toward zero by a target that's "
        "mostly zero."
    )

    with st.expander("The shared LightGBM settings (same across all 3 loss variants)"):
        st.dataframe(pd.DataFrame([
            {"setting": "n_estimators", "value": "400", "controls": "how many trees are added in "
             "sequence."},
            {"setting": "learning_rate", "value": "0.03", "controls": "how much each new tree "
             "corrects the previous trees' errors."},
            {"setting": "num_leaves", "value": "31", "controls": "the max number of leaf nodes in "
             "a single tree - how complex a pattern one tree is allowed to carve out."},
            {"setting": "min_child_samples", "value": "20", "controls": "the minimum training "
             "rows a leaf must contain to be created - stops a leaf being built around one "
             "unusual observation."},
            {"setting": "subsample / colsample_bytree", "value": "0.8 / 0.8", "controls": "the "
             "fraction of rows / features randomly offered to each tree (bagging)."},
        ]), hide_index=True, use_container_width=True)
    st.caption(
        "None of these five were swept - they're fixed, reasonable settings; only the LOSS "
        "FUNCTION varies across the three LightGBM candidates. See the Recovery page for the "
        "scored comparison table and why Tweedie won."
    )

# --------------------------------------------------------------------------------------- TFT
with tab_tft:
    st.markdown(
        "The Temporal Fusion Transformer reads `encoder_days` of a series' own history through "
        "an attention mechanism, then forecasts the next 7 days at **six quantiles at once** "
        "(q0.025 ... q0.975) - one model produces the whole predictive range, not six separate "
        "models. Each series is normalised on its **own** history, so a fast- and a slow-selling "
        "product are learned on their own scale rather than the store-wide average."
    )

    st.markdown("###### Hyperparameters")
    st.dataframe(pd.DataFrame([
        {"hyperparameter": "learning_rate", "swept over": "0.005, 0.01",
         "what it controls": "the optimiser's step size - how much the model's weights move to "
         "correct error after each training batch."},
        {"hyperparameter": "dropout", "swept over": "0.2, 0.3",
         "what it controls": "the fraction of internal connections randomly switched off on each "
         "training pass, so no single pathway can be relied on alone - the main defence against "
         "overfitting for a model this size."},
        {"hyperparameter": "encoder_days", "swept over": "3, 5, 7",
         "what it controls": "how many days of a series' own history it's shown before having to "
         "forecast - its lookback window."},
        {"hyperparameter": "weight_decay", "swept over": "1e-4, 1e-3",
         "what it controls": "a penalty on large weights during training, nudging the model "
         "toward simpler, smoother solutions."},
        {"hyperparameter": "hidden_size", "swept over": "fixed at 32",
         "what it controls": "the width of the model's internal representations - more capacity "
         "to represent complex patterns. 64 was tried in earlier searches and lost in most "
         "pairings, so it was dropped rather than kept as dead weight."},
        {"hyperparameter": "attention_head_size", "swept over": "fixed at 4",
         "what it controls": "how many separate attention \"views\" the model keeps in parallel "
         "when deciding which past days matter most - not tuned."},
        {"hyperparameter": "gradient_clip_val", "swept over": "fixed at 0.1",
         "what it controls": "caps how large a single training update can be, so one unusually "
         "noisy batch can't destabilise the whole fit."},
        {"hyperparameter": "early stopping", "swept over": "patience 8, min_delta 1e-3, ceiling "
         "15 epochs", "what it controls": "training stops once validation loss stops improving "
         "by at least 0.001 for 8 epochs running, or at 15 epochs regardless, whichever comes "
         "first."},
    ]), hide_index=True, use_container_width=True)

    tft_tuning = load_csv(config.tft_tuning("recovered")).drop(columns=["config_id"])
    winner = tft_tuning.iloc[0]
    st.markdown("###### The 24-config grid (learning rate x dropout x encoder length x weight "
               "decay), ranked by `pinball(avg)` on the validation window - the metric the "
               "ordering stage actually reads a quantile off, not WAPE")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winning learning rate", f"{winner['learning_rate']:g}")
    c2.metric("Winning dropout", f"{winner['dropout']:g}")
    c3.metric("Winning encoder length", f"{int(winner['encoder_days'])} days")
    c4.metric("Winning weight decay", f"{winner['weight_decay']:g}")
    with st.expander("Full 24-config grid"):
        st.dataframe(tft_tuning, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------------- XGBoost
with tab_xgb:
    st.markdown(
        "Gradient-boosted trees, the same six quantiles as the TFT - but fit as **one model**, "
        "via XGBoost's `quantile_alpha` taking the whole quantile list at once, rather than one "
        "fit per quantile. That's the reason XGBoost was picked over LightGBM for this role: "
        "LightGBM has no equivalent, so it would need six separate models. Fitting jointly "
        "doesn't stop the six curves crossing on some rows (they still do, and are sorted "
        "defensively afterward), but LightGBM would have added a second decision-tree library "
        "rather than a genuinely different architecture to check the TFT against - the "
        "contrast this project wants is trees vs. the TFT's sequence encoder, not tree library A "
        "vs. tree library B."
    )

    st.markdown("###### Hyperparameters")
    st.dataframe(pd.DataFrame([
        {"hyperparameter": "learning_rate", "swept over": "0.02, 0.03, 0.05, 0.08",
         "what it controls": "same idea as every boosted-tree model here - how much each new "
         "tree corrects the previous trees' errors."},
        {"hyperparameter": "max_depth", "swept over": "5, 6, 7",
         "what it controls": "how many splits deep one tree can go - deeper trees capture more "
         "feature interactions, at higher overfitting risk."},
        {"hyperparameter": "subsample / colsample_bytree", "swept over": "fixed at 0.8 / 0.8",
         "what it controls": "the fraction of rows / features randomly offered to each tree "
         "(bagging) - probed individually against the winning depth and left fixed once nothing "
         "beat them."},
        {"hyperparameter": "reg_lambda", "swept over": "fixed at 1",
         "what it controls": "an L2 penalty on the size of each leaf's prediction, smoothing out "
         "extreme leaf values."},
        {"hyperparameter": "early stopping", "swept over": "40 rounds' patience, 1200-round "
         "ceiling", "what it controls": "each config gets its OWN round count rather than a "
         "fixed one - a fixed round count would ask a slow learning rate to finish in a fast "
         "learner's budget, which handicaps it. Stopping watches the last 14 TRAINING days held "
         "back, never validation - so a config can't size itself using the very window it's "
         "later scored on."},
    ]), hide_index=True, use_container_width=True)

    gbm_tuning = load_csv(config.gbm_tuning("recovered"))
    winner = gbm_tuning.iloc[0]
    st.markdown("###### The 12-config grid (learning rate x max depth), ranked by "
               "`pinball(avg)` on the validation window")
    c1, c2, c3 = st.columns(3)
    c1.metric("Winning learning rate", f"{winner['learning_rate']:g}")
    c2.metric("Winning max depth", f"{int(winner['max_depth'])}")
    c3.metric("Rounds before early stopping", f"{int(winner['best_round'])}")
    with st.expander("Full 12-config grid"):
        st.dataframe(gbm_tuning, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Both forecasters are then trained on **both** targets (raw sales / recovered demand) - four "
    "arms total, same code path, same tuning process. One honest gap: the grid search itself was "
    "only run and saved for the *recovered* target; the *raw* arm reuses the recovered arm's "
    "winning config rather than being independently searched (see Limitations)."
)
