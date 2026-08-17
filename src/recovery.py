"""Latent demand recovery (de-censoring) - the novel layer.

A stocked-out shelf records zero sales, not zero demand. This module fills in what the till never
saw, and the hourly data is the reason it can: the dataset marks WHICH hours each shelf was empty,
so a stockout is distinguishable from a genuine no-sale only at hour resolution.

Hours are therefore an implementation detail, contained here:

    Stage 1     learn the in-stock hourly demand rate from training-period, non-stockout hours -
                the only hours where what sold is what was wanted
    Stage 2     replace each stocked-out hour inside 06:00-22:00 with that prediction (floored at
                what was observed), keep every other hour as-is, sum to a daily `recovered_demand`
    correction  summing 16 predicted hours skews the daily total high, so ONE multiplier
                (`bias_correction`) is fitted on held-out full-stock days and applied to every
                recovered hour

Everything public here takes and returns DAILY frames:

    evaluate        train/test one candidate on held-out full-stock days
    compare_models  the same for every candidate - the table that picks Stage 1
    run             fit the chosen model and write `recovered_demand`

`evaluate` can only score full-stock days, so one check covers what it structurally cannot -
whether the fitted correction survives the calendar:

    correction_by_period   refit the multiplier in each later window - does one number still hold?
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from .utils import config, data_io
from .utils.metrics import bias_scores
from .splits import add_period
from .utils.features import (
    add_lag_roll_features, explode_hourly, HOURLY_FEATURES, HOURLY_CATEGORICAL,
)

KEY = ["store_id", "product_id", "dt"]
ACTIVE_LO, ACTIVE_HI = config.ACTIVE_HOURS
N_TEST_DAYS = 5000   # clean days rebuilt to score a candidate


def load_daily(kind: str = "daily") -> pd.DataFrame:
    """The daily frame with lag/rolling context and the frozen `period` labels attached.

    `kind="recovered"` reads the parquets `run` wrote instead - the same raw columns plus
    `recovered_demand`. Both go through the identical pipeline because NEITHER file stores derived
    columns: `_save_recovered` strips lag/rolling/`period` before writing, so they are rebuilt here
    and cannot describe a subset that no longer exists. Downstream stages take this route rather
    than loading `daily` and joining `recovered_demand` onto it - one pair of parquets, same frame.
    """
    return add_period(add_lag_roll_features(data_io.load(kind)))


_hours = None


def hours(daily: pd.DataFrame) -> pd.DataFrame:
    """The hourly frame, one row per (series, day, hour), built once per session and reused.

    Cached because exploding it costs a minute and several million rows. It is derived purely from
    `daily` plus the hourly parquet, so the cache only goes stale if the subset is rebuilt
    mid-session - restart the kernel after a rebuild.
    """
    global _hours
    if _hours is None:
        _hours = explode_hourly(data_io.load("hourly"), daily)
    return _hours


# ------------------------------------------------------------------ Stage-1 candidates

def stage1_lgbm(random_state: int = config.RANDOM_STATE,
                objective: str = "tweedie") -> lgb.LGBMRegressor:
    """The Stage-1 model config - one source for the pipeline and every comparison.

    `objective` swaps the loss. Squared error fits the conditional median, which for a mostly-zero
    target is often zero; "tweedie" and "poisson" fit the conditional mean, which is what Stage 2
    needs when it sums 16 hours into a daily total. The default was chosen by `compare_models`.
    """
    return lgb.LGBMRegressor(
        objective=objective,
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, verbosity=-1,
    )


def stage1_xgb(random_state: int = config.RANDOM_STATE):
    """The other gradient-boosting library, given the same features and seed."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
        random_state=random_state)


def stage1_poisson_glm(random_state: int = config.RANDOM_STATE):
    """Parametric floor: a Poisson GLM with the ID columns one-hot encoded, so it is given the same
    information as the trees. `random_state` is unused (the fit is deterministic) but kept so every
    candidate has the same signature."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    return make_pipeline(
        ColumnTransformer([("ids", OneHotEncoder(handle_unknown="ignore"), HOURLY_CATEGORICAL)],
                          remainder="passthrough"),
        PoissonRegressor(max_iter=300))


class SeriesHourMean:
    """Non-ML floor: predict each hour as that (store, product, hour)'s mean in-stock sale, falling
    back to (product, hour) then the global hour mean. The 'why a model, not a groupby?' control."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        d = X[["store_id", "product_id", "hour"]].copy()
        d["y"] = y
        self.sp_hour = d.groupby(["store_id", "product_id", "hour"], observed=True)["y"].mean()
        self.p_hour = d.groupby(["product_id", "hour"], observed=True)["y"].mean()
        self.g_hour = d.groupby("hour", observed=True)["y"].mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        def lookup(table, cols):
            return table.reindex(pd.MultiIndex.from_frame(X[cols])).to_numpy()

        pred = lookup(self.sp_hour, ["store_id", "product_id", "hour"])
        pred = np.where(np.isnan(pred), lookup(self.p_hour, ["product_id", "hour"]), pred)
        return np.where(np.isnan(pred), self.g_hour.reindex(X["hour"]).to_numpy(), pred)


# Each value is `make_model(random_state)`, so every candidate goes through one code path.
# HistGradientBoosting is absent: it caps categorical features at 255 levels and `product_id` has
# more, so it cannot be given the same features as the others.
CANDIDATES = {
    "lightgbm_tweedie": stage1_lgbm,   # the shipped config, hence no explicit objective
    "lightgbm_poisson": lambda rs: stage1_lgbm(rs, objective="poisson"),
    "lightgbm_l2": lambda rs: stage1_lgbm(rs, objective="regression"),
    "xgboost": stage1_xgb,
    "poisson_glm": stage1_poisson_glm,
    "series_hour_mean": lambda rs: SeriesHourMean(),
}


# ------------------------------------------------------------------ the two stages (hour-level)

def _model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The Stage-1 feature matrix, ID columns typed as pandas categoricals so LightGBM and XGBoost
    both treat them as categorical without being told."""
    X = df[HOURLY_FEATURES].copy()
    for c in HOURLY_CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def training_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """The only rows Stage 1 may fit on: non-stockout hours from training-period days. Public so
    the leakage audit checks the filter production actually uses."""
    return hourly[(hourly.period == "training") & (hourly.hour_stockout == 0)]


def _stage2(hourly: pd.DataFrame, model, bias_correction: float) -> pd.DataFrame:
    """Replace each stocked-out 06:00-22:00 hour with the corrected prediction, floored at what was
    observed; keep every other hour as-is; sum to a daily `recovered_demand`."""
    expected = np.clip(model.predict(_model_matrix(hourly)), 0, None) * bias_correction
    in_window = (hourly["hour"] >= ACTIVE_LO) & (hourly["hour"] < ACTIVE_HI)
    recovered = np.where((hourly["hour_stockout"] == 1) & in_window,
                         np.maximum(expected, hourly["hour_sale"]),   # floor at observed
                         hourly["hour_sale"])                         # in-stock / off-window
    return (hourly.assign(hour_recovered=recovered)
            .groupby(KEY, as_index=False).agg(recovered_demand=("hour_recovered", "sum")))


# ------------------------------------------------------------------ evaluation

def _recover_test_days(hourly: pd.DataFrame, test_days: pd.DataFrame, make_model, random_state):
    """Recover `test_days` as if every 06:00-22:00 hour had been stocked out; return their daily
    totals. Stage 1 is trained with those days held out, so it predicts days it has never seen -
    without that, the score would measure memorisation."""
    is_test = pd.Series(   # MultiIndex.isin, not a set of tuples: same mask, far faster
        pd.MultiIndex.from_frame(hourly[KEY]).isin(pd.MultiIndex.from_frame(test_days[KEY])),
        index=hourly.index)

    train = training_hours(hourly)
    train = train[~is_test.loc[train.index]]
    model = make_model(random_state)
    model.fit(_model_matrix(train), train["hour_sale"].values)

    held_out = hourly[is_test].copy()
    expected = np.clip(model.predict(_model_matrix(held_out)), 0, None)
    in_window = (held_out["hour"] >= ACTIVE_LO) & (held_out["hour"] < ACTIVE_HI)
    held_out["recovered"] = np.where(in_window, expected, held_out["hour_sale"])
    return held_out.groupby(KEY)["recovered"].sum()


def evaluate(daily: pd.DataFrame, make_model=stage1_lgbm, n_days: int = N_TEST_DAYS,
             random_state: int = config.RANDOM_STATE) -> dict:
    """Train / test one Stage-1 candidate and return its scores.

    A real stockout has no ground truth, so the test set is built from days the shelf was FULL -
    days whose recorded total IS the true demand. Those days are held out of training, recovered as
    if they had been stocked out 06:00-22:00, then compared against the total that was hidden.

    The held-out days are halved again: the first half fits `bias_correction`, the second scores it,
    so the reported error is never measured on days the correction was tuned on.

    Returns `wape`/`mae`/`wpe` on daily totals, plus the fitted `bias_correction` and the
    `wpe_uncorrected` it removes.
    """
    full_stock = daily[(daily.period == "training") & (daily.is_censored == 0)]
    test_days = full_stock.sample(n=min(n_days, len(full_stock)), random_state=random_state)

    t0 = time.perf_counter()
    recovered = _recover_test_days(hours(daily), test_days, make_model, random_state)
    fit_seconds = time.perf_counter() - t0

    truth = test_days.set_index(KEY)["sale_amount"]
    both = pd.concat([recovered.rename("recovered"), truth.rename("true")], axis=1).dropna()
    # split in half: one half fits bias_correction, the other scores it, so the reported error is
    # never measured on the same days the correction was tuned on
    calibration = both.sample(frac=0.5, random_state=random_state)
    test = both.drop(index=calibration.index)

    correction = float(calibration["true"].sum() / calibration["recovered"].sum())
    scores = bias_scores(test["true"].values, test["recovered"].values * correction)
    return dict(n_test_days=len(test), bias_correction=round(correction, 4),
                wpe_uncorrected=round(float(test["recovered"].sum() / test["true"].sum() - 1), 4),
                fit_seconds=round(fit_seconds, 1),
                **{k: round(v, 4) for k, v in scores.items()})


def compare_models(daily: pd.DataFrame, random_state: int = config.RANDOM_STATE,
                   save: bool = True) -> pd.DataFrame:
    """Train / test every Stage-1 candidate - one row each, scored on daily totals.

    How to pick a row:

    1. `wape` is the decider - the error as a share of actual demand. **Lowest wins.**
    2. Check the winner beats `series_hour_mean` by a clear margin. If it does not, the modelling is
       not earning its complexity.
    3. Require `wpe` inside about +-0.02. A model that is accurate but skewed will systematically
       over- or under-order.
    4. Only if two rows are within ~1% on `wape`, break the tie on `fit_seconds`.

    `wpe_uncorrected` and `bias_correction` are diagnostics, not selection criteria: each candidate
    fits its own correction because the bias belongs to the aggregation, not to the model.
    """
    rows = {}
    for name, make in CANDIDATES.items():
        print(f"train/test: {name} ...")
        rows[name] = evaluate(daily, make_model=make, random_state=random_state)
    board = pd.DataFrame(rows).T.sort_values("wape")
    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        board.to_csv(config.RECOVERY_COMPARISON)
    return board


# ------------------------------------------------------------------ running the chosen model

def run(daily: pd.DataFrame, model_name: str = "lightgbm_tweedie",
        random_state: int = config.RANDOM_STATE, save: bool = True):
    """Recover demand for every day using `model_name`, and return the daily frame with a
    `recovered_demand` column.

    Two fits, in this order: `evaluate` supplies `bias_correction` (its model must hold out the days
    the correction is calibrated on), then the shipped model is fitted on EVERY clean training hour.
    `save` writes the model, the fitted parameters and the recovered parquets, so `recovered_demand`
    can always be traced to the model that produced it.
    """
    make_model = CANDIDATES[model_name]
    params = evaluate(daily, make_model=make_model, random_state=random_state)
    correction = params["bias_correction"]

    hourly = hours(daily)
    train = training_hours(hourly)
    model = make_model(random_state)
    model.fit(_model_matrix(train), train["hour_sale"].values)

    recovered = (daily.drop(columns=["recovered_demand"], errors="ignore")
                 .merge(_stage2(hourly, model, correction), on=KEY, how="left"))
    # Stage 2 floors every hour at observed, so a daily total can never fall below recorded sales
    assert (recovered["recovered_demand"] >= recovered["sale_amount"] - 1e-6).all(), \
        "recovered_demand fell below sale_amount - the per-hour floor in _stage2 broke"

    if save:
        _save_params(model_name, params, model)
        _save_recovered(recovered)
        print(f"saved model + parameters (bias correction {correction:.4f}) and the parquets")
    return recovered


def _save_params(model_name: str, params: dict, model) -> None:
    """Record which model recovered the data and the scale it was applied with - without them
    `recovered_demand` is a column nobody can explain or reproduce."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(model, lgb.LGBMRegressor):   # native text format: readable, version-independent
        model.booster_.save_model(str(config.STAGE1_MODEL))
    config.RECOVERY_PARAMS.write_text(json.dumps({"model": model_name, **params}, indent=2))


def load_params() -> dict | None:
    """What the last recovery run used, or None if it has not run."""
    return (json.loads(config.RECOVERY_PARAMS.read_text())
            if config.RECOVERY_PARAMS.exists() else None)


def _save_recovered(recovered: pd.DataFrame) -> None:
    """Persist the raw columns + `recovered_demand`. Lag/rolling and `period` columns are rebuilt on
    load, so they are dropped rather than saved stale."""
    derived = [c for c in recovered.columns
               if c.startswith(("lag", "roll")) or c in ("censoring_frac", "period")]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in [("train", config.RECOVERED_TRAIN), ("eval", config.RECOVERED_EVAL)]:
        (recovered[recovered["split"] == split]
         .drop(columns=["split"] + derived).to_parquet(path, index=False))


# ------------------------------------------------------------------ diagnostics

def bias_by_censoring(recovered: pd.DataFrame) -> pd.DataFrame:
    """Recorded vs recovered demand, split by how censored the day was.

    `uplift_%` grows down the table, which reads as the model adding demand only where trading
    hours were missing. `uplift_%_flat_fill` is the control that stops that being read as
    evidence: it replaces every stocked-out hour with ONE number - the average in-stock hour, no
    model, no features - and reproduces the same climb. The pattern is arithmetic (more missing
    hours, more filled in, on a smaller base), so it cannot tell two fillers apart. What the model
    has to beat is in `compare_models`; this table shows only WHERE the demand is added.
    """
    cens = recovered[recovered.is_censored == 1].copy()
    cens["censoring_band"] = pd.cut(
        cens["censoring_frac"], bins=[0, 0.25, 0.5, 0.75, 1.0], include_lowest=True,
        labels=["low (0-25%)", "med (25-50%)", "high (50-75%)", "severe (75-100%)"])

    hours_open = ACTIVE_HI - ACTIVE_LO
    flat = (recovered["sale_amount"].sum()
            / (hours_open - recovered["stock_hour6_22_cnt"]).sum())   # the average in-stock hour
    cens["constant_fill"] = cens["sale_amount"] + cens["stock_hour6_22_cnt"] * flat

    def summarise(d):
        recorded, rec = d["sale_amount"].sum(), d["recovered_demand"].sum()
        flat_fill = d["constant_fill"].sum()
        model_uplift = (rec / recorded - 1) * 100
        flat_uplift = (flat_fill / recorded - 1) * 100
        return pd.Series({"days": len(d),
                          "recorded_mean": d["sale_amount"].mean(),
                          "recovered_mean": d["recovered_demand"].mean(),
                          "uplift_%": model_uplift,
                          "uplift_%_flat_fill": flat_uplift,
                          "model_vs_flat_%": model_uplift / flat_uplift * 100})

    return cens.groupby("censoring_band", observed=True).apply(summarise, include_groups=False)


def correction_by_period(daily: pd.DataFrame, make_model=stage1_lgbm, n_days: int = N_TEST_DAYS,
                         random_state: int = config.RANDOM_STATE,
                         save: bool = True) -> pd.DataFrame:
    """Does the one fitted multiplier still hold weeks later, where `run` actually applies it?

    `bias_correction` is a single number fitted inside the 62-day training window and then applied
    to every recovered hour out to the test week. This refits it separately in each later window.
    Stable means the single multiplier is sound and needs no per-period patching; drift means the
    shipped number is stale and `recovered_demand` is tilted more the further a day sits from
    training - which matters because that column is what the forecaster is trained on.

    Full-shelf days only and every one of them held out of the fit, so the calendar is the only
    thing changing between rows. The test week is not read.
    """
    clean = daily[(daily.is_censored == 0) & (daily.period != "test")]
    days = pd.concat([d.sample(n=min(n_days, len(d)), random_state=random_state)
                      for _, d in clean.groupby("period", observed=True)])
    recovered = _recover_test_days(hours(daily), days, make_model, random_state)

    scored = days.set_index(KEY).join(recovered.rename("recovered")).dropna(subset=["recovered"])
    board = scored.groupby("period", observed=True).apply(lambda d: pd.Series({
        "n_clean_days": len(d),
        "correction": d["sale_amount"].sum() / d["recovered"].sum()}), include_groups=False)
    board = board.reindex(["training", "validation", "calibration"])   # calendar order, not A-Z
    shipped = (load_params() or {}).get("bias_correction", float("nan"))
    board["shipped_is_off_by_%"] = (shipped / board["correction"] - 1) * 100

    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        board.round(4).to_csv(config.CORRECTION_BY_PERIOD)
    return board.round(4)


def how_much_was_hidden(recovered: pd.DataFrame) -> dict:
    """How much demand the till never recorded, subset-wide - the size of the problem in one line.

    Every figure is signed the same way as the rest of the project: POSITIVE means recovered demand
    exceeds recorded sales, i.e. the till under-counted by that much - the same direction every other
    signed figure in the project uses.
    """
    recorded, rec = recovered["sale_amount"].sum(), recovered["recovered_demand"].sum()
    cens = recovered[recovered.is_censored == 1]
    rec_c, recd_c = cens["recovered_demand"].sum(), cens["sale_amount"].sum()
    return dict(under_recorded_all_days_pct=round((rec / recorded - 1) * 100, 1),
                under_recorded_censored_days_pct=round((rec_c / recd_c - 1) * 100, 1),
                censored_day_share_pct=round((recovered.is_censored == 1).mean() * 100, 1))


def hour_level_table(daily: pd.DataFrame, random_state: int = config.RANDOM_STATE,
                     save: bool = True) -> pd.DataFrame:
    """Appendix only - the same candidates scored on individual held-out hours instead of on days.

    Kept to show WHY the obvious metric cannot be used: most hours sell nothing, so predicting zero
    everywhere scores WAPE = 1.000 exactly, and every real candidate lands near or above that. A
    metric minimised by doing no recovery at all cannot rank de-censoring models - which is the
    argument for ranking them on daily totals in `compare_models`.
    """
    from sklearn.model_selection import train_test_split

    pool = training_hours(hours(daily))
    X_tr, X_val, y_tr, y_val = train_test_split(
        _model_matrix(pool), pool["hour_sale"].values, test_size=0.15, random_state=random_state)

    rows = {}
    for name, make in CANDIDATES.items():
        print(f"hour-level fit: {name} on {len(X_tr):,} clean hours ...")
        model = make(random_state)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        scores = bias_scores(y_val, np.clip(model.predict(X_val), 0, None))
        rows[name] = ({k: round(v, 4) for k, v in scores.items()}
                      | {"fit_seconds": round(time.perf_counter() - t0, 1)})

    board = pd.DataFrame(rows).T.sort_values("wape")
    board.loc["all_zeros"] = dict(bias_scores(y_val, np.zeros_like(y_val)), fit_seconds=0.0)
    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        board.to_csv(config.HOUR_LEVEL_TABLE)
    return board
