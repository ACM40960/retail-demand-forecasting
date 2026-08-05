"""Raw-(censored)-sales baselines: the accuracy bar everything downstream must beat.

Fit on training days - on censored sales, on purpose - and scored on non-stockout validation days,
the dataset's own convention. The test week is never read here.

These are an accuracy bar, not evidence about censoring: scoring skips stockout rows, which are
exactly the rows censoring distorts, so WPE sits near zero however under-counted the series are.
"""
import numpy as np
import pandas as pd

from .utils import config
from .utils.features import add_lagged_features
from .utils.metrics import per_date_scores, pinball, quantile_scores

BASELINE_FEATS = ["dow", "discount", "holiday_flag", "activity_flag", "precpt",
                  "avg_temperature", "avg_humidity", "avg_wind_level",
                  "lag7", "lag14", "roll7_mean", "roll7_std"]


def add_history_features(sub_train: pd.DataFrame, sub_eval: pd.DataFrame) -> pd.DataFrame:
    """Eval rows with lag/rolling features, as a 7-day-ahead forecast would see them.

    Over a 14-day window the second week's lag7 comes from the first week of the eval window, not
    from training. That is correct for a 7-day-ahead forecast and matches what the TFT sees, so the
    comparison stays like-for-like. Rolling stats are frozen at the last 7 training days rather
    than rolled forward, which if anything handicaps the baseline.
    """
    both = add_lagged_features(pd.concat([sub_train, sub_eval], ignore_index=True),
                               source="sale_amount", lags=(7, 14),
                               mean_windows=(), std_windows=(), fill=False)
    ev = (both[both["dt"].isin(sub_eval["dt"].unique())]
          .drop(columns=["roll7_mean", "roll7_std"], errors="ignore"))   # avoid _x/_y on merge

    last7 = (sub_train.sort_values("dt").groupby(["store_id", "product_id"])["sale_amount"]
             .agg(roll7_mean=lambda s: s.tail(7).mean(), roll7_std=lambda s: s.tail(7).std())
             .reset_index())
    return ev.merge(last7, on=["store_id", "product_id"], how="inner")   # drops no-history series


def xgboost_quantile(sub_train: pd.DataFrame, ev_feat: pd.DataFrame,
                     random_state: int = config.RANDOM_STATE) -> dict:
    """Baseline 2: XGBoost quantile (q10/q50/q90) - the strongest baseline, the bar to clear."""
    from xgboost import XGBRegressor

    # fill=False so incomplete start-of-series rows are dropped rather than back-filled
    tr = add_lagged_features(sub_train, source="sale_amount", lags=(7, 14),
                             mean_windows=(7,), std_windows=(7,), fill=False)
    trf = tr.dropna(subset=["lag14", "roll7_std"])

    Xtr, ytr = trf[BASELINE_FEATS], trf["sale_amount"]
    Xev = ev_feat[BASELINE_FEATS].fillna(0)
    xv = ev_feat.copy()
    for tau, col in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=tau,
                         n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=random_state)
        m.fit(Xtr, ytr)
        xv[col] = np.clip(m.predict(Xev), 0, None)
    return quantile_scores(xv, actual="sale_amount")


def sarima_sampled(sub_train: pd.DataFrame, sub_eval: pd.DataFrame,
                   n_series: int = config.SARIMA_SAMPLE,
                   random_state: int = config.RANDOM_STATE) -> dict:
    """Baseline 3: SARIMA on a random sample of series (slow classical-stats reference)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    series = list(sub_train.groupby(["store_id", "product_id"]))
    rng = np.random.default_rng(random_state)
    pick = [series[i] for i in rng.choice(len(series), min(n_series, len(series)), replace=False)]
    rows, n_failed = [], 0
    for (s, p), tr in pick:
        y = tr.sort_values("dt")["sale_amount"].astype(float).values
        ev_sp = sub_eval[(sub_eval.store_id == s) & (sub_eval.product_id == p)].sort_values("dt")
        if len(ev_sp) == 0:
            continue
        try:
            f = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 7),
                        enforce_stationarity=False, enforce_invertibility=False
                        ).fit(disp=False).forecast(len(ev_sp))
        except Exception:   # SARIMA genuinely fails to converge on sparse series; count those
            f = np.repeat(y[-7:].mean(), len(ev_sp))
            n_failed += 1
        rows.append(ev_sp.assign(pred=np.clip(f, 0, None)))
    return per_date_scores(pd.concat(rows)) | {"n_series": len(pick), "n_failed": n_failed}


def score_baselines(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                    random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """All three baselines for one (train -> eval window) pair. Shared by `build_scorecard` and
    the forecasting stage's like-for-like comparison against the forecaster."""
    ev = add_history_features(train_df, eval_df)

    # baseline 1: repeat last week (lag-7), falling back to the last-7-day mean
    naive = ev.assign(pred=ev["lag7"].fillna(ev["roll7_mean"]))
    scores = {"seasonal_naive": per_date_scores(naive) | {
        "pinball@0.5": round(pinball(naive["sale_amount"].values, naive["pred"].values, 0.5), 4)},
        "xgboost_quantile": xgboost_quantile(train_df, ev, random_state),
        "sarima": sarima_sampled(train_df, eval_df, random_state=random_state)}
    return pd.DataFrame(scores).T


def build_scorecard(daily: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Baselines scored Training -> VALIDATION, both cut from the train file by date.

    The shipped eval split is the test week and is not read here, so `daily` may contain it
    without contaminating anything.
    """
    train_file = daily[daily.split == "train"]
    dt = train_file["dt"]
    board = score_baselines(
        train_file[dt <= pd.Timestamp(config.TRAIN_END)],
        train_file[dt.between(pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END))])
    if save:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        board.to_csv(config.BASELINE_SCORECARD)
    return board
