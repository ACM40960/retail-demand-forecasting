"""Lag/rolling features and the hourly explode.

`add_lagged_features` is the one lag implementation, used on raw `sale_amount` (baselines, recovery
context) and later on `recovered_demand` (forecasting inputs), so the two can never drift apart.
"""
import numpy as np
import pandas as pd

GROUP = ["store_id", "product_id"]
_MIN_PERIODS = {7: 3, 14: 5}   # start-of-series tolerance, by window

FEATURES = [
    "dow", "discount", "holiday_flag", "activity_flag",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
    "lag7", "lag14", "roll7_mean", "roll7_std", "roll14_mean",
    "store_id", "product_id", "first_category_id", "second_category_id", "third_category_id",
]
CATEGORICAL = ["store_id", "product_id",
               "first_category_id", "second_category_id", "third_category_id"]

# the hourly Stage-1 model additionally sees which hour of day it is
HOURLY_FEATURES = ["hour"] + FEATURES
HOURLY_CATEGORICAL = CATEGORICAL


def add_lagged_features(df: pd.DataFrame, source: str,
                        lags=(7, 14), mean_windows=(7, 14), std_windows=(7,),
                        fill: bool = True) -> pd.DataFrame:
    """Add `lag{k}` and `roll{w}_mean/std` per series, computed from `source`.

    Everything shifts before aggregating, so day t never enters its own feature. `fill=True`
    back-fills start-of-series gaps; `fill=False` leaves them for the caller to drop.
    """
    df = df.sort_values(GROUP + ["dt"]).copy()
    g = df.groupby(GROUP)[source]   # one group per series (store, product); everything below
                                    # operates within a series, never mixing one product's history
                                    # into another's

    def rolling(w, how):
        # shift(1) first, so the window ends the day BEFORE the current row - the same reason
        # every lag below is a shift, not a lookup: day t must never see its own value.
        # min_periods below `w` (_MIN_PERIODS) is the start-of-series tolerance: a series only 3
        # days into training still gets a 7-day rolling mean, just from fewer than 7 days
        mp = _MIN_PERIODS.get(w, 1)
        return g.transform(lambda s: how(s.shift(1).rolling(w, min_periods=mp)))

    lag_cols = [f"lag{k}" for k in lags]
    mean_cols = [f"roll{w}_mean" for w in mean_windows]
    std_cols = [f"roll{w}_std" for w in std_windows]

    for k, col in zip(lags, lag_cols):
        df[col] = g.shift(k)   # shift(k) inside a groupby reads k rows earlier IN THE SAME SERIES,
                               # so lag7 is literally "what this exact product sold 7 days ago"
    for w, col in zip(mean_windows, mean_cols):
        df[col] = rolling(w, lambda r: r.mean())
    for w, col in zip(std_windows, std_cols):
        df[col] = rolling(w, lambda r: r.std())

    if fill:
        # expanding, not whole-series: a whole-series mean would pull validation/test values
        # backwards into start-of-training rows
        prior_mean = g.transform(lambda s: s.shift(1).expanding().mean())
        for c in lag_cols + mean_cols:
            df[c] = df[c].fillna(prior_mean).fillna(0.0)
        for c in std_cols:
            df[c] = df[c].fillna(0.0)
    return df


def censoring_bucket(daily: pd.DataFrame) -> pd.Series:
    """Each series' share of TRAINING days that sold out, cut into four bands and indexed by series
    so it can be attached to any frame.

    Training rows only. A rate computed over the whole train file would fold validation and
    calibration days into a column results are grouped by, which is the same leak as using it as a
    feature. Bands are the axis `recovery_impact` has to be read on: pooled, the two targets are
    identical on the rows that get scored.
    """
    rate = daily[daily["period"] == "training"].groupby(GROUP)["is_censored"].mean()
    return pd.cut(rate, [0, 0.25, 0.5, 0.75, 1.001], right=False,
                  labels=["<25% censored", "25-50%", "50-75%", ">=75%"])


def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw-sales lag/roll context plus `censoring_frac`, the share of the 16 active-window hours
    that were out of stock."""
    df = add_lagged_features(df, source="sale_amount", lags=(7, 14),
                             mean_windows=(7, 14), std_windows=(7,), fill=True)
    df["censoring_frac"] = (df["stock_hour6_22_cnt"] / 16.0).clip(0, 1)
    return df


def explode_hourly(hourly_df: pd.DataFrame, daily_context: pd.DataFrame) -> pd.DataFrame:
    """Turn the hourly companion parquet (length-24 lists per day) into one row per
    (store_id, product_id, dt, hour), joined with the daily context features."""
    h = hourly_df[["store_id", "product_id", "dt", "hours_sale", "hours_stock_status"]]
    # each cell of hours_sale/hours_stock_status is itself a 24-long list (one value per hour);
    # np.stack turns a column of lists into a proper 2-D array, one row per day, 24 columns
    sale = np.stack(h["hours_sale"].to_numpy())              # (n_days, 24)
    stockout = np.stack(h["hours_stock_status"].to_numpy())  # (n_days, 24)
    n_days, n_hours = sale.shape
    assert n_hours == 24, f"expected 24-length hourly vectors, got {n_hours}"

    # flatten (n_days, 24) down to one row per (day, hour): repeat each day's id/date 24 times,
    # tile the hour numbers 0-23 once per day, then ravel the two 2-D arrays in the same row-major
    # order so hour_sale[i] and hour_stockout[i] line up with the (day, hour) at row i
    long = pd.DataFrame({
        "store_id": np.repeat(h["store_id"].to_numpy(), n_hours),
        "product_id": np.repeat(h["product_id"].to_numpy(), n_hours),
        "dt": np.repeat(h["dt"].to_numpy(), n_hours),
        "hour": np.tile(np.arange(n_hours), n_days),
        "hour_sale": sale.ravel(),
        "hour_stockout": stockout.ravel(),
    })

    # dict.fromkeys dedupes while keeping order (store_id/product_id/dt appear both in the base
    # list and, redundantly, wherever FEATURES also lists them) - then keep only the columns that
    # actually exist on daily_context, since not every caller's frame carries every FEATURES column
    ctx = ["store_id", "product_id", "dt", "split", "period"] + FEATURES
    ctx = [c for c in dict.fromkeys(ctx) if c in daily_context.columns]
    return long.merge(daily_context[ctx], on=["store_id", "product_id", "dt"], how="left")
