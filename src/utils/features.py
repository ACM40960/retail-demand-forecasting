"""Lag/rolling features - one implementation, used on raw `sale_amount` (baselines) and later on
`recovered_demand` (TFT inputs, prefix `r_`), so the two can never drift apart.
"""
import pandas as pd

GROUP = ["store_id", "product_id"]
_MIN_PERIODS = {7: 3, 14: 5}   # start-of-series tolerance, by window


def add_lagged_features(df: pd.DataFrame, source: str, prefix: str = "",
                        lags=(7, 14), mean_windows=(7, 14), std_windows=(7,),
                        fill: bool = True) -> pd.DataFrame:
    """Add `{prefix}lag{k}` and `{prefix}roll{w}_mean/std` per series.

    Everything shifts before aggregating, so day t never enters its own feature. `fill=True`
    back-fills start-of-series gaps; `fill=False` leaves them for the caller to drop.
    """
    df = df.sort_values(GROUP + ["dt"]).copy()
    g = df.groupby(GROUP)[source]

    def rolling(w, how):
        mp = _MIN_PERIODS.get(w, 1)
        return g.transform(lambda s: how(s.shift(1).rolling(w, min_periods=mp)))

    lag_cols = [f"{prefix}lag{k}" for k in lags]
    mean_cols = [f"{prefix}roll{w}_mean" for w in mean_windows]
    std_cols = [f"{prefix}roll{w}_std" for w in std_windows]

    for k, col in zip(lags, lag_cols):
        df[col] = g.shift(k)
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
