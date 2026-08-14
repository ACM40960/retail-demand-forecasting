"""Forecast metrics matching Dingdong's published FreshRetailNet `evaluation()`.

Accuracy is scored per date on non-stockout rows only (`stock_hour6_22_cnt == 0`), then
averaged across dates.
"""
import re

import numpy as np
import pandas as pd


def per_date_scores(df: pd.DataFrame, actual: str = "sale_amount", pred: str = "pred") -> dict:
    """WAPE/WPE/MAE per date on non-stockout rows, averaged across dates."""
    W, P, M = [], [], []
    for _, s in df.query("stock_hour6_22_cnt == 0").groupby("dt"):
        if s[actual].sum() == 0:   # skip degenerate dates
            continue
        ae = (s[actual] - s[pred]).abs()
        W.append(ae.sum() / s[actual].sum())
        P.append((s[pred] - s[actual]).sum() / s[actual].sum())
        M.append(ae.mean())
    if not W:
        raise ValueError("no scorable dates: every non-stockout date had zero total sales")
    return dict(WAPE=round(np.mean(W), 4), WPE=round(np.mean(P), 4), MAE=round(np.mean(M), 4))


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss for one quantile tau in (0,1)."""
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


Q_COLS = (("q10", 0.1), ("q50", 0.5), ("q90", 0.9))


def quantile_scores(df: pd.DataFrame, actual: str = "sale_amount") -> dict:
    """Full scorecard for a q10/q50/q90 forecast: point WAPE/WPE/MAE from q50, plus mean pinball
    and a crude 3-quantile CRPS proxy. Shared by the XGBoost baseline and the TFT."""
    scores = per_date_scores(df, actual=actual, pred="q50")

    nz = df["stock_hour6_22_cnt"] == 0
    y = df.loc[nz, actual].values
    mean_pb = float(np.mean([pinball(y, df.loc[nz, c].values, tau) for c, tau in Q_COLS]))
    scores["pinball(avg)"] = round(mean_pb, 4)
    scores["CRPS~"] = round(2 * mean_pb / (np.abs(y).mean() + 1e-9), 4)
    return scores


def pinball_by_quantile(df: pd.DataFrame, actual: str = "sale_amount") -> dict:
    """Pinball loss at EVERY quantile column present, not just the three `Q_COLS` averages.

    `quantile_scores` reports `pinball(avg)` over q10/q50/q90 only, while the models fit six quantiles -
    so q025, q80 and q975 are trained but never measured. q80 is the one that matters: the ordering
    stage reads its order quantity near there, which makes it the only column whose loss converts
    directly into money. This reports it without changing `pinball(avg)`, so every score already
    computed stays comparable.

    Same row convention as `quantile_scores`: non-stockout rows, against recorded sales.
    """
    nz = df["stock_hour6_22_cnt"] == 0
    y = df.loc[nz, actual].values
    taus = {c: float(c[1:]) / (1000 if len(c) > 3 else 100) for c in df.columns
            if re.fullmatch(r"q\d{2,3}", str(c))}
    return {f"pinball@{tau:g}": round(pinball(y, df.loc[nz, c].values, tau), 5)
            for c, tau in sorted(taus.items(), key=lambda kv: kv[1])}


def attach_bucket(df: pd.DataFrame, bucket: pd.Series) -> pd.Series:
    """`bucket` (a per-series Series, e.g. `censoring_bucket`) aligned to `df`'s rows.

    The ID-dtype cast is the whole reason this is a function rather than a `reindex` at each call
    site: `forecast._prepare` stores IDs as strings for the TFT's embeddings while `censoring_bucket`
    is int-indexed, and reindexing across that matches nothing - silently, leaving a clean-looking
    table with only a pooled row. Every per-band table goes through here, so the guard is unmissable.
    """
    levels = bucket.index.levels
    keys = pd.DataFrame({c: df[c].astype(levels[i].dtype)
                         for i, c in enumerate(["store_id", "product_id"])})
    # set_axis, not to_numpy: keeps the categorical dtype, so bands stay in severity order
    band = bucket.reindex(pd.MultiIndex.from_frame(keys)).set_axis(df.index)
    if band.isna().all():
        raise ValueError(f"no row matched the bucket index - {len(df):,} rows, 0 banded. Frame keys "
                         f"are {df['store_id'].dtype}/{df['product_id'].dtype}, bucket index is "
                         f"{levels[0].dtype}/{levels[1].dtype}")
    return band


def scores_by_bucket(df: pd.DataFrame, bucket: pd.Series, actual: str = "sale_amount"):
    """`quantile_scores` per band of `bucket`, plus a pooled ALL row. The pooled row is the one that
    hides the effect - keep both."""
    band = attach_bucket(df, bucket)

    def row(g):   # n counts the rows that are actually scored, not the rows in the band
        return {"n_scored": int((g["stock_hour6_22_cnt"] == 0).sum()),
                **quantile_scores(g, actual=actual)}

    rows = {str(b): row(g) for b, g in df.groupby(band, observed=True)}
    return pd.DataFrame({**rows, "ALL": row(df)}).T


def lost_sales_vs_waste(frames: dict, bucket: pd.Series, quantile: str = "q50",
                        actual: str = "sale_amount") -> pd.DataFrame:
    """The raw-vs-recovered trade, per band: demand that walked away against units that get binned.

    `frames` is `{"raw": forecast_df, "recovered": forecast_df}`. Scored on non-stockout rows only,
    where recorded sales ARE demand, so both sides are real rather than estimated - which also makes
    this the regime that PENALISES recovery, since those are the quiet days where it over-orders.

    The column that matters is `worth_it_above_ratio` = waste added / lost sales recovered: how much
    worse an empty shelf has to be than a bin before recovery pays. Below 1.0 it pays even when the
    two cost the same. This is the one claim in the project that needs no accuracy improvement to
    hold, and WAPE cannot express it - WAPE charges a unit too many and a unit too few identically.
    """
    def one(df: pd.DataFrame, label: str) -> pd.DataFrame:
        d = df[df["stock_hour6_22_cnt"] == 0].copy()
        d["band"] = attach_bucket(d, bucket)
        d["missed"] = (d[actual] - d[quantile]).clip(lower=0)
        d["binned"] = (d[quantile] - d[actual]).clip(lower=0)
        g = d.groupby("band", observed=True).agg(demand=(actual, "sum"),
                                                 missed=("missed", "sum"),
                                                 binned=("binned", "sum"))
        return pd.DataFrame({f"lost_sales_%_{label}": (g.missed / g.demand * 100).round(1),
                             f"waste_%_{label}": (g.binned / g.demand * 100).round(1)})

    trade = pd.concat([one(df, label) for label, df in frames.items()], axis=1)
    raw, rec = list(frames)
    trade["lost_sales_recovered_pts"] = (trade[f"lost_sales_%_{raw}"]
                                         - trade[f"lost_sales_%_{rec}"]).round(1)
    trade["waste_added_pts"] = (trade[f"waste_%_{rec}"] - trade[f"waste_%_{raw}"]).round(1)
    trade["worth_it_above_ratio"] = (trade["waste_added_pts"]
                                     / trade["lost_sales_recovered_pts"]).round(2)
    return trade


def bias_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """WAPE / MAE / signed WPE over flat arrays - used by `recovery.evaluate`, which scores a flat
    set of held-out days rather than the per-date frame `per_date_scores` needs."""
    err = y_pred - y_true
    return dict(wape=float(np.abs(err).sum() / np.abs(y_true).sum()),
                mae=float(np.abs(err).mean()),
                wpe=float(err.sum() / y_true.sum()))
