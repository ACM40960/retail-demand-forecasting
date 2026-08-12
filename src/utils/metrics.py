"""Forecast metrics matching Dingdong's published FreshRetailNet `evaluation()`.

Accuracy is scored per date on non-stockout rows only (`stock_hour6_22_cnt == 0`), then
averaged across dates.
"""
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


def scores_by_bucket(df: pd.DataFrame, bucket: pd.Series, actual: str = "sale_amount"):
    """`quantile_scores` per band of `bucket` (a per-series Series, e.g. `censoring_bucket`), plus
    a pooled ALL row. The pooled row is the one that hides the effect - keep both."""
    # Cast the keys to the bucket index's own dtypes first. `forecast._prepare` stores the IDs as
    # STRINGS so the TFT treats them as embeddings, while `censoring_bucket` is indexed on the daily
    # frame's integer IDs - reindexing across that matches nothing, and it fails SILENTLY: every band
    # comes back NaN, groupby yields no groups, and the caller gets a clean-looking table holding only
    # the pooled ALL row. That happened, and cost a run. Hence the cast and the check below it.
    levels = bucket.index.levels
    keys = pd.DataFrame({c: df[c].astype(levels[i].dtype)
                         for i, c in enumerate(["store_id", "product_id"])})
    # set_axis, not to_numpy: keeps the categorical dtype, so bands stay in severity order
    band = bucket.reindex(pd.MultiIndex.from_frame(keys)).set_axis(df.index)
    if band.isna().all():
        raise ValueError(f"no row matched the bucket index - {len(df):,} rows, 0 banded. Frame keys "
                         f"are {df['store_id'].dtype}/{df['product_id'].dtype}, bucket index is "
                         f"{levels[0].dtype}/{levels[1].dtype}")

    def row(g):   # n counts the rows that are actually scored, not the rows in the band
        return {"n_scored": int((g["stock_hour6_22_cnt"] == 0).sum()),
                **quantile_scores(g, actual=actual)}

    rows = {str(b): row(g) for b, g in df.groupby(band, observed=True)}
    return pd.DataFrame({**rows, "ALL": row(df)}).T


def bias_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """WAPE / MAE / signed WPE over flat arrays - used by `recovery.evaluate`, which scores a flat
    set of held-out days rather than the per-date frame `per_date_scores` needs."""
    err = y_pred - y_true
    return dict(wape=float(np.abs(err).sum() / np.abs(y_true).sum()),
                mae=float(np.abs(err).mean()),
                wpe=float(err.sum() / y_true.sum()))
