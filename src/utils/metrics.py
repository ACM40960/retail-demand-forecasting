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


def quantile_scores(df: pd.DataFrame, actual: str = "sale_amount",
                    q_cols=(("q10", 0.1), ("q50", 0.5), ("q90", 0.9)),
                    stockout_col: str = "stock_hour6_22_cnt") -> dict:
    """Full scorecard for a q10/q50/q90 forecast: point WAPE/WPE/MAE from the median, plus mean
    pinball and a crude 3-quantile CRPS proxy. Shared by the XGBoost baseline and the TFT."""
    median_col = next(c for c, tau in q_cols if tau == 0.5)
    scores = per_date_scores(df, actual=actual, pred=median_col)

    nz = df[stockout_col] == 0
    y = df.loc[nz, actual].values
    mean_pb = float(np.mean([pinball(y, df.loc[nz, c].values, tau) for c, tau in q_cols]))
    scores["pinball(avg)"] = round(mean_pb, 4)
    scores["CRPS~"] = round(2 * mean_pb / (np.abs(y).mean() + 1e-9), 4)
    return scores
