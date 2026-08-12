"""M5 - Honest bands (PLAN.md Phase 5). Split conformalized quantile regression (CQR, Romano
2019) applied post-hoc to the saved TFT forecasts.

The Phase-4 80% band (q10/q90) is overconfident - it covers ~73% of actuals, not 80%. CQR fixes
this without retraining: calibrate one offset Q on the CALIBRATION period (never seen in
training), then widen the band to [q10-Q, q90+Q]. Under exchangeability that restores >= nominal
coverage. Scored on non-stockout days only (ground rule 3); reads saved forecasts and writes
corrected test intervals + a results JSON for the Trust-check tab (ground rule 7).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .utils import config
from . import forecast

LOWER, UPPER, MEDIAN = "q10", "q90", "q50"   # q10/q90 = the 80% central band
NOMINAL = 0.80
RELIABILITY_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

def load_forecasts(periods=("calibration", "validation", "test"), master: pd.DataFrame = None,
                   forecast_tag: str = None, observed_only: bool = True):
    """Saved forecast per period, joined to actuals. `forecast_tag` selects which Phase-4 target
    to read - "recovered" (default) or "raw"; every current run already carries all six
    quantiles (q025/q10/q50/q80/q90/q975) in that ONE file, so the 80% and 95% bands are read
    from the same saved forecast, just corrected by separate `run()` calls below.

    Adds `observed` (True on non-stockout days, where demand was truly seen), `y` (= sale_amount,
    meaningful ONLY where observed) and `recovered_demand` (the filled-in series, meaningful
    everywhere - what Phase 6 simulates against).

    `observed_only=True` (default) returns just the non-stockout rows, which is what calibration
    and accuracy scoring need (ground rule 3). Pass False to keep every product-day: the ORDER
    simulation runs on all days, including the censored ones the method exists to fix.
    Returns ({period: frame}, master)."""
    master = master if master is not None else forecast.build_master_frame()
    cols = ["store_id", "product_id", "dt", "sale_amount", "recovered_demand",
            "stock_hour6_22_cnt"]
    paths = config.forecast_paths(forecast_tag or "recovered")
    id_dtypes = master[["store_id", "product_id"]].dtypes.to_dict()

    out = {}
    for p in periods:
        if not paths[p].exists():
            raise FileNotFoundError(
                f"{paths[p].name} missing - run the Phase-4 forecast first "
                f"(forecast.run(..., tag='{forecast_tag or 'recovered'}') or, for the test "
                f"period, forecast.forecast_test).")
        # `forecast._prepare` casts store_id/product_id to STRING categoricals for the TFT's
        # embeddings, so every saved forecast parquet carries them as strings - but `master`
        # (and everything else in the pipeline: recovery, data_io, orders.py's KEY merges) keeps
        # them as the raw integer IDs. Merging string-vs-int keys doesn't silently mismatch, it
        # raises (pandas refuses to merge on differently-typed columns) - cast back here, once,
        # rather than at every call site (`02_forecasting.ipynb`'s `int_keys` is the same fix,
        # needed there because that notebook merges forecast frames directly instead of through
        # this loader).
        # `forecast.CARRY` already saves sale_amount/stock_hour6_22_cnt/recovered_demand INTO
        # every forecast parquet (so forecast.py's own scoring can run standalone) - merging
        # `master[cols]` on top without dropping them first doesn't error, it silently suffixes
        # every one of them to `_x`/`_y`, and every line below reading the bare column name
        # would raise (or worse, silently read the wrong one if the suffix ever changed).
        # `master` is treated as authoritative here; the carried copies are redundant duplicates
        # of the exact same source rows, dropped rather than merge-suffixed.
        fc = pd.read_parquet(paths[p]).astype(id_dtypes)
        fc = fc.drop(columns=[c for c in cols[3:] if c in fc.columns])
        merged = fc.merge(master[cols], on=cols[:3], how="left")
        merged["observed"] = merged["stock_hour6_22_cnt"] == 0
        merged["y"] = merged["sale_amount"]
        out[p] = merged[merged["observed"]].copy() if observed_only else merged
    return out, master


def nonconformity(df: pd.DataFrame, lower=LOWER, upper=UPPER) -> np.ndarray:
    """CQR score E_i = max(q_lo - y, y - q_hi): how far y falls outside the band, negative when
    comfortably inside (Romano 2019 eq. 6)."""
    return np.maximum(df[lower].values - df["y"].values, df["y"].values - df[upper].values)


def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample split-conformal quantile - ceil((n+1)(1-alpha))/n of the scores - i.e.
    the smallest widening that guarantees >= 1-alpha coverage on exchangeable data."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def fit_offsets(calib: pd.DataFrame, alpha: float, group_col: str = None, min_group: int = 100,
                lower=LOWER, upper=UPPER):
    """Global CQR offset plus optional per-group (Mondrian) offsets. Groups with fewer than
    `min_group` calibration rows fall back to the global Q, so a sparse group cannot produce a
    wild one. Returns (global_Q, {group: Q})."""
    global_q = conformal_offset(nonconformity(calib, lower, upper), alpha)
    group_q = {}
    if group_col:
        for g, s in calib.groupby(group_col):
            group_q[str(g)] = (conformal_offset(nonconformity(s, lower, upper), alpha)
                               if len(s) >= min_group else global_q)
    return global_q, group_q


def interval_coverage(df: pd.DataFrame, lower: str, upper: str) -> dict:
    """Empirical coverage and mean width of [lower, upper] on the observed rows."""
    y = df["y"].values
    return dict(coverage=round(float(((y >= df[lower].values) & (y <= df[upper].values)).mean()), 4),
                mean_width=round(float((df[upper].values - df[lower].values).mean()), 4),
                n=int(len(df)))


def kupiec_pof(n: int, failures: int, alpha: float) -> dict:
    """Kupiec proportion-of-failures LR test. H0: true miss-rate == alpha, chi-square(1).
    p >= 0.05 means calibration cannot be rejected (PLAN.md §7 pass mark)."""
    x = int(failures)
    pi = x / n if n else 0.0
    if x == 0 or x == n:   # the LR is undefined through the logs at the extremes
        return dict(miss_rate=round(pi, 4), expected=alpha, LR_pof=None, p_value=None, calibrated=None)
    lr = float(-2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi)))
    p = float(chi2.sf(lr, df=1))
    return dict(miss_rate=round(pi, 4), expected=alpha, LR_pof=round(lr, 4),
                p_value=round(p, 4), calibrated=bool(p >= 0.05))


def reliability_curve(calib: pd.DataFrame, evalset: pd.DataFrame, levels=RELIABILITY_LEVELS,
                      median=MEDIAN) -> pd.DataFrame:
    """Nominal-vs-empirical coverage at every level, via *symmetric* split conformal on q50:
    calibrate a half-width h from |y - q50|, measure coverage of [q50-h, q50+h] on `evalset`.
    A different estimator from the asymmetric CQR band, which only gives the 80% point - three
    saved quantiles cannot produce a full curve. Honest = points on the diagonal."""
    resid = np.abs(calib["y"].values - calib[median].values)
    y = evalset["y"].values
    rows = []
    for lvl in levels:
        h = conformal_offset(resid, alpha=1 - lvl)
        lo = np.clip(evalset[median].values - h, 0, None)
        hi = evalset[median].values + h
        rows.append(dict(nominal=lvl,
                         empirical=round(float(((y >= lo) & (y <= hi)).mean()), 4),
                         mean_width=round(float((hi - lo).mean()), 4)))
    return pd.DataFrame(rows)


def run(nominal: float = NOMINAL, lower: str = LOWER, upper: str = UPPER, group_col: str = None,
        calibration_periods=("calibration",), save: bool = True, verbose: bool = True,
        tag: str = None, forecast_tag: str = None) -> dict:
    """Fit CQR on the calibration set, apply it to the held-out periods, and report before/after
    coverage, Kupiec and band width. Saves the corrected TEST intervals + a results JSON.

    - `lower`/`upper`: the band to correct. The 80% band is the default
      (`run(0.80, "q10", "q90", tag="wide80")`); the 95% band is
      `run(0.95, "q025", "q975", tag="wide95")` - same saved forecast, a different pair of
      columns and a separate offset, which is why `orders.py` reads two conformal *outputs*
      even though there is only one Phase-4 forecast file behind them.
    - `calibration_periods`: pass ("validation", "calibration") to POOL a larger, more temporally
      representative calibration set (the exchangeability fix).
    - `group_col`: switches on Mondrian (group-conditional) CQR.
    - `forecast_tag`: which Phase-4 *target* to read - "recovered" (default) or "raw"; `tag`:
      suffixes the outputs written here (e.g. "wide80"/"wide95" so both band corrections can be
      saved side by side).
    """
    alpha = round(1 - nominal, 4)
    calibration_periods = tuple(calibration_periods)
    if "test" in calibration_periods:
        raise ValueError("the test week is never a calibration set (PLAN.md §5)")
    eval_periods = [p for p in ("validation", "test") if p not in calibration_periods]

    # every product-day is loaded; `observed` marks the non-stockout rows. The offset is FIT and
    # SCORED on those only (ground rule 3 - demand was not seen on the rest), but it is APPLIED
    # to and saved for all of them, because Phase 6 orders on every day and the censored days are
    # exactly where the fill-in layer pays off.
    data, _ = load_forecasts(tuple(dict.fromkeys(calibration_periods + tuple(eval_periods))),
                             forecast_tag=forecast_tag, observed_only=False)
    missing = [c for c in (lower, upper) if c not in data["test"].columns]
    if missing:
        raise KeyError(f"forecast missing {missing} for the {nominal:.0%} band - re-run "
                       f"forecast.run(..., tag='{forecast_tag or 'recovered'}') including the "
                       f"'test' period (forecast.QUANTILES already covers q025/q10/q50/q80/q90/"
                       f"q975, so no separate model is needed for a wider band).")

    calib = pd.concat([data[p] for p in calibration_periods], ignore_index=True)
    calib = calib[calib["observed"]].copy()
    global_q, group_q = fit_offsets(calib, alpha, group_col=group_col, lower=lower, upper=upper)

    results = {"method": "CQR" + (f"-mondrian[{group_col}]" if group_col else ""),
               "nominal": nominal, "alpha": alpha, "band": [lower, upper], "group_col": group_col,
               "calibration_set": list(calibration_periods), "calibration_n": int(len(calib)),
               "global_offset": round(global_q, 4), "periods": {}}

    corrected_test = None
    for name in eval_periods:
        corr = data[name].copy()                       # ALL product-days
        Q = (corr[group_col].astype(str).map(group_q).fillna(global_q).values
             if group_q and group_col else global_q)
        corr[f"{lower}_c"] = np.clip(corr[lower].values - Q, 0, None)   # demand >= 0
        corr[f"{upper}_c"] = corr[upper].values + Q

        obs = corr[corr["observed"]]                   # scoring set (ground rule 3)
        y = obs["y"].values
        fail = int(((y < obs[f"{lower}_c"].values) | (y > obs[f"{upper}_c"].values)).sum())
        results["periods"][name] = dict(
            n_rows=int(len(corr)), n_observed=int(len(obs)),
            uncorrected=interval_coverage(obs, lower, upper),
            corrected=interval_coverage(obs, f"{lower}_c", f"{upper}_c"),
            kupiec_corrected=kupiec_pof(len(obs), fail, alpha))
        if name == "test":
            corrected_test = corr

    results["reliability_test"] = reliability_curve(
        calib, data["test"][data["test"]["observed"]]).to_dict("records")

    if save:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        corrected_test.to_parquet(config.OUTPUTS_DIR / f"forecast_test_conformal{suffix}.parquet",
                                  index=False)
        (config.OUTPUTS_DIR / f"conformal_results{suffix}.json").write_text(json.dumps(results, indent=2))

    if verbose:
        print(f"\n{results['method']} {nominal:.0%} [{lower}/{upper}]  |  "
              f"calib={'+'.join(calibration_periods)} ({len(calib):,} observed rows)  |  "
              f"offset Q = {global_q:.4f}")
        for name, p in results["periods"].items():
            u, c, k = p["uncorrected"], p["corrected"], p["kupiec_corrected"]
            print(f"  [{name:10s}] scored on {p['n_observed']:>6,} of {p['n_rows']:>6,} rows"
                  f"  uncorrected {u['coverage']:.3f} (w {u['mean_width']:.3f})"
                  f"  ->  corrected {c['coverage']:.3f} (w {c['mean_width']:.3f})"
                  f"  Kupiec p={k['p_value']} ok={k['calibrated']}")
        print(f"  saved test intervals for ALL {results['periods']['test']['n_rows']:,} "
              f"product-days (Phase 6 orders on every day, not just the observed ones)")
    return results
