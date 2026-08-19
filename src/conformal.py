"""Honest bands. Split conformalized quantile regression (CQR, Romano 2019) applied post-hoc to a
saved quantile forecast - no retraining.

The raw 80% band (q10/q90) is overconfident: it covers ~73% of actuals on validation, not 80%. CQR
calibrates one offset Q on the CALIBRATION period and widens the band to [q10-Q, q90+Q], which under
EXCHANGEABILITY restores >= nominal coverage.

That assumption is where it goes wrong, and the diagnosis drove the fix. A forecast horizon walks
forward in time, so a later week is not exchangeable with the window the offset was fitted on, and
coverage decays with distance: plain CQR overshoots on the near window and undershoots on the far one.

Two things were measured before changing anything, both WITHOUT opening the test week:

    the functional form is not the problem   multiplicative band-scaling and Mondrian per-band
                                             offsets land within 0.0003 of plain additive CQR
    the DISTANCE is the problem              coverage falls ~2.5 points per window of separation

Hence `FORWARD_DRIFT_INFLATION`: calibrate at 0.83 to land on 0.80 a window later. Applied
automatically and only where it belongs - `_is_forward` reads the frozen calendar, so a window
BEFORE the calibration set (validation) is never inflated and a window after it (test) always is.

ONE method, no switches. Mondrian per-band offsets and multiplicative band-scaling were both
measured on the harness above and land within 0.0003 of this, so they are recorded here rather than
left in the code as options nobody should pick.

Coverage is reported with a DAY-BLOCK bootstrap interval (`coverage_ci`), not a Kupiec test. Kupiec
assumes independent rows; these are clustered by date at a design effect of 13.7, so it rejects
misses far too small to matter and was removed rather than reported alongside a valid statistic.

Scored on non-stockout days only, where recorded sales are true demand. Corrected intervals are
saved for EVERY evaluated period, not just test, so ordering can be iterated on validation without
opening the test week.
"""
import json

import numpy as np
import pandas as pd

from .utils import config
from . import forecast

LOWER, UPPER = "q10", "q90"   # q10/q90 = the 80% central band
NOMINAL = 0.80

# Windows in calendar order. Used to decide whether an eval window sits AFTER the calibration set,
# which is the only thing that changes how the offset is fitted - see FORWARD_DRIFT_INFLATION.
PERIOD_ORDER = ("training", "validation", "calibration", "test")

# How much wider to calibrate when the eval window sits FORWARD of the calibration set. Applied by
# `run` automatically via `_is_forward`, never by hand: validation sits BEFORE the calibration
# window, has no forward drift to absorb, and inflating it there overshoots (0.826 -> 0.858,
# measured). That mistake is the reason this is derived from the calendar rather than passed in.
#
# Plain CQR loses ~2.5 coverage points per window of forward distance, because the offset is fitted
# on one fortnight and applied to the next. Asking for a slightly wider band absorbs that.
#
# MEASURED, not guessed, and measured WITHOUT the test week: fitting on validation and applying to
# calibration - the same one-window-forward shift as calibration -> test - gives
#
#     calibrate at   0.80    0.83    0.85    0.87    0.90
#     land on        0.775   0.804   0.822   0.840   0.870
#
# so +0.03 is what lands on 0.80 a window later. Chosen on validation -> calibration and applied
# unchanged, which is why using it on the test week is not tuning on the test week.
FORWARD_DRIFT_INFLATION = 0.03

def load_forecasts(periods=("calibration", "validation", "test"), master: pd.DataFrame = None,
                   forecast_tag: str = None, observed_only: bool = True, family: str = "tft"):
    """Saved forecast per period, joined to actuals. `forecast_tag` selects which target to read -
    "recovered" (default) or "raw" - and `family` which model wrote it ("tft" or "xgb"). Every
    current run carries all six quantiles (q025/q10/q50/q80/q90/q975) in that ONE file, so the 80%
    and 95% bands are read from the same saved forecast, just corrected by separate `run()` calls
    below.

    Adds `observed` (True on non-stockout days, where demand was truly seen), `y` (= sale_amount,
    meaningful ONLY where observed) and `recovered_demand` (the filled-in series, meaningful
    everywhere - what the ordering stage simulates against).

    `observed_only=True` (default) returns just the non-stockout rows - the only rows where demand
    was actually seen, and so the only rows an offset may be fitted or scored on. Pass False to keep
    every product-day: ordering runs on all days, and the censored ones are exactly where the
    recovery layer pays off.
    Returns ({period: frame}, master)."""
    master = master if master is not None else forecast.build_master_frame()
    cols = ["store_id", "product_id", "dt", "sale_amount", "recovered_demand",
            "stock_hour6_22_cnt"]
    paths = config.forecast_paths(forecast_tag or "recovered", family)
    id_dtypes = master[["store_id", "product_id"]].dtypes.to_dict()

    out = {}
    for p in periods:
        if not paths[p].exists():
            raise FileNotFoundError(
                f"{paths[p].parent.name}/{paths[p].name} missing - forecast period '{p}' for "
                f"tag '{forecast_tag or 'recovered'}' first. TFT: forecast.run(..., "
                f"periods=(...,'{p}')) or forecast.forecast_test. XGBoost: "
                f"gbm.run(..., periods=(...,'{p}')).")
        # `forecast._prepare` casts store_id/product_id to STRING categoricals for the TFT's
        # embeddings, so every saved forecast parquet carries them as strings - but `master`
        # (and everything else in the pipeline: recovery, data_io, orders.py's KEY merges) keeps
        # them as the raw integer IDs. Merging string-vs-int keys doesn't silently mismatch, it
        # raises (pandas refuses to merge on differently-typed columns) - cast back here, once,
        # rather than at every call site. `metrics.attach_bucket` is the same fix for the same
        # reason, on the path where a forecast frame is grouped by censoring band instead of merged.
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
    # if y sits below q_lo, the first term is positive and the row scores its distance below the
    # band; if y sits above q_hi, the second term is positive instead. A row comfortably inside
    # the band makes both terms negative, and max() picks whichever is less negative (closest to 0).
    return np.maximum(df[lower].values - df["y"].values, df["y"].values - df[upper].values)


def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample split-conformal quantile - ceil((n+1)(1-alpha))/n of the scores - i.e.
    the smallest widening that guarantees >= 1-alpha coverage on exchangeable data."""
    n = len(scores)
    # not simply the alpha-quantile of the scores: the (n+1) and ceil() are the finite-sample
    # correction from the CQR paper, which is what makes the coverage guarantee hold exactly rather
    # than only in the limit of infinite calibration data
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def interval_coverage(df: pd.DataFrame, lower: str, upper: str) -> dict:
    """Empirical coverage and mean width of [lower, upper] on the observed rows."""
    y = df["y"].values
    return dict(coverage=round(float(((y >= df[lower].values) & (y <= df[upper].values)).mean()), 4),
                mean_width=round(float((df[upper].values - df[lower].values).mean()), 4),
                n=int(len(df)))


def coverage_ci(df: pd.DataFrame, lower: str, upper: str, n_boot: int = 2000,
                seed: int = 0) -> dict:
    """Coverage with a 95% interval from a DAY-BLOCK bootstrap - resample whole dates, not rows.

    This is the honest replacement for Kupiec here. Kupiec assumes independent draws, and these
    rows are not remotely independent: 5,601 series share each date, so a busy Saturday moves
    thousands of rows the same way. Measured, the between-day spread is 13.7x what independence
    predicts, which puts the EFFECTIVE sample near 3,500 rather than 48,000 - so Kupiec is handed
    fourteen times more evidence than actually exists and rejects deviations far too small to
    matter. It returns p = 0.0 on a 2.5-point miss.

    Resampling whole days keeps the within-day correlation intact, so the interval reflects the
    evidence that is really there. Read it as "covers 82.6%, and 80% is outside the interval":
    a small real miss, rather than a catastrophic one.
    """
    y = df["y"].values
    hit = ((y >= df[lower].values) & (y <= df[upper].values)).astype(float)
    # blocks = a list of arrays, one per day, of the 5,601 hits on that day. The bootstrap resamples
    # whole days, not rows, so the within-day correlation is preserved.
    blocks = [g.to_numpy() for _, g in pd.Series(hit, index=df["dt"].values).groupby(level=0)]
    rng = np.random.default_rng(seed)
    # draws is the mean coverage of a bootstrap sample of the day-blocks, repeated n_boot times
    draws = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(coverage=round(float(hit.mean()), 4), ci_low=round(float(lo), 4),
                ci_high=round(float(hi), 4), n_days=len(blocks))


def _is_forward(eval_period: str, calibration_periods) -> bool:
    """Does `eval_period` sit after every calibration window?

    The one thing that changes how the offset is fitted, and it is read off the frozen calendar
    rather than asked of the caller - because the caller (me) got it wrong once, inflating a
    BACKWARD window and overshooting 0.826 -> 0.858.
    """
    return (PERIOD_ORDER.index(eval_period)
            > max(PERIOD_ORDER.index(p) for p in calibration_periods))


def run(nominal: float = NOMINAL, lower: str = LOWER, upper: str = UPPER,
        calibration_periods=("calibration",), eval_periods=("validation",), save: bool = True,
        verbose: bool = True, tag: str = None, forecast_tag: str = None,
        family: str = "tft", master: pd.DataFrame = None) -> dict:
    """Fit CQR on the calibration set, apply it to `eval_periods`, report coverage, and save.

    ONE method, no method arguments. Split CQR with a single additive offset, widened by
    `FORWARD_DRIFT_INFLATION` when - and only when - the eval window sits after the calibration set.
    Everything below selects WHICH DATA to run it on, never which technique:

    - `lower`/`upper`: the band to correct. 80% is the default (`run(0.80, "q10", "q90",
      tag="wide80")`); 95% is `run(0.95, "q025", "q975", tag="wide95")` - the same saved forecast,
      a different pair of columns and its own offset, which is why `orders.py` reads two conformal
      outputs from one forecast file.
    - `eval_periods`: defaults to validation ALONE. The test week opens by naming it, the same rule
      the forecast and ordering stages follow.
    - `calibration_periods`: which window the offset is fitted on.
    - `forecast_tag` / `family`: which target ("recovered"/"raw") and model ("tft"/"xgb") to read.
      With `tag`, all three appear in the output filenames, so the eight (family x target x band)
      runs a full comparison needs cannot overwrite each other.
    - `master`: pass `forecast.build_master_frame()` when correcting several arms in a row - it is
      the same frame every time and rebuilding it per call dominates the stage.

    Rejected alternatives are recorded in the module docstring rather than left in as options:
    Mondrian per-band offsets and multiplicative band-scaling were both measured against this and
    land within 0.0003, so keeping them would be three switches for one answer.
    """
    target = forecast_tag or "recovered"
    calibration_periods, eval_periods = tuple(calibration_periods), tuple(eval_periods)
    if "test" in calibration_periods:
        raise ValueError("the test week is never a calibration set")
    overlap = set(calibration_periods) & set(eval_periods)
    if overlap:
        raise ValueError(f"{sorted(overlap)} is both calibrated and scored on - the offset would be "
                         f"fitted on the rows it is then measured against")

    # every product-day is loaded; `observed` marks the non-stockout rows. The offset is FIT and
    # SCORED on those only - demand was not seen on the rest - but it is APPLIED to and saved for
    # all of them, because a store orders every day and the censored days are exactly where the
    # fill-in layer pays off.
    data, _ = load_forecasts(tuple(dict.fromkeys(calibration_periods + eval_periods)),
                             master=master, forecast_tag=target, observed_only=False, family=family)
    missing = [c for c in (lower, upper)
               if any(c not in data[p].columns for p in calibration_periods + eval_periods)]
    if missing:
        raise KeyError(f"forecast missing {missing} for the {nominal:.0%} band - re-run the "
                       f"forecast for tag '{target}' (both families already "
                       f"fit q025/q10/q50/q80/q90/q975, so no separate model is needed for a "
                       f"wider band).")

    calib = pd.concat([data[p] for p in calibration_periods], ignore_index=True)
    calib = calib[calib["observed"]].copy()
    scores = nonconformity(calib, lower, upper)   # one nonconformity score per calibration row,
                                                   # fitted once and reused for every eval window below

    results = {"method": "split CQR", "nominal": nominal, "band": [lower, upper],
               "family": family, "forecast_tag": target,
               "calibration_set": list(calibration_periods), "calibration_n": int(len(calib)),
               "periods": {}}

    for name in eval_periods:
        # The offset is fitted per eval window because the ONLY thing that varies is whether that
        # window is forward of the calibration set, which decides the inflation.
        inflation = FORWARD_DRIFT_INFLATION if _is_forward(name, calibration_periods) else 0.0
        # asking for a bit more than the nominal coverage (nominal + inflation) on forward windows
        # is what "calibrate at 83% to land on 80%" means mechanically: alpha shrinks, so
        # conformal_offset reads a higher, wider-covering percentile off the same calibration scores
        Q = conformal_offset(scores, round(1 - nominal - inflation, 4))

        corr = data[name].copy()                       # ALL product-days
        corr[f"{lower}_c"] = np.clip(corr[lower].values - Q, 0, None)   # demand >= 0
        corr[f"{upper}_c"] = corr[upper].values + Q

        obs = corr[corr["observed"]]                   # scoring set: demand was seen on these rows
        results["periods"][name] = dict(
            n_rows=int(len(corr)), n_observed=int(len(obs)),
            offset=round(Q, 4), drift_inflation=inflation,
            uncorrected=interval_coverage(obs, lower, upper),
            corrected=interval_coverage(obs, f"{lower}_c", f"{upper}_c"),
            coverage_ci=coverage_ci(obs, f"{lower}_c", f"{upper}_c"))

        # Every evaluated period is saved, not just test: ordering reads these files, and it has to
        # be iterable on validation. ALL product-days go in, including the censored ones - ordering
        # happens every day, and the censored days are exactly where the recovery layer pays off.
        if save:
            path = config.conformal_parquet(name, tag, family=family, target=target)
            path.parent.mkdir(parents=True, exist_ok=True)
            corr.to_parquet(path, index=False)

    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.conformal_results(tag, family=family, target=target)
        # Merge into whatever's already saved rather than overwrite it - otherwise a later
        # test-week run erases an earlier validation record for the same arm.
        if out_path.exists():
            results["periods"] = {**json.loads(out_path.read_text())["periods"], **results["periods"]}
        out_path.write_text(json.dumps(results, indent=2))

    if verbose:
        print(f"\n{results['method']} {nominal:.0%} [{lower}/{upper}] on {family}/"
              f"{results['forecast_tag']}  |  calib={'+'.join(calibration_periods)} "
              f"({len(calib):,} observed rows)")
        for name, p in results["periods"].items():
            u, c, ci = p["uncorrected"], p["corrected"], p["coverage_ci"]
            drift = f" +drift {p['drift_inflation']:.2f}" if p["drift_inflation"] else ""
            print(f"  [{name:10s}] {p['n_observed']:>6,} scored rows  Q={p['offset']:.4f}{drift}  "
                  f"covers {u['coverage']:.3f} -> {c['coverage']:.3f} "
                  f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  target {nominal:.2f}  "
                  f"width {c['mean_width']:.3f}")
    return results
