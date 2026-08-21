"""Honest bands. Split conformalized quantile regression (CQR, Romano 2019) applied post-hoc to a
saved quantile forecast, with no retraining.

The raw 80% band (q10/q90) is overconfident: it covers ~73% of actuals on validation, not 80%. CQR
calibrates one offset Q on the CALIBRATION period and widens the band to [q10-Q, q90+Q], which under
EXCHANGEABILITY restores >= nominal coverage.

Exchangeability is what a forecast horizon breaks. A later week is not exchangeable with the window
the offset was fitted on, so coverage decays with distance: plain CQR overshoots on the near window
and undershoots on the far one, by about 2.5 coverage points per window of separation.

`FORWARD_DRIFT_INFLATION` absorbs that drift, calibrating at 0.83 to land on 0.80 a window later.
`_is_forward` reads the frozen calendar, so only a window after the calibration set is inflated.

One method, no switches: multiplicative band-scaling and Mondrian per-censoring-band offsets both
land within 0.0003 of plain additive CQR on the same harness, so the functional form is not what
limits coverage here.

Coverage carries a day-block bootstrap interval (`coverage_ci`) instead of a Kupiec test. Kupiec
assumes independent rows, and these cluster by date at a design effect of 13.7, so it rejects misses
far too small to matter.

Scored on non-stockout days only, where recorded sales are true demand. Corrected intervals are saved
for every evaluated period, so ordering can be iterated on validation without opening the test week.
"""
import json

import numpy as np
import pandas as pd

from .utils import config
from . import forecast

LOWER, UPPER = "q10", "q90"   # q10/q90 = the 80% central band
NOMINAL = 0.80

# Calendar order, so `_is_forward` can tell whether an eval window sits after the calibration set.
# That is the only thing that changes how the offset is fitted.
PERIOD_ORDER = ("training", "validation", "calibration", "test")

# How much wider to calibrate when the eval window sits forward of the calibration set. Measured by
# fitting on validation and applying to calibration, the same one-window shift as calibration -> test,
# so the test week plays no part in choosing it:
#
#     calibrate at   0.80    0.83    0.85    0.87    0.90
#     land on        0.775   0.804   0.822   0.840   0.870
#
# +0.03 lands on 0.80 a window later. Forward windows only: validation sits before the calibration
# set, has no drift to absorb, and inflating it there overshoots to 0.858.
FORWARD_DRIFT_INFLATION = 0.03


def load_forecasts(periods=("calibration", "validation", "test"), master: pd.DataFrame = None,
                   forecast_tag: str = None, observed_only: bool = True, family: str = "tft"):
    """Saved forecast per period, joined to actuals. Returns ({period: frame}, master).

    `forecast_tag` selects the target, "recovered" (default) or "raw"; `family` selects the model
    that wrote it, "tft" or "xgb". One file carries all six quantiles (q025/q10/q50/q80/q90/q975),
    so both bands read the same forecast and only their offsets differ.

    Adds `observed` (True on non-stockout days), `y` (= sale_amount, meaningful only where observed)
    and `recovered_demand`, which the ordering stage simulates against.

    `observed_only=True` keeps just the non-stockout rows, the only ones an offset may be fitted or
    scored on. Pass False for every product-day: ordering runs daily, and the censored days are
    where the recovery layer pays off.
    """
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
        # Two fixes the merge needs. `forecast._prepare` stores the IDs as string categoricals for
        # the TFT's embeddings while the rest of the pipeline keeps raw ints, and pandas refuses to
        # merge differently-typed keys, so cast them back once here (`metrics.attach_bucket` is the
        # same fix on the grouping path). `forecast.CARRY` also writes sale_amount /
        # stock_hour6_22_cnt / recovered_demand into each parquet so forecast.py can score
        # standalone; those duplicate `master`, and left in place they suffix to `_x`/`_y` and break
        # every bare column name below. `master` is authoritative, so drop the copies.
        fc = pd.read_parquet(paths[p]).astype(id_dtypes)
        fc = fc.drop(columns=[c for c in cols[3:] if c in fc.columns])
        merged = fc.merge(master[cols], on=cols[:3], how="left")
        merged["observed"] = merged["stock_hour6_22_cnt"] == 0
        merged["y"] = merged["sale_amount"]
        out[p] = merged[merged["observed"]].copy() if observed_only else merged
    return out, master


def nonconformity(df: pd.DataFrame, lower=LOWER, upper=UPPER) -> np.ndarray:
    """CQR score E_i = max(q_lo - y, y - q_hi): how far y falls outside the band, negative for a row
    inside it, where max() gives the distance to the nearer edge (Romano 2019 eq. 6)."""
    return np.maximum(df[lower].values - df["y"].values, df["y"].values - df[upper].values)


def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample split-conformal quantile - ceil((n+1)(1-alpha))/n of the scores - i.e.
    the smallest widening that guarantees >= 1-alpha coverage on exchangeable data."""
    n = len(scores)
    # the (n+1) and ceil() are the finite-sample correction: without them the guarantee holds only
    # in the limit of infinite calibration data
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def interval_coverage(df: pd.DataFrame, lower: str, upper: str) -> dict:
    """Empirical coverage and mean width of [lower, upper] on the observed rows."""
    y = df["y"].values
    return dict(coverage=round(float(((y >= df[lower].values) & (y <= df[upper].values)).mean()), 4),
                mean_width=round(float((df[upper].values - df[lower].values).mean()), 4),
                n=int(len(df)))


def coverage_ci(df: pd.DataFrame, lower: str, upper: str, n_boot: int = 2000,
                seed: int = 0) -> dict:
    """Coverage with a 95% interval from a day-block bootstrap: resample whole dates, not rows.

    5,601 series share each date, so a busy Saturday moves thousands of rows together. The measured
    between-day spread is 13.7x what independence predicts, putting the effective sample near 3,500
    against a nominal 48,000, which is why Kupiec returns p = 0.0 on a 2.5-point miss.

    Resampling whole days keeps the within-day correlation, so the interval reflects the evidence
    that is there. Read it as "covers 82.6%, and 80% sits outside the interval": a small real miss.
    """
    y = df["y"].values
    hit = ((y >= df[lower].values) & (y <= df[upper].values)).astype(float)
    # one array of hits per day, so the resample below draws whole days and keeps their correlation
    blocks = [g.to_numpy() for _, g in pd.Series(hit, index=df["dt"].values).groupby(level=0)]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(coverage=round(float(hit.mean()), 4), ci_low=round(float(lo), 4),
                ci_high=round(float(hi), 4), n_days=len(blocks))


def _is_forward(eval_period: str, calibration_periods) -> bool:
    """Does `eval_period` sit after every calibration window?

    Read off the frozen calendar, never passed in: only forward windows take the drift inflation.
    """
    return (PERIOD_ORDER.index(eval_period)
            > max(PERIOD_ORDER.index(p) for p in calibration_periods))


def run(nominal: float = NOMINAL, lower: str = LOWER, upper: str = UPPER,
        calibration_periods=("calibration",), eval_periods=("validation",), save: bool = True,
        verbose: bool = True, tag: str = None, forecast_tag: str = None,
        family: str = "tft", master: pd.DataFrame = None) -> dict:
    """Fit CQR on the calibration set, apply it to `eval_periods`, report coverage, and save.

    Split CQR with one additive offset, widened by `FORWARD_DRIFT_INFLATION` only when the eval
    window sits after the calibration set. No argument here picks a technique; each picks data:

    - `lower`/`upper`: the band to correct. 80% is the default (`run(0.80, "q10", "q90",
      tag="wide80")`), 95% is `run(0.95, "q025", "q975", tag="wide95")`. Same saved forecast, a
      different column pair and its own offset, which is why `orders.py` reads two conformal files.
    - `eval_periods`: validation only by default. The test week opens by naming it, as in the
      forecast and ordering stages.
    - `calibration_periods`: which window the offset is fitted on.
    - `forecast_tag` / `family`: which target ("recovered"/"raw") and model ("tft"/"xgb") to read.
      With `tag`, all three appear in the output filenames, so the eight (family x target x band)
      runs of a full comparison cannot overwrite each other.
    - `master`: pass `forecast.build_master_frame()` when correcting several arms in a row, since
      rebuilding it per call dominates the stage.
    """
    target = forecast_tag or "recovered"
    calibration_periods, eval_periods = tuple(calibration_periods), tuple(eval_periods)
    if "test" in calibration_periods:
        raise ValueError("the test week is never a calibration set")
    overlap = set(calibration_periods) & set(eval_periods)
    if overlap:
        raise ValueError(f"{sorted(overlap)} is both calibrated and scored on - the offset would be "
                         f"fitted on the rows it is then measured against")

    # Every product-day loads; `observed` marks the non-stockout rows. The offset is fitted and
    # scored on those alone, since demand was not seen on the rest, but applied and saved for all of
    # them: a store orders every day, and the censored days are where the fill-in layer pays off.
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

        # Every evaluated period is saved, because ordering reads these files and has to be
        # iterable on validation without opening the test week.
        if save:
            path = config.conformal_parquet(name, tag, family=family, target=target)
            path.parent.mkdir(parents=True, exist_ok=True)
            corr.to_parquet(path, index=False)

    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.conformal_results(tag, family=family, target=target)
        # Merge into what is already saved, so a validation run and a test-week run for the same
        # arm both survive in one file.
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
