"""The frozen train / validation / calibration windows.

    Training    2024-03-28 .. 2024-05-28   all models may fit here
    Validation  2024-05-29 .. 2024-06-11   evaluation only (early stopping, model choice)
    Calibration 2024-06-12 .. 2024-06-25   band widths only, never trained on
    Test        the shipped eval file      looked at once, at the final evaluation

The boundary dates live in `config` and are read here, never redeclared.
"""
import pandas as pd

from .utils import config


def add_period(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a `period` column: training/validation/calibration for `split=="train"` rows (cut by
    date), "test" for `split=="eval"` rows. Raises if a train-file date falls outside the frozen
    90-day window. `df` needs `split` and `dt`, e.g. from `data_io.load(...)`."""
    df = df.copy()
    df["period"] = "test"

    is_train_file = df["split"] == "train"
    dt = pd.to_datetime(df.loc[is_train_file, "dt"])
    period = pd.Series("unknown", index=dt.index, dtype="object")
    for name, start, end in [("training", config.TRAIN_START, config.TRAIN_END),
                             ("validation", config.VAL_START, config.VAL_END),
                             ("calibration", config.CAL_START, config.CAL_END)]:
        period[(dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))] = name

    if (period == "unknown").any():
        raise ValueError("dates outside the frozen 90-day window: "
                         f"{sorted(dt[period == 'unknown'].unique())[:5]}")

    df.loc[is_train_file, "period"] = period
    return df


def check_leakage(daily: pd.DataFrame, verbose: bool = True) -> dict:
    """The anti-cheating checks; returns {check_name: passed}. Reads the hourly frame through
    `recovery.hours`, so the checks run on the same rows the pipeline fitted on."""
    from . import recovery   # imported here, not at module scope: recovery imports add_period

    train_only = daily[daily["split"] == "train"]
    results = {}

    def record(name: str, passed: bool, detail: str) -> None:
        results[name] = bool(passed)
        if verbose:
            print(f"[{'PASS' if passed else 'FAIL'}] {detail}")

    # 1. the frozen boundaries produce the expected day counts
    counts = train_only.groupby("period")["dt"].nunique().to_dict()
    expected = {"training": 62, "validation": 14, "calibration": 14}   # the frozen calendar
    record("reproducible_day_counts", counts == expected,
           f"reproducible day counts: got {counts}, expected {expected}")

    # 2. lag features look backwards only: every row's lag7 must equal the actual sale_amount
    # from exactly 7 days earlier in the same series (checked by self-join, all rows)
    truth = (train_only[["store_id", "product_id", "dt", "sale_amount"]]
             .assign(dt=train_only["dt"] + pd.Timedelta(days=7))
             .rename(columns={"sale_amount": "true_lag7"}))
    joined = train_only.merge(truth, on=["store_id", "product_id", "dt"], how="inner")
    max_err = (joined["lag7"] - joined["true_lag7"]).abs().max()
    record("lag_features_backward_only", max_err < 1e-9,
           f"lag7 == actual sale_amount 7 days earlier, same series "
           f"(n={len(joined):,} rows, max error {max_err:.2e})")

    # 3. lag/rolling fills must not pull in future days: start-of-series rows are filled from an
    # expanding mean of earlier days, so no feature may exceed the series' running maximum
    starts = train_only.groupby(["store_id", "product_id"], as_index=False).head(14)
    running_max = train_only.groupby(["store_id", "product_id"])["sale_amount"].cummax()
    filled_from_future = (starts[["lag7", "lag14", "roll7_mean", "roll14_mean"]]
                          .gt(running_max.loc[starts.index], axis=0).any().any())
    record("fills_use_past_only", not filled_from_future,
           f"start-of-series feature fills never exceed the series' running max "
           f"(n={len(starts):,} rows checked)")

    # 4-5. the Stage-1 pool: training-period hours only, zero calibration hours. Calls the
    # shared pool-builder production uses, so a regression there is actually caught here.
    pool = recovery.training_hours(recovery.hours(daily))
    leaked = pool.loc[pool["period"] != "training", "period"].value_counts().to_dict()
    record("stage1_trained_on_training_only", not leaked,
           f"Stage-1 training pool holds training-period hours only: "
           f"{'none leaked' if not leaked else f'LEAKED {leaked}'}")
    n_cal = leaked.get("calibration", 0)
    record("calibration_never_trained_on", n_cal == 0,
           f"zero calibration-period hours in the Stage-1 pool ({n_cal:,} found)")

    if verbose:
        print(f"\n{sum(results.values())}/{len(results)} checks passed")
    return results
