"""Central config: paths + the shared knobs, resolved relative to the repo root so the
same imports work from the notebook and the Streamlit app.

Artifact paths are added here as each phase starts producing them; nothing else in the
codebase is allowed to hard-code a path.
"""
from pathlib import Path

# this file lives at src/utils/config.py, so the repo root is three parents up
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# outputs/ is split by artifact KIND, not by stage, so a reader looking for "the numbers" or "the
# pictures" has one place to look rather than sifting a flat directory.
FORECAST_DIR = OUTPUTS_DIR / "forecasts"   # parquets, one subdirectory per model family
REPORTS_DIR = OUTPUTS_DIR / "reports"      # csv / json / md - everything with numbers in it
PLOTS_DIR = OUTPUTS_DIR / "plots"          # png
MODEL_DIR = OUTPUTS_DIR / "models"         # fitted models that must outlive the kernel that fit them

# ---- data artifacts: the working subset ----
DAILY_TRAIN = DATA_DIR / "daily_train.parquet"
DAILY_EVAL = DATA_DIR / "daily_eval.parquet"
HOURLY_TRAIN = DATA_DIR / "hourly_train.parquet"
HOURLY_EVAL = DATA_DIR / "hourly_eval.parquet"

# ---- data artifacts: recovered demand ----
RECOVERED_TRAIN = DATA_DIR / "daily_train_recovered.parquet"
RECOVERED_EVAL = DATA_DIR / "daily_eval_recovered.parquet"

# ---- outputs ----
# The subset itself is rebuildable from (n_stores, RANDOM_STATE), so it is not committed. This
# summary is: it is the committed record of what the reported numbers describe, and a rebuild
# rewrites it.
SUBSET_SUMMARY = REPORTS_DIR / "subset_summary.md"
BASELINE_SCORECARD = REPORTS_DIR / "baseline_scorecard.csv"

# recovery: the Stage-1 model and the bias correction that together produced the recovered parquets.
# Without them `recovered_demand` is an unexplainable column - the correction is a fitted parameter
# applied to every recovered hour, so it belongs on disk next to its output.
STAGE1_MODEL = MODEL_DIR / "recovery_stage1_lgbm.txt"   # LightGBM native text format
RECOVERY_PARAMS = REPORTS_DIR / "recovery_params.json"
RECOVERY_COMPARISON = REPORTS_DIR / "recovery_model_comparison.csv"   # the table that picks Stage 1
BIAS_BUCKET_CSV = REPORTS_DIR / "recovery_by_censoring_bucket.csv"
LEAKAGE_CHECKS = REPORTS_DIR / "leakage_checks.json"
RECOVERY_PLOT = PLOTS_DIR / "recovery_bias_and_example.png"
HOUR_LEVEL_TABLE = REPORTS_DIR / "hour_level_appendix.csv"   # why daily totals rank, not hours

# the check that tests recovery where its headline score cannot: across time
CORRECTION_BY_PERIOD = REPORTS_DIR / "recovery_correction_by_period.csv"

# the 2x2's missing cell: one baseline, two targets, scored per censoring band
RECOVERY_IMPACT = REPORTS_DIR / "recovery_impact.csv"

# forecasting: one saved forecast per (period, target), plus the fitted TFT for each target.
# `tag` is "recovered" or "raw" - the two targets being compared - so both sets of files sit side by
# side and neither can quietly overwrite the other.
FORECAST_SCORECARD = REPORTS_DIR / "forecast_vs_baselines.csv"

# ordering: per-product-day results at the headline cost ratio, and the full cost-ratio sweep that
# shows whether a claim holds everywhere rather than at one lucky ratio.
# Parquet, not CSV, and not in REPORTS_DIR: 39K rows x 20 columns of per-product-day simulation output is
# row-level DATA, which is what FORECAST_DIR and parquet are for. As a CSV it was 7.8 MB - by itself more
# than the rest of reports/ put together, in a directory meant to hold summary tables a person reads.
# `cost_sweep.csv` is the readable summary of this file and stays a CSV.
ORDER_SIMULATION = FORECAST_DIR / "order_simulation.parquet"
COST_SWEEP_CSV = REPORTS_DIR / "cost_sweep.csv"


def forecast_parquet(period: str, target: str, family: str = "tft"):
    """Saved quantile forecast for one period and target, filed under its model family.

    `target` is "recovered" or "raw" - the two things being compared. `family` is which model made it,
    so the TFT and the gradient-boosted forecaster can hold the same (period, target) without one
    overwriting the other.
    """
    return FORECAST_DIR / family / f"forecast_{period}_{target}.parquet"


def conformal_parquet(period: str, tag: str = None, family: str = "tft",
                      target: str = "recovered"):
    """Conformally corrected intervals, named like the forecast they were computed from.

    `family` AND `target` are both in the path for the same reason they are in `forecast_parquet`:
    an arm is (model, target), and correcting the raw arm used to overwrite the recovered arm's
    file because only `tag` varied. That silently turned any multi-arm comparison into the same
    arm scored twice.
    """
    suffix = f"_{tag}" if tag else ""
    return FORECAST_DIR / family / f"forecast_{period}_{target}_conformal{suffix}.parquet"


def conformal_results(tag: str = None, family: str = "tft", target: str = "recovered"):
    """The coverage / Kupiec / width / reliability report for one conformal run - one file per arm
    per band, for the same reason `conformal_parquet` is."""
    suffix = f"_{tag}" if tag else ""
    return REPORTS_DIR / f"conformal_results_{family}_{target}{suffix}.json"


def forecast_paths(tag: str = "recovered", family: str = "tft") -> dict:
    """{period: path} for every period a forecast can be saved for, one target and one family. The
    single place `conformal.py` (and anything else that needs more than one period at once) asks for
    saved forecast locations, so a path is never rebuilt by hand at the call site.

    `family` is here because calibrating the gradient-boosted arms means pointing the same conformal
    code at `forecasts/xgb/`; without it the correction could only ever be applied to the TFT.
    """
    return {p: forecast_parquet(p, tag, family) for p in ("validation", "calibration", "test")}


def tft_checkpoint(tag: str):
    """The fitted TFT for one target."""
    return MODEL_DIR / f"tft_best_{tag}.ckpt"


def gbm_tuning(tag: str):
    """The gradient-boosted forecaster's config ranking. Separate file from the TFT's so the two
    searches cannot overwrite each other."""
    return REPORTS_DIR / f"gbm_tuning_{tag}.csv"


def learning_curve_csv(family: str, tag: str):
    """Train-vs-eval loss per training step, for one model family.

    The one measurement a hyperparameter search cannot make: a grid scores configs on validation only,
    so it finds the best config without ever showing whether that config memorises. Saved because the
    TFT's costs a GPU fit to reproduce.

    The two families' curves are NOT on a shared axis - the TFT's x is epochs and the tree's is boosting
    rounds, and each uses its own loss implementation. Plot them as two panels and compare shape.
    """
    return REPORTS_DIR / f"{family}_learning_curve_{tag}.csv"


def tft_tuning(tag: str):
    """The hyperparameter search ranking. Rewritten after every config, so an interrupted grid
    resumes from it rather than restarting."""
    return REPORTS_DIR / f"tft_tuning_{tag}.csv"

# Its wording lives here rather than in data_io, so that module holds only the numbers. Markdown
# because the file is committed and read on GitHub.
SUBSET_SUMMARY_MD = """# Working subset

*Rebuilt {built_at} by `data_io.build_subset`. Regenerated on every rebuild - do not edit by hand.*

{selection}

| | corpus | subset |
|---|---|---|
{table}

The subset is **{share} of the corpus** at ~{per_store} series per store. Matching the corpus on
units/day and censored-day share is what makes it representative: nothing is selected on sales
volume, so no scope restriction has to be declared.

A further **{eval_rows} rows** of shipped eval data are held back for the final evaluation and are
not read by any earlier stage.

Categories present: {categories}

Store IDs: {stores}
"""

# ---- frozen calendar (ISO) ----
# The 90-day train file splits here; the shipped eval file is the TEST week and is opened once,
# at the final evaluation. Declared once, in full, even though the calibration window is not read
# until the calibration stage - a calendar split across files is a calendar that drifts.
TRAIN_START = "2024-03-28"   # training: the train file starts here
TRAIN_END = "2024-05-28"
VAL_START = "2024-05-29"   # validation: early stopping, model choice, baseline scorecard
VAL_END = "2024-06-11"
CAL_START = "2024-06-12"   # calibration: conformal band widths only
CAL_END = "2024-06-25"
TEST_START = "2024-06-26"   # test: the shipped eval file - opened once, at the final evaluation
TEST_END = "2024-07-02"

# ---- knobs ----
HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
ACTIVE_HOURS = (6, 22)   # the dataset's annotated 06:00-22:00 censoring window

SARIMA_SAMPLE = 30       # series sampled for the SARIMA baseline (it is slow per series)

# Which stores `build_subset` draws. FROZEN - changing it redraws the whole dataset and every
# artifact downstream stops describing the data it was computed on. Kept separate from
# RANDOM_STATE so a model-seed experiment can never move the subset by accident.
SUBSET_SEED = 123

# Model fits and train/val splits. Safe to vary: it changes results, never the data.
RANDOM_STATE = 123
