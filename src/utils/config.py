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
MODEL_DIR = OUTPUTS_DIR / "models"   # fitted models that must outlive the kernel that fit them

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
SUBSET_SUMMARY = OUTPUTS_DIR / "subset_summary.md"
BASELINE_SCORECARD = OUTPUTS_DIR / "baseline_scorecard.csv"

# recovery: the Stage-1 model and the bias correction that together produced the recovered parquets.
# Without them `recovered_demand` is an unexplainable column - the correction is a fitted parameter
# applied to every recovered hour, so it belongs on disk next to its output.
STAGE1_MODEL = MODEL_DIR / "recovery_stage1_lgbm.txt"   # LightGBM native text format
RECOVERY_PARAMS = OUTPUTS_DIR / "recovery_params.json"
RECOVERY_COMPARISON = OUTPUTS_DIR / "recovery_model_comparison.csv"   # the table that picks Stage 1
BIAS_BUCKET_CSV = OUTPUTS_DIR / "recovery_by_censoring_bucket.csv"
LEAKAGE_CHECKS = OUTPUTS_DIR / "leakage_checks.json"
RECOVERY_PLOT = OUTPUTS_DIR / "recovery_bias_and_example.png"
HOUR_LEVEL_TABLE = OUTPUTS_DIR / "hour_level_appendix.csv"   # why daily totals rank, not hours

# the check that tests recovery where its headline score cannot: across time
CORRECTION_BY_PERIOD = OUTPUTS_DIR / "recovery_correction_by_period.csv"

# the 2x2's missing cell: one baseline, two targets, scored per censoring band
RECOVERY_IMPACT = OUTPUTS_DIR / "recovery_impact.csv"

# forecasting: one saved forecast per (period, target), plus the fitted TFT for each target.
# `tag` is "recovered" or "raw" - the two targets being compared - so both sets of files sit side by
# side and neither can quietly overwrite the other.
FORECAST_SCORECARD = OUTPUTS_DIR / "forecast_vs_baselines.csv"

# ordering (Phase 6): per-product-day results at the headline cost ratio, and the full cost-ratio
# sweep that proves (or doesn't) the waste/stockout KPI holds everywhere, not just at one ratio.
ORDER_SIMULATION_CSV = OUTPUTS_DIR / "order_simulation.csv"
COST_SWEEP_CSV = OUTPUTS_DIR / "cost_sweep.csv"


def forecast_parquet(period: str, tag: str):
    """Saved q10/q50/q90 for one period and one target."""
    return OUTPUTS_DIR / f"forecast_{period}_{tag}.parquet"


def forecast_paths(tag: str = "recovered") -> dict:
    """{period: path} for every period a forecast can be saved for, one target. The single place
    `conformal.py` (and anything else that needs more than one period at once) asks for saved
    forecast locations, so a path is never rebuilt by hand at the call site."""
    return {p: forecast_parquet(p, tag) for p in ("validation", "calibration", "test")}


def tft_checkpoint(tag: str):
    """The fitted TFT for one target."""
    return MODEL_DIR / f"tft_best_{tag}.ckpt"


def tft_tuning(tag: str):
    """The hyperparameter search ranking. Rewritten after every config, so an interrupted grid
    resumes from it rather than restarting."""
    return OUTPUTS_DIR / f"tft_tuning_{tag}.csv"


def tft_seed_spread(tag: str):
    """Repeat-seed fits at the winning config: mean + spread, the noise floor any reported margin
    has to clear. Saved so the check doesn't have to be re-run (~3 fits) just to be trusted."""
    return OUTPUTS_DIR / f"tft_seed_spread_{tag}.csv"

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
