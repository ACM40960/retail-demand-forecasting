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

# ---- data artifacts: the working subset ----
DAILY_TRAIN = DATA_DIR / "daily_train.parquet"
DAILY_EVAL = DATA_DIR / "daily_eval.parquet"
HOURLY_TRAIN = DATA_DIR / "hourly_train.parquet"
HOURLY_EVAL = DATA_DIR / "hourly_eval.parquet"
CENSORING_RATE = DATA_DIR / "censoring_rate.parquet"

# ---- outputs ----
# The subset itself is rebuildable from (n_stores, RANDOM_STATE), so it is not committed. This
# summary is: it is the committed record of what the reported numbers describe, and a rebuild
# rewrites it.
SUBSET_SUMMARY = OUTPUTS_DIR / "subset_summary.md"
BASELINE_SCORECARD = OUTPUTS_DIR / "baseline_scorecard.csv"

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
TRAIN_END = "2024-05-28"   # training: file start .. here
VAL_START = "2024-05-29"   # validation: early stopping, model choice, baseline scorecard
VAL_END = "2024-06-11"
CAL_START = "2024-06-12"   # calibration: conformal band widths only
CAL_END = "2024-06-25"

# ---- knobs ----
HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
ACTIVE_HOURS = (6, 22)   # the dataset's annotated 06:00-22:00 censoring window

SARIMA_SAMPLE = 30       # series sampled for the SARIMA baseline (it is slow per series)

RANDOM_STATE = 0
