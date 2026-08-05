[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23975980&assignment_repo_type=AssignmentRepo)

# Perishable Demand Forecasting & Zero-Waste Inventory Engine

Fresh food is ordered against a forecast, and a bad forecast is paid for twice: order too
much and it is binned, order too little and the shelf is empty. This project builds an
ordering assistant for perishable goods on **FreshRetailNet-50K** (Dingdong Inc.), and turns
a calibrated forecast into an actual order quantity.

## The problem the data hides

**A sold-out shelf records zero sales, not zero demand.** When a product runs out at 3pm, the
rest of that day's demand is never observed — the sale is simply absent from the data. Around
half of the product-days in the working subset have at least one out-of-stock hour inside the
dataset's annotated 06:00–22:00 trading window.

Train a forecaster directly on recorded sales and it learns those censored zeros as if they
were real, so it systematically **under-forecasts** exactly the products that keep selling
out. Ordering to that forecast keeps them selling out. The status quo is stable and wrong.

## The approach

Recover the demand that censoring hid, then forecast *that*, then say honestly how uncertain
the answer is, then convert it into an order.

| Stage | What it does |
|---|---|
| **Data** | Ingest, assert the schema from the bytes, cut a seeded reproducible subset, and score seasonal-naive / XGBoost-quantile / SARIMA baselines on raw sales — the bar to clear |
| **Recovery** | Two-stage latent-demand recovery at hour resolution: learn the normal in-stock hourly demand rate, then replace each stocked-out hour with its prediction (floored at observed) and sum to a daily `recovered_demand` |
| **Splits** | Freeze the train / validation / calibration windows, then prove no leakage — lag features look backwards only, and nothing fits on calibration data |
| **Forecasting** | A Temporal Fusion Transformer trained on recovered demand, emitting quantiles (q10/q50/q90) rather than a single number |
| **Calibration** | Split conformal / CQR widens the interval so a promised 80% band actually contains the truth 80% of the time, checked with a Kupiec test and a reliability diagram |
| **Ordering** | Newsvendor order quantities off the calibrated quantiles, with a waste-vs-stockout cost slider |
| **Final evaluation** | Scored on the shipped test week, looked at exactly once |

Everything downstream of recovery learns from filled-in demand, never raw sales. Scoring waste
against recorded sales would reward under-ordering, which is the whole failure being fixed.

Because no ground truth exists for a real stockout, the recovery layer is validated on a test set
built from the days we *do* trust: days the shelf stayed full, whose recorded total **is** the true
demand. Those days are held out of training, recovered as if they had been stocked out all day, and
compared against the total that was hidden. The same test ranks the candidate models.

## How success is measured

| Question | Metric | Target |
|---|---|---|
| Did the fill-in remove the under-counting? | WPE (bias) | ≈ 0 |
| How far off is the best guess? | WAPE | beat every baseline |
| Is the range good, not just the middle? | pinball / CRPS | ≥10% better than the best baseline |
| Is "80% sure" true 80% of the time? | coverage @80/95 | within ±2.5 points |
| **Do our orders bin less food?** | simulated waste | **≥15% below a naive order** |
| …without starving the shelf? | stockout rate | ≤5% |

Accuracy is scored the dataset's own way — per date, on non-stockout rows only — so the
numbers stay comparable to the published FreshRetailNet evaluation.

## Design rules

- **All logic lives in `src/`.** The notebook and the dashboard are thin callers, so the
  science has exactly one implementation.
- **The dashboard reads saved files only** and never recomputes. What it shows is what was
  produced by a recorded run.
- **The subset is seeded and self-describing.** It rebuilds from two numbers (store count and
  seed), and every rebuild rewrites `outputs/subset_summary.md`, so reported numbers can always
  be traced back to the data behind them.
- **Splits, once frozen, never move.** The test week is touched once, at the final evaluation.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
nbstripout --install                               # keeps notebook outputs out of git
```

## Running it

```bash
jupyter lab notebooks/                      # heavy fits sit behind RUN_* toggles
```

| notebook | covers | runtime |
|---|---|---|
| `01_data_and_recovery.ipynb` | subset + baselines, demand recovery, frozen splits | ~15 min |
| `02_forecasting.ipynb` | TFT tuning + fit, conformal calibration | hours (GPU advised) |
| `03_ordering_and_results.ipynb` | order quantities + cost sweep, final scoreboard | ~5 min |

Each notebook is independent — it reads what it needs from disk and tells you which notebook to
run if something is missing. Only a cold start needs them in order.

The working subset and how it compares to the full corpus is recorded in
[outputs/subset_summary.md](outputs/subset_summary.md), rewritten on every rebuild so it can never
drift from the data it describes.

`src/` is a library with no CLI entry points — every stage is a plain function call, so the
notebook and the dashboard go through identical code. The working subset is rebuilt by
`data_io.build_subset(n_stores)` and the draw is seeded from `config.RANDOM_STATE`, so the same
two numbers always reproduce the same data.

The repo is built up one stage at a time — the layout, the dashboard and the results tables land
as each stage does.
