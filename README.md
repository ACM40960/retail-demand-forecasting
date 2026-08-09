[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23975980&assignment_repo_type=AssignmentRepo)

# Retail Demand Forecasting & Zero-Waste Inventory Engine

An ordering assistant for fresh food, built on **FreshRetailNet-50K** (Dingdong Inc.). It recovers
demand that stockouts hid from the data, forecasts that demand as a range, and turns the range into
an order quantity.

**The contribution is the recovery layer, not the forecaster.**

---

## 1. Problem statement

Fresh food is ordered against a forecast, and a bad forecast is paid for twice: order too much and
it is binned, order too little and the shelf is empty.

Underneath that sits a data problem that makes the forecast wrong in a specific, self-reinforcing
direction:

> **A sold-out shelf records zero sales, not zero demand.**

When a product runs out at 3pm, the rest of that day's demand is never observed — the sale is simply
absent from the file. **43.8%** of product-days in the working subset have at least one out-of-stock
hour inside the dataset's annotated 06:00–22:00 trading window.

Train a forecaster directly on recorded sales and it learns those censored zeros as if they were
real demand, so it **systematically under-forecasts exactly the products that keep selling out**.
Ordering to that forecast keeps them selling out, which produces more censored zeros. The status quo
is stable and wrong.

This is *censored demand*, and the dataset is unusually well suited to attacking it: it records
**which hours** each shelf was empty, so a stockout is distinguishable from a genuine no-sale at hour
resolution. That annotation is what makes recovery possible at all.

## 2. Objectives

| # | Objective | How it is judged | Status |
|---|---|---|---|
| 1 | Establish what the status quo achieves on raw sales | WAPE against seasonal-naive / SARIMA / XGBoost-quantile | ✅ done |
| 2 | Recover the demand censoring hid, and validate it on data that could have falsified it | WAPE on held-out full-shelf days; must beat a no-model control | ✅ done |
| 3 | Show recovery changes the forecast where it should | raw-vs-recovered twins, split by how often a series sells out | ✅ done |
| 4 | Forecast as a **range**, not a point | pinball / CRPS vs the best baseline | ⚠️ built; margin was inside seed noise, awaiting re-run at the current subset |
| 5 | Make the range honest | 80% band contains truth 80% of the time (Kupiec, reliability diagram) | ❌ not built |
| 6 | Convert the range into an order that bins less food | simulated waste vs a naive order, across a cost sweep | ❌ not built |
| 7 | Score once on the sealed test week | all of the above | ❌ not opened |

## 3. Dataset

**FreshRetailNet-50K** — Dingdong Inc., via Hugging Face (`Dingdong-Inc/FreshRetailNet-50K`).

| | corpus | working subset |
|---|---|---|
| series (store × product) | 50,000 | 5,601 |
| stores | 898 | 100 |
| products | 865 | 557 |
| categories | 32 | 30 |
| train rows | 4,500,000 | 504,090 |
| units/day | 0.9986 | 0.997 |
| censored days | 44.3% | 43.8% |

The subset is **11.2% of the corpus**, drawn as 100 whole stores sampled uniformly. Whole stores, not
individual series: nothing is selected on sales volume, so the sample stays representative and no
scope restriction has to be declared — and each store keeps a full assortment, which the ordering
stage needs to recommend a realistic basket.

A further **39,207 rows** of shipped eval data are held back and read by no earlier stage.

The draw is **nested in the store count** — the stores are one fixed shuffle and the first *N* are
taken, so raising the count adds stores without swapping the ones already there. See
[outputs/subset_summary.md](outputs/subset_summary.md), regenerated on every rebuild so it can never
drift from the data it describes.

**Calendar** (frozen; boundaries live in `config` and are never redeclared):

| window | dates | used for |
|---|---|---|
| training | 2024-03-28 … 05-28 (62d) | all model fitting |
| validation | 2024-05-29 … 06-11 (14d) | early stopping, model choice, scorecards |
| calibration | 2024-06-12 … 06-25 (14d) | conformal band widths only, never trained on |
| test | the shipped eval file | opened once, at the final evaluation |

## 4. What counts as ground truth here — and what does not

This is the most important section in the README. Nearly every limitation downstream traces back to
a row in this table.

### Observed and verifiable

| what | where | why it can be trusted |
|---|---|---|
| **Sales on non-stockout days** | `sale_amount` where `stock_hour6_22_cnt == 0` | nothing was lost, so **recorded sales *are* true demand**. This is the anchor for every test in the project |
| **Which hours each shelf was empty** | `hours_stock_status` | the dataset's own annotation — the fact that makes censoring detectable rather than assumed |
| **Hourly sales vectors** | `hours_sale` | 24 values per product-day |
| **Discount, holiday, activity flags** | daily columns | recorded, not inferred |
| **Weather** | `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level` | observed — see the caveat below |
| **Test-week actuals** | shipped eval file | sealed; read once |

### Not observable, and never will be from this data

| what | consequence |
|---|---|
| **Demand during a stockout** | the target of the entire recovery layer. **No ground truth exists and none can be constructed.** Every claim about it is indirect by necessity |
| **Stock on hand, deliveries** | recovery's central assumption cannot be checked directly against inventory |
| **Spoilage / waste** | never recorded, so never predicted — it is computed arithmetically as the gap between an order and a demand estimate |
| **Shelf life per product** | the newsvendor stays single-period; leftovers cannot carry forward |
| **Physical unit counts** | `sale_amount` is normalised by an undisclosed coefficient. **Every figure in this project is a ratio or a percentage**; no absolute quantity is ever asserted |
| **The true waste:stockout cost ratio** | an assumption, swept across nine values — a claim only counts if it survives the sweep |

### Model output, not measurement

| what | how to read it |
|---|---|
| **`recovered_demand`** | a model output floored at recorded sales. **A floor on true demand, not an estimate of it.** Demonstrably closer than recorded sales, and if it errs it errs short |
| **Simulated waste** | arithmetic on an order and a demand *estimate*, not an observed outcome |

### Where each result is anchored

| stage | scored against | is that ground truth? |
|---|---|---|
| Baselines | recorded sales, non-stockout rows only | **yes** |
| Recovery | held-out full-shelf days, whose recorded total *is* true demand | **yes** — and the test could have failed |
| Raw-vs-recovered impact | recorded sales, non-stockout rows only | **yes** |
| Forecast accuracy | recorded sales, non-stockout rows only | **yes** |
| Ordering simulation | `recovered_demand` | **no** — a model output. The dependency is stated, never presented as neutral |

**Two caveats that belong here rather than buried in Limitations:**

- **Weather enters as *realised*, not forecast.** In production you would have a weather *forecast*.
  This flatters absolute accuracy; baselines receive the same covariates, so relative comparisons
  hold.
- **The recovery test is a worst case.** Held-out days are blanked across all 16 active hours, which
  is harsher than roughly 87% of real censored days. The reported error is therefore an upper bound;
  the typical-day figure is unmeasured. The bound errs against the model, not for it.

## 5. Architecture

```
FreshRetailNet-50K (HF)
        │
        ▼
  data_io.build_subset        seeded, nested, self-describing
        │
        ├─────────────► baselines.py        seasonal-naive · SARIMA · XGBoost-quantile
        │                                   (the bar to clear, on RAW sales)
        ▼
  recovery.py   ── Stage 1 ──► hourly in-stock demand rate (LightGBM, Tweedie)
                ── Stage 2 ──► stocked-out 06:00–22:00 hours replaced by prediction,
                               floored at observed, summed to a daily total
                ── correction ► one fitted multiplier on the aggregation bias
        │
        ▼
  recovered_demand ──────────► forecast.py   Temporal Fusion Transformer → q10/q50/q90
        │                            │
        │                            ▼
        │                      calibration    split conformal / CQR   [not built]
        │                            │
        │                            ▼
        └──────────────────────► orders.py    newsvendor + cost sweep  [not built]
```

Hours are an implementation detail confined to `recovery.py`; every public function in the project
takes and returns **daily** frames.

Everything downstream of recovery learns from filled-in demand, never raw sales. Scoring waste
against recorded sales would reward under-ordering, which is the failure being fixed.

**Design rules**

- **All logic lives in `src/`.** Notebooks and the dashboard are thin callers, so the science has
  exactly one implementation.
- **The dashboard reads saved files only** and never recomputes.
- **The subset is seeded and self-describing** — it rebuilds from two numbers.
- **Splits, once frozen, never move.** The test week is touched once.

## 6. Results

*30-store subset, training → validation. The test week has not been opened.*

**Baselines on raw sales** — the bar to clear

| | WAPE | WPE |
|---|---|---|
| seasonal_naive | 0.4213 | +0.0187 |
| **xgboost_quantile** | **0.3423** | −0.0187 |
| sarima *(sampled series)* | 0.5210 | −0.2521 |

`WPE` near zero here is **not** evidence that censoring is harmless — these are non-stockout rows,
exactly the rows censoring does not touch. The scorecard structurally cannot see the problem.

**Recovery** — validated on held-out full-shelf days

| | |
|---|---|
| day WAPE | **0.2966** |
| vs `series_hour_mean` (no-model control) | 0.3684 — **19.5% better** |
| day WPE after correction | +0.0186 |
| aggregation bias before correction | +13.2%, corrected by ×0.9001 |
| correction refit per period | 0.892 / 0.857 / 0.923 — no monotone drift |
| leakage checks | **5 / 5 pass** |

Six candidates on the same test: **lightgbm_tweedie 0.2966** · xgboost 0.3009 · lightgbm_poisson
0.3051 · lightgbm_l2 0.3057 · series_hour_mean 0.3684 · poisson_glm 0.5692.

**Does recovery change the forecast?** One model, one seed, two targets, scored identically.

| series band | n scored | WAPE raw → recovered | WPE raw → recovered |
|---|---|---|---|
| **≥75% censored** | 1,375 | **0.2875 → 0.2256** *(−21.5%)* | **−0.196 → +0.001** |
| 50–75% | 12,319 | 0.3084 → 0.3124 | −0.015 → +0.112 |
| 25–50% | 30,345 | 0.3601 → 0.3702 | +0.008 → +0.113 |
| <25% | 3,955 | 0.4712 → 0.4604 | +0.020 → +0.116 |
| ALL | 47,994 | 0.3423 → 0.3426 | −0.019 → +0.101 |

On chronic-stockout series the raw forecaster is **19.6% low** — trained on sales that never
happened. Recovery removes essentially all of it. Read per band, never pooled: scoring uses
non-stockout rows only, exactly the rows the two targets agree on, so the pooled row structurally
cannot show the effect in either direction. Honest cost — outside the top band the recovered twin
runs ~11% high, which the ordering stage has to earn back.

**Forecasting** — built and run, but **not yet re-run at the current subset size**. The earlier
result showed the TFT ahead of every baseline by a margin *smaller than its own seed-to-seed spread*,
so "the TFT beats the baseline" was not established and is reported as mean ± spread, never the best
seed. Those figures are withheld here rather than restated, because they describe a smaller subset
than the one above.

Every number above is written by the code into [outputs/](outputs/) — the CSVs are the evidence, not
a transcription of it.

## 7. Limitations

This is a **proof of concept built on a dataset that records sales, not inventory.** That single
fact sets the ceiling on what any result here can claim — see §4 for the full ground-truth map.

| Limitation | Effect |
|---|---|
| **Demand during a stockout is unobservable** | the load-bearing one. Validation happens on full-shelf days but recovery is only ever *applied* to days the shelf emptied. No test can close that gap, so the layer is built to fall on the safe side of it |
| One subset draw (11.2% of corpus) | only one draw has ever been evaluated. The headline recovery result rests on **1,375 scored rows** — the smallest band in the table |
| Recovery's headline is a worst case | held-out days are blanked across all 16 active hours, harsher than the large majority of real censored days; the typical-day error is unmeasured. The bound errs against the model |
| Stage-1 history features are themselves censored | `lag7`/`roll7` are built from recorded sales, so the model's sense of a series' normal level is slightly depressed — biasing recovery **downward**, the same direction as the floor |
| The classical baseline is a sampled one | SARIMA is fitted on a sample of series, not all of them — a reference point, not a like-for-like competitor |
| Seed spread exceeds the forecaster's margin | "the TFT beats the baseline" is not established; reported as mean ± spread |
| Weather is realised, not forecast | flatters absolute accuracy; relative comparisons hold |
| Pinball ≥10% target vs ~4% achieved | reported as measured — the target was set before any evidence existed |
| Ordering is simulated against `recovered_demand` | not a neutral referee; the dependency is stated |
| Cost ratio is an assumption | swept across nine ratios |

**What richer data would unlock.** Each layer can be replaced without touching the others:

- **Stock on hand and deliveries** → recovery's assumption becomes directly testable; known starting
  stock plus known replenishment gives true unmet demand, converting today's *floor* into a
  *measurement*.
- **Recorded spoilage** → waste becomes an observed outcome rather than an arithmetic one.
- **Shelf life** → the newsvendor becomes multi-period.
- **Real unit costs** → the cost sweep collapses to one operating point, and savings can be quoted in
  currency.

None of these change the architecture. They replace assumptions with measurements at the exact points
§4 names.

## 8. Repo structure

```
src/                        all logic — a library, no CLI entry points
  utils/
    config.py               every path and knob; nothing else hard-codes a path
    data_io.py              HF ingest, schema assertions, build_subset
    features.py             lag/rolling features, hourly explode
    metrics.py              WAPE / WPE / MAE / pinball / CRPS, matching the dataset's protocol
    plots.py                figures
  splits.py                 frozen calendar + the leakage suite
  baselines.py              seasonal-naive · SARIMA · XGBoost-quantile · recovery_impact
  recovery.py               the novel layer: Stage 1, Stage 2, correction, model selection
  forecast.py               TFT: grid search, fit, quantile forecasts

notebooks/
  01_data_and_recovery.ipynb    subset + baselines + recovery + leakage      ~15 min
  02_forecasting.ipynb          TFT search and fit, raw-vs-recovered         hours (GPU)
  03_ordering_and_results.ipynb order quantities, cost sweep, scoreboard     ~5 min

outputs/                    every reported number, as written by the code
  subset_summary.md         what the numbers describe; rewritten by every rebuild
  models/                   fitted models that must outlive the kernel

data/processed/             the working subset (rebuildable, mostly gitignored)
                            recovered parquets ARE committed, so notebook 02 runs in Colab

requirements.txt            pinned, except torch — see §9
```

This README is the project's only prose document. Everything else committed here is either code or
an artifact the code wrote, so nothing in the repo can fall out of date with the numbers.

Each notebook is independent — it reads what it needs from disk and says which notebook to run if
something is missing. Only a cold start needs them in order.

## 9. Stack

| layer | library |
|---|---|
| data | `numpy` 2.3.2 · `pandas` 2.3.1 · `pyarrow` 21.0.0 · `datasets` 4.8.5 |
| gradient boosting | `lightgbm` 4.6.0 (Stage-1 recovery) · `xgboost` 3.3.0 (quantile baseline) |
| classical | `statsmodels` 0.14.5 (SARIMA) · `scikit-learn` 1.8.0 |
| deep learning | `torch` · `lightning` · `pytorch-forecasting` (TFT) — unpinned so Colab's CUDA-matched build wins |
| figures / notebooks | `matplotlib` 3.10.8 · `ipykernel` 7.1.0 |

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
jupyter lab notebooks/                             # heavy fits sit behind RUN_* / TUNE toggles
```

`02_forecasting.ipynb` also runs in Colab straight from a clone — its first cell clones and installs,
and the recovered-demand parquets are committed so no rebuild is needed.

## 10. References

**Data**

- Dingdong Inc. — *FreshRetailNet-50K*. Hugging Face: `Dingdong-Inc/FreshRetailNet-50K`.
  Accuracy here is scored the dataset's own way — per date, on non-stockout rows only — so the
  numbers stay comparable to the published evaluation.

**Methods**

- Lim, Arık, Loeff & Pfister (2021). *Temporal Fusion Transformers for interpretable multi-horizon
  time series forecasting.* — the forecasting architecture.
- Romano, Patterson & Candès (2019). *Conformalized Quantile Regression.* — the calibration layer.
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.* — split conformal
  prediction.
- Kupiec (1995). *Techniques for verifying the accuracy of risk measurement models.* — the coverage
  test.
- Arrow, Harris & Marschak (1951). *Optimal inventory policy.* — the newsvendor quantile rule.
- Jørgensen (1987). *Exponential dispersion models.* — the Tweedie loss used in Stage 1, chosen for a
  zero-inflated target.
- Gneiting & Raftery (2007). *Strictly proper scoring rules, prediction, and estimation.* — pinball
  loss and CRPS.
