[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23975980&assignment_repo_type=AssignmentRepo)

# Perishable Demand Forecasting & Zero-Waste Inventory Engine

A fresh-food ordering pipeline that recovers the demand a stockout hides, forecasts it as an
honest range, and turns that range into an order quantity that beats simply ordering what sold
last week — built on **FreshRetailNet-50K** (Dingdong Inc.).

**The contribution is the recovery layer, not the forecaster.**

---

## 1. Problem statement

Fresh food is ordered against a forecast, and a bad forecast is paid for twice: order too much and
it is binned, order too little and the shelf is empty.

Underneath that sits a data problem that makes the forecast wrong in a specific, self-reinforcing
direction:

> **A sold-out shelf records zero sales, not zero demand.**

When a product runs out at 3pm, the rest of that day's demand is never observed; the sale is simply
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

| # | Objective | How it is judged |
|---|---|---|
| 1 | Establish what the status quo achieves on raw sales | WAPE against seasonal-naive / SARIMA / XGBoost-quantile |
| 2 | Recover the demand censoring hid, and validate it on data that could have falsified it | WAPE on held-out full-shelf days; must beat a no-model control |
| 3 | Show recovery changes the forecast where it should | raw-vs-recovered twins, split by how often a series sells out, **replicated across two model families** |
| 4 | Forecast as a **range**, not a point | pinball / CRPS vs the best baseline (TFT and gradient-boosted trees, ~4% apart) |
| 5 | Make the range honest | 80% band contains truth 80% of the time, with a clustered confidence interval. Coverage improves but still misses by 2–3 points (**a reported finding, not a fix**), §6 |
| 6 | Convert the range into an order that bins less food | cost, waste and demand met vs a naive order, across a cost sweep: cheaper than the naive rule at every cost ratio tested, and less waste at matched demand met, §6 |
| 7 | Score once on the sealed test week | all of the above, opened once: cheaper than the naive rule at the headline ratio for all four arms, by a smaller margin than validation suggested; two of three hand-picked operating points reverse on it, §6 |

## 3. Dataset

**FreshRetailNet-50K**, Dingdong Inc., via Hugging Face (`Dingdong-Inc/FreshRetailNet-50K`).

| | corpus | working subset |
|---|---|---|
| series (store × product) | 50,000 | 5,601 |
| stores | 898 | 100 |
| products | 865 | 557 |
| categories | 32 | 30 |
| train rows | 4,500,000 | 504,090 |
| normalised sale amount/day | 0.9986 | 0.997 |
| censored product-days | 44.3% | 43.8% |

The subset is **11.2% of the corpus**, drawn as 100 whole stores sampled uniformly. Whole stores, not
individual series: nothing is selected on sales volume, so the sample stays representative and no
scope restriction has to be declared, and each store keeps a full assortment, which the ordering
stage needs to recommend a realistic basket.

A further **39,207 rows** of shipped eval data are held back and read by no earlier stage.

The draw is **nested in the store count**: the stores are one fixed shuffle and the first *N* are
taken, so raising the count adds stores without swapping the ones already there. See
[outputs/reports/subset_summary.md](outputs/reports/subset_summary.md), regenerated on every rebuild so it can never
drift from the data it describes.

**Calendar** (frozen; boundaries live in `config` and are never redeclared):

| window | dates | used for |
|---|---|---|
| training | 2024-03-28 … 05-28 (62d) | all model fitting |
| validation | 2024-05-29 … 06-11 (14d) | early stopping, model choice, scorecards |
| calibration | 2024-06-12 … 06-25 (14d) | conformal band widths only, never trained on |
| test | the shipped eval file | opened once, at the final evaluation |

## 4. What counts as ground truth here, and what does not

This is the most important section in the README. Nearly every limitation downstream traces back to
a row in this table.

### Observed and verifiable

| what | where | why it can be trusted |
|---|---|---|
| **Sales on non-stockout days** | `sale_amount` where `stock_hour6_22_cnt == 0` | nothing was lost, so **recorded sales *are* true demand**. This is the anchor for every test in the project |
| **Which hours each shelf was empty** | `hours_stock_status` | the dataset's own annotation, the fact that makes censoring detectable rather than assumed |
| **Hourly sales vectors** | `hours_sale` | 24 values per product-day |
| **Discount, holiday, activity flags** | daily columns | recorded, not inferred |
| **Weather** | `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level` | observed (see the caveat below) |
| **Test-week actuals** | shipped eval file | sealed; read once |

### Not observable, and never will be from this data

| what | consequence |
|---|---|
| **Demand during a stockout** | the target of the entire recovery layer. **No ground truth exists and none can be constructed.** Every claim about it is indirect by necessity |
| **Stock on hand, deliveries** | recovery's central assumption cannot be checked directly against inventory |
| **Spoilage / waste** | never recorded, so never predicted; it is computed arithmetically as the gap between an order and a demand estimate |
| **Shelf life per product** | the newsvendor stays single-period; leftovers cannot carry forward |
| **Physical unit counts** | `sale_amount` is normalised by an undisclosed coefficient. **Every figure in this project is a ratio or a percentage**; no absolute quantity is ever asserted |
| **What a product actually is** | only an anonymised `product_id` is available - no name, description, or human-readable label. Category is the same: `first_category_id` etc. are numbers, not words like "dairy" |
| **The true waste:stockout cost ratio** | an assumption, swept across nine values; a claim only counts if it survives the sweep |

### Model output, not measurement

| what | how to read it |
|---|---|
| **`recovered_demand`** | a model output floored at recorded sales. **A floor on true demand, not an estimate of it.** Demonstrably closer than recorded sales, and if it errs it errs short |
| **Simulated waste** | arithmetic on an order and a demand *estimate*, not an observed outcome |

### Where each result is anchored

| stage | scored against | is that ground truth? |
|---|---|---|
| Baselines | recorded sales, non-stockout rows only | **yes** |
| Recovery | held-out full-shelf days, whose recorded total *is* true demand | **yes**, and the test could have failed |
| Raw-vs-recovered impact | recorded sales, non-stockout rows only | **yes** |
| Forecast accuracy | recorded sales, non-stockout rows only | **yes** |
| Ordering simulation | `recovered_demand` | **no**, a model output. The dependency is stated, never presented as neutral |

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
  recovered_demand ──┬───────► forecast.py   Temporal Fusion Transformer ─┐
                     │                                                    │  six quantiles
                     └───────► gbm.py        XGBoost, deterministic ──────┤  q025 … q975
                                             (the cross-architecture check)│
                                                                          ▼
                                              conformal.py   split CQR, one offset per band
                                                                          │
                                                                          ▼
                                              orders.py      newsvendor + cost sweep
```

Both forecasters are run on both targets, giving **four arms** (`tft_recovered`, `tft_raw`,
`xgb_recovered`, `xgb_raw`), which are calibrated and ordered through the same two functions. That
is what makes "recovery helps" a claim about the data rather than about one architecture.

Hours are an implementation detail confined to `recovery.py`; every public function in the project
takes and returns **daily** frames.

Everything downstream of recovery learns from filled-in demand, never raw sales. Scoring waste
against recorded sales would reward under-ordering, which is the failure being fixed.

**Design rules**

- **All logic lives in `src/`.** Notebooks and the dashboard are thin callers, so the science has
  exactly one implementation.
- **The dashboard reads saved files only** and never recomputes.
- **The subset is seeded and self-describing**: it rebuilds from two numbers.
- **Splits, once frozen, never move.** The test week is touched once.

## 6. Results

*100-store subset. Training → validation drives every model and architecture choice below; the
sealed test week appears only in the two subsections that say so, and was opened once.*

**Baselines on raw sales**, the bar to clear

| | WAPE | WPE |
|---|---|---|
| seasonal_naive | 0.4213 | +0.0187 |
| **xgboost_quantile** | **0.3423** | −0.0187 |
| sarima *(sampled series)* | 0.5210 | −0.2521 |

`WPE` near zero here is **not** evidence that censoring is harmless: these are non-stockout rows,
exactly the rows censoring does not touch. The scorecard structurally cannot see the problem.

**Recovery**, validated on held-out full-shelf days

| | |
|---|---|
| day WAPE | **0.2966** |
| vs `series_hour_mean` (no-model control) | 0.3684 (**19.5% better**) |
| day WPE after correction | +0.0186 |
| aggregation bias before correction | +13.2%, corrected by ×0.9001 |
| correction refit per period | 0.892 / 0.857 / 0.923, no monotone drift |
| leakage checks | **5 / 5 pass** |

Six candidates on the same test: **lightgbm_tweedie 0.2966** · xgboost 0.3009 · lightgbm_poisson
0.3051 · lightgbm_l2 0.3057 · series_hour_mean 0.3684 · poisson_glm 0.5692.

**Does recovery change the forecast?** One model, two targets, scored identically.

| series band | n scored | WAPE raw → recovered | WPE raw → recovered |
|---|---|---|---|
| **≥75% censored** | 1,375 | **0.2875 → 0.2256** *(−21.5%)* | **−0.196 → +0.001** |
| 50–75% | 12,319 | 0.3084 → 0.3124 | −0.015 → +0.112 |
| 25–50% | 30,345 | 0.3601 → 0.3702 | +0.008 → +0.113 |
| <25% | 3,955 | 0.4712 → 0.4604 | +0.020 → +0.116 |
| ALL | 47,994 | 0.3423 → 0.3426 | −0.019 → +0.101 |

On chronic-stockout series the raw forecaster is **19.6% low**, trained on sales that never
happened. Recovery removes essentially all of it. Read per band, never pooled: scoring uses
non-stockout rows only, exactly the rows the two targets agree on, so the pooled row structurally
cannot show the effect in either direction. Honest cost: outside the top band the recovered twin
runs ~11% high, which the ordering stage has to earn back.

**Forecasting**: two architectures, close but not tied.

| | WAPE | pinball(avg) | pinball@0.8 |
|---|---|---|---|
| **TFT on recovered demand** | **0.3284** | **0.1074** | 0.1281 |
| XGBoost on recovered demand | 0.3417 | 0.1116 | 0.1330 |
| `xgboost_quantile` baseline *(untuned)* | 0.3423 | 0.1121 | n/a |

The TFT wins by **0.0042 on pinball, about 3.9%**. That is a real margin, not the under-1% figure
reported here previously (0.1103 vs 0.1113): those numbers were read off a forecast the TFT has
since been re-run to produce, and the prose was never updated to match. The comparison is still set
up in the TFT's favour: it early-stops on the window it is then scored on, while the tree
early-stops on held-back *training* days, so the gap under an equally strict comparison is
probably smaller than 3.9%, but "the two are indistinguishable" is no longer the right summary.
The tree still reproduces exactly on a CPU, which is what lets anyone re-run the recovery result
without a GPU, and it stays within about 4% of the more expensive model.

**Does recovery help the forecast?** Pooled, no, and that is a property of the metric, not of the
layer. Recovery is a no-op on full-shelf days (largest difference: 3.6e-15) and those are the only
days that get graded. Split by how often a series sells out, **two unrelated model families
independently give the same answer**:

| products that sell out | TFT WAPE change | XGBoost WAPE change |
|---|---|---|
| rarely (<25% of days) | −0.6% | −1.5% |
| sometimes (25–50%) | +3.2% | +2.9% |
| often (50–75%) | +0.2% | +1.2% |
| **constantly (75%+)** | **−25.2%** | **−22.9%** |

Only the chronic-stockout group survives, and there it improves a lot: on that group the raw
model under-forecasts by 20.2% and the recovered model by 0.3%. That is the mechanism behaving as
designed: censoring hides demand only where shelves empty, so products that rarely sell out have
nothing to give back.

**The trade that decides it**: at q50, on full-shelf days only, the regime that *penalises*
recovery. XGBoost arm:

| products that sell out | lost sales recovered | waste added | pays once a stockout costs more than |
|---|---|---|---|
| rarely | 5.3 pts | 4.6 pts | **0.87×** a bin |
| sometimes | 4.8 pts | 5.9 pts | **1.23×** a bin |
| often | 6.1 pts | 6.6 pts | **1.08×** a bin |
| **constantly** | **13.7 pts** | 6.9 pts | **0.50×** a bin |

The TFT arm agrees: 0.94× / 1.13× / 1.01× / **0.49×**.

**This is the strongest defensible claim in the project**, and it needs no accuracy improvement to
be true. Recovery pays wherever an empty shelf costs more than roughly 1.2× a binned item, and on
chronic-stockout products once a stockout costs even half a bin. For fresh food a stockout is
always worse than a bin. WAPE cannot express this, because it charges a unit too many and a unit
too few identically.

**Calibration**, split-CQR, offset fitted on the calibration window only:

| band | window | before | after | target |
|---|---|---|---|---|
| 80% | validation | 0.738 | 0.826 | 0.80 |
| 80% | test | 0.678 | 0.791 | 0.80 |
| 95% | validation | 0.918 | 0.970 | 0.95 |
| 95% | test | 0.858 | 0.961 | 0.95 |

Coverage improves everywhere but **overshoots on the near window and undershoots on the far one**.
One offset cannot fit both, because coverage decays with distance in time: conformal prediction
guarantees coverage only under exchangeability, and a forecast horizon that walks forward violates
it. That is a **distribution shift, not a broken method**.

Two alternatives were measured before accepting it (multiplicative band-scaling and Mondrian
per-censoring-band offsets), on a harness that never touches the test week. **All three land within
0.0003 of each other**, so the functional form is not the problem and only one method is kept in the
code. The fix that does work is a measured *drift inflation*: calibrate at 83% to land on 80% a
window later, applied automatically to forward windows only.

Coverage is reported with a **day-block bootstrap interval** rather than a Kupiec test. Kupiec
assumes independent rows; these are 5,601 products sharing each date, and the measured day-to-day
spread is **13.7×** what independence predicts, so it rejects misses far too small to matter.

**Ordering**: all four arms through one function, at `c_u/c_o = 4` (a stockout assumed to cost 4×
a wasted unit), on validation:

| arm | stockouts | demand met | cost vs. the naive rule |
|---|---|---|---|
| **xgb_recovered** | **7.7%** | 97.2% | −30.7% |
| tft_recovered | 9.3% | 96.9% | −34.7% |
| xgb_raw | 12.6% | 95.2% | −32.8% |
| tft_raw | 14.6% | 95.1% | −37.3% |

*Naive rule = "order what sold on this day last week." Demand met = share of demand actually sold,
not lost to a stockout.*

Ordering from recovered demand cuts stockouts by ~5 percentage points against the raw arm, in both
families and under both demand regimes: **the sign never flips**. Cost beats the naive rule at all
nine cost ratios, by 21–73%.

**Does it hold on the sealed test week?** Same four arms, same headline ratio, scored once:

| arm | validation | test |
|---|---|---|
| tft_recovered | −34.7% | **−28.0%** |
| xgb_recovered | −30.7% | −13.4% |
| xgb_raw | −32.8% | −0.8% |
| tft_raw | −37.3% | −9.2% |

*Cost vs. the naive rule, both windows scored the same way against demand that actually happened.*

On validation the four arms sit within 7 points of each other and recovery looks like the smaller
factor. The test week reorders that: both **raw** arms nearly collapse to the naive rule's own cost
(xgb_raw: −0.8%, barely better than doing nothing), while both **recovered** arms stay well ahead.
Recovery's own effect (recovered − raw, same architecture) is **+18.8 points for TFT and +12.6 for
XGBoost** on the test week, against −2.6 and −2.1 points on validation. Recovery costs a little on
the weeks used to pick a model, and earns it back on the week nothing was tuned against. That is the
result the project exists to produce, and validation alone would not have shown it.

The 34.7%/28.0% headline figures are each an average over a handful of calendar days (14 validation,
7 test), not independent rows; every product sharing a date moves together. A day-block bootstrap
puts the true saving at roughly **30–39%** on validation and **25–31%** on the test week.

**Choosing an operating point, and does it survive the test week?** `c_u/c_o = 4` is one assumption
among nine tested. Three named points on the recovered-TFT forecast, *waste-focused* (q0.46),
*balanced* (q0.51) and *stockout-focused* (q0.90), were picked on validation, where Balanced beat the
naive rule on stockout days (35% vs. 42%) at essentially the same waste (21.8% vs. 21.9%). Holding
those exact quantiles fixed and applying them to the test week reverses two of the three: Balanced's
stockout days rise to 54% (naive: 43%) and Waste-focused's to 61%, both worse than doing nothing on
availability, cheaper only on waste. Stockout-focused (q0.90) is the only one of the three that still
beats the naive rule on stockout days in **both** windows and **both** model families, at the cost of
far more waste (40% vs. naive's 19%). All three still beat the naive rule on cost in both windows;
the reversal is about which failure mode a store is exposed to, not about money.

Every number above is written by the code into [outputs/](outputs/); the CSVs are the evidence, not
a transcription of it, and every figure in this section was re-verified directly against the saved
parquets/JSON/CSVs on 2026-08-17 by re-running the notebook that produces them, rather than copied
from an earlier draft. One structural gap remains: `baseline_scorecard.csv` has no pinball for two of
its five rows (`seasonal_naive` and `sarima` are point forecasts, not quantile ones, so pinball does
not apply to them); that is expected, not stale.

## 7. Limitations

This is a **proof of concept built on a dataset that records sales, not inventory.** That single
fact sets the ceiling on what any result here can claim; see §4 for the full ground-truth map.

| Limitation | Effect |
|---|---|
| **Demand during a stockout is unobservable** | the load-bearing one. Validation happens on full-shelf days but recovery is only ever *applied* to days the shelf emptied. No test can close that gap, so the layer is built to fall on the safe side of it |
| One subset draw (11.2% of corpus) | only one draw has ever been evaluated. The headline recovery result rests on **1,375 scored rows**, the smallest band in the table |
| Recovery's headline is a worst case | held-out days are blanked across all 16 active hours, harsher than the large majority of real censored days; the typical-day error is unmeasured. The bound errs against the model |
| Stage-1 history features are themselves censored | `lag7`/`roll7` are built from recorded sales, so the model's sense of a series' normal level is slightly depressed, biasing recovery **downward**, the same direction as the floor |
| The classical baseline is a sampled one | SARIMA is fitted on a sample of series, not all of them: a reference point, not a like-for-like competitor |
| The TFT beats the tree by only ~4% | a real but modest margin, and the comparison favours the TFT, so treat "the TFT is somewhat better" as the claim, not "clearly better" |
| Weather is realised, not forecast | flatters absolute accuracy; relative comparisons hold |
| Pinball ≥10% target vs ~4% achieved | reported as measured; the target was set before any evidence existed |
| Ordering is simulated against `recovered_demand` | not a neutral referee; the dependency is stated |
| Cost ratio is an assumption | swept across nine ratios |
| An operating point chosen on validation doesn't automatically hold | two of three named quantiles (waste-focused, balanced) beat the naive rule on stockout days on validation and lose to it on the test week, in both model families; only the most stockout-averse of the three held up in both. One test week is a check, not a certificate: a point worth re-checking, not a setting to fix and forget |
| The one-store walkthrough is illustrative, not evidence | n=1, picked as closest-to-median rather than best-case, but a single store's week is not the result; the 100-store aggregate above it is |

**What richer data would unlock.** Each layer can be replaced without touching the others:

- **Stock on hand and deliveries** → recovery's assumption becomes directly testable; known starting
  stock plus known replenishment gives true unmet demand, converting today's *floor* into a
  *measurement*.
- **Recorded spoilage** → waste becomes an observed outcome rather than an arithmetic one.
- **Shelf life** → the newsvendor becomes multi-period.
- **Real unit costs** → the cost sweep collapses to one operating point, and savings can be quoted in
  currency.
- **More sealed weeks, not just one** → whether an operating point picked on validation holds is
  currently answered by a single 7-day test window; a rolling series of them would show whether
  Stockout-focused's test-week win is the stable one or itself a one-window draw.

None of these change the architecture. They replace assumptions with measurements at the exact points
§4 names.

## 8. Repo structure

```
src/                        all logic, a library, no CLI entry points
  utils/
    config.py               every path and knob; nothing else hard-codes a path
    data_io.py              HF ingest, schema assertions, build_subset
    features.py             lag/rolling features, hourly explode, censoring bands
    metrics.py              WAPE / WPE / MAE / pinball / CRPS, matching the dataset's protocol,
                            plus the per-band and lost-sales-vs-waste tables
    plots.py                screen-sized figures, shared by the dashboard and the notebooks
    store_view.py           picks one representative store/products for the one-store walkthrough
  splits.py                 frozen calendar + the leakage suite
  baselines.py              seasonal-naive · SARIMA · XGBoost-quantile · recovery_impact
  recovery.py               the novel layer: Stage 1, Stage 2, correction, model selection
  forecast.py               TFT: grid search, fit, six-quantile forecasts
  gbm.py                    XGBoost quantile forecaster, same interface, deterministic, no GPU
  conformal.py              split CQR: one offset per band, coverage with a clustered interval
  orders.py                 newsvendor rule, realised-demand simulation, cost sweep, arm compare

notebooks/
  01_data_and_recovery.ipynb      subset + baselines + recovery + leakage      ~15 min
  02_forecast_tft.ipynb           TFT search and fit, raw-vs-recovered         hours (GPU)
  03_forecast_xgb.ipynb           the tree, and the cross-architecture check   ~1 h (CPU)
  04_ordering_and_results.ipynb   calibration, the four arms, the cost sweep   ~5 min
  05_poster_results.ipynb         reproduces every number/figure on the poster ~1 min

app/                        Streamlit reviewer dashboard, reads committed outputs only, never
                            recomputes; `streamlit run app/Introduction.py`
  Introduction.py            entry page: objective + subset summary
  pages/                     in reading order: Pipeline (approach) -> EDA (interactive
                            store/product-level stockout rankings and lookups, live over the
                            committed recovered parquet - how uneven the problem is, before the
                            fix) -> Recovery -> Calibration -> Ordering Decision -> Operating
                            Point -> Why TFT and Recovered -> Model Architecture & Tuning (every
                            model, every hyperparameter explained, tuning-grid winners read live
                            off the saved tuning tables) -> Limitations -> One Store (a single
                            store's real products, ordered and scored on the sealed test week).
                            EDA's default pick and One Store's default view are the same function
                            call, so they can't drift apart even though the two pages aren't
                            adjacent

outputs/                    every reported number, as written by the code, split by artifact KIND
  reports/                  csv / json / md, everything with numbers in it
    subset_summary.md       what the numbers describe; rewritten by every rebuild
  forecasts/{tft,xgb}/      saved quantile forecasts and their corrected intervals, per arm
  plots/                    png
  models/                   fitted models that must outlive the kernel

poster/                     poster.html (the submitted poster) and its exported PDF; every figure
                            and number in it is reproduced by notebook 05, not hand-drawn

data/processed/             the working subset (rebuildable, mostly gitignored)
                            recovered parquets ARE committed, so notebook 02 runs in Colab

requirements.txt            pinned, except torch (see §9)
```

**One implementation per idea.** Anything two notebooks both need lives in `src/` and is imported,
never pasted. The per-band scorecard and the lost-sales-vs-waste table are shared by notebooks 02
and 03 for exactly this reason: a second copy is how the transformer's table and the tree's table
would silently stop being the same measurement. The dashboard follows the same rule: it imports
`src/` and `plots.py` rather than keeping its own copy of any chart or number.

This README is the only prose document meant for a reader outside the project; everything else
committed here is either code, a saved artifact the code wrote, or the poster itself (whose numbers
are transcribed from those same artifacts by notebook 05, not computed independently of them).

Each notebook is independent: it reads what it needs from disk and says which notebook to run if
something is missing. Only a cold start needs them in order.

## 9. Stack

| layer | library |
|---|---|
| data | `numpy` 2.3.2 · `pandas` 2.3.1 · `pyarrow` 21.0.0 · `datasets` 4.8.5 |
| gradient boosting | `lightgbm` 4.6.0 (Stage-1 recovery) · `xgboost` 3.3.0 (quantile baseline) |
| classical | `statsmodels` 0.14.5 (SARIMA) · `scikit-learn` 1.8.0 |
| deep learning | `torch` · `lightning` · `pytorch-forecasting` (TFT), unpinned so Colab's CUDA-matched build wins |
| figures / notebooks | `matplotlib` 3.10.8 · `ipykernel` 7.1.0 |
| reviewer dashboard | `streamlit` 1.47.1 |

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
jupyter lab notebooks/                             # heavy fits sit behind RUN_* / TUNE toggles
streamlit run app/Introduction.py                  # reviewer dashboard, reads saved outputs only
```

`02_forecast_tft.ipynb` also runs in Colab straight from a clone: its first cell clones and installs,
and the recovered-demand parquets are committed so no rebuild is needed.

## 10. References

**Data**

- Dingdong Inc., *FreshRetailNet-50K*. Hugging Face: `Dingdong-Inc/FreshRetailNet-50K`.
  Accuracy here is scored the dataset's own way (per date, on non-stockout rows only), so the
  numbers stay comparable to the published evaluation.
- Wang et al. (2025). *FreshRetailNet-50K* (arXiv:2505.16319): the dataset paper. It names
  perishable inventory optimization a direction the dataset opens, not one it addresses; recovery
  through to an order quantity is this project's answer to that gap, not a re-run of their benchmark.

**Methods**

- Lim, Arık, Loeff & Pfister (2021). *Temporal Fusion Transformers for interpretable multi-horizon
  time series forecasting*: the forecasting architecture.
- Romano, Patterson & Candès (2019). *Conformalized Quantile Regression*: the calibration layer.
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*: split conformal
  prediction.
- Arrow, Harris & Marschak (1951). *Optimal inventory policy*: the newsvendor quantile rule.
- Jørgensen (1987). *Exponential dispersion models*: the Tweedie loss used in Stage 1, chosen for a
  zero-inflated target.
- Gneiting & Raftery (2007). *Strictly proper scoring rules, prediction, and estimation*: pinball
  loss and CRPS.
