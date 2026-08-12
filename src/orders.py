"""M6 - Orders (PLAN.md Phase 6). Turn a calibrated demand quantile into a single order
quantity via the newsvendor critical fractile, then prove it beats the status quo with a Monte
Carlo simulation swept across every plausible cost assumption.

    q* = c_u / (c_u + c_o)                    the newsvendor rule (critical fractile)
    order = F^-1(q*)                          read straight off the (corrected) forecast

`F^-1` is not a parametric distribution - it is a piecewise-linear inverse-CDF built from
whichever corrected quantiles Phase 5 saved (`_tau_breakpoints`), so the order quantity and the
Monte Carlo demand draws are read off the *same* curve by construction.

Reads Phase-5 output only (ground rule 7 - this module writes files, Streamlit/the notebook only
read them; nothing here retrains a forecaster). Simulates against `recovered_demand` (ground
rule 2) on EVERY test product-day, censored or not - a store orders daily, and the censored days
are exactly where the fill-in layer earns its keep (PLAN.md §8 Phase 7). The naive competitor is
built from raw `sale_amount` on purpose: it is what a store could actually see without this
pipeline.
"""
import numpy as np
import pandas as pd

from .utils import config

KEY = ["store_id", "product_id", "dt"]

# {tau: corrected-quantile column} for the two Phase-5 outputs orders.py can read.
CANONICAL_QCOLS = {0.1: "q10_c", 0.5: "q50", 0.9: "q90_c"}
WIDE_QCOLS = {0.025: "q025_c", 0.1: "q10_c", 0.5: "q50", 0.9: "q90_c", 0.975: "q975_c"}

# c_u/c_o (c_o fixed at 1) spanning waste-averse to stockout-averse -> q* = 0.2 .. 0.95.
DEFAULT_RATIOS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 19.0]
HEADLINE_RATIO = 4.0   # c_u/c_o = 4 -> q* = 0.8, the canonical 80% band's own service level

WASTE_TARGET_PCT = 15.0   # PLAN.md §7 business KPI
STOCKOUT_TARGET = 0.05


# --------------------------------------------------------------- the newsvendor rule
def critical_fractile(c_u: float, c_o: float) -> float:
    """q* = c_u / (c_u + c_o) - the demand quantile that minimises expected cost when a missed
    sale costs `c_u` (under-order) and a wasted unit costs `c_o` (over-order). Real costs are
    not in the data (ground rule 4), so both are assumptions - `cost_sweep` sweeps them."""
    if c_u <= 0 or c_o <= 0:
        raise ValueError("both costs must be positive")
    return c_u / (c_u + c_o)


# --------------------------------------------------------------- reading Phase-5 forecasts
def load_test_forecast(wide: bool = True) -> tuple:
    """The Phase-5 conformal test frame Phase 6 simulates against, plus the {tau: column} map
    describing which corrected quantiles it carries.

    `wide=True` (default, PLAN.md's recommendation) merges the two conformal *correction* runs -
    `..._wide80.parquet` (corrected q10/q90) and `..._wide95.parquet` (corrected q025/q975) -
    into one 5-point corrected grid. Both are `conformal.run()` calls over the SAME saved Phase-4
    forecast (one target already carries all six quantiles - see `forecast.QUANTILES`), so their
    raw q10/q50/q90 columns are identical and only the conformal correction differs; asserted
    below rather than assumed. `wide=False` falls back to the canonical 3-point 80%-band file,
    which PLAN.md's Phase-6 input audit calls "nearly self-sufficient" on its own."""
    w80_path = config.OUTPUTS_DIR / "forecast_test_conformal_wide80.parquet"
    w95_path = config.OUTPUTS_DIR / "forecast_test_conformal_wide95.parquet"
    have_wide = wide and w80_path.exists() and w95_path.exists()

    # the canonical (untagged) file is only actually read on the non-wide paths below - do not
    # require it to exist when both wide files are already there, or a run that only ever called
    # conformal.run(tag="wide80"/"wide95") would be forced into a third, redundant call just to
    # satisfy an existence check nothing downstream reads.
    if not have_wide:
        canon_path = config.OUTPUTS_DIR / "forecast_test_conformal.parquet"
        if not canon_path.exists():
            raise FileNotFoundError(f"{canon_path.name} missing - run conformal.run() first "
                                    f"(Phase 5), or run it for both 'wide80'/'wide95' tags.")
        canon = pd.read_parquet(canon_path)

    if not wide:
        return canon, CANONICAL_QCOLS

    if not have_wide:
        print("wide (5-quantile) conformal files not found - falling back to the 3-point "
              "canonical band (still a valid Phase-6 input, just coarser tails). Run "
              "conformal.run(0.80, 'q10', 'q90', tag='wide80') and "
              "conformal.run(0.95, 'q025', 'q975', tag='wide95') to unlock it.")
        return canon, CANONICAL_QCOLS

    w80, w95 = pd.read_parquet(w80_path), pd.read_parquet(w95_path)
    mismatch = (w80.set_index(KEY)[["q10", "q50", "q90"]]
                .sub(w95.set_index(KEY)[["q10", "q50", "q90"]]).abs().to_numpy().max())
    assert mismatch < 1e-6, ("wide80/wide95 raw quantiles disagree - they should be the same "
                              "underlying forecast, just different conformal corrections")

    merged = w80[KEY + ["q50", "q10_c", "q90_c", "recovered_demand", "sale_amount",
                        "observed", "stock_hour6_22_cnt"]].merge(
        w95[KEY + ["q025_c", "q975_c"]], on=KEY, how="inner")
    assert len(merged) == len(w80) == len(w95), "wide80/wide95 have mismatched rows"
    return merged, WIDE_QCOLS


# --------------------------------------------------------------- reading a demand quantile
def _tau_breakpoints(df: pd.DataFrame, qcols: dict) -> tuple:
    """Fixed inverse-CDF breakpoints shared by every row: tau=0 -> 0 units, each saved quantile
    in order, tau=1 -> a linear extrapolation past the highest saved quantile (so demand isn't
    hard-capped at, say, q975_c - we just have no direct estimate of what lies beyond it).
    `np.maximum.accumulate` enforces monotonicity per row (a corrected band can, rarely, cross a
    neighbouring uncorrected quantile - "quantile crossing", the standard fix).
    Returns (taus sorted incl. 0/1, values array shape (n_rows, n_taus))."""
    taus = sorted(qcols)
    vals = np.column_stack([df[qcols[t]].to_numpy(dtype=float) for t in taus])
    top_gap = vals[:, -1] - vals[:, -2]                     # slope of the last saved segment
    extrapolated_top = vals[:, -1] + top_gap * ((1 - taus[-1]) / (taus[-1] - taus[-2]))
    vals = np.column_stack([np.zeros(len(df)), vals, np.maximum(extrapolated_top, vals[:, -1])])
    vals = np.maximum(np.maximum.accumulate(vals, axis=1), 0.0)
    return np.array([0.0] + taus + [1.0]), vals


def order_quantity(df: pd.DataFrame, q_star: float, qcols: dict = None) -> np.ndarray:
    """The order quantity at the newsvendor critical fractile `q_star`, read off each row's
    (corrected) predictive distribution by linear interpolation between saved quantiles."""
    qcols = qcols or CANONICAL_QCOLS
    taus, vals = _tau_breakpoints(df, qcols)
    q_star = float(np.clip(q_star, 0.0, 1.0))
    idx = int(np.clip(np.searchsorted(taus, q_star, side="left"), 1, len(taus) - 1))
    t0, t1 = taus[idx - 1], taus[idx]
    frac = 0.0 if t1 == t0 else (q_star - t0) / (t1 - t0)
    return vals[:, idx - 1] + frac * (vals[:, idx] - vals[:, idx - 1])


def sample_demand(df: pd.DataFrame, qcols: dict = None, n_draws: int = 10_000,
                  random_state: int = None) -> np.ndarray:
    """`n_draws` Monte Carlo demand samples per row, drawn from the same piecewise-linear
    inverse-CDF `order_quantity` reads from - so the simulated demand is consistent with the
    quantiles the order was chosen against. Returns shape (len(df), n_draws)."""
    qcols = qcols or CANONICAL_QCOLS
    rng = np.random.default_rng(config.RANDOM_STATE if random_state is None else random_state)
    taus, vals = _tau_breakpoints(df, qcols)
    u = rng.random((len(df), n_draws))
    idx = np.clip(np.searchsorted(taus, u.ravel(), side="left"), 1, len(taus) - 1).reshape(u.shape)
    t0, t1 = taus[idx - 1], taus[idx]
    v0 = np.take_along_axis(vals, idx - 1, axis=1)
    v1 = np.take_along_axis(vals, idx, axis=1)
    return v0 + (u - t0) / (t1 - t0) * (v1 - v0)


# --------------------------------------------------------------- the naive competitor
def naive_orders(test_df: pd.DataFrame, train_recovered: pd.DataFrame = None) -> np.ndarray:
    """The status-quo ordering rule (§2b): 'order what sold same day last week' - raw
    `sale_amount` seven days before the test date, from the RAW history a store actually has
    (no fill-in, no calibration - that is the point of the comparison). Every test date's -7
    falls inside the frozen calibration window (2024-06-19..06-25), so this is exact, not
    approximate; the rare row with no match (a series absent from that window) falls back to its
    own q50 forecast so every row still gets an order."""
    train_recovered = (pd.read_parquet(config.RECOVERED_TRAIN) if train_recovered is None
                       else train_recovered)
    lag = test_df[KEY].assign(dt=test_df["dt"] - pd.Timedelta(days=7))
    joined = lag.merge(train_recovered[KEY + ["sale_amount"]], on=KEY, how="left")
    naive = joined["sale_amount"].to_numpy()
    missing = np.isnan(naive)
    if missing.any():
        print(f"naive_orders: {missing.sum()} of {len(test_df)} test rows have no exact "
              f"7-day-back match - falling back to that row's own q50 forecast")
        naive[missing] = test_df["q50"].to_numpy()[missing]
    return naive


# --------------------------------------------------------------- simulate + sweep
def simulate(orders: np.ndarray, demand_draws: np.ndarray, c_u: float = 1.0,
            c_o: float = 1.0) -> pd.DataFrame:
    """Monte Carlo tally for one order policy: order too much -> waste (cost `c_o`/unit); order
    too little -> a lost sale (cost `c_u`/unit). `orders` is one quantity per row;
    `demand_draws` is (n_rows, n_draws) simulated demand for that same row (from
    `sample_demand`). Returns per-row expectations, averaged over the draws."""
    orders = np.asarray(orders)[:, None]                     # broadcast over draws
    waste = np.maximum(orders - demand_draws, 0)
    shortfall = np.maximum(demand_draws - orders, 0)
    return pd.DataFrame({
        "order_quantity": orders.ravel(),
        "expected_waste": waste.mean(axis=1),
        "expected_shortfall": shortfall.mean(axis=1),
        "stockout_prob": (demand_draws > orders).mean(axis=1),
        "expected_cost": (c_o * waste + c_u * shortfall).mean(axis=1),
    })


def cost_sweep(df: pd.DataFrame, qcols: dict, naive_order: np.ndarray, demand_draws: np.ndarray,
               ratios=DEFAULT_RATIOS) -> pd.DataFrame:
    """PLAN.md §7's 'does the win survive any cost assumption?' check: run the newsvendor policy
    (and the naive one) at every `c_u/c_o` ratio (c_o fixed at 1), against the SAME
    `demand_draws` throughout (common random numbers - a fair, low-variance comparison both
    within a ratio and across the sweep). One row per ratio."""
    rows = []
    for c_u in ratios:
        q_star = critical_fractile(c_u, 1.0)
        ours = simulate(order_quantity(df, q_star, qcols), demand_draws, c_u, 1.0)
        naive = simulate(naive_order, demand_draws, c_u, 1.0)
        w_ours, w_naive = ours["expected_waste"].sum(), naive["expected_waste"].sum()
        cost_ours, cost_naive = ours["expected_cost"].sum(), naive["expected_cost"].sum()
        rows.append(dict(
            c_u=c_u, c_o=1.0, q_star=round(q_star, 4),
            waste_ours=round(float(w_ours), 2), waste_naive=round(float(w_naive), 2),
            waste_reduction_pct=round(100 * (1 - w_ours / w_naive), 2) if w_naive else np.nan,
            stockout_rate_ours=round(float(ours["stockout_prob"].mean()), 4),
            stockout_rate_naive=round(float(naive["stockout_prob"].mean()), 4),
            cost_ours=round(float(cost_ours), 2), cost_naive=round(float(cost_naive), 2),
            cost_savings_pct=round(100 * (1 - cost_ours / cost_naive), 2) if cost_naive else np.nan,
        ))
    return pd.DataFrame(rows)


def run(c_u: float = HEADLINE_RATIO, c_o: float = 1.0, ratios=DEFAULT_RATIOS, wide: bool = True,
       n_draws: int = 10_000, random_state: int = None, save: bool = True,
       verbose: bool = True) -> dict:
    """End to end: load the Phase-5 test forecast, build the naive competitor, draw demand once
    and reuse it everywhere (common random numbers), then produce (1) a per-product-day table at
    one headline cost ratio - the Order tab's input - and (2) the full cost-ratio sweep - the
    business-KPI proof. Saves `outputs/order_simulation.csv` + `outputs/cost_sweep.csv`."""
    df, qcols = load_test_forecast(wide=wide)
    naive_order = naive_orders(df)
    demand_draws = sample_demand(df, qcols, n_draws=n_draws, random_state=random_state)

    q_star = critical_fractile(c_u, c_o)
    ours = simulate(order_quantity(df, q_star, qcols), demand_draws, c_u, c_o)
    naive = simulate(naive_order, demand_draws, c_u, c_o)

    per_day = df[KEY + ["recovered_demand", "sale_amount", "observed",
                        "stock_hour6_22_cnt"]].reset_index(drop=True).copy()
    per_day["c_u"], per_day["c_o"], per_day["q_star"] = c_u, c_o, round(q_star, 4)
    per_day["ours_order_quantity"] = ours["order_quantity"].to_numpy()
    per_day["naive_order_quantity"] = naive_order
    for col in ("expected_waste", "expected_shortfall", "stockout_prob", "expected_cost"):
        per_day[f"ours_{col}"] = ours[col].to_numpy()
        per_day[f"naive_{col}"] = naive[col].to_numpy()

    sweep = cost_sweep(df, qcols, naive_order, demand_draws, ratios=ratios)
    passes = bool(((sweep["waste_reduction_pct"] >= WASTE_TARGET_PCT)
                  & (sweep["stockout_rate_ours"] <= STOCKOUT_TARGET)).all())
    headline_waste_pct = round(float(100 * (1 - ours["expected_waste"].sum()
                                            / naive["expected_waste"].sum())), 2)
    kpi = dict(headline=dict(c_u=c_u, c_o=c_o, q_star=round(q_star, 4),
                             waste_reduction_pct=headline_waste_pct,
                             stockout_rate=round(float(ours["stockout_prob"].mean()), 4)),
              sweep_passes_everywhere=passes,
              ratios_failing=sweep.loc[~((sweep["waste_reduction_pct"] >= WASTE_TARGET_PCT)
                                        & (sweep["stockout_rate_ours"] <= STOCKOUT_TARGET)),
                                       "c_u"].tolist())

    if save:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        per_day.to_csv(config.ORDER_SIMULATION_CSV, index=False)
        sweep.to_csv(config.COST_SWEEP_CSV, index=False)

    if verbose:
        print(f"newsvendor headline (c_u={c_u}, c_o={c_o} -> q*={q_star:.3f}, n_draws={n_draws:,}):"
              f" waste {headline_waste_pct:+.1f}% vs. naive, "
              f"stockout rate {kpi['headline']['stockout_rate']:.1%}")
        print(f"\ncost sweep ({len(ratios)} ratios, {qcols is WIDE_QCOLS and 'wide' or 'canonical'}"
              f" quantile grid):")
        print(sweep.to_string(index=False))
        verdict = "PASS" if passes else f"NOT MET at c_u={kpi['ratios_failing']}"
        print(f"\nKPI (PLAN.md §7): >={WASTE_TARGET_PCT:.0f}% less waste at "
              f"<={STOCKOUT_TARGET:.0%} stockouts, holding across the whole sweep -> {verdict}")
    return dict(per_day=per_day, sweep=sweep, kpi=kpi)
