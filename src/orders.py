"""Orders - turn a demand quantile into one order quantity, then test it against what happened.

    q* = c_u / (c_u + c_o)      the newsvendor rule
    order = F^-1(q*)            read off the (corrected) forecast

Scored against DEMAND THAT ACTUALLY OCCURRED. An earlier version drew simulated demand from the same
forecast it was testing, which made every result an identity: the measured stockout rate came out at
exactly 1 - q* at every cost ratio, because ordering the q*-th percentile of a distribution you also
sampled from misses (1 - q*) of the time by construction. That version could not tell one forecaster
from another, or a good forecast from a random one.

Demand is unknown on stockout days - that is the problem the project exists for - so no single scoring
choice is neutral. Both are run and the pair reported as a BOUND:

    observed    full-shelf days only, demand = recorded sales (true there). ADVERSE to recovery,
                since those are the quiet days where the recovered model over-orders
    recovered   every day, demand = `recovered_demand`. FAVOURABLE, since the recovered arm is then
                partly graded against its own training target

A conclusion holding at both ends is robust. One that flips between them is undetermined, and should
be reported that way.

Takes a forecast FRAME, so every arm - tft_recovered, tft_raw, xgb_recovered, xgb_raw - runs through
one function. Defaults to the validation window: the test week opens once, and only by an explicit
`period="test"`.
"""
import numpy as np
import pandas as pd

from .utils import config

KEY = ["store_id", "product_id", "dt"]

# {tau: corrected-quantile column} for the two conformal outputs orders.py can read.
CANONICAL_QCOLS = {0.1: "q10_c", 0.5: "q50", 0.9: "q90_c"}
WIDE_QCOLS = {0.025: "q025_c", 0.1: "q10_c", 0.5: "q50", 0.9: "q90_c", 0.975: "q975_c"}

# c_u/c_o (c_o fixed at 1) spanning waste-averse to stockout-averse -> q* = 0.2 .. 0.95.
DEFAULT_RATIOS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 19.0]
HEADLINE_RATIO = 4.0   # c_u/c_o = 4 -> q* = 0.8, the canonical 80% band's own coverage target


# --------------------------------------------------------------- the newsvendor rule
def critical_fractile(c_u: float, c_o: float) -> float:
    """q* = c_u / (c_u + c_o) - the demand quantile that minimises expected cost when a missed
    sale costs `c_u` (under-order) and a wasted unit costs `c_o` (over-order). Real costs are
    not in the data, so both are assumptions - `cost_sweep` sweeps them rather than picking one."""
    if c_u <= 0 or c_o <= 0:
        raise ValueError("both costs must be positive")
    return c_u / (c_u + c_o)


# --------------------------------------------------------------- reading corrected forecasts
def load_forecast(period: str = "validation", wide: bool = True, family: str = "tft",
                  target: str = "recovered") -> tuple:
    """One arm's conformally-corrected forecast frame, plus the {tau: column} map for its quantiles.

    An arm is (`family`, `target`) - which model, trained on which of the two demand columns. All
    four combinations are on disk under distinct names, so any caller can put them side by side.

    `period` defaults to validation on purpose. The test week is the final exam and works once, so
    reaching it requires naming it.

    `wide=True` merges the two conformal runs - `_wide80` (corrected q10/q90) and `_wide95` (corrected
    q025/q975) - into one 5-point grid. Both correct the SAME underlying forecast, so their raw
    quantiles must agree; asserted, not assumed.
    """
    w80_path = config.conformal_parquet(period, "wide80", family=family, target=target)
    w95_path = config.conformal_parquet(period, "wide95", family=family, target=target)

    if wide and w80_path.exists() and w95_path.exists():
        w80, w95 = pd.read_parquet(w80_path), pd.read_parquet(w95_path)
        mismatch = (w80.set_index(KEY)[["q10", "q50", "q90"]]
                    .sub(w95.set_index(KEY)[["q10", "q50", "q90"]]).abs().to_numpy().max())
        assert mismatch < 1e-6, ("wide80/wide95 raw quantiles disagree - they should be the same "
                                 "underlying forecast, differing only in the correction")
        merged = w80[KEY + ["q50", "q10_c", "q90_c", "recovered_demand", "sale_amount",
                            "stock_hour6_22_cnt"]].merge(w95[KEY + ["q025_c", "q975_c"]],
                                                         on=KEY, how="inner")
        assert len(merged) == len(w80) == len(w95), "wide80/wide95 have mismatched rows"
        return merged, WIDE_QCOLS

    canon = config.conformal_parquet(period, family=family, target=target)
    if not canon.exists():
        raise FileNotFoundError(
            f"{family}/{canon.name} missing - run conformal.run(eval_periods=('{period}',), "
            f"family='{family}', forecast_tag='{target}') first, or run it for both the 'wide80' "
            f"and 'wide95' tags to use the 5-point grid.")
    return pd.read_parquet(canon), CANONICAL_QCOLS


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


# --------------------------------------------------------------- the naive competitor
def naive_orders(df: pd.DataFrame, train_recovered: pd.DataFrame = None) -> np.ndarray:
    """The status-quo ordering rule: 'order what sold the same day last week' - raw `sale_amount`
    seven days back, from the RAW history a store actually has. No fill-in and no calibration,
    which is the entire point of the comparison.

    Exact for every window this is scored on, not approximate: validation and test both sit at
    least seven days inside the train file, so each row's -7 has a real recorded value. The rare
    row with no match (a series absent that day) falls back to its own q50 forecast, so every row
    still gets an order."""
    train_recovered = (pd.read_parquet(config.RECOVERED_TRAIN) if train_recovered is None
                       else train_recovered)
    lag = df[KEY].assign(dt=df["dt"] - pd.Timedelta(days=7))
    joined = lag.merge(train_recovered[KEY + ["sale_amount"]], on=KEY, how="left")
    naive = joined["sale_amount"].to_numpy()
    missing = np.isnan(naive)
    if missing.any():
        print(f"naive_orders: {missing.sum()} of {len(df)} rows have no exact 7-day-back match "
              f"- falling back to that row's own q50 forecast")
        naive[missing] = df["q50"].to_numpy()[missing]
    return naive


# --------------------------------------------------------------- what demand to score against
DEMAND_REGIMES = {
    "observed": "full-shelf days only, demand = recorded sales. Adverse to recovery.",
    "recovered": "every day, demand = recovered_demand. Favourable to recovery.",
}


def realised_demand(df: pd.DataFrame, regime: str = "observed") -> tuple:
    """The demand column to score against, and the row mask it applies to.

    `observed` keeps only days the shelf never emptied, where recorded sales ARE demand - so waste and
    lost sales are both real, not estimated. It is the conservative regime: those are the quiet days,
    which is exactly where a recovered-demand model over-orders, so its cost shows and its benefit does
    not.

    `recovered` uses every day against the recovered column. Bigger sample and it covers the days
    recovery changes, but the recovered arm is then partly graded against its own training target.
    """
    if regime not in DEMAND_REGIMES:
        raise ValueError(f"regime must be one of {list(DEMAND_REGIMES)}, got {regime!r}")
    if regime == "observed":
        mask = (df["stock_hour6_22_cnt"] == 0).to_numpy()
        return df.loc[mask, "sale_amount"].to_numpy(dtype=float), mask
    return df["recovered_demand"].to_numpy(dtype=float), np.ones(len(df), dtype=bool)


# --------------------------------------------------------------- simulate + sweep
def simulate(orders: np.ndarray, demand: np.ndarray, c_u: float = 1.0,
             c_o: float = 1.0) -> pd.DataFrame:
    """Per-row outcome of one order policy against REALISED demand.

    Order above demand and the surplus is binned; order below and the shortfall walks away. Unlike the
    Monte Carlo version this replaced, `stockout_rate` is now a measurement - it can come out anywhere,
    and a forecast that misjudges which days are busy will be punished for it.
    """
    orders, demand = np.asarray(orders, dtype=float), np.asarray(demand, dtype=float)
    if orders.shape != demand.shape:
        raise ValueError(f"orders {orders.shape} and demand {demand.shape} must align - the mask from "
                         "`realised_demand` has to be applied to the orders too")
    waste = np.maximum(orders - demand, 0)
    shortfall = np.maximum(demand - orders, 0)
    return pd.DataFrame({"order_quantity": orders, "demand": demand,
                         "waste": waste, "shortfall": shortfall,
                         "stockout": (demand > orders).astype(float),
                         "cost": c_o * waste + c_u * shortfall})


def cost_sweep(df: pd.DataFrame, qcols: dict, naive_order: np.ndarray, demand: np.ndarray,
               mask: np.ndarray, ratios=DEFAULT_RATIOS) -> pd.DataFrame:
    """The newsvendor policy against the naive one at every cost ratio, on the same realised demand.

    `demand_met_pct` is the share of demand actually met, and it's what makes the arms comparable: an
    arm that simply orders more will always waste more and stock out less, so waste only means
    something read next to how much demand it met.
    """
    rows = []
    for c_u in ratios:
        q_star = critical_fractile(c_u, 1.0)
        model = simulate(order_quantity(df, q_star, qcols)[mask], demand, c_u, 1.0)
        naive = simulate(naive_order[mask], demand, c_u, 1.0)
        total = demand.sum()
        rows.append(dict(
            c_u=c_u, c_o=1.0, q_star=round(q_star, 4),
            waste_model=round(float(model.waste.sum()), 2),
            waste_naive=round(float(naive.waste.sum()), 2),
            waste_vs_naive_pct=round(100 * (1 - model.waste.sum() / naive.waste.sum()), 2)
            if naive.waste.sum() else np.nan,
            stockout_pct_model=round(100 * float(model.stockout.mean()), 2),
            stockout_pct_naive=round(100 * float(naive.stockout.mean()), 2),
            demand_met_pct_model=round(100 * float(1 - model.shortfall.sum() / total), 2),
            demand_met_pct_naive=round(100 * float(1 - naive.shortfall.sum() / total), 2),
            cost_model=round(float(model.cost.sum()), 2),
            cost_naive=round(float(naive.cost.sum()), 2),
            cost_vs_naive_pct=round(100 * (1 - model.cost.sum() / naive.cost.sum()), 2)
            if naive.cost.sum() else np.nan,
        ))
    return pd.DataFrame(rows)


def run(forecast_df: pd.DataFrame = None, qcols: dict = None, period: str = "validation",
        regime: str = "observed", ratios=DEFAULT_RATIOS, c_u: float = HEADLINE_RATIO,
        c_o: float = 1.0, wide: bool = True, family: str = "tft", target: str = "recovered",
        save: bool = True, verbose: bool = True) -> dict:
    """Order from one forecast, score it against realised demand, sweep the cost ratio.

    Pass `forecast_df`/`qcols` to score an arm already in memory; omit them to load `period` from disk.
    Defaults to validation - the test week needs `period="test"` spelled out.
    """
    if forecast_df is None:
        forecast_df, qcols = load_forecast(period=period, wide=wide, family=family,
                                           target=target)
    qcols = qcols or CANONICAL_QCOLS
    
    demand, mask = realised_demand(forecast_df, regime)
    naive_order = naive_orders(forecast_df)
    sweep = cost_sweep(forecast_df, qcols, naive_order, demand, mask, ratios=ratios)

    q_star = critical_fractile(c_u, c_o)
    model = simulate(order_quantity(forecast_df, q_star, qcols)[mask], demand, c_u, c_o)
    naive = simulate(naive_order[mask], demand, c_u, c_o)

    per_day = forecast_df.loc[mask, KEY + ["sale_amount", "recovered_demand",
                                           "stock_hour6_22_cnt"]].reset_index(drop=True)
    per_day["regime"], per_day["q_star"] = regime, round(q_star, 4)
    for col in ("order_quantity", "waste", "shortfall", "cost"):
        per_day[f"model_{col}"] = model[col].to_numpy()
        per_day[f"naive_{col}"] = naive[col].to_numpy()

    # `cost_vs_naive_pct` belongs next to `waste_vs_naive_pct`, not below it: read alone, waste is
    # actively misleading here. The naive rule wastes little because it under-orders and stocks out
    # ~42% of the time, so ANY policy that keeps shelves full loses on waste while winning on cost.
    # Cost is the only column that prices both mistakes on the terms the cost ratio declares.
    kpi = dict(period=period, regime=regime, n_scored=int(mask.sum()),
               headline=dict(c_u=c_u, c_o=c_o, q_star=round(q_star, 4),
                             waste_vs_naive_pct=round(float(
                                 100 * (1 - model.waste.sum() / naive.waste.sum())), 2),
                             cost_vs_naive_pct=round(float(
                                 100 * (1 - model.cost.sum() / naive.cost.sum())), 2),
                             stockout_pct=round(float(100 * model.stockout.mean()), 2),
                             demand_met_pct=round(float(
                                 100 * (1 - model.shortfall.sum() / demand.sum())), 2)))

    if save:
        # Per-product-day rows are DATA (39K x 20), so parquet beside the forecasts; the sweep is a
        # summary table a person reads, so CSV in reports/. As a CSV the per-day file was 7.8 MB -
        # by itself more than the whole reports/ directory it was sitting in.
        config.ORDER_SIMULATION.parent.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        per_day.to_parquet(config.ORDER_SIMULATION, index=False)
        sweep.to_csv(config.COST_SWEEP_CSV, index=False)

    if verbose:
        print(f"[{period} | {regime}] {mask.sum():,} rows scored - {DEMAND_REGIMES[regime]}")
        print(f"  headline c_u={c_u}:{c_o} -> q*={q_star:.3f} | "
              f"waste {kpi['headline']['waste_vs_naive_pct']:+.1f}% vs naive | "
              f"cost {kpi['headline']['cost_vs_naive_pct']:+.1f}% vs naive | "
              f"stockouts {kpi['headline']['stockout_pct']:.1f}% | "
              f"demand met {kpi['headline']['demand_met_pct']:.1f}%")
        print(sweep.to_string(index=False))
    return dict(per_day=per_day, sweep=sweep, kpi=kpi)


def at_demand_met(frames: dict, target_demand_met: float = 0.95, regime: str = "observed",
                  verbose: bool = True) -> pd.DataFrame:
    """Every arm held to meeting the SAME share of demand - what does each one waste to get there?

    `run`'s headline compares arms at one fixed cost ratio, where an arm that simply orders MORE
    stocks out less and wastes more - so it can't tell "forecasts better" from "orders harder".
    Fixing demand-met removes that: same availability, so waste is what the forecast bought.

    Each arm's own order percentile is solved for and reported too - a lower one for the same
    demand-met share means the forecast is better centred and needs less over-ordering to get there.
    """
    grid = np.arange(0.50, 0.996, 0.005)
    rows = []
    for name, (df, qcols) in frames.items():
        demand, mask = realised_demand(df, regime)
        met, waste, stockout = [], [], []
        for q in grid:
            sim = simulate(order_quantity(df, q, qcols)[mask], demand)
            met.append(1 - sim.shortfall.sum() / demand.sum())
            waste.append(100 * sim.waste.sum() / demand.sum())
            stockout.append(100 * sim.stockout.mean())
        rows.append({"arm": name,
                     "order_percentile": round(float(np.interp(target_demand_met, met, grid)), 3),
                     "waste_pct": round(float(np.interp(target_demand_met, met, waste)), 1),
                     "stockout_pct": round(float(np.interp(target_demand_met, met, stockout)), 2)})

    out = pd.DataFrame(rows).sort_values("waste_pct").reset_index(drop=True)
    if verbose:
        print(f"every arm held to {target_demand_met:.0%} of demand met | {regime} regime "
              f"({DEMAND_REGIMES[regime].split('.')[0]})")
        print(out.to_string(index=False))
    return out
