"""XGBoost quantile forecasting - the deterministic alternative to the TFT.

Same interface and saved-parquet format as `src/forecast.py`, so every downstream stage works unchanged.

Within about 4% of the TFT on pinball, reproducing exactly and needing no GPU, which is what makes
the recovery result independent of the transformer: anyone can re-run this arm and get the same
numbers. Fit time tracks the config, since early stopping lets a low learning rate run several times
longer than a high one.

XGBoost over LightGBM because `quantile_alpha` fits the whole quantile list in one model where
LightGBM needs a fit each. Both are boosted trees, so the contrast that earns its place is trees
against the TFT's sequence encoder, not one boosting library against another.
"""
import itertools

import numpy as np
import pandas as pd

from .baselines import BASELINE_FEATS, add_history_features
from .utils.features import add_lagged_features
from .forecast import CARRY, QCOL, QUANTILES, TARGETS
from .utils import config
from .utils.metrics import pinball_by_quantile, quantile_scores

KEY = ["store_id", "product_id", "dt"]

MAX_ROUNDS = 1200       # ceiling only; every config is expected to stop well below it
EARLY_STOP_ROUNDS = 40  # rounds without improvement before giving up
EARLY_STOP_DAYS = 14    # training days held back to stop on - matches the forecast horizon

# 12 configs. Each is early-stopped to its own round count, so the learning-rate axis is fair: a
# pinned `n_estimators` asks a low rate to finish in a high rate's budget, and in boosting that is
# where the last accuracy hides.
#
#   learning_rate    0.001 is excluded on measurement: it needs ~10,000 rounds, hits the MAX_ROUNDS
#                    ceiling, and scores 0.2014 against 0.1113 even when given them.
#   max_depth        3, 4, 6, 8 and 10 have been measured and 6 won, so 5 and 7 are the unmeasured
#                    neighbours of the winner.
GRID = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "max_depth": [5, 6, 7],
}

# Held constant so the search moves only what is open. Each was probed at depth 6 against its own
# best round count, and every alternative scored worse (validation quantile loss, baseline 0.09082):
#
#   subsample 0.5         0.09126      colsample_bytree 0.5   0.09099
#   reg_lambda 10 / 50    0.09109 / 0.09119               reg_alpha 1   0.09118
#
# Each one narrowed the train/validation gap and cost accuracy, so those two move in opposite
# directions here. `min_child_weight` was flat to 0.0003 and stays at the XGBoost default. `tune`
# accepts any grid, so re-opening one of these is a keyword argument, not a second constant.
FIXED = {"subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1}

_INT_PARAMS = {"max_depth", "min_child_weight", "n_estimators"}
_NOT_A_PARAM = {"best_round", "WAPE", "WPE", "MAE", "pinball(avg)", "CRPS~", "pinball@0.8"}


def _train_rows(train_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Training rows with lag/rolling features built, for one target.

    `add_lagged_features` directly, NOT `add_history_features` as the eval side uses: that helper
    concatenates its two arguments, so passing the training frame twice would duplicate every row and
    break the lags. Eval rows do go through it, so the model sees only the history a 7-day-ahead
    forecast would have.

    Depends on the target alone and not on which period is being forecast, which is why `run` fits
    once and loops only over prediction.
    """
    return add_lagged_features(train_df, source=target, lags=(7, 14), mean_windows=(7,),
                               std_windows=(7,), fill=False).dropna(subset=BASELINE_FEATS)


def _model(**params):
    """An XGBoost quantile regressor. `quantile_alpha` takes the whole list, so one fit covers all six
    quantiles where LightGBM would need a fit each. Caller params override `FIXED`."""
    from xgboost import XGBRegressor

    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES,
                        eval_metric="quantile", random_state=config.RANDOM_STATE, verbosity=0,
                        **{**FIXED, **params})


def _to_frame(model, ev: pd.DataFrame) -> pd.DataFrame:
    """`ev` plus one column per quantile, predictions clipped at zero and sorted across each row.

    Sorting matters: a joint fit does not guarantee q10 <= q50 <= q90, and they do cross here. That is
    fatal downstream, since `orders.py` reads an order by interpolating along this curve. Sorting
    cannot damage a row that was already ordered.
    """
    preds = np.clip(model.predict(ev[BASELINE_FEATS].fillna(0)), 0, None)   # demand cannot be negative
    preds = np.sort(preds, axis=1)

    out = ev[KEY + [c for c in CARRY if c in ev.columns]].copy()
    for i, q in enumerate(QUANTILES):
        out[QCOL[q]] = preds[:, i]
    return out.reset_index(drop=True)


def tune(daily: pd.DataFrame, tag: str = "recovered", grid: dict = None,
         save: bool = True) -> pd.DataFrame:
    """Score every config in `grid` on the VALIDATION window; return the table, best first.

    Early stopping gives each config its own round count, reported as `best_round`, which is what keeps
    the learning-rate axis fair: a fixed `n_estimators` asks a low rate to finish in a high one's budget.

    Stopping watches the LAST `EARLY_STOP_DAYS` of TRAINING, never validation. Stopping on validation
    would let a model size itself on the window it is then scored on, which is what the TFT does and
    what would make this arm's number flattering instead of comparable.

    One fit per config, on 14 fewer days than the final model gets, since stopping needs a held-back
    slice. Every config is handicapped identically, so the ranking holds and only the absolute scores
    run slightly pessimistic; `run` trains the winner on all training days.

    Ranked on pinball, not WAPE: the ordering stage reads a quantile off this forecast, and pinball at
    that quantile is the newsvendor cost function up to a constant, where WAPE is direction-blind.
    `pinball@0.8` is reported alongside because that is the quantile the order is read at.

    Budget roughly two fits per config. A low learning rate stops late, so the slowest configs run
    several times longer than the fastest.
    """
    grid = GRID if grid is None else grid
    target = TARGETS[tag]
    train_df = daily[daily["period"] == "training"]
    tr = _train_rows(train_df, target)
    ev = add_history_features(train_df, daily[daily["period"] == "validation"], source=target)

    cutoff = tr["dt"].max() - pd.Timedelta(days=EARLY_STOP_DAYS)
    fit_rows, stop_rows = tr[tr["dt"] <= cutoff], tr[tr["dt"] > cutoff]
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]

    rows = []
    for i, cfg in enumerate(combos, 1):
        print(f"[{tag}] config {i}/{len(combos)}: {cfg}", flush=True)

        # MAX_ROUNDS is a ceiling; `stop_rows` decides the real round count.
        model = _model(n_estimators=MAX_ROUNDS, early_stopping_rounds=EARLY_STOP_ROUNDS, **cfg)
        model.fit(fit_rows[BASELINE_FEATS], fit_rows[target], verbose=False,
                  eval_set=[(stop_rows[BASELINE_FEATS], stop_rows[target])])
        best_round = int(model.best_iteration) + 1   # best_iteration is 0-indexed, best_round isn't
        if best_round >= MAX_ROUNDS:
            print(f"    !! never stopped - hit the {MAX_ROUNDS}-round ceiling, so this score is a "
                  f"floor rather than this config's best, and is not comparable to the rows that "
                  f"converged. Raise MAX_ROUNDS or drop the learning rate from the grid.", flush=True)

        fc = _to_frame(model, ev)
        rows.append({**cfg, "best_round": best_round, **quantile_scores(fc),
                     "pinball@0.8": pinball_by_quantile(fc)["pinball@0.8"]})
        print(f"    -> best_round={best_round}  pinball(avg)={rows[-1]['pinball(avg)']}  "
              f"pinball@0.8={rows[-1]['pinball@0.8']}  WAPE={rows[-1]['WAPE']}", flush=True)

        # Written after every config: a grid this slow gets interrupted, and a table that only
        # appears on clean completion turns an interruption into a total loss.
        board = pd.DataFrame(rows).sort_values("pinball(avg)").reset_index(drop=True)
        if save:
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            board.to_csv(config.gbm_tuning(tag), index=False)

    return board


def learning_curve(daily: pd.DataFrame, tag: str = "recovered", n_estimators: int = 600,
                   label: str = None, **params) -> pd.DataFrame:
    """Per-round quantile loss on the TRAINING rows and the VALIDATION rows, from a single fit.

    A grid ranks configs on validation alone, so it locates the optimum without showing the gap to
    training. This shows the gap, which is the overfitting question.

    `n_estimators` defaults past the tuned round count so the curve covers the region beyond the chosen
    config: if the validation line turns up, that turning point is where this model starts memorising.

    Two things the y-axis is not. It is loss against `recovered_demand` on both sides, the training
    signal, where the scorecard scores recorded `sale_amount` on non-stockout rows only, so the level
    is not comparable to pinball 0.1113. And it is XGBoost's own averaging over `QUANTILES`, not
    `metrics.quantile_scores`.
    """
    target = TARGETS[tag]
    train_df = daily[daily["period"] == "training"]
    tr = _train_rows(train_df, target)
    ev = add_history_features(train_df, daily[daily["period"] == "validation"],
                              source=target).dropna(subset=[target])

    # Through `_model`, so `FIXED` applies here as it does in `tune` and `run`. A regressor built
    # separately would default `subsample` to XGBoost's 1.0 against the grid's 0.8, giving a curve for
    # a config nothing else ever fits.
    model = _model(n_estimators=n_estimators, **params)
    # XGBoost names eval sets "validation_0", "validation_1" by position, so this order is what makes
    # the two columns read below mean train and validation respectively.
    model.fit(tr[BASELINE_FEATS], tr[target], verbose=False,
              eval_set=[(tr[BASELINE_FEATS], tr[target]),
                        (ev[BASELINE_FEATS].fillna(0), ev[target])])

    hist = model.evals_result()
    curve = pd.DataFrame({"round": np.arange(1, n_estimators + 1),
                          "train": hist["validation_0"]["quantile"],       # eval_set[0] = train rows
                          "validation": hist["validation_1"]["quantile"]}) # eval_set[1] = validation rows
    curve["gap"] = curve["validation"] - curve["train"]

    # `label`, not `tag`, names the file: the curve belongs to a config, so two configs of the same
    # target would otherwise overwrite each other.
    path = config.learning_curve_csv("xgb", label or tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(path, index=False)
    best = curve.loc[curve["validation"].idxmin()]
    print(f"[xgb {tag}] validation loss bottoms at round {int(best['round'])} "
          f"({best['validation']:.5f}), gap to train {best['gap']:.5f} -> {path.name}", flush=True)
    return curve


def best_params(tuning: pd.DataFrame) -> dict:
    """The winning row as keyword arguments for `run`, including the round count it stopped at.

    Hyperparameters come off the table, not off `GRID`, so a ranking from any one-off grid still
    resolves; keying off the module-level grid would reject a perfectly valid table.

    `best_round` becomes `n_estimators`, because with early stopping the round count is part of the
    answer: a caller that dropped it would train a differently-sized model than the one scored.
    """
    if "best_round" not in tuning.columns:
        raise KeyError("tuning table has no `best_round`, so the round count behind its scores is "
                       "unrecoverable. Re-run `tune`.")
    top = tuning.iloc[0]
    # back to int or float per param: a saved CSV gives generic column types, and XGBoost needs
    # max_depth as an int, not a numpy float
    cfg = {c: (int(top[c]) if c in _INT_PARAMS else float(top[c]))
           for c in tuning.columns if c not in _NOT_A_PARAM}
    return {**FIXED, **cfg, "n_estimators": int(top["best_round"])}


def run(daily: pd.DataFrame, tag: str = "recovered",
        periods=("validation", "calibration"), save: bool = True, curve: bool = True,
        **fit_kwargs) -> dict:
    """Fit on one target and return `{period: forecast frame}`, saving each.

    Pass the same params for both targets: the raw-vs-recovered comparison must change only the
    target. Files land under `forecasts/xgb/` so the TFT's cannot be clobbered.

    One fit covers every period: the training rows depend on the target alone, so refitting per period
    would train the identical model twice.

    `periods` excludes the test week by default; pass `periods=(..., "test")` to open it, and only at
    the final evaluation.

    `curve` records the per-round learning curve from this fit, the same measurement `learning_curve`
    makes but as a by-product rather than a second model: the extra cost is one prediction per round
    over rows already in hand. It ends at this config's own `n_estimators`, which is far enough to see
    the turn, since validation loss bottoms at round 290 and the tuned winner runs 430. Looking past
    the chosen round count still needs `learning_curve`.
    """
    train_df = daily[daily["period"] == "training"]
    tr = _train_rows(train_df, TARGETS[tag])
    model = _model(**fit_kwargs)   # no early stopping: fit_kwargs carries the tuned round count

    # Curve only. Both sides score the training signal on all rows, where the scorecard scores
    # recorded `sale_amount` on non-stockout rows, so this level is not comparable to `pinball(avg)`.
    eval_set = None
    if curve:
        # same (train, validation) order as `learning_curve`, for the same positional-naming reason
        ev_curve = add_history_features(train_df, daily[daily["period"] == "validation"],
                                        source=TARGETS[tag]).dropna(subset=[TARGETS[tag]])
        eval_set = [(tr[BASELINE_FEATS], tr[TARGETS[tag]]),
                    (ev_curve[BASELINE_FEATS].fillna(0), ev_curve[TARGETS[tag]])]
    model.fit(tr[BASELINE_FEATS], tr[TARGETS[tag]], eval_set=eval_set, verbose=False)

    if curve:
        hist = model.evals_result()
        rounds = len(hist["validation_0"]["quantile"])
        cur = pd.DataFrame({"round": np.arange(1, rounds + 1),
                            "train": hist["validation_0"]["quantile"],       # eval_set[0] = train rows
                            "validation": hist["validation_1"]["quantile"]}) # eval_set[1] = validation rows
        cur["gap"] = cur["validation"] - cur["train"]
        if save:
            path = config.learning_curve_csv("xgb", tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            cur.to_csv(path, index=False)
        turn = int(cur.loc[cur["validation"].idxmin(), "round"])
        print(f"[xgb {tag}] curve: {rounds} rounds, validation bottoms at {turn} "
              f"({cur['validation'].min():.5f}), gap at the end {cur['gap'].iloc[-1]:.5f}", flush=True)

    out = {}
    for period in periods:
        ev = add_history_features(train_df, daily[daily["period"] == period], source=TARGETS[tag])
        out[period] = _to_frame(model, ev)
        if save:
            path = config.forecast_parquet(period, tag, family="xgb")
            path.parent.mkdir(parents=True, exist_ok=True)
            out[period].to_parquet(path, index=False)

    print(f"[xgb {tag}] {periods[0]}:", quantile_scores(out[periods[0]]))
    return out
