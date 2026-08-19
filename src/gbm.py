"""XGBoost quantile forecasting - the deterministic alternative to the TFT.

Same interface and saved-parquet format as `src/forecast.py`, so every downstream stage works unchanged.

Scores level with the TFT while repeating EXACTLY and needing no GPU, which is what makes the
recovery result independent of the transformer: this arm can be re-run by anyone and returns the same
numbers. Fit time depends on the config - early stopping lets a low learning rate run several times
longer than a high one, so a full grid is not the minute a single fit used to take.

XGBoost rather than LightGBM because `quantile_alpha` takes the whole quantile list and fits them in one
model; LightGBM needs one fit each. Both are boosted trees, so LightGBM would be a duplicate rather than
a second opinion - the architecture contrast is trees against the TFT's sequence encoder.
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

# 9 configs over the ONE interaction earlier searches could not see.
#
# Every previous grid pinned `n_estimators`, which quietly rigged the learning-rate axis: a lower rate
# needs more trees to reach the same place, so judging 0.02 and 0.05 at the same 200 rounds asks the slow
# learner to finish in the fast learner's budget. `tune` now early-stops, giving each config its own round
# count, which is the first time low learning rates get a fair run - and in boosting that is where the
# last accuracy usually hides.
#
#   learning_rate    0.05 won at fixed 200 rounds; 0.02 and 0.03 are the ones that were handicapped.
#                    0.001 was tried and REMOVED, not overlooked: it needs ~10,000 rounds, hit the
#                    MAX_ROUNDS ceiling, and scores 0.2014 against 0.1113 when finally given them.
#                    Raising the ceiling to accommodate it buys a config that is already known to lose
#   max_depth        5 and 7 have never been fitted. 3, 4, 6, 8 and 10 have, and 6 won - so the optimum
#                    is bracketed but its immediate neighbours are still unmeasured
GRID = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "max_depth": [5, 6, 7],
}

# Held constant, so the search moves only what is still in question. These are not defaults inherited
# without checking - each was probed individually at depth 6 against its own best round count, and every
# alternative scored worse (validation quantile loss, baseline 0.09082):
#
#   subsample 0.5         0.09126      colsample_bytree 0.5   0.09099
#   reg_lambda 10 / 50    0.09109 / 0.09119               reg_alpha 1   0.09118
#
# Every one narrowed the train/validation gap and cost accuracy - the two move in opposite directions, so
# the settings kept here are the measured optimum rather than an untested starting point.
# `min_child_weight` was flat to 0.0003 and is left at the XGBoost default rather than pinned to a number
# that never mattered. `tune` takes any grid, so re-opening one of these is a keyword argument, not a
# second constant to keep in sync with this one.
FIXED = {"subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1}

_INT_PARAMS = {"max_depth", "min_child_weight", "n_estimators"}
_NOT_A_PARAM = {"best_round", "WAPE", "WPE", "MAE", "pinball(avg)", "CRPS~", "pinball@0.8"}


def _train_rows(train_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Training rows with lag/rolling features built, for one target.

    `add_lagged_features` directly, NOT `add_history_features` as the eval side uses: that helper
    concatenates its two arguments, so passing the training frame twice duplicates every row and
    corrupts the lags. Eval rows do go through it, so the model sees only the history a 7-day-ahead
    forecast would have.

    Depends on the target and nothing else - not on which period is being forecast - which is why
    `run` fits once and loops only over prediction.
    """
    return add_lagged_features(train_df, source=target, lags=(7, 14), mean_windows=(7,),
                               std_windows=(7,), fill=False).dropna(subset=BASELINE_FEATS)


def _model(**params):
    """An XGBoost quantile regressor. One fit covers all six quantiles - `quantile_alpha` takes the whole
    list, where LightGBM would need a fit each. Caller params override `FIXED`."""
    from xgboost import XGBRegressor

    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES,
                        eval_metric="quantile", random_state=config.RANDOM_STATE, verbosity=0,
                        **{**FIXED, **params})


def _to_frame(model, ev: pd.DataFrame) -> pd.DataFrame:
    """`ev` plus one column per quantile, predictions clipped at zero and sorted across each row.

    Sorting matters: fitting the quantiles jointly does not guarantee q10 <= q50 <= q90 and they DO cross
    here - fatal downstream, since `orders.py` reads an order by interpolating along this curve and a curve
    that reverses returns nonsense. Sorting cannot damage an already-ordered row.
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

    Each config gets its OWN round count from early stopping, reported as `best_round`. That is what makes
    the learning-rate axis honest: a fixed `n_estimators` asks a low learning rate to finish in a high
    one's budget, so the fast rate wins on the budget rather than on merit.

    Stopping watches the LAST `EARLY_STOP_DAYS` of TRAINING, never validation. Stopping on validation would
    let the model size itself using the window it is then scored on - which is what the TFT does, and
    copying it here would make this model's number flattering instead of comparable.

    ONE fit per config. Stopping needs a held-back slice, so every model here trains on 14 fewer days than
    the final one will - but every model here is handicapped identically, so the RANKING is unaffected and
    only the absolute scores are slightly pessimistic. `run` trains the winner on all training days, which
    is where a full-data fit belongs. An earlier version refit each config on full data before scoring,
    which doubled the cost of the grid to make numbers nobody reads slightly prettier.

    Ranked on pinball, not WAPE: the ordering stage reads a quantile off this forecast, and pinball at that
    quantile IS the newsvendor cost function up to a constant. WAPE is direction-blind and cannot rank a
    policy whose whole job is choosing a direction to lean. `pinball@0.8` is reported alongside because
    that is the quantile the order is actually read at, and `pinball(avg)` never covered it.

    Budget roughly two fits per config, and note that a low learning rate stops late: the slowest configs
    here run several times longer than the fastest.
    """
    grid = GRID if grid is None else grid
    target = TARGETS[tag]   # "recovered_demand" or "sale_amount", whichever this tag points to
    train_df = daily[daily["period"] == "training"]
    tr = _train_rows(train_df, target)
    ev = add_history_features(train_df, daily[daily["period"] == "validation"], source=target)

    # the last EARLY_STOP_DAYS of training become the held-back slice early stopping watches;
    # everything before the cutoff is what each config actually fits on
    cutoff = tr["dt"].max() - pd.Timedelta(days=EARLY_STOP_DAYS)
    fit_rows, stop_rows = tr[tr["dt"] <= cutoff], tr[tr["dt"] > cutoff]
    # every combination of grid values, e.g. {learning_rate: 0.02, max_depth: 5}, {0.02, 6}, ...
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]

    rows = []
    for i, cfg in enumerate(combos, 1):
        print(f"[{tag}] config {i}/{len(combos)}: {cfg}", flush=True)

        # MAX_ROUNDS is just a ceiling here; early stopping decides the real round count. Fit on
        # fit_rows, watch loss on stop_rows, and stop once EARLY_STOP_ROUNDS pass with no improvement.
        model = _model(n_estimators=MAX_ROUNDS, early_stopping_rounds=EARLY_STOP_ROUNDS, **cfg)
        model.fit(fit_rows[BASELINE_FEATS], fit_rows[target], verbose=False,
                  eval_set=[(stop_rows[BASELINE_FEATS], stop_rows[target])])
        best_round = int(model.best_iteration) + 1   # best_iteration is 0-indexed, best_round isn't
        if best_round >= MAX_ROUNDS:
            print(f"    !! never stopped - hit the {MAX_ROUNDS}-round ceiling, so this score is a "
                  f"floor rather than this config's best, and is NOT comparable to the rows that "
                  f"converged. Raise MAX_ROUNDS or drop the learning rate from the grid.", flush=True)

        fc = _to_frame(model, ev)
        rows.append({**cfg, "best_round": best_round, **quantile_scores(fc),
                     "pinball@0.8": pinball_by_quantile(fc)["pinball@0.8"]})
        print(f"    -> best_round={best_round}  pinball(avg)={rows[-1]['pinball(avg)']}  "
              f"pinball@0.8={rows[-1]['pinball@0.8']}  WAPE={rows[-1]['WAPE']}", flush=True)

        # Written after EVERY config, not once at the end: a grid this slow gets interrupted, and a
        # table that only appears on clean completion turns an interruption into a total loss.
        board = pd.DataFrame(rows).sort_values("pinball(avg)").reset_index(drop=True)
        if save:
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            board.to_csv(config.gbm_tuning(tag), index=False)

    return board


def learning_curve(daily: pd.DataFrame, tag: str = "recovered", n_estimators: int = 600,
                   label: str = None, **params) -> pd.DataFrame:
    """Per-round quantile loss on the TRAINING rows and the VALIDATION rows, from a single fit.

    The measurement `tune` structurally cannot make. A grid ranks configs on validation only, which
    locates the optimum on the validation surface but never shows the gap to training - so it can say
    which config is best without saying whether that config memorises. This says whether it memorises.

    `n_estimators` defaults well past the tuned 200 so the curve covers the region beyond the chosen
    config: if the validation line turns up, the turning point IS where this model starts overfitting,
    and whether 200 sits before or after it is then a fact rather than an inference.

    Two things the y-axis is NOT. It is loss against `recovered_demand` on both sides (the training
    signal, so train and validation are the same quantity), where the scorecard scores against recorded
    `sale_amount` on non-stockout rows only - so the level here is not comparable to pinball 0.1113.
    And it is XGBoost's own averaging over `QUANTILES`, not `metrics.quantile_scores`.
    """
    target = TARGETS[tag]
    train_df = daily[daily["period"] == "training"]
    tr = _train_rows(train_df, target)
    # dropna: a row with no target value can't be scored, so it would just add noise to the eval loss
    ev = add_history_features(train_df, daily[daily["period"] == "validation"],
                              source=target).dropna(subset=[target])

    # `_model`, so `FIXED` applies here exactly as it does in `tune` and `run`. Building the regressor
    # separately let this function default `subsample` to XGBoost's 1.0 while the grid used 0.8 - a curve
    # for a config no other function would ever fit, which is the kind of mismatch nothing would surface.
    model = _model(n_estimators=n_estimators, **params)
    # eval_set order matters: XGBoost names them "validation_0", "validation_1", ... by POSITION, not by
    # what they actually are - so passing (train, val) in that order is what makes "validation_0" below
    # mean the training rows and "validation_1" mean the actual validation rows.
    model.fit(tr[BASELINE_FEATS], tr[target], verbose=False,
              eval_set=[(tr[BASELINE_FEATS], tr[target]),
                        (ev[BASELINE_FEATS].fillna(0), ev[target])])

    hist = model.evals_result()
    curve = pd.DataFrame({"round": np.arange(1, n_estimators + 1),
                          "train": hist["validation_0"]["quantile"],       # eval_set[0] = train rows
                          "validation": hist["validation_1"]["quantile"]}) # eval_set[1] = validation rows
    curve["gap"] = curve["validation"] - curve["train"]

    # `label`, not `tag`, names the file: the curve belongs to a CONFIG, not just a target, and two
    # configs of the same target would otherwise silently overwrite each other - which is exactly how
    # the regularisation probe destroyed the baseline curve it was being compared against.
    path = config.learning_curve_csv("xgb", label or tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(path, index=False)
    best = curve.loc[curve["validation"].idxmin()]   # the round where validation loss was lowest
    print(f"[xgb {tag}] validation loss bottoms at round {int(best['round'])} "
          f"({best['validation']:.5f}), gap to train {best['gap']:.5f} -> {path.name}", flush=True)
    return curve


def best_params(tuning: pd.DataFrame) -> dict:
    """The winning row as keyword arguments for `run`, including the round count it stopped at.

    Reads the hyperparameters off the TABLE rather than off `GRID`, so a ranking produced by a different
    grid - a regularisation sweep, or any one-off - still resolves. Keying off the module-level grid would
    raise on a table that is perfectly valid.

    `best_round` becomes `n_estimators`: the whole point of early stopping is that the round count is part
    of the answer, so a caller that dropped it would train a differently-sized model than the one scored.
    """
    if "best_round" not in tuning.columns:
        raise KeyError("tuning table has no `best_round` - it predates early stopping, so the round "
                       "count that produced its scores is unrecoverable. Re-run `tune`.")
    top = tuning.iloc[0]   # the table is already sorted by pinball(avg), so the first row is the winner
    # cast back to int or float depending on the param: everything came out of a saved CSV as a
    # generic column type, and XGBoost is picky about max_depth being an int, not a numpy float
    cfg = {c: (int(top[c]) if c in _INT_PARAMS else float(top[c]))
           for c in tuning.columns if c not in _NOT_A_PARAM}
    return {**FIXED, **cfg, "n_estimators": int(top["best_round"])}


def run(daily: pd.DataFrame, tag: str = "recovered",
        periods=("validation", "calibration"), save: bool = True, curve: bool = True,
        **fit_kwargs) -> dict:
    """Fit on one target and return `{period: forecast frame}`, saving each.

    Pass the SAME params for both targets - the raw-vs-recovered comparison must change only the
    target. Files land under `forecasts/xgb/` so the TFT's cannot be clobbered.

    ONE fit for every period. The training rows are a function of the target alone, so refitting per
    period would train the identical model twice and forecast different weeks with it.

    `periods` excludes the test week by default; pass `periods=(..., "test")` to open it, and only at
    the final evaluation.

    `curve` records the per-round learning curve from THIS fit and writes it beside the reports. It
    is the same measurement `learning_curve` makes, taken as a by-product instead of as a second fit:
    the only extra cost is one prediction per round over rows already in hand, where a standalone
    call pays for a whole additional model. The curve ends at this config's own `n_estimators`, which
    is enough - validation loss bottoms at round 290 and the tuned winner runs 430, so the turning
    point is already inside the fitted range. `learning_curve` remains the way to look PAST the
    chosen round count, which is the one thing this cannot show.
    """
    train_df = daily[daily["period"] == "training"]
    tr = _train_rows(train_df, TARGETS[tag])
    # no early stopping here - fit_kwargs already carries the winning config, including the
    # n_estimators best_params picked, so this trains for exactly that many rounds and no more
    model = _model(**fit_kwargs)

    # Both sides scored against the TRAINING signal (`TARGETS[tag]`) on all rows, which is NOT what
    # the scorecard measures - it scores recorded `sale_amount` on non-stockout rows only. The level
    # here is therefore not comparable to `pinball(avg)`; read the shape.
    eval_set = None
    if curve:
        # same (train, validation) eval_set order as learning_curve - see the comment there on why
        # the order matters for how "validation_0"/"validation_1" get read below
        ev_curve = add_history_features(train_df, daily[daily["period"] == "validation"],
                                        source=TARGETS[tag]).dropna(subset=[TARGETS[tag]])
        eval_set = [(tr[BASELINE_FEATS], tr[TARGETS[tag]]),
                    (ev_curve[BASELINE_FEATS].fillna(0), ev_curve[TARGETS[tag]])]
    # eval_set=None when curve=False, so this is just a normal fit with nothing extra recorded
    model.fit(tr[BASELINE_FEATS], tr[TARGETS[tag]], eval_set=eval_set, verbose=False)

    if curve:
        hist = model.evals_result()   # per-round loss XGBoost tracked during the fit above, for free
        rounds = len(hist["validation_0"]["quantile"])
        cur = pd.DataFrame({"round": np.arange(1, rounds + 1),
                            "train": hist["validation_0"]["quantile"],       # eval_set[0] = train rows
                            "validation": hist["validation_1"]["quantile"]}) # eval_set[1] = validation rows
        cur["gap"] = cur["validation"] - cur["train"]
        if save:
            path = config.learning_curve_csv("xgb", tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            cur.to_csv(path, index=False)
        turn = int(cur.loc[cur["validation"].idxmin(), "round"])   # round validation loss was lowest at
        print(f"[xgb {tag}] curve: {rounds} rounds, validation bottoms at {turn} "
              f"({cur['validation'].min():.5f}), gap at the end {cur['gap'].iloc[-1]:.5f}", flush=True)

    # one forecast per requested period, all from the SAME fitted model above - only the history
    # window changes, so this is prediction, not retraining
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
