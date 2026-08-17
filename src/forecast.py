"""Quantile demand forecasting - the Forecast stage.

A Temporal Fusion Transformer trained on `recovered_demand`, emitting q10/q50/q90 over a 7-day
horizon rolled across a target period. `tag="raw"` trains the identical model on raw `sale_amount`,
so "was filling in the hidden demand worth it?" is two calls to one code path, not a second model.

Scoring uses the dataset's convention: per date, non-stockout rows only, against recorded
`sale_amount`. On those rows recorded sales ARE demand, so neither version is judged against the
target it was trained on.

    tuning = tune(daily)                              # rank configs on the validation window
    forecasts = run(daily, **best_params(tuning))     # train the winner, forecast, save
"""
import itertools
import json
import logging
import os
import warnings

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from . import recovery
from .utils import config
from .utils.features import CATEGORICAL, FEATURES
from .utils.metrics import quantile_scores

HORIZON = 7
QUANTILES = [0.025, 0.1, 0.5, 0.8, 0.9, 0.975]

# Column name per quantile, stated explicitly rather than derived as f"q{int(q*100)}". That
# expression truncates: 0.025 lands on "q2" and 0.975 on "q97" (float, so 97.4999...), which name the
# 2nd and 97th percentiles rather than the 2.5th and 97.5th - and neither matches the `q025`/`q975`
# the ordering stage reads. Two-decimal quantiles have no safe integer name, so they get spelled out.
#
# 0.8 is here because the ordering stage reads tau* = c_u/(c_u+c_o), and the headline cost ratio of
# 4:1 puts that at exactly 0.8. Without it the order is interpolated between q50 and q90.
QCOL = {0.025: "q025", 0.1: "q10", 0.5: "q50", 0.8: "q80", 0.9: "q90", 0.975: "q975"}
assert set(QUANTILES) <= set(QCOL), f"no column name for {set(QUANTILES) - set(QCOL)}"
TARGETS = {"recovered": "recovered_demand", "raw": "sale_amount"}   # the two targets compared

# The same feature lists the rest of the pipeline uses, split the way the TFT needs them: IDs become
# embeddings, and the covariates known before the day starts go in the decoder. Lag/rolling features
# are excluded - the encoder reads history itself.
KNOWN_REALS = [c for c in FEATURES if c not in CATEGORICAL and not c.startswith(("lag", "roll"))]

# carried into the saved forecast: scoring needs the first two, the ordering stage simulates
# against `recovered_demand`
CARRY = ["sale_amount", "stock_hour6_22_cnt", "recovered_demand", "is_censored"]


def build_master_frame() -> pd.DataFrame:
    """The daily frame every saved forecast gets joined back to for scoring: raw `sale_amount`,
    the stockout flag and `recovered_demand`, across every split (train/eval, i.e. including the
    test week) - not just the periods a given forecast run happened to save.

    One read, no merge. The recovered parquets are a strict superset of the raw daily ones - same
    rows, same columns, plus `recovered_demand` - because `recovery._save_recovered` writes the
    whole frame rather than just the new column. This used to load the raw daily file and merge the
    recovered column onto it, which was both a redundant read and a REPRODUCIBILITY BUG: the raw
    daily parquets are rebuildable and therefore gitignored, while the recovered ones are committed,
    so from a fresh clone the merge failed on a file that was never shipped.
    """
    return recovery.load_daily("recovered")

# 24 configs. Across three searches only `weight_decay` has moved the score materially, and it moves
# monotonically (1e-4 > 1e-3 > 1e-2) - so the open question is whether something below 1e-4 is better
# still. `encoder_days` moves non-monotonically and the rest barely move, so read the ranking as a
# sweep rather than a precise selection.
# `hidden_size` is fixed at 32 in `train`: 64 lost in most pairings.
GRID = {
    "learning_rate": [0.005, 0.01],
    "dropout": [0.2, 0.3],
    "encoder_days": [3, 5, 7],
    "weight_decay":[1e-3, 1e-4],
}


class _EpochLine(pl.Callback):
    """One line per epoch instead of a live progress bar, and the epoch's losses kept in `history`.

    The bar redraws on every batch, which in a notebook is an output update per batch - hundreds per
    epoch, and the single biggest cost of a small model on a GPU. A printed line also survives being
    scrolled or saved.

    `history` exists because the Trainer runs with `logger=False`, so Lightning persists no metrics and
    the printed lines were the only record - lost as soon as a cell was cleared. Collecting them here
    keeps the learning curve available without turning on a logger and its per-run directories.
    """

    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, module):
        def value(key):
            v = trainer.callback_metrics.get(key)
            return float(v) if v is not None else float("nan")

        row = {"epoch": trainer.current_epoch,
               "train_loss": value("train_loss_epoch"), "val_loss": value("val_loss")}
        self.history.append(row)
        print(f"  epoch {row['epoch']:>2}  train_loss={row['train_loss']:.4f}  "
              f"val_loss={row['val_loss']:.4f}", flush=True)


def _select_accelerator(accelerator: str) -> str:
    """Resolve `"auto"` to the best accelerator Lightning can actually use: CUDA, then Apple's MPS
    (Colab gives CUDA; a Mac laptop gives MPS or nothing), falling back to CPU. An explicit choice
    ("cpu"/"gpu"/"mps") is passed through unchanged.

    MPS coverage in pytorch-forecasting's ops is incomplete, so PYTORCH_ENABLE_MPS_FALLBACK is set
    (without overriding a value the caller already set) - unsupported ops silently run on CPU instead
    of crashing the fit.
    """
    if accelerator != "auto":
        return accelerator
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def _device_name(accel: str) -> str:
    return {"gpu": lambda: torch.cuda.get_device_name(0), "mps": lambda: "Apple GPU (MPS)"} \
        .get(accel, lambda: "CPU")()


def _quiet_lightning():
    """Lightning announces GPU/TPU availability and a cloud-logging tip on every fit; across a grid
    that is hundreds of lines burying the scores. Called from `train` because notebooks reset
    warning filters between cells."""
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    for name in ("lightning.pytorch.utilities.rank_zero",
                 "lightning.pytorch.accelerators.cuda", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """The two columns PyTorch Forecasting needs that the daily frame lacks: an integer day counter
    to index time on, and the IDs typed as strings so they become embeddings, not magnitudes."""
    df = daily.sort_values(["store_id", "product_id", "dt"]).copy()
    df["time_idx"] = (df["dt"] - df["dt"].min()).dt.days
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).astype("category")
    return df


def _day(frame: pd.DataFrame, date: str) -> int:
    """A calendar date as its `time_idx`, so the frozen windows in `config` drive the splits."""
    return int((pd.Timestamp(date) - frame["dt"].min()).days)


def _dataset(frame: pd.DataFrame, target: str, encoder_days: int):
    """Training-period rows only. Normalised per series, so a fast and a slow product are scaled on
    their own history rather than against the store-wide average."""
    return TimeSeriesDataSet(
        frame[frame.time_idx <= _day(frame, config.TRAIN_END)],
        time_idx="time_idx", target=target, group_ids=["store_id", "product_id"],
        max_encoder_length=encoder_days, max_prediction_length=HORIZON,
        static_categoricals=CATEGORICAL,
        time_varying_known_reals=["time_idx"] + KNOWN_REALS,
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(groups=["store_id", "product_id"]),
        add_relative_time_idx=True, add_target_scales=True, allow_missing_timesteps=True,
    )


def train(daily: pd.DataFrame, tag: str = "recovered", learning_rate: float = 0.01,
          encoder_days: int = 14, hidden_size: int = 32, dropout: float = 0.1,
          weight_decay: float = 1e-4,
          max_epochs: int = 15, patience: int = 8, seed: int = config.RANDOM_STATE,
          batch_size: int = None, num_workers: int = None, accelerator: str = "auto",
          save_checkpoint: bool = True):
    """Fit the TFT and return `(model, training_dataset)`.

    Fitted on the training window, early-stopped on `val_loss` - so validation decides when to stop
    and is never trained on. The best epoch is checkpointed and reloaded, so a run that overshoots
    still returns its best weights rather than its last.

    `max_epochs` is a ceiling, not a target - `patience` usually stops first. It stays low because
    `val_loss` covers all validation rows including censored days while the scorecard scores only
    non-stockout ones, so training long optimises a target the scorecard does not measure. The proper
    fix is a non-stockout validation loss; until then the ceiling is the guard.

    `batch_size` and `num_workers` default by device - this model is small enough that a GPU spends
    most of its time waiting for batches. Workers stay at 0 on Windows, where >0 deadlocks in
    notebooks. `seed` is fixed so a repeated fit reproduces, not so it can be varied.
    """
    _quiet_lightning()
    accel = _select_accelerator(accelerator)
    on_accelerator = accel != "cpu"
    batch_size = batch_size or (1024 if on_accelerator else 256)
    if num_workers is None:
        num_workers = 4 if on_accelerator else 0

    print(f"[{tag}] {_device_name(accel)} | batch={batch_size} workers={num_workers} seed={seed}",
          flush=True)

    # workers=True also seeds the dataloader workers; without it, batch order varies between
    # otherwise identical runs
    pl.seed_everything(seed, workers=True, verbose=False)
    frame = _prepare(daily)
    training = _dataset(frame, TARGETS[tag], encoder_days)
    validation = TimeSeriesDataSet.from_dataset(
        training, frame[frame.time_idx <= _day(frame, config.VAL_END)],
        min_prediction_idx=_day(frame, config.VAL_START), stop_randomization=True)

    # Lightning suffixes -v1, -v2, ... rather than overwriting, so a grid fills the model dir with
    # dead checkpoints. Clear this name first; repeat fits use a scratch name so they cannot replace
    # the model the saved forecasts came from.
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = config.tft_checkpoint(tag).stem if save_checkpoint else "tft_scratch"
    for stale in config.MODEL_DIR.glob(f"{stem}*.ckpt"):
        stale.unlink()
    ckpt = ModelCheckpoint(dirpath=str(config.MODEL_DIR), monitor="val_loss", save_top_k=1,
                           filename=stem)

    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=learning_rate, hidden_size=hidden_size, dropout=dropout,
        hidden_continuous_size=max(hidden_size // 2, 4), attention_head_size=4,
        weight_decay=weight_decay,
        loss=QuantileLoss(quantiles=QUANTILES), output_size=len(QUANTILES))

    loader = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    epoch_line = _EpochLine()
    pl.Trainer(max_epochs=max_epochs, accelerator=accel, devices=1,
               gradient_clip_val=0.1, logger=False, enable_model_summary=False,
               enable_progress_bar=False,   # replaced by _EpochLine - see its docstring
               # patience 8 against a ceiling of 15 means nearly every config runs the full budget,
               # so configs are compared at EQUAL training rather than at whatever epoch a noisy dip
               # happened to end them - the previous run stopped configs anywhere from 8 to 15 epochs.
               # `min_delta` stops a swing smaller than the metric's own jitter counting as progress
               # and resetting the counter. Overshooting costs only compute: ModelCheckpoint keeps the
               # best epoch and `train` reloads it.
               callbacks=[EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-3),
                          ckpt, epoch_line]).fit(
        model,
        train_dataloaders=training.to_dataloader(train=True, batch_size=batch_size, **loader),
        val_dataloaders=validation.to_dataloader(train=False, batch_size=batch_size * 2, **loader))

    # Only the kept fit writes its curve. A tuning grid fits dozens of scratch models, and each would
    # overwrite the file with a curve belonging to a config that was not chosen.
    if save_checkpoint and epoch_line.history:
        path = config.learning_curve_csv("tft", tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(epoch_line.history).to_csv(path, index=False)
        print(f"  learning curve -> {path.name}", flush=True)

    return TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path), training


def _forecast_week(model, training, frame: pd.DataFrame, last_day: int) -> pd.DataFrame:
    """Forecast the 7 days ending at `last_day`, for every series.

    Cutting the frame at `last_day` and asking for `predict=True` yields exactly one window per
    series - the final one - so the week is predicted from history that stops before it begins.
    """
    ds = TimeSeriesDataSet.from_dataset(training, frame[frame.time_idx <= last_day],
                                        predict=True, stop_randomization=True)
    batch = 1024 if _select_accelerator("auto") != "cpu" else 256
    out = model.predict(ds.to_dataloader(train=False, batch_size=batch), mode="quantiles",
                        return_index=True)
    preds, index = out.output.cpu().numpy(), out.index.reset_index(drop=True)

    days = []
    for step in range(preds.shape[1]):
        day = index[["store_id", "product_id"]].copy()
        day["time_idx"] = index["time_idx"] + step
        for i, q in enumerate(QUANTILES):
            day[QCOL[q]] = preds[:, step, i].clip(min=0)   # demand cannot be negative
        days.append(day)
    return pd.concat(days, ignore_index=True)


def forecast_period(model, training, daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Roll the 7-day horizon across [start, end]; return q10/q50/q90 per series per day."""
    frame = _prepare(daily)
    lo, hi = _day(frame, start), _day(frame, end)

    fc = pd.concat([_forecast_week(model, training, frame, d)
                    for d in range(lo + HORIZON - 1, hi + 1, HORIZON)], ignore_index=True)
    fc = fc[(fc.time_idx >= lo) & (fc.time_idx <= hi)]

    keys = ["store_id", "product_id", "time_idx"]
    actuals = frame[keys + ["dt"] + [c for c in CARRY if c in frame.columns]]
    return actuals.merge(fc, on=keys, how="inner").sort_values(keys).reset_index(drop=True)


def _config_id(cfg: dict) -> str:
    """A config's identity, saved as a column so resuming compares stored ids rather than
    re-deriving them from numbers that have been through a CSV."""
    return json.dumps({k: cfg[k] for k in sorted(GRID)}, sort_keys=True)


def tune(daily: pd.DataFrame, tag: str = "recovered", max_epochs: int = 15,
         **train_kwargs) -> pd.DataFrame:
    """Score every config in `GRID` on the VALIDATION window; return the table, best first.

    One fit per config, at the fixed `config.RANDOM_STATE` that `train` defaults to, so the same
    grid returns the same ranking.

    Ranked by pinball(avg), not WAPE: the ordering stage consumes the whole q10/q50/q90 range, so
    the metric that scores the range should choose the model.

    Resumes automatically: the CSV is rewritten after every config and configs already in it are
    skipped, so a disconnect costs one config.

    Rows left over from a DIFFERENT grid are discarded on load. `config_id` only covers the keys of
    the grid that wrote it, so editing `GRID` mid-search would otherwise leave old rows in the table,
    ranked against the new ones - and `best_params` reads `iloc[0]`, so a stale winner would be asked
    for a column it predates and hand back NaN hyperparameters.
    """
    path = config.tft_tuning(tag)
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]

    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    valid = {_config_id(cfg) for cfg in combos}
    rows, dropped = [r for r in rows if r["config_id"] in valid], \
                    [r for r in rows if r["config_id"] not in valid]
    if dropped:
        print(f"discarded {len(dropped)} row(s) from a previous grid", flush=True)
    done = {r["config_id"] for r in rows}

    for i, cfg in enumerate(combos, 1):
        if _config_id(cfg) in done:
            continue
        print(f"\n=== config {i}/{len(combos)}: {cfg} ===", flush=True)
        scores = quantile_scores(forecast_period(
            *train(daily, tag=tag, max_epochs=max_epochs, save_checkpoint=False,
                   **cfg, **train_kwargs),
            daily, config.VAL_START, config.VAL_END))

        row = {"config_id": _config_id(cfg), **cfg,
               **{k: round(v, 4) for k, v in scores.items()}}
        print(f"    -> pinball(avg)={row['pinball(avg)']}  WAPE={row['WAPE']}", flush=True)

        rows.append(row)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("pinball(avg)").to_csv(path, index=False)

    return pd.DataFrame(rows).sort_values("pinball(avg)").reset_index(drop=True)


def best_params(tuning: pd.DataFrame) -> dict:
    """The winning config as keyword arguments for `train`, cast back to plain Python types using
    the grid's own values - so a new grid entry needs no change here."""
    top = tuning.iloc[0]
    return {k: type(values[0])(top[k]) for k, values in GRID.items()}


def run(daily: pd.DataFrame, tag: str = "recovered", periods=("validation", "calibration"),
        save: bool = True, **train_kwargs) -> dict:
    """Train on one target and return `{period: forecast frame}`.

    Pass the SAME params for both targets - the comparison must change only the target. `periods`
    defaults to excluding the test week - pass `periods=(..., "test")` explicitly to open it, and
    only at the final evaluation. `save=False` writes neither forecasts nor model.
    """
    windows = {"validation": (config.VAL_START, config.VAL_END),
               "calibration": (config.CAL_START, config.CAL_END),
               "test": (config.TEST_START, config.TEST_END)}
    model, training = train(daily, tag=tag, save_checkpoint=save, **train_kwargs)

    out = {}
    for period in periods:
        out[period] = forecast_period(model, training, daily, *windows[period])
        if save:
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = config.forecast_parquet(period, tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            out[period].to_parquet(path, index=False)

    print(f"[{tag}] validation:", quantile_scores(out["validation"]))
    return out


def load(tag: str, daily: pd.DataFrame, encoder_days: int):
    """Reload a checkpoint `train()` already fit and saved, paired with the exact
    `TimeSeriesDataSet` it was fit against - so a later notebook can forecast a NEW period from
    the saved model without paying for a retrain (`outputs/models/` "must outlive the kernel" -
    README §8).

    Rebuilding `training` here is safe: `_dataset` is a deterministic function of the
    training-window rows and `encoder_days`, neither of which involves the model's learned
    weights, so the object built here is the training dataset `train` used, not an
    approximation of it. `encoder_days` isn't saved on the checkpoint itself, so pass the value
    actually used to fit it - e.g. `best_params(pd.read_csv(config.tft_tuning(tag)))["encoder_days"]`.
    """
    ckpt = config.tft_checkpoint(tag)
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt.name} missing - run forecast.run(..., tag='{tag}') first.")
    training = _dataset(_prepare(daily), TARGETS[tag], encoder_days)
    # `map_location` pins the WEIGHTS to this machine's device. It is not by itself sufficient: a
    # checkpoint also pickles `loss` and `logging_metrics` into `hyper_parameters`, and torchmetrics
    # objects store their own `_device`. Fit on an Apple GPU, those come back tagged `mps:0`, and
    # Lightning's `.to()` walks into `Metric._apply`, which builds `torch.zeros(1, device=self._device)`
    # BEFORE moving anything - so any machine without an MPS backend dies with
    # `NotImplementedError: Could not run 'aten::empty.memory_format' ... 'MPS' backend`.
    # The committed checkpoints have had those attributes normalised to CPU (weights untouched and
    # verified identical); this argument keeps a freshly-fit one portable too.
    model = TemporalFusionTransformer.load_from_checkpoint(
        ckpt, map_location=torch.device(_select_accelerator("auto").replace("gpu", "cuda")))
    return model, training


def forecast_test(daily: pd.DataFrame, tag: str = "recovered", save: bool = True,
                  encoder_days: int = None) -> pd.DataFrame:
    """Forecast the TEST week from the ALREADY-FIT checkpoint for `tag` - no retraining, and
    nothing about the test week informs the model, which was early-stopped on validation alone.

    Ground rule: the test week is opened once, at the final evaluation (README §5, "the test week
    is touched once") - call this only when you mean to score or order against it, not as part of
    routine tuning.

    `encoder_days` defaults to the value in this tag's tuning table. Pass it explicitly for a tag
    whose checkpoint exists but whose tuning table does not - which is the raw arm's situation: it
    was fit alongside the recovered arm (the comparison changes only the target, never the config)
    but only the recovered search was saved. `load` notes that the encoder length is NOT stored on
    the checkpoint, so it cannot be recovered from the file.

    A wrong `encoder_days` does not raise - it silently builds a differently-conditioned dataset and
    returns a plausible-looking forecast. So a borrowed value has to be VERIFIED, by re-forecasting
    the window already on disk and diffing; notebook 04 section 4 does exactly that before it trusts
    the raw arm's test forecast, and it reads nothing sealed to do so.
    """
    if encoder_days is None:
        tuning_path = config.tft_tuning(tag)
        if not tuning_path.exists():
            raise FileNotFoundError(
                f"{tuning_path.name} missing - run forecast.tune(..., tag='{tag}') first, so the "
                f"encoder length the checkpoint was fit with is known, or pass encoder_days=... "
                f"explicitly and verify it reproduces this tag's saved validation forecast.")
        encoder_days = best_params(pd.read_csv(tuning_path))["encoder_days"]
    model, training = load(tag, daily, encoder_days)
    fc = forecast_period(model, training, daily, config.TEST_START, config.TEST_END)
    if save:
        path = config.forecast_parquet("test", tag)
        path.parent.mkdir(parents=True, exist_ok=True)   # forecasts/, not reports/
        fc.to_parquet(path, index=False)
    return fc
