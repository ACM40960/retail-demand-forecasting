"""Quantile demand forecasting - the Forecast stage.

A Temporal Fusion Transformer trained on `recovered_demand`, emitting q10/q50/q90 over a 7-day
horizon rolled across a target period. `tag="raw"` trains the identical model on raw `sale_amount`,
so "was filling in the hidden demand worth it?" is two calls to one code path, not a second model.

Scoring uses the dataset's convention: per date, non-stockout rows only, against recorded
`sale_amount`. On those rows recorded sales ARE demand, so neither version is judged against the
target it was trained on.

    tuning = tune(daily)                              # score every config on validation
    forecasts = run(daily, **best_params(tuning))     # train the winner, forecast, save
"""
import itertools
import json
import logging
import warnings

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from .utils import config
from .utils.features import CATEGORICAL, FEATURES
from .utils.metrics import quantile_scores

HORIZON = 7
QUANTILES = [0.1, 0.5, 0.9]
TARGETS = {"recovered": "recovered_demand", "raw": "sale_amount"}   # the two targets compared

# The same feature lists the rest of the pipeline uses, split the way the TFT needs them: IDs become
# embeddings, and the covariates known before the day starts go in the decoder. Lag/rolling features
# are excluded - the encoder reads history itself.
KNOWN_REALS = [c for c in FEATURES if c not in CATEGORICAL and not c.startswith(("lag", "roll"))]

# carried into the saved forecast: scoring needs the first two, the ordering stage simulates
# against `recovered_demand`
CARRY = ["sale_amount", "stock_hour6_22_cnt", "recovered_demand", "is_censored"]

# 48 configs, all of them scored. `target_transform` matters most: the target is zero-inflated, so
# normalising raw values lets the few big days dominate - the problem Tweedie solved for the
# recovery model, fixed here by reshaping the target instead of the loss.
GRID = {
    "learning_rate": [0.003, 0.01, 0.03],
    "hidden_size": [16, 32],
    "encoder_days": [14, 28],
    "target_transform": ["none", "log1p"],
    "dropout": [0.1, 0.2],
}


class _EpochLine(pl.Callback):
    """One line per epoch instead of a live progress bar. The bar redraws on every batch, which in
    a notebook is an output update per batch - hundreds per epoch, and the single biggest cost of a
    small model on a GPU. A printed line also survives being scrolled or saved."""

    def on_train_epoch_end(self, trainer, module):
        def metric(key):
            v = trainer.callback_metrics.get(key)
            return f"{float(v):.4f}" if v is not None else "n/a"
        print(f"  epoch {trainer.current_epoch:>2}  train_loss={metric('train_loss_epoch')}  "
              f"val_loss={metric('val_loss')}", flush=True)


def _quiet_lightning():
    """Lightning announces GPU/TPU availability and a cloud-logging tip on every single fit; across
    48 configs that is hundreds of lines burying the scores. Called from `train` because notebooks
    reset warning filters between cells."""
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


def _dataset(frame: pd.DataFrame, target: str, encoder_days: int, target_transform: str):
    """Training-period rows only. The transform is applied per series, so each product is
    normalised on its own scale rather than against the store-wide average."""
    return TimeSeriesDataSet(
        frame[frame.time_idx <= _day(frame, config.TRAIN_END)],
        time_idx="time_idx", target=target, group_ids=["store_id", "product_id"],
        max_encoder_length=encoder_days, max_prediction_length=HORIZON,
        static_categoricals=CATEGORICAL,
        time_varying_known_reals=["time_idx"] + KNOWN_REALS,
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(
            groups=["store_id", "product_id"],
            transformation=None if target_transform == "none" else target_transform),
        add_relative_time_idx=True, add_target_scales=True, allow_missing_timesteps=True,
    )


def train(daily: pd.DataFrame, tag: str = "recovered", learning_rate: float = 0.01,
          hidden_size: int = 32, encoder_days: int = 28, target_transform: str = "log1p",
          dropout: float = 0.1, max_epochs: int = 30, patience: int = 5,
          batch_size: int = None, num_workers: int = None, accelerator: str = "auto"):
    """Fit the TFT and return `(model, training_dataset)`.

    Fitted on the training window, early-stopped on validation - so validation decides when to stop
    and is never trained on. The best epoch is checkpointed and reloaded, so a run that ends does
    not take the model with it.

    `batch_size` and `num_workers` default by device: this model is small enough that a GPU spends
    most of its time waiting for batches, so on GPU it takes big batches and background workers.
    Workers stay at 0 on Windows, where >0 deadlocks inside notebooks.
    """
    _quiet_lightning()
    on_gpu = torch.cuda.is_available() if accelerator == "auto" else accelerator == "gpu"
    batch_size = batch_size or (1024 if on_gpu else 256)
    if num_workers is None:
        num_workers = 4 if on_gpu else 0

    device = torch.cuda.get_device_name(0) if on_gpu else "CPU"
    print(f"[{tag}] {device} | batch_size={batch_size} num_workers={num_workers}", flush=True)

    pl.seed_everything(config.RANDOM_STATE, verbose=False)
    frame = _prepare(daily)
    training = _dataset(frame, TARGETS[tag], encoder_days, target_transform)
    validation = TimeSeriesDataSet.from_dataset(
        training, frame[frame.time_idx <= _day(frame, config.VAL_END)],
        min_prediction_idx=_day(frame, config.VAL_START), stop_randomization=True)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=str(config.MODEL_DIR), monitor="val_loss", save_top_k=1,
                           filename=config.tft_checkpoint(tag).stem)
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=learning_rate, hidden_size=hidden_size, dropout=dropout,
        hidden_continuous_size=max(hidden_size // 2, 4),
        attention_head_size=4, weight_decay=1e-4,   # fixed: neither moved anything worth tuning
        loss=QuantileLoss(quantiles=QUANTILES), output_size=len(QUANTILES))

    loader = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    pl.Trainer(max_epochs=max_epochs, accelerator="gpu" if on_gpu else "cpu", devices=1,
               gradient_clip_val=0.1, logger=False, enable_model_summary=False,
               enable_progress_bar=False,   # replaced by _EpochLine - see its docstring
               callbacks=[EarlyStopping(monitor="val_loss", patience=patience), ckpt,
                          _EpochLine()]).fit(
        model,
        train_dataloaders=training.to_dataloader(train=True, batch_size=batch_size, **loader),
        val_dataloaders=validation.to_dataloader(train=False, batch_size=batch_size * 2, **loader))

    return TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path), training


def _forecast_week(model, training, frame: pd.DataFrame, last_day: int) -> pd.DataFrame:
    """Forecast the 7 days ending at `last_day`, for every series.

    Cutting the frame at `last_day` and asking for `predict=True` yields exactly one window per
    series - the final one - so the week is predicted from history that stops before it begins.
    """
    ds = TimeSeriesDataSet.from_dataset(training, frame[frame.time_idx <= last_day],
                                        predict=True, stop_randomization=True)
    batch = 1024 if torch.cuda.is_available() else 256
    out = model.predict(ds.to_dataloader(train=False, batch_size=batch), mode="quantiles",
                        return_index=True)
    preds, index = out.output.cpu().numpy(), out.index.reset_index(drop=True)

    days = []
    for step in range(preds.shape[1]):
        day = index[["store_id", "product_id"]].copy()
        day["time_idx"] = index["time_idx"] + step
        for i, q in enumerate(QUANTILES):
            day[f"q{int(q * 100)}"] = preds[:, step, i].clip(min=0)   # demand cannot be negative
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


def tune(daily: pd.DataFrame, tag: str = "recovered", max_epochs: int = 10,
         **train_kwargs) -> pd.DataFrame:
    """Score every config in `GRID` on the VALIDATION window; return the table, best first.

    Ranked by **pinball(avg)**, not WAPE: the ordering stage consumes the whole q10/q50/q90 range,
    so the metric that scores the range is the one that should choose the model.

    Only ranks - each config stops at `max_epochs`, enough to separate configs but not to finish a
    fit. Retrain the winner with `run(daily, **best_params(tuning))`.

    Resumes automatically: the CSV is rewritten after every config and configs already in it are
    skipped, so a disconnect costs one config. Just call `tune` again.
    """
    path = config.tft_tuning(tag)
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["config_id"] for r in rows}
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]

    for i, cfg in enumerate(combos, 1):
        if _config_id(cfg) in done:
            continue
        print(f"\n=== config {i}/{len(combos)}: {cfg} ===", flush=True)
        model, training = train(daily, tag=tag, max_epochs=max_epochs, **cfg, **train_kwargs)
        scores = quantile_scores(
            forecast_period(model, training, daily, config.VAL_START, config.VAL_END))
        print(f"    -> pinball(avg)={scores['pinball(avg)']}  WAPE={scores['WAPE']}", flush=True)

        rows.append({"config_id": _config_id(cfg), **cfg, **scores})
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("pinball(avg)").to_csv(path, index=False)

    return pd.DataFrame(rows).sort_values("pinball(avg)").reset_index(drop=True)


def best_params(tuning: pd.DataFrame) -> dict:
    """The winning config as keyword arguments for `train`, cast back to plain Python types using
    the grid's own values - so a new grid entry needs no change here."""
    top = tuning.iloc[0]
    return {k: type(values[0])(top[k]) for k, values in GRID.items()}


def run(daily: pd.DataFrame, tag: str = "recovered", periods=("validation", "calibration"),
        **train_kwargs) -> dict:
    """Train on one target at full epochs and return `{period: forecast frame}`, saving each.

    Pass the SAME params for both targets - the comparison must change only the target. `periods`
    excludes the test week, which is opened once at the final evaluation.
    """
    windows = {"validation": (config.VAL_START, config.VAL_END),
               "calibration": (config.CAL_START, config.CAL_END)}
    model, training = train(daily, tag=tag, **train_kwargs)

    out = {}
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for period in periods:
        out[period] = forecast_period(model, training, daily, *windows[period])
        out[period].to_parquet(config.forecast_parquet(period, tag), index=False)

    print(f"[{tag}] validation:", quantile_scores(out["validation"]))
    return out
