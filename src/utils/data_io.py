"""Load FreshRetailNet-50K, verify its schema, cut the working subset, and persist it.

`build_subset(n_stores)` rebuilds data/processed/ (driven from the notebook); everything
downstream reads the persisted parquets via `load(kind)`.
"""
import datetime

import numpy as np
import pandas as pd

from . import config

ID_COLS = ["city_id", "store_id", "management_group_id",
           "first_category_id", "second_category_id", "third_category_id", "product_id"]

KEEP = ID_COLS + ["dt", "sale_amount", "stock_hour6_22_cnt",
                  "discount", "holiday_flag", "activity_flag",
                  "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]

# train/eval parquet pairs, keyed by the `kind` passed to load()
_PAIRS = {
    "daily": (config.DAILY_TRAIN, config.DAILY_EVAL),
    "hourly": (config.HOURLY_TRAIN, config.HOURLY_EVAL),
    "recovered": (config.RECOVERED_TRAIN, config.RECOVERED_EVAL),
}


def validate_schema(train: pd.DataFrame) -> dict:
    """Assert the schema facts the rest of the pipeline relies on: 24-length hourly vectors,
    binary stock status, `stock_hour6_22_cnt` derivable from it, and `sale_amount` equal to one
    of the two hourly aggregations. Raises on any violation."""
    lo, hi = config.ACTIVE_HOURS
    # each row's hours_sale/hours_stock_status is a 24-long list; stacking the whole column turns
    # it into one (n_rows, 24) array so the hour axis can be sliced and summed directly below
    hs = np.array(train["hours_sale"].tolist(), dtype=float)
    hss = np.array(train["hours_stock_status"].tolist(), dtype=float)

    assert hs.shape[1] == 24 and hss.shape[1] == 24, "expected 24-length hourly vectors"
    assert set(np.unique(hss)).issubset({0, 1}), "stock status must be binary"

    # recompute stock_hour6_22_cnt ourselves from the raw hourly flags, and check it matches the
    # column the dataset shipped - the count of active-window (06:00-22:00) hours flagged "stocked"
    derived_cnt = (hss[:, lo:hi] == 1).sum(axis=1)
    assert np.array_equal(derived_cnt, train["stock_hour6_22_cnt"].values), \
        "stock_hour6_22_cnt != #(hours_stock_status[6:22]==1)"

    # sale_amount could be either convention depending on the data variant: the sum of all 24
    # hours, or just the sum over the active window. Only one needs to hold, but at least one must.
    eq_full = np.allclose(train["sale_amount"].values, hs.sum(axis=1), atol=1e-6)
    eq_win = np.allclose(train["sale_amount"].values, hs[:, lo:hi].sum(axis=1), atol=1e-6)
    assert eq_full or eq_win, "sale_amount matches neither hourly aggregation"

    return dict(sale_amount_is_full_24h_sum=eq_full,
                oos_zero_share=float((hs[hss == 1] == 0).mean()),
                n_rows=len(train))


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the daily columns, add `is_censored` + `dow`, sort by series and date."""
    d = df[KEEP].copy()
    d["dt"] = pd.to_datetime(d["dt"])
    d["is_censored"] = (d["stock_hour6_22_cnt"] > 0).astype(int)
    d["dow"] = d["dt"].dt.dayofweek
    return d.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)


def _persist_hourly(sub: pd.DataFrame, raw: pd.DataFrame, path) -> None:
    """Persist the 24h vectors `to_daily` drops, keyed on (store_id, product_id, dt). The
    hour-level demand recovery needs them."""
    key = ["store_id", "product_id", "dt"]
    cols = ["hours_sale", "hours_stock_status"]
    raw = raw.assign(dt=pd.to_datetime(raw["dt"]))
    hourly = sub[key].merge(raw[key + cols], on=key, how="left")
    assert not hourly["hours_sale"].isna().any(), \
        "hourly join dropped rows - subset keys must match raw exactly"
    hourly.to_parquet(path, index=False)


def build_subset(n_stores: int = 30, random_state: int = config.SUBSET_SEED) -> None:
    """Rebuild data/processed/ from a seeded random sample of `n_stores` stores, keeping every
    series (store x product) they carry, across all categories.

    Whole stores, not individual series: nothing is selected on sales volume, so the sample stays
    representative and no scope restriction has to be declared - and each store keeps a full
    assortment, which the ordering stage needs to recommend a realistic basket.

    The draw is nested in `n_stores`: at a fixed seed, 30 stores are the first 30 of the 100, so
    growing the subset adds stores without swapping the existing ones. Without that, "the estimate
    stopped moving as the sample grew" would be a statement about which stores were drawn, and subset
    choice moves WAPE by ~6%, more than most model choices.
    """
    from datasets import load_dataset
    ds = load_dataset(config.HF_DATASET)   # cached locally after the first pull
    train_raw, eval_raw = ds["train"].to_pandas(), ds["eval"].to_pandas()

    daily_train, daily_eval = to_daily(train_raw), to_daily(eval_raw)

    # One fixed shuffle, take the first n. `choice(all_stores, n)` redraws per n and would share
    # only 2 of 30 stores between a 30- and a 100-store subset. A permutation prefix is still a
    # uniform sample, so nesting costs nothing in representativeness.
    all_stores = np.sort(daily_train["store_id"].unique())
    picked = np.random.default_rng(random_state).permutation(all_stores)[:n_stores]
    stores = sorted(int(s) for s in picked)

    sub_train = daily_train[daily_train["store_id"].isin(stores)].copy()
    sub_eval = daily_eval[daily_eval["store_id"].isin(stores)].copy()
    categories = sorted(int(c) for c in sub_train["first_category_id"].unique())

    # kept rows only, and on the raw frames: `to_daily` drops the hourly vectors these checks read
    for label, raw in [("train", train_raw), ("eval", eval_raw)]:
        print(f"schema facts verified ({label} subset):",
              validate_schema(raw[raw["store_id"].isin(stores)]))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    sub_train.to_parquet(config.DAILY_TRAIN, index=False)
    sub_eval.to_parquet(config.DAILY_EVAL, index=False)
    _persist_hourly(sub_train, train_raw, config.HOURLY_TRAIN)
    _persist_hourly(sub_eval, eval_raw, config.HOURLY_EVAL)

    summary = _summary_md(
        selection=(f"Seeded random sample of **{n_stores} of {len(all_stores)} stores** "
                   f"(seed {random_state}), keeping every series they carry, all categories."),
        corpus=_describe(daily_train), subset=_describe(sub_train),
        eval_rows=len(sub_eval), stores=stores, categories=categories)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBSET_SUMMARY.write_text(summary, encoding="utf-8")
    print(summary)


def _describe(daily: pd.DataFrame) -> dict:
    """The same shape counts for the corpus and the subset, so the two are comparable."""
    return dict(
        n_series=int(daily.groupby(["store_id", "product_id"]).ngroups),
        n_stores=int(daily["store_id"].nunique()),
        n_categories=int(daily["first_category_id"].nunique()),
        n_products=int(daily["product_id"].nunique()),
        rows=int(len(daily)),
        censored_day_share=round(float(daily["is_censored"].mean()), 4),
        mean_daily_sales=round(float(daily["sale_amount"].mean()), 4),
    )


def _summary_md(selection: str, corpus: dict, subset: dict, eval_rows: int,
                stores: list, categories: list) -> str:
    """Fill `config.SUBSET_SUMMARY_MD` - corpus and subset side by side, so representativeness is
    visible rather than asserted."""
    rows = [("series", f"{corpus['n_series']:,}", f"{subset['n_series']:,}"),
            ("stores", f"{corpus['n_stores']:,}", f"{subset['n_stores']:,}"),
            ("categories", corpus["n_categories"], subset["n_categories"]),
            ("products", f"{corpus['n_products']:,}", f"{subset['n_products']:,}"),
            ("train rows", f"{corpus['rows']:,}", f"{subset['rows']:,}"),
            ("normalised sale amount/day", corpus["mean_daily_sales"], subset["mean_daily_sales"]),
            ("censored product-days", f"{corpus['censored_day_share']:.1%}",
             f"{subset['censored_day_share']:.1%}")]
    return config.SUBSET_SUMMARY_MD.format(
        built_at=datetime.datetime.now().isoformat(timespec="seconds"),
        selection=selection,
        table="\n".join(f"| {name} | {c} | {s} |" for name, c, s in rows),
        share=f"{subset['n_series'] / corpus['n_series']:.1%}",
        per_store=subset["n_series"] // subset["n_stores"],
        eval_rows=f"{eval_rows:,}",
        categories=", ".join(str(c) for c in categories),
        stores=", ".join(str(s) for s in stores),
    )


def load(kind: str) -> pd.DataFrame:
    """Load a train/eval parquet pair ("daily" | "hourly" | "recovered") into one frame with a
    `split` column ("train"/"eval") and parsed dates."""
    train_path, eval_path = _PAIRS[kind]
    # relative to the repo root: these lines land in committed notebooks, where an absolute
    # user-home path would leak whose machine ran it
    print(f"loading {kind} subset from {train_path.relative_to(config.ROOT).parent.as_posix()}")
    train = pd.read_parquet(train_path).assign(split="train")
    eval_ = pd.read_parquet(eval_path).assign(split="eval")
    combined = pd.concat([train, eval_], ignore_index=True)
    combined["dt"] = pd.to_datetime(combined["dt"])
    return combined
