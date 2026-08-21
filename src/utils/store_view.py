"""Per-store and per-product rollups for the EDA page (app/pages/2_EDA.py) and the one-store
walkthrough it feeds (app/pages/10_One_Store.py).

Every groupby elsewhere in `src/` is at the `(store_id, product_id)` series level, which is what a
demand forecast is scored on. Picking one store to show a reviewer, or ranking stores and products on
how often they go empty, needs these coarser cuts. Pure pandas, no Streamlit import.

Both `*_stats` functions read the full available history (train+eval, ~90 days), not just training,
unlike `features.censoring_bucket`, which is training-only because its bands group scored rows and
folding later windows into that grouping would leak. These are description rather than a feature or a
scoring axis, so the longer window only makes the number a more stable structural fact.
"""
import pandas as pd


def store_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per store: how many products it carries and how often its shelf empties
    (`is_censored` share), over the full ~90-day history rather than the test week alone."""
    return daily.groupby("store_id").agg(
        n_products=("product_id", "nunique"),
        censored_share=("is_censored", "mean"),
    ).reset_index()


def product_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per product, pooled across every store that carries it: how many stores stock it, how
    many product-days that is, and how often it goes out of stock.

    Pooling answers "does this product go out of stock wherever it is sold", where `store_stats`
    answers "does this one store manage it badly"."""
    return daily.groupby("product_id").agg(
        n_stores=("store_id", "nunique"),
        n_days=("dt", "size"),
        censored_share=("is_censored", "mean"),
    ).reset_index()


def pick_store(stats: pd.DataFrame) -> int:
    """The store closest to the subset median on `censored_share`, ties broken by `n_products`
    closest to its own median: a typical performer, not a best case."""
    med_censored = stats["censored_share"].median()
    dist = (stats["censored_share"] - med_censored).abs()
    tied = stats[dist == dist.min()]
    if len(tied) == 1:
        return int(tied.iloc[0]["store_id"])
    med_products = stats["n_products"].median()
    closest = (tied["n_products"] - med_products).abs().idxmin()
    return int(tied.loc[closest, "store_id"])


def pick_products(store_daily: pd.DataFrame, per_day: pd.DataFrame,
                  min_observed_days: int = 3) -> dict:
    """{"chronic", "typical", "lowest_waste"} product_ids for one store.

    `store_daily`: that store's full-history rows (`is_censored` per product-day, all periods) -
    used for the two structural picks. `per_day`: that store's test-week rows already scored by
    `orders.run(..., regime="observed")` - i.e. already restricted to full-shelf days, with
    `model_waste` on it - used for the one model-outcome pick.

    `chronic` is picked only among products with >=1 scoreable (full-shelf) day in the test week,
    so its card is never empty under the "observed" regime `per_day` was scored under. `lowest_waste`
    is picked only among products with >=`min_observed_days` scoreable days, so a 1-day sample can't
    win it by chance. Both guards fall back to the unrestricted pick if nothing clears the bar.
    """
    censored_share = store_daily.groupby("product_id")["is_censored"].mean()
    n_observed = per_day.groupby("product_id").size()

    scoreable = censored_share.index[censored_share.index.isin(n_observed[n_observed >= 1].index)]
    chronic_pool = censored_share.loc[scoreable] if len(scoreable) else censored_share
    chronic = int(chronic_pool.idxmax())

    typical = int((censored_share - censored_share.median()).abs().idxmin())

    waste_by_product = per_day.groupby("product_id")["model_waste"].sum()
    eligible = waste_by_product.index[
        waste_by_product.index.isin(n_observed[n_observed >= min_observed_days].index)]
    waste_pool = waste_by_product.loc[eligible] if len(eligible) else waste_by_product
    lowest_waste = int(waste_pool.idxmin())

    return {"chronic": chronic, "typical": typical, "lowest_waste": lowest_waste}
