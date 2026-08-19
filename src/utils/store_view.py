"""Per-store and per-product analysis - the EDA page (app/pages/2_EDA.py) and the one-store demo
page it feeds into (app/pages/10_One_Store.py).

Nobody upstream groups by `store_id` alone, or by `product_id` pooled across the stores that
carry it - every existing groupby in `src/` is at the `(store_id, product_id)` series level
(that's what a demand forecast is scored on). Picking one store to show a reviewer, or comparing
stores/products on how often they go out of stock, needs these coarser rollups, which is what
this module adds. Pure pandas/numpy, no Streamlit import - same role as `poster.py` for the
poster: analysis that's shared and importable from a notebook too, just not currently called
from one.

Both `*_stats` functions read the FULL available history (train+eval, ~90 days), not just
training - deliberately, unlike `features.censoring_bucket` (which is training-only because its
bands are used to GROUP scored rows, and folding validation/test into that grouping would leak).
These are pure description - "how often has this store/product actually gone empty" - not a
feature or a scoring axis, so there is nothing to leak; using the longer window just makes the
number a more stable structural fact and not a noisy artifact of whichever week you looked at.
"""
import pandas as pd


def store_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per store: how many products it carries and how often its shelf actually empties
    (`is_censored` share), over the FULL available history (train+eval, ~90 days) - a stable,
    structural fact about the store, not a noisy 7-day artifact from the test week alone."""
    return daily.groupby("store_id").agg(
        n_products=("product_id", "nunique"),
        censored_share=("is_censored", "mean"),
    ).reset_index()


def product_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per product, POOLED ACROSS every store that carries it: how many stores stock it,
    how many product-days that is in total, and how often it goes out of stock.

    Pooling across stores is deliberate: it answers "does this product itself go out of stock a
    lot, wherever it's sold" rather than "does this one store manage it badly" - the store view
    answers the second question."""
    return daily.groupby("product_id").agg(
        n_stores=("store_id", "nunique"),
        n_days=("dt", "size"),
        censored_share=("is_censored", "mean"),
    ).reset_index()


def pick_store(stats: pd.DataFrame) -> int:
    """The store closest to the subset median on `censored_share` - ties broken by `n_products`
    closest to ITS median. The honest "typical performer", not a cherry-picked best case."""
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
