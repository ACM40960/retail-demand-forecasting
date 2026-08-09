"""Figures shared by the notebooks and the Streamlit app.

Every function RETURNS the figure and only saves when `save_path` is given, so the app renders
exactly what the notebook shows inline - the app never recomputes anything.

Each builder CLOSES the figure before returning it. Returning it is what displays it in a notebook;
leaving it open as well makes the inline backend draw it a second time at the end of the cell, so
the reader sees the same chart twice. A closed figure still renders from the return value and still
works with `st.pyplot`.
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_recovery_overlay(series_df: pd.DataFrame, title_suffix: str = ""):
    """Recorded sales vs. recovered demand for ONE series (already filtered), stockout days
    shaded. The gap on shaded days is the demand the till never saw."""
    ex = series_df.sort_values("dt")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ex["dt"], ex["recovered_demand"], label="recovered demand",
            color="tab:blue", linewidth=1.5)
    ax.plot(ex["dt"], ex["sale_amount"], label="recorded sales", color="tab:red", linewidth=1.5)
    ax.fill_between(ex["dt"], 0, ex["recovered_demand"].max() * 1.05,
                    where=ex["is_censored"] == 1, color="grey", alpha=0.15, label="stockout day")
    ax.set_title(f"Recorded sales vs. recovered demand{title_suffix}")
    ax.set_ylabel("units/day")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.close(fig) 
    return fig


def plot_recovery_example(recovered: pd.DataFrame, save_path=None):
    """The most-censored series' overlay - the report figure, where the effect is largest."""
    cens = recovered.groupby(["store_id", "product_id"])["is_censored"].mean()
    store_id, product_id = cens.sort_values(ascending=False).index[0]
    ex = recovered[(recovered.store_id == store_id) & (recovered.product_id == product_id)]
    fig = plot_recovery_overlay(ex, title_suffix=f" - store {store_id}, product {product_id}")
    if save_path is not None:
        fig.savefig(save_path, dpi=110)
    return fig
