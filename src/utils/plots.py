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
    ax.set_ylabel("sale_amount (normalized, not physical units - README §4)")
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
        fig.savefig(save_path, dpi=300)   # print resolution - A0 poster, not just notebook display
    return fig


def plot_cost_sweep(sweep: pd.DataFrame, save_path=None):
    """The ordering stage's business chart: waste and stockouts across the whole `c_u/c_o` cost
    sweep (`orders.cost_sweep`), model vs. naive, x-axis the newsvendor fractile `q_star` a ratio
    implies - not the ratio itself, which is not on a comparable scale across rows. Both figures
    report PERCENTAGES only, never an absolute unit count: `sale_amount` is normalised by an
    undisclosed coefficient (README §4), so summed waste and shortfall are relative quantities,
    not physical ones."""
    sweep = sweep.sort_values("q_star")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(sweep["q_star"], sweep["waste_vs_naive_pct"], marker="o", color="tab:blue",
             label="model vs. naive")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_xlabel("q* (newsvendor fractile)")
    ax1.set_ylabel("waste vs. naive (%)")
    ax1.set_title("Waste across the cost sweep")
    ax1.legend(loc="best")

    ax2.plot(sweep["q_star"], sweep["stockout_pct_model"], marker="o", color="tab:blue",
             label="model")
    ax2.plot(sweep["q_star"], sweep["stockout_pct_naive"], marker="o", color="tab:red",
             linestyle=":", label="naive")
    ax2.set_xlabel("q* (newsvendor fractile)")
    ax2.set_ylabel("stockouts (%)")
    ax2.set_title("Stockouts across the cost sweep")
    ax2.legend(loc="best")

    fig.tight_layout()
    plt.close(fig)
    if save_path is not None:
        fig.savefig(save_path, dpi=300)   # print resolution - A0 poster, not just notebook display
    return fig


# Column names differ per family because each writes what its own trainer reports: XGBoost counts
# boosting rounds, Lightning counts epochs. Normalised here rather than at write time so each saved
# curve stays a faithful record of what its trainer produced.
_CURVE_COLS = {"xgb": ("round", "train", "validation", "boosting round"),
               "tft": ("epoch", "train_loss", "val_loss", "epoch")}


def plot_learning_curves(curves: dict, save_path=None):
    """Train-vs-validation loss per family, ONE PANEL EACH, with the validation minimum marked.

    `curves` maps a family in `_CURVE_COLS` to its saved curve frame; families whose file does not
    exist yet are simply left out, so this renders with one panel or two.

    Deliberately NOT one shared pair of axes. The x-axes count different things (boosting rounds
    against epochs) and each y is that family's own loss implementation over its own quantile set, so a
    single plot would invite reading a vertical gap between two curves as a difference in accuracy when
    it is a difference in units. Separate panels, and the y-labels say so.

    The marked minimum is the whole point of the figure: everything left of it is the model still
    learning, everything right of it is the model memorising. Where the CHOSEN config sits relative to
    that mark is the overfitting question, answered by looking.
    """
    have = [(fam, curves[fam]) for fam in _CURVE_COLS if fam in curves and curves[fam] is not None]
    if not have:
        raise ValueError(f"no curves given - expected any of {list(_CURVE_COLS)}")

    fig, axes = plt.subplots(1, len(have), figsize=(6.5 * len(have), 4.5), squeeze=False)
    for ax, (fam, curve) in zip(axes[0], have):
        x, train_col, val_col, xlabel = _CURVE_COLS[fam]
        best = curve.loc[curve[val_col].idxmin()]   # the row where validation loss is lowest

        ax.plot(curve[x], curve[train_col], color="tab:blue", linewidth=2, label="training")
        ax.plot(curve[x], curve[val_col], color="tab:red", linewidth=2, label="validation")
        ax.axvline(best[x], color="grey", linestyle="--", linewidth=1)
        # Annotate only the minimum. A label on every point is unreadable and the shape carries the
        # rest of the story on its own.
        ax.annotate(f"validation best\n{xlabel} {int(best[x])} · {best[val_col]:.4f}",
                    xy=(best[x], best[val_col]), xytext=(0.30, 0.70), textcoords="axes fraction",
                    fontsize=9, color="dimgrey",
                    arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.8))

        # The turn upward is the finding, and on the full y-range it is invisible - a rise of ~0.001 on
        # an axis spanning 0.15 reads as a flat line. The inset is that region at a scale where the U
        # is legible; without it the figure appears to show a model that simply converges.
        tail = curve[curve[x] >= best[x] * 0.35]
        if len(tail) > 5 and tail[val_col].max() - tail[val_col].min() > 0:
            zoom = ax.inset_axes([0.53, 0.13, 0.44, 0.36])
            zoom.plot(tail[x], tail[val_col], color="tab:red", linewidth=1.6)
            zoom.axvline(best[x], color="grey", linestyle="--", linewidth=0.9)
            zoom.set_title("validation, zoomed", fontsize=8, color="dimgrey")
            zoom.tick_params(labelsize=7, colors="dimgrey")
            zoom.grid(alpha=0.2, linewidth=0.5)
            zoom.set_axisbelow(True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{fam.upper()} quantile loss (own scale)")
        ax.set_title(f"{fam.upper()}: training vs validation loss")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="best")

    fig.tight_layout()
    plt.close(fig)
    if save_path is not None:
        fig.savefig(save_path, dpi=300)   # print resolution - A0 poster, not just notebook display
    return fig
