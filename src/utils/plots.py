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

from .poster import PALETTE, _QUANTILE_COLORS


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


def plot_store_product_week(week_df: pd.DataFrame, store_id, product_id):
    """One store-product's test week (app/pages/8_One_Store.py): recorded sales, stockout days
    shaded, the naive order and the recommended order at the chosen cost ratio - both decided
    BEFORE the day, so both are drawn as lines against what actually sold that day, stockout day
    or not. Screen figure, not a poster one - no save_path, no print-resolution export."""
    d = week_df.sort_values("dt")
    ceiling = pd.concat([d["sale_amount"], d["naive_order_quantity"],
                        d["model_order_quantity"]]).max() * 1.08

    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(d["dt"], d["sale_amount"], color=PALETTE["flat"], width=0.7, label="recorded sales")
    ax.fill_between(d["dt"], 0, ceiling, where=d["stock_hour6_22_cnt"] > 0,
                    color="grey", alpha=0.15, step="mid", label="stockout day")
    ax.plot(d["dt"], d["naive_order_quantity"], marker="o", linestyle=":",
            color=PALETTE["naive"], label="naive order (last week's sales)")
    ax.plot(d["dt"], d["model_order_quantity"], marker="o",
            color=PALETTE["tft"], label="recommended order")
    ax.set_title(f"Store {store_id}, product {product_id} - test week")
    ax.set_ylabel("units (normalized)")
    ax.set_ylim(0, ceiling)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=4, fontsize=8, frameon=False)
    fig.autofmt_xdate(rotation=30)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.30)
    plt.close(fig)
    return fig


def plot_store_week(store_df: pd.DataFrame, store_id, save_path=None):
    """Whole store's test week (app/pages/8_One_Store.py): every product's recorded sales, naive
    order and recommended order summed per day, stockout share shaded. Store-level sibling of
    `plot_store_product_week` - same three series, summed across products instead of one.

    A store carries dozens of products, so on any given day SOME product is almost always out of
    stock - the boolean shading `plot_store_product_week` uses at product level would shade every
    single day here and say nothing (verified: 28-56% of this store's products are out on EVERY
    day of its test week). Shading intensity instead tracks the SHARE of products stocked out that
    day, so a quiet day and a bad day look different."""
    d = store_df.groupby("dt", as_index=False).agg(
        sale_amount=("sale_amount", "sum"),
        naive_order_quantity=("naive_order_quantity", "sum"),
        model_order_quantity=("model_order_quantity", "sum"),
        stockout_share=("stock_hour6_22_cnt", lambda s: (s > 0).mean()),
    ).sort_values("dt")
    ceiling = pd.concat([d["sale_amount"], d["naive_order_quantity"],
                        d["model_order_quantity"]]).max() * 1.08

    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(d["dt"], d["sale_amount"], color=PALETTE["flat"], width=0.7,
           label="recorded sales (all products)")
    half_day = pd.Timedelta(hours=12)
    for dt, share in zip(d["dt"], d["stockout_share"]):
        ax.axvspan(dt - half_day, dt + half_day, color="grey", alpha=0.5 * share, linewidth=0)
    ax.plot(d["dt"], d["naive_order_quantity"], marker="o", linestyle=":",
            color=PALETTE["naive"], label="naive order (last week's sales)")
    ax.plot(d["dt"], d["model_order_quantity"], marker="o",
            color=PALETTE["tft"], label="recommended order")
    ax.set_title(f"Store {store_id} - all products, test week")
    ax.set_ylabel("units (normalized)")
    ax.set_ylim(0, ceiling)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=10, frameon=False)
    fig.autofmt_xdate(rotation=30)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.30)
    plt.close(fig)
    if save_path is not None:
        # tight bbox trims the dead canvas below the legend instead of relying on a hand-tuned
        # bottom margin - the poster copy is a saved PNG, not an interactive figure, so cropping to
        # content only here (not in the returned `fig`) doesn't affect the Streamlit app's display.
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    return fig


def plot_operating_points(table: pd.DataFrame):
    """Three operating points against the naive rule, on the three metrics that matter at once
    (app/pages/5_Operating_Point.py) - a screen-sized sibling of `poster.fig_quantile_anchors`:
    same data, same colors (`poster._QUANTILE_COLORS`), normal screen size instead of
    300dpi/24pt poster type. `table` is indexed ["naive", "waste-focused", "balanced",
    "stockout-focused"] in that order, columns "ran out %"/"demand met %"/"waste % of demand"."""
    fig, axes = plt.subplots(1, len(table.columns), figsize=(10.5, 3.6))
    colors = _QUANTILE_COLORS[:len(table)]
    for ax, col in zip(axes, table.columns):
        bars = ax.bar(table.index, table[col], color=colors)
        for bar, v in zip(bars, table[col]):
            ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylabel(col)
        ax.set_ylim(0, table[col].max() * 1.28)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Three operating points against today", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------- EDA (app/pages/2_EDA.py)
def plot_censoring_distribution(values: pd.Series, unit: str, highlight_value: float = None,
                                highlight_label: str = "picked"):
    """Histogram of one population's censored-day share (per-store or per-product), median
    marked, with an optional second marker - e.g. where the One Store pick sits among all 100
    stores, so that choice reads as "typical", not asserted as typical."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(values * 100, bins=20, color=PALETTE["recovered"], alpha=0.85,
           edgecolor="white", linewidth=0.6, zorder=3)
    top = ax.get_ylim()[1]

    med = float(values.median()) * 100
    ax.axvline(med, color=PALETTE["muted"], linestyle="--", linewidth=1.3, zorder=4)
    ax.annotate(f"median {med:.0f}%", xy=(med, top * 0.96), xytext=(6, -4),
               textcoords="offset points", fontsize=9, color=PALETTE["muted"])

    if highlight_value is not None:
        hv = float(highlight_value) * 100
        ax.axvline(hv, color=PALETTE["raw"], linewidth=2.2, zorder=5)
        ax.annotate(highlight_label, xy=(hv, top * 0.8), xytext=(6, 0),
                   textcoords="offset points", fontsize=9, fontweight="bold",
                   color=PALETTE["raw"])

    ax.set_xlabel(f"share of days censored, per {unit} (%)")
    ax.set_ylabel(f"number of {unit}s")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_ranked_bar(df: pd.DataFrame, id_col: str, value_col: str, n: int = 15,
                    direction: str = "highest", highlight_id=None, id_prefix: str = "#",
                    xlabel: str = ""):
    """Horizontal bar of the N `direction` (`"highest"` or `"lowest"`) `value_col` rows,
    direct-labeled, largest-magnitude entry at the top either way. Lets the EDA page's own toggle
    decide whether this reads as "most affected" or "least affected" rather than baking a value
    judgement into the function. `highlight_id`, if it lands inside the N shown, prints in a
    second color so one specific store/product stands out against the rest of the ranking."""
    picker = df.nlargest if direction == "highest" else df.nsmallest
    top = picker(n, value_col)
    top = top.iloc[::-1] if direction == "highest" else top   # barh draws bottom-up either way
    colors = [PALETTE["raw"] if highlight_id is not None and row[id_col] == highlight_id
             else PALETTE["recovered"] for _, row in top.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 0.36 * len(top) + 1.4))
    bars = ax.barh([f"{id_prefix}{v}" for v in top[id_col]], top[value_col] * 100,
                  color=colors, zorder=3)
    for bar, v in zip(bars, top[value_col]):
        ax.annotate(f"{v * 100:.0f}%", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                   xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, top[value_col].max() * 100 * 1.16)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig
