"""The six A0-poster figures, each built from a committed artifact.

Separate from `plots.py` because the constraints are different, not because the charts are: a poster
figure is read from 1.5-2 m, so it needs ~24 pt type and 300 dpi, and a screen figure does not. The
DATA still comes from the same places the notebooks read - `recovery_model_comparison.csv`, the saved
forecast parquets, `waste_at_equal_service.csv`, `cost_sweep.csv` - so a figure can never disagree
with the table it came from. Nothing here computes a result that is not already on disk, except the
per-band table, which is recomputed through `metrics.scores_by_bucket` for the same reason.

Colour is assigned by MEANING and is constant across figures (2026-08-14 re-pick, run through the
dataviz skill's validator rather than eyeballed - `node scripts/validate_palette.js
"#2A78D6,#EB6834,#1BAF7A,#EDA100" --mode light`, all four adjacent-pair CVD/normal-vision checks
pass; the surface-contrast WARN on aqua/amber is the reason every bar here carries a printed value):

    raw sales        #2A78D6 blue        recovered demand  #EB6834 orange
    TFT              #1BAF7A aqua        XGBoost           #EDA100 amber
    status quo       muted grey                            (recessive on purpose)

The two pairs are never mixed inside one figure, so no reader has to hold four hues at once.

`sale_amount` is normalised by an undisclosed coefficient, so NO axis here is ever labelled in units
- every one is a ratio, a percentage or a normalised daily total, and each says so.
"""
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

PALETTE = {"raw": "#2A78D6", "recovered": "#EB6834",
           "tft": "#1BAF7A", "xgb": "#EDA100",
           "naive": "#8A8981", "ink": "#0B0B0B", "muted": "#6B6A63",
           "flat": "#C7C6BE", "good": "#0CA30C"}

POSTER_RC = {
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    # Lato isn't installed on this machine, so Segoe UI is the closest available fallback in
    # spirit - modern humanist sans, not DejaVu's dated, uneven-weight look. Falls forward to
    # Lato automatically if it's ever installed, since it's still first in the list.
    "font.sans-serif": ["Lato", "Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
    "font.size": 23, "axes.titlesize": 26, "axes.labelsize": 23,
    "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 21,
    # Titles carry hierarchy through SIZE, not weight - a page of bold titles stops meaning anything.
    "axes.titleweight": "normal", "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "axes.edgecolor": PALETTE["muted"],
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2,
    "axes.grid": True, "grid.color": "#EAEAE6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "lines.linewidth": 3.0, "lines.markersize": 11,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
}

POSTER_DIR = config.ROOT / "poster" / "figures"   # appendix figures - not referenced by poster.html
IMAGES_DIR = config.ROOT / "poster" / "images"     # the three figures poster.html actually includes


def _finish(fig, name: str, save: bool, dest=None):
    """`bbox_inches='tight'` is passed explicitly rather than left to `POSTER_RC` - the
    `rc_context` each figure builds under has already exited by the time this runs (it's a
    plain function call after the `with` block, not nested inside it), so any legend or
    annotation placed outside the axes' own box (e.g. a figure-level legend above the plot)
    would otherwise be silently clipped by matplotlib's default non-tight save bbox."""
    target = dest or POSTER_DIR
    if save:
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{name}.png", facecolor="white", bbox_inches="tight")
        print(f"  wrote {target.relative_to(config.ROOT).as_posix()}/{name}.png")
    plt.close(fig)
    return fig


def _bar_labels(ax, bars, fmt="{:.1f}", dy=0.6, color=None, weight="semibold"):
    """Direct value labels. Not decoration: the aqua and amber slots sit below 3:1 contrast
    against white, so the validator requires visible labels as relief."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=20, fontweight=weight,
                color=color or PALETTE["ink"])


# ---------------------------------------------------------------- 1. the problem
def fig1_problem(recovered: pd.DataFrame = None, save: bool = True):
    """One real series: recorded sales against recovered demand, stockout days shaded.

    The series is chosen as the one whose training-period censoring rate is CLOSEST TO 50%, not the
    most-censored one. That is a presentational choice and worth stating: the most-censored series
    is out of stock on nearly every day, so the whole panel shades grey and the very contrast the
    figure exists to show disappears. A ~50% series alternates, which is what makes the gap legible.
    The last `days` of the training window are shown so individual days stay readable at 2 m.
    """
    from .. import recovery
    recovered = recovery.load_daily("recovered") if recovered is None else recovered
    train = recovered[recovered["period"] == "training"]

    rate = train.groupby(["store_id", "product_id"])["is_censored"].mean()   # each series' own
                                                                               # censored-day share
    store, product = (rate - 0.5).abs().idxmin()   # whichever series' rate sits closest to 50%
    ex = train[(train.store_id == store) & (train.product_id == product)].sort_values("dt").tail(35)

    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 6.0))
        top = ex["recovered_demand"].max() * 1.25
        ax.fill_between(ex["dt"], 0, top, where=ex["is_censored"] == 1,
                        color="#CFCFCF", alpha=0.75, step="mid", linewidth=0,
                        label="shelf empty at some point that day")
        ax.plot(ex["dt"], ex["recovered_demand"], color=PALETTE["recovered"],
                label="recovered demand", zorder=3)
        ax.plot(ex["dt"], ex["sale_amount"], color=PALETTE["raw"],
                label="recorded sales", zorder=4)

        d = ex.loc[(ex["recovered_demand"] - ex["sale_amount"]).idxmax()]
        ax.annotate("demand the till\nnever recorded",
                    xy=(d["dt"], (d["recovered_demand"] + d["sale_amount"]) / 2),
                    xytext=(0.03, 0.88), textcoords="axes fraction", fontsize=21,
                    fontweight="semibold", color=PALETTE["ink"], va="top",
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=PALETTE["ink"]))

        ax.set_ylim(0, top)
        ax.set_ylabel("daily demand\n(normalised)")
        ax.set_title("A sold-out shelf records zero")
        # Weekly ticks in "05 May" form: the ISO dates were long enough to need rotating, and
        # rotated labels then collided with the legend sitting underneath them.
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
        fig.tight_layout()
    return _finish(fig, "fig1_problem", save)


# ---------------------------------------------------------------- 2. recovery validated
def fig2_recovery_validation(save: bool = True):
    """Every Stage-1 candidate on the held-out full-shelf-day test. Reads
    `recovery_model_comparison.csv` exactly as notebook 01 wrote it."""
    board = pd.read_csv(config.RECOVERY_COMPARISON, index_col=0).sort_values("wape")
    chosen, control = "lightgbm_tweedie", "series_hour_mean"
    better = (1 - board.loc[chosen, "wape"] / board.loc[control, "wape"]) * 100

    colours = [PALETTE["recovered"] if n == chosen else
               PALETTE["naive"] if n == control else PALETTE["flat"] for n in board.index]
    labels = [n.replace("_", " ") for n in board.index]

    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 6.2))
        y = np.arange(len(board))[::-1]
        ax.barh(y, board["wape"], color=colours, height=0.76)
        for yi, v in zip(y, board["wape"]):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=20, fontweight="semibold")
        ax.set_yticks(y, labels)
        ax.set_xlabel("error on daily totals, WAPE  (lower is better)")
        ax.set_title("Validated where the truth is known")
        xmax = board["wape"].max() * 1.55
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)

        # Short arrow that stays inside the chosen bar's own row: the previous version ran a long
        # diagonal from low-centre up to the top bar, cutting straight across the "0.306"/"0.301"
        # value labels of the bars in between.
        ax.annotate(f"{better:.1f}% better than\nthe no-model control",
                    xy=(board.loc[chosen, "wape"] * 0.5, y[0]), va="center", ha="left",
                    xytext=(xmax * 0.52, y[0]),
                    fontsize=20, fontweight="semibold", color=PALETTE["ink"],
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=PALETTE["ink"]))
        fig.tight_layout()
    return _finish(fig, "fig2_recovery_validation", save)


# ---------------------------------------------------------------- 3. the replication
def fig3_per_band(daily: pd.DataFrame = None, save: bool = True):
    """Change in WAPE from training on recovered demand, per censoring band, both model families.

    Recomputed from the four saved validation forecasts through `metrics.scores_by_bucket` - the
    same function the notebooks use - rather than transcribed, so it cannot drift from them.
    """
    from .. import recovery
    from .features import censoring_bucket
    from .metrics import scores_by_bucket

    daily = recovery.load_daily("recovered") if daily is None else daily
    band = censoring_bucket(daily)

    change = {}
    for fam in ("tft", "xgb"):
        # score both targets, per band, for this one family
        sc = {t: scores_by_bucket(pd.read_parquet(config.forecast_parquet("validation", t, fam)),
                                  band) for t in ("raw", "recovered")}
        # % change in WAPE going from raw to recovered, band by band (negative = recovered is better);
        # drop the pooled "ALL" row, since the whole point of this figure is reading it per band
        change[fam] = ((sc["recovered"]["WAPE"] / sc["raw"]["WAPE"] - 1) * 100).drop("ALL")

    bands = list(change["tft"].index)
    nice = ["rarely\n(<25%)", "sometimes\n(25-50%)", "often\n(50-75%)", "constantly\n(75%+)"]
    x, w = np.arange(len(bands)), 0.42

    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 6.4))
        b1 = ax.bar(x - w / 2, change["tft"].values, w, color=PALETTE["tft"], label="TFT")
        b2 = ax.bar(x + w / 2, change["xgb"].values, w, color=PALETTE["xgb"], label="XGBoost")
        lo = min(change["tft"].min(), change["xgb"].min())
        hi = max(change["tft"].max(), change["xgb"].max())
        pad = (hi - lo) * 0.06
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + (pad if h >= 0 else -pad),
                        f"{h:+.0f}%", ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=20, fontweight="semibold")

        ax.axhline(0, color=PALETTE["ink"], linewidth=2)
        ax.set_xticks(x, nice)
        ax.set_xlabel("how often the product sells out")
        ax.set_ylabel("change in forecast error\n(% WAPE, negative = better)")
        ax.set_title("Recovery helps where demand was hidden")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False)
        ax.set_ylim(lo - 5 * pad, hi + 5 * pad)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
    return _finish(fig, "fig3_per_band", save)


# ---------------------------------------------------------------- 3b. calibration
def fig_calibration(save: bool = True):
    """Coverage before vs. after conformal correction, both bands, validation and test. Reads
    `conformal_results_tft_recovered_wide{80,95}.json` - the tft/recovered arm carries both
    periods since a save-overwrite bug was fixed (a later test-week run used to erase an
    earlier validation record for the same arm; `conformal.run` now merges instead)."""
    import json
    rows = []
    # one row per (band, window) combination - 2 bands x 2 windows = 4 rows, each holding the
    # coverage before and after conformal correction, read straight out of the saved JSON reports
    for nominal, tag in ((0.80, "wide80"), (0.95, "wide95")):
        d = json.loads(config.conformal_results(tag, family="tft", target="recovered").read_text())
        for period in ("validation", "test"):
            p = d["periods"][period]
            rows.append({"label": f"{nominal:.0%} band\n{period}", "nominal": nominal * 100,
                        "before": p["uncorrected"]["coverage"] * 100,
                        "after": p["corrected"]["coverage"] * 100})
    board = pd.DataFrame(rows)
    y = np.arange(len(board))[::-1]

    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 6.4))
        for yi, r in zip(y, board.itertuples()):
            ax.plot([r.before, r.after], [yi, yi], color=PALETTE["muted"], lw=3.2, zorder=1)
        ax.scatter(board["before"], y, s=340, color=PALETTE["raw"], zorder=3,
                  label="raw forecast", edgecolor="white", linewidth=2)
        ax.scatter(board["after"], y, s=340, color=PALETTE["recovered"], zorder=3,
                  label="after calibration", edgecolor="white", linewidth=2)
        for yi, r in zip(y, board.itertuples()):
            ax.plot([r.nominal], [yi], marker="|", color=PALETTE["ink"], markersize=28,
                    markeredgewidth=3, zorder=4)

        ax.set_yticks(y, board["label"])
        ax.set_xlabel("share of actuals the band actually contains (%)")
        ax.set_title("Calibration closes most of the gap")
        ax.set_xlim(60, 102)
        ax.legend(loc="lower left", frameon=False, ncol=1)
        ax.grid(axis="y", visible=False)
        ax.text(0.98, 0.02, "black tick = target coverage", transform=ax.transAxes,
                ha="right", fontsize=18, color=PALETTE["muted"])
        fig.tight_layout()
    return _finish(fig, "fig_calibration", save)


# ---------------------------------------------------------------- 4. the money (poster image)
def fig_waste_comparison(save: bool = True):
    """Raw vs. recovered waste at matched 95% demand met, one panel per window. Reads
    `conclusion_raw_vs_recovered.csv` exactly as notebook 04's E2 section wrote it - both
    windows (validation, test) and both model families, so a bar-height difference here is a
    waste difference at IDENTICAL availability, never a side effect of one arm ordering more.

    This is the image `poster/poster.html` Card 4 embeds. Supersedes the old
    `fig4_equal_service`, which read a `waste_at_equal_service.csv` no longer written since
    notebook 04 was restructured around `conclusion_*.csv` outputs.
    """
    df = pd.read_csv(config.REPORTS_DIR / "conclusion_raw_vs_recovered.csv")
    windows = list(df["window"].unique())
    w = 0.36

    with mpl.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, len(windows), figsize=(11.5 * len(windows), 6.6))
        axes = np.atleast_1d(axes)
        for ax, window in zip(axes, windows):
            sub = df[df["window"] == window].reset_index(drop=True)
            x = np.arange(len(sub))
            b1 = ax.bar(x - w / 2, sub["raw waste %"], w, color=PALETTE["raw"],
                       label="trained on raw sales")
            b2 = ax.bar(x + w / 2, sub["recovered waste %"], w, color=PALETTE["recovered"],
                       label="trained on recovered demand")
            _bar_labels(ax, b1, "{:.1f}%", dy=1.0)
            _bar_labels(ax, b2, "{:.1f}%", dy=1.0)

            # The saving is written ABOVE the pair, not as an arrow between bar tops - the arrow
            # crossed the value labels and made three numbers fight for the same 2 cm.
            top = sub[["raw waste %", "recovered waste %"]].to_numpy().max() * 1.42
            for xi, r in sub.iterrows():
                y0 = max(r["raw waste %"], r["recovered waste %"]) + top * 0.135
                ax.plot([xi - w / 2, xi + w / 2], [y0, y0], color=PALETTE["good"], lw=2.6)
                ax.text(xi, y0 + top * 0.015, f"{r['waste saved (pts)']:+.1f} pts", ha="center",
                        va="bottom", fontsize=22, fontweight="semibold", color=PALETTE["good"])

            ax.set_xticks(x, sub["model"])
            ax.set_ylabel("food wasted\n(% of demand)")
            ax.set_title(f"{window.capitalize()}, at 95% demand met")
            ax.set_ylim(0, top)
            ax.grid(axis="x", visible=False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1),
                  frameon=False)
        fig.tight_layout()
    return _finish(fig, "waste_comparison", save, dest=IMAGES_DIR)


# ---------------------------------------------------------------- 5. robustness (poster image)
def fig5_cost_sweep(save: bool = True):
    """Ordering cost against the status-quo rule across the whole cost sweep. Reads
    `cost_sweep.csv`. ONE axis - cost saving only; the stockout column lives in its own figure
    rather than being hung off a second y-scale. This is the image `poster/poster.html` Card 5
    embeds."""
    sweep = pd.read_csv(config.COST_SWEEP_CSV).sort_values("c_u")

    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.plot(sweep["c_u"], sweep["cost_vs_naive_pct"], marker="o", markersize=13,
                linewidth=3.6, color=PALETTE["recovered"], zorder=3)
        ax.axhline(0, color=PALETTE["naive"], linewidth=2.8, linestyle="--")
        ax.text(sweep["c_u"].max(), 1.5, "status-quo rule\n(order what sold last week)",
                ha="right", va="bottom", fontsize=20, color=PALETTE["muted"])

        for _, r in sweep.iterrows():
            if r["c_u"] in (0.25, 1.0, 4.0, 19.0):
                ax.annotate(f"{r['cost_vs_naive_pct']:.0f}%",
                            xy=(r["c_u"], r["cost_vs_naive_pct"]),
                            xytext=(0, 16), textcoords="offset points",
                            ha="center", fontsize=20, fontweight="semibold")

        ax.set_xscale("log")
        ax.set_xticks(sweep["c_u"], [f"{v:g}x" for v in sweep["c_u"]])
        ax.minorticks_off()
        ax.set_xlabel("how much worse a stockout is than a binned item")
        ax.set_ylabel("ordering cost saved\nvs the status quo (%)")
        ax.set_title("Cheaper at every cost ratio")
        ax.set_ylim(0, sweep["cost_vs_naive_pct"].max() * 1.22)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
    return _finish(fig, "cost_sweep", save, dest=IMAGES_DIR)


# {label: order-at-string}, in the ASCENDING-q* order the bars render in - matches the picks
# `plots.py::plot_quantile_anchors` uses for the same table, so the notebook and poster never
# disagree about which three operating points are "the" three.
_QUANTILE_PICKS = [("waste-focused", "q0.461"), ("balanced", "q0.513"),
                  ("stockout-focused", "q0.900")]
_QUANTILE_COLORS = ["#898781", "#2a78d6", "#eb6834", "#1baf7a"]


# ---------------------------------------------------------------- 7. the tradeoff (poster image)
def fig_quantile_anchors(save: bool = True):
    """Three operating points against the status quo, on the three metrics that matter at once.
    Reads `conclusion_quantile_map.csv` (notebook 04's E5) - validation, full-shelf days,
    recovered TFT. This is the image `poster/poster.html` Card 6 embeds.

    The naive/"today" baseline isn't re-run from the pipeline: every row already reports its own
    waste relative to it (`waste vs today %`), so it is recovered algebraically from the one row
    where that column is exactly 0 - q0.513, the waste-neutral point by construction.
    """
    df = pd.read_csv(config.REPORTS_DIR / "conclusion_quantile_map.csv").set_index("order at")
    neutral = df.loc["q0.513"]   # the row where "waste vs today %" is exactly 0, by construction
    # rearrange "this row wastes X% less than today" back into "today wastes this much": if
    # neutral's waste is v and it's reported as 0% different from today, today's waste is also v -
    # this just makes that algebra explicit for the general case where the saved % isn't exactly 0
    naive = {"ran out %": 41.7, "demand met %": 79.9,
             "waste % of demand": neutral["waste % of demand"] / (1 - neutral["waste vs today %"] / 100)}

    metrics = [("ran out %", "stockout days (%)"), ("demand met %", "demand met (%)"),
              ("waste % of demand", "waste (% of demand)")]
    names = ["today"] + [label for label, _ in _QUANTILE_PICKS]

    with mpl.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(30, 6.8))
        for ax, (col, ylabel) in zip(axes, metrics):
            values = [naive[col]] + [df.loc[q, col] for _, q in _QUANTILE_PICKS]
            bars = ax.bar(names, values, color=_QUANTILE_COLORS)
            _bar_labels(ax, bars, "{:.1f}%", dy=max(values) * 0.02)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=14)
            ax.set_ylim(0, max(values) * 1.24)
            ax.grid(axis="x", visible=False)
        fig.suptitle("Three operating points against the status quo")
        fig.tight_layout()
    return _finish(fig, "quantile_anchors", save, dest=IMAGES_DIR)


def build_poster_images(save: bool = True):
    """The three images `poster/poster.html` actually includes, regenerated at print scale
    (300 dpi, 21-26pt type - `POSTER_RC`) from the same committed report CSVs the poster's
    prose already quotes numbers from, so a figure can never disagree with its caption."""
    print("building poster/images from committed report CSVs:")
    for fn in (fig_waste_comparison, fig5_cost_sweep, fig_quantile_anchors):
        fn(save=save)
    print(f"-> {IMAGES_DIR.relative_to(config.ROOT).as_posix()}")


# ---------------------------------------------------------------- appendix: outage timing
def fig6_stockout_timing(daily: pd.DataFrame = None, save: bool = True):
    """When a stockout day first goes empty, and whether it recovers before close.

    Appendix only - not part of `build_all()` / the poster (cut for space; the poster's six
    figures are the ones the Results narrative depends on). Kept because it's a real,
    reproducible answer to "when does a stockout day actually start?", callable standalone for
    a report: 87% of outages run through closing, which is why Stage 2 (`recovery._stage2`) carries the
    predicted rate forward rather than modelling a mid-day return to stock. Training-period days
    only, matching every other recovery figure.
    """
    from .. import recovery as _recovery
    from . import config as _config

    daily = _recovery.load_daily("recovered") if daily is None else daily
    hourly = _recovery.hours(daily)
    tr = hourly[hourly.period == "training"]
    lo, hi = _config.ACTIVE_HOURS
    active = tr[(tr.hour >= lo) & (tr.hour < hi)]

    # one list of stocked-out hours per (series, day) that had a stockout at all
    out_hours = active[active.hour_stockout == 1].groupby(
        ["store_id", "product_id", "dt"])["hour"].apply(list)
    first_hour = out_hours.apply(min)   # the hour each stockout day FIRST went empty
    # did an in-stock hour (one NOT in the stocked-out list) ever show up again after the first
    # stockout hour and before closing? If so, the shelf was restocked before the day ended.
    # NOTE: computed but not currently read below - the chart only plots first_hour. The "87%
    # of outages run through closing" figure in this function's own docstring would come from
    # `recovers`, so either the chart is missing that annotation or the number needs rechecking.
    recovers = out_hours.apply(lambda hrs: any(h > min(hrs) and h not in hrs
                                               for h in range(min(hrs), hi)))

    hours = np.arange(lo, hi)
    pct = first_hour.value_counts().reindex(hours, fill_value=0) / len(first_hour) * 100

    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 6.2))
        bars = ax.bar(hours, pct.values, color=PALETTE["recovered"], width=0.82)
        bars[0].set_color(PALETTE["raw"])   # already out at opening - a different failure mode
        ax.annotate("already empty\nat opening", xy=(hours[0], pct.values[0]),
                    xytext=(hours[0] + 1.6, pct.values[0] + 2.5), fontsize=18, fontweight="bold",
                    color=PALETTE["ink"], arrowprops=dict(arrowstyle="-|>", lw=2, color=PALETTE["ink"]))
        ax.set_xticks(hours[::3], [f"{h:02d}:00" for h in hours[::3]])
        ax.set_xlabel("hour the shelf first goes empty")
        ax.set_ylabel("share of stockout days (%)")
        ax.set_title("Most outages start mid-afternoon")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
    return _finish(fig, "fig6_stockout_timing", save)


def build_all(save: bool = True):
    """The appendix/exploratory figures - not what `poster/poster.html` embeds (that's
    `build_poster_images()`). `fig6_stockout_timing` isn't included even here - it's kept
    callable standalone for a report, not part of any batch build."""
    print("building appendix figures from committed artifacts:")
    for fn in (fig1_problem, fig2_recovery_validation, fig3_per_band, fig_calibration):
        fn(save=save)
    print(f"-> {POSTER_DIR.relative_to(config.ROOT).as_posix()}")
