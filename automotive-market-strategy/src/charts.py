"""Charts for the strategy report (saved to outputs/charts/)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, LIGHT, ORANGE, GREY = "#2f5d8a", "#9bb7d4", "#e0a458", "#8c8c8c"
SCEN_COLORS = {"slow": LIGHT, "base": BLUE, "fast": ORANGE}


def _finish(fig, ax, path: Path):
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ev_share_forecast(ev_monthly, forecast, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ev_monthly["t"], 100 * ev_monthly["ev_share"], color=GREY, lw=1, label="Actual (monthly)")
    for name, g in forecast.groupby("scenario", sort=False):
        ax.plot(g["year"] + 0.5, g["ev_share_pct"], marker="o", ms=3, color=SCEN_COLORS[name],
                label=f"{name} (saturation {g['saturation_L'].iloc[0]:.0%})")
    ax.set_ylabel("EV share of new car sales (%)")
    ax.set_title("EV adoption: history and 2025-2030 scenarios")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, out / "ev_share_forecast.png")


def ev_vs_petrol_units(units, forecast, out: Path):
    base = forecast[forecast["scenario"] == "base"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(units.index, units["Petrol"] / 1e6, color=ORANGE, marker="o", ms=3, label="Petrol (actual)")
    ax.plot(base["year"], base["petrol_units"] / 1e6, color=ORANGE, ls="--", label="Petrol (base forecast)")
    ax.plot(units.index, units["EV"] / 1e6, color=BLUE, marker="o", ms=3, label="EV (actual)")
    ax.plot(base["year"], base["ev_units"] / 1e6, color=BLUE, ls="--", label="EV (base forecast)")
    ax.set_ylabel("Units sold (millions)")
    ax.set_title("EV vs petrol sales")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, out / "ev_vs_petrol_units.png")


def ev_segment_share(seg, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [BLUE, ORANGE, "#6aa36f", "#c0504d", GREY]
    for color, brand in zip(colors, ["Company X"] + [c for c in seg.columns if c != "Company X"]):
        ax.plot(seg.index, seg[brand], marker="o", ms=3, color=color, lw=2.5 if brand == "Company X" else 1.2,
                label=brand)
    ax.set_ylabel("% of EV segment")
    ax.set_ylim(0, 100)
    ax.set_title("Who sells the country's EVs")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _finish(fig, ax, out / "ev_segment_share.png")


def ev_model_mix(mix, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(mix))
    for color, col in zip([BLUE, LIGHT, ORANGE], ["X-City EV", "X-Range EV", "X-Pro EV"]):
        ax.bar(mix.index, mix[col], bottom=bottom, color=color, label=col)
        bottom += mix[col].to_numpy()
    ax.set_ylabel("% of Company X EV sales")
    ax.set_title("Company X EV sales by model")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    _finish(fig, ax, out / "ev_model_mix.png")


def review_heatmap(scores, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    data = scores.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn", vmin=2.5, vmax=4.7, aspect="auto")
    ax.set_xticks(range(len(scores.columns)), [c.replace("_", " ") for c in scores.columns])
    ax.set_yticks(range(len(scores.index)), scores.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Customer scores by model (out of 5)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "review_scores.png", dpi=130)
    plt.close(fig)
