"""Static report figures (matplotlib), one consistent visual system.

Palette: the validated reference categorical order (blue, orange, aqua,
yellow, magenta, green, violet, red) on a near-white surface; thin bars
with a surface gap, hairline solid grid, legend for >= 2 series.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _round_bars(ax, bars, radius=0.012):
    """Rounded data-end, square baseline: draw rounded patch, then cover base."""
    for bar in bars:
        x, y, w, h = bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()
        if h <= 0:
            continue
        color = bar.get_facecolor()
        bar.set_visible(False)
        r = min(radius, h / 2)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    linewidth=0, facecolor=color, mutation_aspect=1))
        ax.add_patch(plt.Rectangle((x, y), w, h / 2, linewidth=0, facecolor=color))


def grouped_bars(path, categories, series: dict, title, ylabel, ylim=(0, 1), subtitle=None,
                 value_labels=False, figsize=(10, 4.6), rotate=None):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    group_w = 0.8
    bar_w = min(group_w / n, 0.22)
    gap = 0.015
    x = np.arange(len(categories))
    for k, (name, values) in enumerate(series.items()):
        offs = (k - (n - 1) / 2) * bar_w
        vals = [v if v is not None else 0 for v in values]
        ax.bar(x + offs, vals, width=bar_w - gap, color=SERIES[k % len(SERIES)], label=name, zorder=3,
               linewidth=0)
        if value_labels:
            for xi, v in zip(x + offs, vals):
                ax.text(xi, v + 0.01 * (ylim[1] - ylim[0]), f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=TEXT_2)
    ax.set_xticks(x)
    tilt = (len(categories) >= 6) if rotate is None else rotate
    ax.set_xticklabels(categories, rotation=20 if tilt else 0, ha="right" if tilt else "center", fontsize=8.5 if not tilt else 10)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)
    if n >= 2:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def horizontal_bars(path, labels, values, title, xlabel, highlight=None, subtitle=None, xlim=None,
                    errors=None, figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))[::-1]
    colors = [SERIES[1] if (highlight and lab in highlight) else SERIES[0] for lab in labels]
    ax.barh(y, values, height=0.55, color=colors, zorder=3, linewidth=0)
    if errors is not None:
        lo = [v - e[0] for v, e in zip(values, errors)]
        hi = [e[1] - v for v, e in zip(values, errors)]
        ax.errorbar(values, y, xerr=[lo, hi], fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=3, zorder=4)
    span = (xlim[1] - xlim[0]) if xlim else max(abs(v) for v in values)
    for i, (yi, v) in enumerate(zip(y, values)):
        # place the label beyond the bar end or the whisker, whichever is further out
        if errors is not None:
            edge = errors[i][0] if v < 0 else errors[i][1]
        else:
            edge = v
        pad = 0.012 * span
        ax.text(edge + (pad if v >= 0 else -pad), yi, f"{v:+.3f}" if xlim and xlim[0] < 0 else f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    if xlim:
        ax.set_xlim(*xlim)
        if xlim[0] < 0:
            ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def confusion(path, matrix, labels, title, subtitle=None):
    m = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=max(1, m.max()))
    short = [l.replace("_PRECEDENT", " PREC.").replace("_", " ").title() for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Gold intent")
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if m[i, j] > 0.55 * m.max() else TEXT)
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def histogram(path, values, title, xlabel, bins=40, log=False, subtitle=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.log10(values) if log else values
    ax.hist(data, bins=bins, color=SERIES[0], rwidth=0.85, zorder=3)
    if log:
        ticks = [3, 3.5, 4, 4.5, 5, 5.5, 6]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{10**t/1000:.0f}K" if 10**t < 1e6 else f"{10**t/1e6:.1f}M" for t in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Judgments")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
