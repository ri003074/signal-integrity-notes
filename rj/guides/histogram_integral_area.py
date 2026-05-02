"""
Shows exactly where the "area from integrating histogram" appears in FailCount.
Single focused figure: histogram right-side area = FailCount value at that x.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

matplotlib.rcParams["axes.unicode_minus"] = False


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "Area from integrating histogram = FailCount value\n"
        "histogram right-side area  =  F(x) height",
        fontsize=13,
        fontweight="bold",
    )

    x_bins = np.array([1, 2, 3, 4, 5])
    h = np.array([10, 30, 50, 20, 5])
    F = np.array([np.sum(h[i:]) for i in range(len(h))])

    bar_width = 0.7
    x0 = 3  # threshold we highlight
    i0 = list(x_bins).index(x0)
    F_at_x0 = F[i0]  # = 75

    # =========================================================
    # Left: Histogram — highlight area to the RIGHT of x0
    # =========================================================
    ax = axes[0]

    for i, (xi, hi) in enumerate(zip(x_bins, h)):
        if xi >= x0:
            color = "tomato"
            edge = "darkred"
            lw = 2.0
        else:
            color = "lightgray"
            edge = "black"
            lw = 1.0
        ax.bar(xi, hi, width=bar_width, color=color, edgecolor=edge,
               linewidth=lw, alpha=0.85)
        ax.text(xi, hi + 1.5, f"{hi}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="darkred" if xi >= x0 else "gray")

    # vertical line at x0
    ax.axvline(x0 - bar_width / 2, color="darkred", lw=2.5, ls="--")

    # annotation: sum of red bars
    ax.annotate(
        f"Right-side area\n= {h[i0]}+{h[i0+1]}+{h[i0+2]}\n= {F_at_x0}",
        xy=(4.0, 40),
        xytext=(1.2, 48),
        fontsize=10,
        fontweight="bold",
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="darkred"),
    )

    ax.text(
        x0 - bar_width / 2 - 0.1,
        55,
        f"x = {x0}",
        ha="right",
        fontsize=10,
        color="darkred",
        fontweight="bold",
    )

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Count h(x)", fontsize=11)
    ax.set_title(
        "1) Histogram h(x)\n"
        "Red bars = area for x >= 3",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0.3, 6.0)
    ax.set_ylim(0, 62)
    ax.grid(True, alpha=0.3, axis="y")

    # formula box at bottom
    ax.text(
        0.5, 0.04,
        f"F(x=3) = integral[3 to +inf] h(t)dt = {F_at_x0}",
        transform=ax.transAxes,
        ha="center", fontsize=10,
        fontweight="bold", color="darkred",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="darkred"),
    )

    # =========================================================
    # Right: FailCount — show that F(x0) equals the area above
    # =========================================================
    ax = axes[1]

    x_step = np.concatenate([[0.5], np.repeat(x_bins, 2), [5.5]])
    F_step = np.concatenate([[F[0]], np.repeat(np.append(F, 0), 2)[1:-1], [0]])

    ax.plot(x_step, F_step, "r-", linewidth=2.5, label="F(x): FailCount")
    ax.fill_between(x_step, 0, F_step, alpha=0.08, color="red")

    # Mark all FailCount points
    for xi, Fi in zip(x_bins, F):
        ax.plot(xi, Fi, "o", color="gray", markersize=6)
        ax.text(xi, Fi + 3, f"{Fi}", ha="center", va="bottom",
                fontsize=9, color="gray")

    # Highlight the specific point F(x0)
    ax.plot(x0, F_at_x0, "o", color="darkred", markersize=11, zorder=5)

    # horizontal dashed line at F(x0)
    ax.plot([0.3, x0], [F_at_x0, F_at_x0],
            color="darkred", lw=2, ls="--")
    ax.text(0.35, F_at_x0 + 3, f"F(3) = {F_at_x0}",
            fontsize=11, color="darkred", fontweight="bold")

    # vertical dashed line at x0
    ax.axvline(x0, color="darkred", lw=2.5, ls="--")
    ax.text(x0 + 0.1, 5, f"x = {x0}",
            fontsize=10, color="darkred", fontweight="bold")

    # annotation linking to the area
    ax.annotate(
        f"This height = {F_at_x0}\n= right-side area\n  in histogram",
        xy=(x0, F_at_x0),
        xytext=(3.8, 95),
        fontsize=10,
        fontweight="bold",
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="darkred"),
    )

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("FailCount F(x)", fontsize=11)
    ax.set_title(
        "2) FailCount F(x)\n"
        "F(3) = right-side area of histogram",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0.3, 6.0)
    ax.set_ylim(0, 130)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax.text(
        0.5, 0.04,
        "FailCount value = area of red bars in left panel",
        transform=ax.transAxes,
        ha="center", fontsize=10,
        fontweight="bold", color="darkred",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="darkred"),
    )

    out = "histogram_integral_area.png"
    fig.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

