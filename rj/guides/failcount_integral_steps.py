"""
FailCount-specific visualization: Integration from histogram to FailCount to second integral

Shows:
1) Histogram h(x) - the raw distribution
2) FailCount F(x) = integral of h(x) - cumulative from right
3) Second integral A(x) = integral of F(x) - stacked FailCount area
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 日本語フォント設定
try:
    import japanize_matplotlib  # noqa: F401
    # japanize_matplotlib が成功した場合はスキップ
except ImportError:
    # フォント設定（フォールバック）
    import matplotlib.font_manager as fm
    try:
        # Windows の日本語フォント探索
        font_names = ['Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'Noto Sans CJK JP']
        for font_name in font_names:
            try:
                matplotlib.rcParams['font.family'] = font_name
                break
            except Exception:
                continue
    except Exception:
        pass

matplotlib.rcParams['axes.unicode_minus'] = False


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "FailCount の積分関係\n"
        "ヒストグラム → FailCount → 第2積分",
        fontsize=13,
        fontweight="bold",
    )

    # Data: simple histogram
    x_bins = np.array([1, 2, 3, 4, 5])
    h = np.array([10, 30, 50, 20, 5])  # histogram counts

    # Compute FailCount (right cumsum)
    F = np.array([np.sum(h[i:]) for i in range(len(h))])

    # Compute second integral (cumsum of FailCount from left)
    dx = 1  # bin width
    A = np.cumsum(F) * dx

    # =========================================================================
    # 左図：ヒストグラム h(x)
    # =========================================================================
    ax = axes[0]

    bar_width = 0.7
    bars = ax.bar(
        x_bins,
        h,
        width=bar_width,
        color="steelblue",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
        label="h(x): histogram",
    )

    # Label each bar with its height
    for xi, hi in zip(x_bins, h):
        ax.text(xi, hi + 2, f"{int(hi)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Shade under curve (to show "area")
    for xi, hi in zip(x_bins, h):
        rect = Rectangle((xi - bar_width / 2, 0), bar_width, hi, facecolor="steelblue", alpha=0.2)
        ax.add_patch(rect)

    ax.set_xlabel("パラメータ x", fontsize=11)
    ax.set_ylabel("個数", fontsize=11)
    ax.set_title("① ヒストグラム\nh(x)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.2, 5.8)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax.text(
        0.02,
        0.97,
        "h(x) = データの分布\n\nこれを右から足すと\nFailCount になる",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95),
        fontweight="bold",
    )

    # =========================================================================
    # 中央図：FailCount F(x)
    # =========================================================================
    ax = axes[1]

    # Plot FailCount as a decreasing curve (step-like)
    x_plot = np.repeat(x_bins, 2)
    F_plot = np.repeat(F, 2)
    F_plot = np.concatenate([[F_plot[0]], F_plot, [0]])
    x_plot = np.concatenate([[x_bins[0] - 1], x_plot, [x_bins[-1] + 1]])

    ax.plot(x_plot, F_plot, "r-", linewidth=2.5, marker="o", markersize=6, label="F(x): FailCount")
    ax.fill_between(x_plot, 0, F_plot, alpha=0.2, color="red", label="FailCount area")

    # Label each FailCount value
    for xi, Fi in zip(x_bins, F):
        ax.text(xi, Fi + 5, f"{int(Fi)}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="red")

    ax.set_xlabel("パラメータ x", fontsize=11)
    ax.set_ylabel("個数（累積）", fontsize=11)
    ax.set_title("② FailCount\nF(x) = ∫[x→∞] h(t)dt", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Relationship explanation
    relationship_text = (
        "F(x) = h(x)を右から足す\n\n"
        "F(1) = 10+30+50+20+5 = 115\n"
        "F(2) = 30+50+20+5 = 105\n"
        "F(3) = 50+20+5 = 75\n"
        "...\n\n"
        "微分: h(x) = -dF/dx"
    )
    ax.text(
        0.98,
        0.5,
        relationship_text,
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95),
        fontweight="bold",
        family="monospace",
    )

    # =========================================================================
    # 右図：FailCount の積分 A(x)
    # =========================================================================
    ax = axes[2]

    # Plot second integral
    x_plot2 = np.repeat(x_bins, 2)
    A_plot2 = np.repeat(A, 2)
    A_plot2 = np.concatenate([[0], A_plot2, [A[-1]]])
    x_plot2 = np.concatenate([[x_bins[0] - 1], x_plot2, [x_bins[-1] + 1]])

    ax.plot(x_plot2, A_plot2, "g-", linewidth=2.5, marker="s", markersize=6, label="A(x): 第2積分")
    ax.fill_between(x_plot2, 0, A_plot2, alpha=0.2, color="green")

    # Label each value
    for xi, Ai in zip(x_bins, A):
        ax.text(xi, Ai + 15, f"{int(Ai)}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="darkgreen")

    ax.set_xlabel("パラメータ x", fontsize=11)
    ax.set_ylabel("累積量", fontsize=11)
    ax.set_title("③ FailCount の積分\nA(x) = ∫ F(u)du", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, max(A_plot2) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    second_integral_text = (
        "F(x)をさらに積分すると\n"
        "新しい累積量になる\n\n"
        "A(1) = 115\n"
        "A(2) = 115+105 = 220\n"
        "A(3) = 220+75 = 295\n"
        "...\n\n"
        "ヒストグラムには\n"
        "戻らない！"
    )
    ax.text(
        0.02,
        0.97,
        second_integral_text,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95),
        fontweight="bold",
        family="monospace",
    )

    out = "failcount_integral_steps.png"
    fig.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

