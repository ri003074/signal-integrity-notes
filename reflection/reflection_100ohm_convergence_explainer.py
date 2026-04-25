"""Visual explainer for convergence from 0.444 V to 0.5 V in 50->100->50 line."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


def draw_circuit(ax):
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("回路: 入力 -> 50Ω -> 線路1(50Ω) -> 線路2(100Ω) -> 観測点 -> 50Ω終端", fontsize=13, fontweight="bold")

    y = 2.5

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.3, y - 0.7),
            1.6,
            1.4,
            boxstyle="round,pad=0.1",
            facecolor="#b3e5fc",
            edgecolor="#1976d2",
            linewidth=2,
        )
    )
    ax.text(1.1, y + 0.15, "$V_s$", ha="center", fontsize=12, fontweight="bold")
    ax.text(1.1, y - 0.3, "入力", ha="center", fontsize=10)

    ax.plot([1.9, 2.4], [y, y], color="black", linewidth=2)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (2.4, y - 0.4),
            1.8,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor="#fff9c4",
            edgecolor="#f9a825",
            linewidth=2,
        )
    )
    ax.text(3.3, y, "$R_s=50\\,\\Omega$", ha="center", va="center", fontsize=10)

    ax.plot([4.2, 4.8], [y, y], color="black", linewidth=2)

    x0, xj, x1 = 4.8, 8.0, 10.8

    ax.plot([x0, xj], [y + 0.35, y + 0.35], color="black", linewidth=2.3)
    ax.plot([x0, xj], [y - 0.35, y - 0.35], color="black", linewidth=2.3)
    ax.plot([x0, x0], [y - 0.35, y + 0.35], color="black", linewidth=1.4)
    ax.plot([xj, xj], [y - 0.35, y + 0.35], color="black", linewidth=1.4)
    ax.text(6.4, y, "$Z_{01}=50\\,\\Omega$", ha="center", va="center", fontsize=10)

    ax.plot([xj, x1], [y + 0.35, y + 0.35], color="#283593", linewidth=2.3)
    ax.plot([xj, x1], [y - 0.35, y - 0.35], color="#283593", linewidth=2.3)
    ax.plot([x1, x1], [y - 0.35, y + 0.35], color="#283593", linewidth=1.4)
    ax.text(9.4, y, "$Z_{02}=100\\,\\Omega$", ha="center", va="center", fontsize=10)

    ax.plot(xj, y, "D", color="#ef6c00", markersize=8)
    ax.annotate(
        "$\\Gamma_{12}=+0.333$",
        xy=(xj, y),
        xytext=(7.0, 3.9),
        fontsize=10,
        color="#e65100",
        arrowprops=dict(arrowstyle="-|>", color="#ef6c00", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ef6c00", alpha=0.95),
    )

    ax.plot(x1, y, "o", color="#8e24aa", markersize=9)
    ax.annotate(
        "観測点",
        xy=(x1, y),
        xytext=(10.3, 4.05),
        fontsize=9,
        color="#6a1b9a",
        arrowprops=dict(arrowstyle="-|>", color="#8e24aa", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3e5f5", edgecolor="#8e24aa", alpha=0.95),
    )

    ax.plot([x1, 11.2], [y, y], color="black", linewidth=2)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (11.2, y - 0.4),
            1.8,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor="#fff9c4",
            edgecolor="#f9a825",
            linewidth=2,
        )
    )
    ax.text(12.1, y, "$R_L=50\\,\\Omega$", ha="center", va="center", fontsize=10)

    ax.plot([13.0, 13.5], [y, y], color="black", linewidth=2)
    ax.plot([13.5, 13.5], [y, y - 0.2], color="black", linewidth=2)
    ax.plot([13.2, 13.8], [y - 0.2, y - 0.2], color="black", linewidth=2)
    ax.plot([13.27, 13.73], [y - 0.35, y - 0.35], color="black", linewidth=1.4)
    ax.plot([13.35, 13.65], [y - 0.48, y - 0.48], color="black", linewidth=1.0)

    ax.text(
        7.5,
        0.55,
        "$\\Gamma_L=-0.333,\\;\\Gamma_{21}=-0.333\\Rightarrow\\Gamma_L\\Gamma_{21}=1/9$",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff8e1", edgecolor="#ef6c00", alpha=0.95),
    )


def compute_levels(vs=1.0):
    rs = 50.0
    z1 = 50.0
    z2 = 100.0
    zl = 50.0

    gamma_l = (zl - z2) / (zl + z2)
    gamma_21 = (z1 - z2) / (z1 + z2)
    tau_12 = 2 * z2 / (z1 + z2)

    a1 = vs * z1 / (rs + z1)
    a_load = tau_12 * a1

    first = (1 + gamma_l) * a_load  # 0.444...
    ratio = gamma_l * gamma_21      # 1/9

    increments = [first * (ratio ** n) for n in range(6)]
    levels = np.cumsum(increments)
    return first, ratio, increments, levels


def draw_waveform(ax):
    first, ratio, increments, levels = compute_levels(vs=1.0)

    t0 = 2.0
    dt = 2.0
    arrivals = np.array([t0 + n * dt for n in range(len(increments))])

    t = np.linspace(0, 14, 3000)
    v = np.zeros_like(t)
    for tm, dv in zip(arrivals, increments):
        v += dv * (t >= tm)

    ax.plot(t, v, color="#6a1b9a", linewidth=2.8, label="観測点電圧")
    ax.axhline(0.5, color="#546e7a", linestyle="--", linewidth=1.6, label="最終値 0.5 V")
    ax.axhline(first, color="#ef6c00", linestyle=":", linewidth=1.8, label=f"初回 {first:.3f} V")

    for tm in arrivals[:4]:
        ax.axvline(tm, color="#cfd8dc", linestyle=":", linewidth=1.0)

    ax.plot(arrivals[:5], levels[:5], "o", color="#4a148c", markersize=5)

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 0.56)
    ax.set_xlabel("時間 (ns)", fontsize=11)
    ax.set_ylabel("観測点電圧 (V)", fontsize=11)
    ax.set_title("① 観測点は 0.444V から始まり、反射往復ごとに 0.5V へ近づく", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax.annotate(
        "初回到達: 0.444V",
        xy=(arrivals[0], levels[0]),
        xytext=(2.8, 0.49),
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color="#ef6c00", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ef6c00", alpha=0.95),
    )
    ax.annotate(
        "以後の増分は毎回 1/9 倍",
        xy=(arrivals[1], levels[1]),
        xytext=(6.6, 0.35),
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#2e7d32", alpha=0.95),
    )


def draw_increment_panel(ax):
    first, ratio, increments, levels = compute_levels(vs=1.0)

    n = np.arange(len(increments))
    labels = [f"n={i}" for i in n]

    ax.bar(n, increments, color="#90caf9", edgecolor="#1565c0")
    ax.set_xticks(n)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(increments) * 1.15)
    ax.set_title("② 追加される電圧増分 ΔV", fontsize=12, fontweight="bold")
    ax.set_ylabel("ΔV (V)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    for i, dv in enumerate(increments[:4]):
        ax.text(i, dv + 0.005, f"{dv:.3f}", ha="center", va="bottom", fontsize=8)

    explain = (
        "$\\Delta V_0 = 0.444$ V\n"
        "$\\Delta V_n = \\Delta V_0 (1/9)^n$\n"
        "$V_{obs}(\\infty)=\\sum\\Delta V_n=0.5$ V"
    )
    ax.text(
        0.98,
        0.95,
        explain,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ede7f6", edgecolor="#5e35b1", alpha=0.95),
    )


def main():
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.45, 1.0])

    ax_circuit = fig.add_subplot(gs[0, :])
    ax_wave = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    draw_circuit(ax_circuit)
    draw_waveform(ax_wave)
    draw_increment_panel(ax_bar)

    fig.suptitle("なぜ 0.444V から 0.5V へ上がるのか（50Ω -> 100Ω -> 50Ω）", fontsize=17, fontweight="bold")

    out_path = Path(__file__).with_name("reflection_100ohm_convergence_explainer.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
