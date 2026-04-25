"""Two-segment transmission line example (50 ohm -> 100 ohm, 16:9)."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


def draw_circuit(ax):
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("回路例: 入力 -> 50Ω -> 伝送路1(50Ω) -> 伝送路2(100Ω) -> 観測点 -> 50Ω終端", fontsize=14, fontweight="bold")

    y = 2.5

    # Source box
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.3, y - 0.8),
            1.6,
            1.6,
            boxstyle="round,pad=0.1",
            facecolor="#b3e5fc",
            edgecolor="#1976d2",
            linewidth=2,
        )
    )
    ax.text(1.1, y + 0.2, "$V_s$", ha="center", fontsize=13, fontweight="bold")
    ax.text(1.1, y - 0.35, "入力", ha="center", fontsize=10)

    # Wire to Rs
    ax.plot([1.9, 2.4], [y, y], color="black", linewidth=2)

    # Rs box
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (2.4, y - 0.45),
            1.8,
            0.9,
            boxstyle="round,pad=0.1",
            facecolor="#fff9c4",
            edgecolor="#f9a825",
            linewidth=2,
        )
    )
    ax.text(3.3, y, "$R_s=50\\,\\Omega$", ha="center", va="center", fontsize=11)

    # Wire to line
    ax.plot([4.2, 4.8], [y, y], color="black", linewidth=2)

    # Transmission line 1 (50 ohm)
    x0, xj = 4.8, 8.0
    ax.plot([x0, xj], [y + 0.35, y + 0.35], color="black", linewidth=2.5)
    ax.plot([x0, xj], [y - 0.35, y - 0.35], color="black", linewidth=2.5)
    ax.plot([x0, x0], [y - 0.35, y + 0.35], color="black", linewidth=1.6)
    ax.plot([xj, xj], [y - 0.35, y + 0.35], color="black", linewidth=1.6)
    ax.text(
        6.4,
        y,
        "$Z_{01}=50\\,\\Omega$,  $T_{D1}=1\\,ns$",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffde7", edgecolor="#fbc02d", alpha=0.95),
    )

    # Transmission line 2 (100 ohm)
    x1 = 10.8
    ax.plot([xj, x1], [y + 0.35, y + 0.35], color="#283593", linewidth=2.5)
    ax.plot([xj, x1], [y - 0.35, y - 0.35], color="#283593", linewidth=2.5)
    ax.plot([x1, x1], [y - 0.35, y + 0.35], color="#283593", linewidth=1.6)
    ax.text(
        9.4,
        y,
        "$Z_{02}=100\\,\\Omega$,  $T_{D2}=1\\,ns$",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8eaf6", edgecolor="#3949ab", alpha=0.95),
    )

    # Junction marker
    ax.plot(xj, y, "D", color="#ef6c00", markersize=8)
    ax.annotate(
        "$\\Gamma_{12}=+0.333$",
        xy=(xj, y),
        xytext=(7.3, 3.9),
        fontsize=10,
        color="#e65100",
        arrowprops=dict(arrowstyle="-|>", color="#ef6c00", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ef6c00", alpha=0.95),
    )

    # Observation point
    ax.plot(10.8, y, "o", color="#8e24aa", markersize=11)
    ax.annotate(
        "観測点",
        xy=(10.8, y),
        xytext=(9.8, 4.1),
        fontsize=10,
        color="#6a1b9a",
        arrowprops=dict(arrowstyle="-|>", color="#8e24aa", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f3e5f5", edgecolor="#8e24aa", alpha=0.95),
    )

    # RL
    ax.plot([10.8, 11.2], [y, y], color="black", linewidth=2)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (11.2, y - 0.45),
            1.8,
            0.9,
            boxstyle="round,pad=0.1",
            facecolor="#fff9c4",
            edgecolor="#f9a825",
            linewidth=2,
        )
    )
    ax.text(12.1, y, "$R_L=50\\,\\Omega$", ha="center", va="center", fontsize=11)

    # Ground
    ax.plot([13.0, 13.5], [y, y], color="black", linewidth=2)
    ax.plot([13.5, 13.5], [y, y - 0.25], color="black", linewidth=2)
    ax.plot([13.2, 13.8], [y - 0.25, y - 0.25], color="black", linewidth=2)
    ax.plot([13.27, 13.73], [y - 0.42, y - 0.42], color="black", linewidth=1.5)
    ax.plot([13.35, 13.65], [y - 0.57, y - 0.57], color="black", linewidth=1.2)

    # Arrows and summary
    ax.annotate("", xy=(10.5, y + 1.2), xytext=(2.1, y + 1.2), arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.2))
    ax.text(6.2, y + 1.45, "前進波", color="#1565c0", fontsize=11)
    ax.annotate("", xy=(7.9, y - 1.0), xytext=(10.2, y - 1.0), arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=2.0))
    ax.text(9.2, y - 1.3, "小さな反射波", color="#c62828", fontsize=10, ha="center")

    ax.text(
        7.5,
        0.6,
        "$\\Gamma_{12}=(100-50)/(100+50)=+0.333$,  $\\Gamma_L=(50-100)/(50+100)=-0.333$",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e1", edgecolor="#ef6c00", alpha=0.95),
    )


def draw_waveform(ax):
    vs = 1.0
    rs = 50.0
    z1 = 50.0
    z2 = 100.0
    zl = 50.0
    td1_ns = 1.0
    td2_ns = 1.0

    gamma_12 = (z2 - z1) / (z2 + z1)
    gamma_21 = (z1 - z2) / (z1 + z2)
    gamma_l = (zl - z2) / (zl + z2)
    tau_12 = 2 * z2 / (z1 + z2)

    # Initial launched wave in Z1.
    a1 = vs * z1 / (rs + z1)
    # Forward wave entering Z2.
    a_load = tau_12 * a1

    t_ns = np.linspace(-0.5, 8.0, 3000)
    v_obs = np.zeros_like(t_ns)

    # At load/observation point, each arrival adds (1+gamma_l)*A_n.
    t0 = td1_ns + td2_ns
    round_trip = 2 * td2_ns
    a_n = a_load
    for n in range(7):
        arrival = t0 + n * round_trip
        dv = (1 + gamma_l) * a_n
        v_obs += dv * (t_ns >= arrival)
        a_n *= gamma_l * gamma_21

    v_ideal = np.where(t_ns >= t0, 0.5, 0.0)

    ax.plot(t_ns, v_obs, color="#6a1b9a", linewidth=2.8, label="観測点電圧")
    ax.plot(t_ns, v_ideal, color="#90a4ae", linestyle="--", linewidth=1.8, label="整合時の基準(0.5V)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(t0, color="gray", linestyle="--", linewidth=1.5)
    ax.axvline(t0 + round_trip, color="#2e7d32", linestyle=":", linewidth=1.5)

    ax.set_title("観測点波形: 50Ω->100Ωのミスマッチで反射ステップが出る", fontsize=14, fontweight="bold")
    ax.set_xlabel("時間 (ns)", fontsize=12)
    ax.set_ylabel("電圧 (V)", fontsize=12)
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-0.05, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)

    ax.annotate(
        "$t=T_{D1}+T_{D2}=2\\,ns$ で初回到達",
        xy=(t0, (1 + gamma_l) * a_load),
        xytext=(2.7, 0.58),
        fontsize=10,
        color="#37474f",
        arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eceff1", edgecolor="#607d8b", alpha=0.95),
    )

    ax.annotate(
        "$t=4\\,ns$ で反射再到達",
        xy=(t0 + round_trip, v_obs[np.searchsorted(t_ns, t0 + round_trip)]),
        xytext=(4.8, 0.54),
        fontsize=10,
        color="#1b5e20",
        arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f5e9", edgecolor="#2e7d32", alpha=0.95),
    )

    ax.text(
        5.3,
        0.09,
        "$V_{obs}$ は最初 $\\approx0.444\\,V$、その後 0.5V に漸近",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ede7f6", edgecolor="#5e35b1", alpha=0.95),
    )


def main():
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    draw_circuit(ax0)
    draw_waveform(ax1)

    fig.suptitle("50Ω->100Ω を含む2段伝送路の基本例（ミスマッチあり）", fontsize=18, fontweight="bold")

    out_path = Path(__file__).with_name("reflection_matched_50ohm_example.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
