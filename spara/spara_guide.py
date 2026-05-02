"""
S-parameters (Sパラメータ) 解説グラフ生成スクリプト
2ポートネットワーク（例：2次バターワースLPF fc=1GHz）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import signal
import japanize_matplotlib  # noqa: F401  日本語フォント自動設定


# ─────────────────────────────────────────────────────────────
# Sパラメータデータ生成（2次バターワースLPF）
# ─────────────────────────────────────────────────────────────
def create_sparam_data(f_c: float = 1e9, order: int = 2):
    """バターワースLPFのSパラメータを生成"""
    b, a = signal.butter(order, 2 * np.pi * f_c, btype="low", analog=True)
    freqs = np.logspace(7, 11, 2000)  # 10 MHz ～ 100 GHz
    w = 2 * np.pi * freqs
    _, H = signal.freqs(b, a, worN=w)

    # S21（通過特性）
    S21_mag_dB = 20 * np.log10(np.abs(H))
    S21_phase_deg = np.unwrap(np.angle(H)) * 180 / np.pi

    # S11（反射特性）：無損失ネットワーク近似 |S11|^2 + |S21|^2 = 1
    S11_lin = np.sqrt(np.clip(1 - np.abs(H) ** 2, 0, 1))
    S11_mag_dB = 20 * np.log10(np.clip(S11_lin, 1e-15, 1))

    # グループ遅延 = -dφ/dω
    phase_rad = np.unwrap(np.angle(H))
    gd_s = -np.gradient(phase_rad, w)  # 単位: 秒
    gd_ns = np.clip(gd_s * 1e9, 0, None)  # 単位: ns（負値をクリップ）

    return freqs, S21_mag_dB, S11_mag_dB, S21_phase_deg, gd_ns


# ─────────────────────────────────────────────────────────────
# ① 概念図
# ─────────────────────────────────────────────────────────────
def draw_2port_concept(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("① 2ポートネットワークとSパラメータの定義", fontsize=11)

    # DUT ボックス
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (3.8, 1.3),
            2.4,
            1.4,
            boxstyle="round,pad=0.15",
            facecolor="lightyellow",
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.text(
        5.0,
        2.0,
        "DUT\n（被測定デバイス）",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Port 1 / Port 2
    for x0, label in [(0.2, "Port 1"), (8.8, "Port 2")]:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, 1.5),
                1.0,
                1.0,
                boxstyle="round,pad=0.1",
                facecolor="lightblue",
                edgecolor="steelblue",
                linewidth=1.5,
            )
        )
        ax.text(x0 + 0.5, 2.0, label, ha="center", va="center", fontsize=9, fontweight="bold")

    # a1: Port1 → DUT（入射波、上段→）
    ax.annotate(
        "", xy=(3.8, 2.5), xytext=(1.2, 2.5), arrowprops=dict(arrowstyle="->", color="blue", lw=2.0)
    )
    ax.text(2.5, 2.75, "a1  入射波", color="blue", fontsize=9, ha="center")

    # b1: DUT → Port1（反射波、下段←）
    ax.annotate(
        "", xy=(1.2, 1.5), xytext=(3.8, 1.5), arrowprops=dict(arrowstyle="->", color="red", lw=2.0)
    )
    ax.text(2.5, 1.22, "b1  反射波  →  S11 = b1/a1", color="red", fontsize=9, ha="center")

    # b2: DUT → Port2（透過波、上段→）
    ax.annotate(
        "",
        xy=(8.8, 2.5),
        xytext=(6.2, 2.5),
        arrowprops=dict(arrowstyle="->", color="green", lw=2.0),
    )
    ax.text(7.5, 2.75, "b2  透過波  →  S21 = b2/a1", color="green", fontsize=9, ha="center")

    # a2: Port2 → DUT（逆入射、下段←、破線）
    ax.annotate(
        "",
        xy=(6.2, 1.5),
        xytext=(8.8, 1.5),
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.5, linestyle="dashed"),
    )
    ax.text(7.5, 1.22, "a2  逆入射（通常 = 0）", color="purple", fontsize=9, ha="center")

    # S-matrix まとめ
    ax.text(
        5.0,
        0.45,
        "[ b1 ]   [ S11  S12 ] [ a1 ]          "
        "S11: ポート1反射（入力反射損失）\n"
        "[ b2 ] = [ S21  S22 ] [ a2 ]          "
        "S21: ポート1→2透過（挿入損失）",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange", alpha=0.9),
    )


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────
def main():
    F_C = 1e9  # カットオフ 1 GHz

    freqs, S21_dB, S11_dB, S21_phase, gd_ns = create_sparam_data(f_c=F_C)
    freqs_GHz = freqs / 1e9

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Sパラメータ (S-parameters) 解説\n" "2ポートネットワーク例：2次バターワースLPF  fc = 1 GHz",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", edgecolor="purple", alpha=0.85),
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig, hspace=0.70, top=0.92, bottom=0.04, height_ratios=[1.2, 1.5, 1.0, 1.0]
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # ─── ① 概念図 ────────────────────────────────────────
    draw_2port_concept(ax0)

    # ─── ② 振幅特性 (dB) ─────────────────────────────────
    ax1.semilogx(
        freqs_GHz, S21_dB, "-", color="blue", linewidth=2, label="S21  透過特性（挿入損失）"
    )
    ax1.semilogx(
        freqs_GHz, S11_dB, "-", color="red", linewidth=2, label="S11  反射特性（入力反射損失）"
    )
    ax1.axhline(-3, color="gray", linestyle="--", linewidth=1.0, label="-3 dB ライン")
    ax1.axvline(
        F_C / 1e9, color="orange", linestyle="--", linewidth=1.5, label=f"fc = {F_C/1e9:.1f} GHz"
    )
    ax1.set_xlabel("周波数 (GHz)")
    ax1.set_ylabel("振幅 (dB)")
    ax1.set_title("② S21・S11 振幅特性（大きさ）")
    ax1.set_ylim(-65, 8)
    ax1.set_xlim(freqs_GHz[0], freqs_GHz[-1])
    ax1.grid(True, which="both", alpha=0.4)
    ax1.legend(fontsize=8, loc="lower left")

    # ② アノテーション
    ax1.annotate(
        "パスバンド\nS21 ≈ 0 dB（ほぼ通過）\nS11 << 0 dB（反射少）",
        xy=(0.1, -0.5),
        xytext=(0.013, -28),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        fontsize=8,
        color="blue",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", alpha=0.92
        ),
    )
    ax1.annotate(
        "ストップバンド\nS21 << 0 dB（大きく減衰）\nS11 ≈ 0 dB（ほぼ全反射）",
        xy=(8.0, -48),
        xytext=(3.5, -40),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=8,
        color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", edgecolor="red", alpha=0.92),
    )
    ax1.annotate(
        f"-3 dB 点 = fc = {F_C/1e9:.1f} GHz\n電力が半分（電圧は 1/√2）になる周波数",
        xy=(F_C / 1e9, -3),
        xytext=(F_C / 1e9 * 3.5, -18),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
        fontsize=8,
        color="darkorange",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange", alpha=0.92
        ),
    )
    ax1.text(
        0.99,
        0.97,
        "0 dB   = 信号がそのまま通過\n-3 dB  = 電力が半分\n-20 dB = 電力が 1/100\n-40 dB = 電力が 1/10000",
        transform=ax1.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange", alpha=0.95
        ),
    )

    # ─── ③ 位相特性 ──────────────────────────────────────
    ax2.semilogx(freqs_GHz, S21_phase, "-", color="blue", linewidth=2, label="S21 位相")
    ax2.axvline(
        F_C / 1e9, color="orange", linestyle="--", linewidth=1.5, label=f"fc = {F_C/1e9:.1f} GHz"
    )
    ax2.axhline(-90, color="gray", linestyle=":", linewidth=1.0, label="-90° ライン")
    ax2.set_xlabel("周波数 (GHz)")
    ax2.set_ylabel("位相 (°)")
    ax2.set_title("③ S21 位相特性")
    ax2.set_xlim(freqs_GHz[0], freqs_GHz[-1])
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend(fontsize=8)

    # ③ アノテーション
    fc_idx = np.argmin(np.abs(freqs - F_C))
    ax2.annotate(
        "低周波: 位相ずれ ≈ 0°\n（信号がほぼそのまま通過）",
        xy=(0.02, -2),
        xytext=(0.013, -55),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        fontsize=8,
        color="blue",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", alpha=0.92
        ),
    )
    ax2.annotate(
        f"fc 付近: 位相 ≈ {S21_phase[fc_idx]:.0f}°\n（2次フィルタでは -90°）",
        xy=(F_C / 1e9, S21_phase[fc_idx]),
        xytext=(F_C / 1e9 * 4, S21_phase[fc_idx] + 40),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
        fontsize=8,
        color="darkorange",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange", alpha=0.92
        ),
    )
    ax2.annotate(
        "高周波: 位相 → -180°\n（2次フィルタの極限）",
        xy=(50, S21_phase[-50]),
        xytext=(5, S21_phase[-50] + 45),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        fontsize=8,
        color="blue",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", alpha=0.92
        ),
    )

    # ─── ④ グループ遅延 ──────────────────────────────────
    mask = freqs < F_C * 8
    ax3.semilogx(
        freqs_GHz[mask], gd_ns[mask], "-", color="purple", linewidth=2, label="グループ遅延 (GD)"
    )
    ax3.axvline(
        F_C / 1e9, color="orange", linestyle="--", linewidth=1.5, label=f"fc = {F_C/1e9:.1f} GHz"
    )
    ax3.set_xlabel("周波数 (GHz)")
    ax3.set_ylabel("グループ遅延 (ns)")
    ax3.set_title("④ グループ遅延  ( GD = -dφ/dω )")
    ax3.set_xlim(freqs_GHz[0], freqs_GHz[mask][-1])
    ax3.set_ylim(bottom=0)
    ax3.grid(True, which="both", alpha=0.4)
    ax3.legend(fontsize=8)

    # ④ アノテーション
    gd_dc = gd_ns[0]
    ax3.annotate(
        f"低周波の遅延 ≈ {gd_dc:.3f} ns\n（信号が何 ns 遅れるか）",
        xy=(freqs_GHz[10], gd_ns[10]),
        xytext=(freqs_GHz[10] * 8, gd_dc * 0.5),
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
        fontsize=8,
        color="purple",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", edgecolor="purple", alpha=0.92),
    )
    ax3.text(
        0.99,
        0.97,
        "グループ遅延が一定 → 位相が周波数に比例（線形位相）\n"
        "                   → 波形の形が歪まない\n"
        "グループ遅延が変動 → 周波数ごとに遅れが違う\n"
        "                   → 波形歪み発生",
        transform=ax3.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange", alpha=0.95
        ),
    )

    plt.savefig("spara_guide.png", dpi=150)
    print("Saved: spara_guide.png")


if __name__ == "__main__":
    main()
