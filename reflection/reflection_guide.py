"""
伝送線路の反射 (Reflection) 解説グラフ生成スクリプト
反射係数 Γ・リターンロス・TDR・定在波パターン（VSWR）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import japanize_matplotlib  # noqa: F401  日本語フォント自動設定


# ─────────────────────────────────────────────────────────────
# ① 概念図
# ─────────────────────────────────────────────────────────────
def draw_reflection_concept(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("① 伝送線路における反射の概念", fontsize=11)

    # 信号源ボックス
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.1, 1.9), 1.3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor="lightblue", edgecolor="steelblue", linewidth=1.5,
    ))
    ax.text(0.75, 2.5, "信号源\nVs, Zs=Z0",
            ha="center", va="center", fontsize=8, fontweight="bold")

    # 伝送線路（上下の線）
    for y in [2.1, 2.9]:
        ax.plot([1.4, 7.4], [y, y], "-", color="black", linewidth=2)
    ax.text(4.4, 2.5, "伝送線路  (特性インピーダンス Z0 = 50 Ω)",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.8))

    # 負荷ボックス
    ax.add_patch(mpatches.FancyBboxPatch(
        (7.4, 1.9), 1.3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor="mistyrose", edgecolor="red", linewidth=1.5,
    ))
    ax.text(8.05, 2.5, "負荷\nZL ≠ Z0",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # 入射波（→）
    ax.annotate("", xy=(7.3, 3.55), xytext=(1.5, 3.55),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.0))
    ax.text(4.4, 3.9, "入射波  Vi  （電力を送り込む）", color="blue",
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="aliceblue",
                      edgecolor="blue", alpha=0.8))

    # 反射波（←）
    ax.annotate("", xy=(1.5, 1.45), xytext=(7.3, 1.45),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0))
    ax.text(4.4, 1.05, "反射波  Vr = Γ · Vi  （インピーダンス不整合により発生）",
            color="red", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="mistyrose",
                      edgecolor="red", alpha=0.8))

    # Γ 数式
    ax.text(5.0, 4.9,
            "反射係数  Γ = (ZL − Z0) / (ZL + Z0)    [ −1 ≤ Γ ≤ +1 ]",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                      edgecolor="purple", alpha=0.9))

    # 代表的な3ケース
    for x, label, color in [
        (1.8,  "短絡端  ZL = 0\nΓ = −1\n全反射・逆位相",   "purple"),
        (5.0,  "整合終端  ZL = Z0\nΓ = 0\n反射なし",       "darkgreen"),
        (8.2,  "開放端  ZL = ∞\nΓ = +1\n全反射・同位相",   "red"),
    ]:
        ax.text(x, 0.08, label, ha="center", va="bottom", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9))


# ─────────────────────────────────────────────────────────────
# ② 反射係数 Γ vs ZL/Z0（+ リターンロス）
# ─────────────────────────────────────────────────────────────
def plot_gamma_vs_zl(ax):
    ratio = np.logspace(-2, 2, 1000)          # ZL/Z0 (0.01 〜 100)
    gamma = (ratio - 1) / (ratio + 1)

    # リターンロス: RL = −20 log10|Γ|  (正値; 大=反射少=良好)
    with np.errstate(divide="ignore", invalid="ignore"):
        rl_dB = np.where(np.abs(gamma) > 0,
                         -20 * np.log10(np.abs(gamma)), 60.0)
    rl_dB = np.clip(rl_dB, 0, 60)

    ax2 = ax.twinx()
    ax.semilogx(ratio, gamma,  "-",  color="royalblue", linewidth=2.5,
                label="反射係数 Γ")
    ax2.semilogx(ratio, rl_dB, "--", color="crimson",   linewidth=1.8,
                 label="リターンロス RL = −20log|Γ| (dB)")

    # 特徴点アノテーション
    for zl_r, lbl, col, dx_mul, dy in [
        (0.01, "短絡\nZL≈0, Γ=−1",  "purple",    0.25, +0.30),
        (1.0,  "整合\nZL=Z0, Γ=0",   "darkgreen", 3.0,  +0.28),
        (100,  "開放\nZL→∞, Γ=+1",  "red",        0.3,  -0.45),
    ]:
        g = (zl_r - 1) / (zl_r + 1)
        ax.plot(zl_r, g, "o", color=col, ms=10, zorder=5)
        ax.annotate(lbl, xy=(zl_r, g),
                    xytext=(zl_r * dx_mul, g + dy),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                    fontsize=8, color=col,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=col, alpha=0.92))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ZL / Z0")
    ax.set_ylabel("反射係数 Γ",                       color="royalblue")
    ax2.set_ylabel("リターンロス (dB)",                color="crimson")
    ax.set_title("② 反射係数 Γ とリターンロス  vs  負荷インピーダンス比 ZL / Z0")
    ax.set_ylim(-1.3, 1.3)
    ax2.set_ylim(0, 65)
    ax.grid(True, which="both", alpha=0.4)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax.text(0.01, 0.97,
            "Γ > 0 : ZL > Z0（高インピーダンス）→ 同位相で反射\n"
            "Γ = 0 : ZL = Z0（完全整合）         → 反射なし\n"
            "Γ < 0 : ZL < Z0（低インピーダンス）→ 逆位相で反射",
            transform=ax.transAxes, fontsize=8, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.95))


# ─────────────────────────────────────────────────────────────
# ③ 時間領域波形（TDR）
# ─────────────────────────────────────────────────────────────
def plot_tdr(ax, t_delay: float = 1.0):
    """
    送信端でのステップ応答を模擬。
    入射ステップ (t=0) + 反射ステップ (t=2·TD, 振幅 Γ) の重ね合わせ。
    """
    t = np.linspace(-0.3, 5.0, 6000)

    def step(t0):
        return np.where(t >= t0, 1.0, 0.0)

    cases = [
        (+1.0, "Γ = +1  開放端 (ZL = ∞)",    "red",         "-"),
        (+0.5, "Γ = +0.5  (ZL = 3·Z0)",       "darkorange",  "--"),
        ( 0.0, "Γ =  0   整合端 (ZL = Z0)",   "green",       "-"),
        (-0.5, "Γ = −0.5  (ZL = Z0/3)",       "blueviolet",  "--"),
        (-1.0, "Γ = −1  短絡端 (ZL = 0)",     "purple",      "-"),
    ]
    for gamma_val, lbl, col, ls in cases:
        wv = step(0) + gamma_val * step(2 * t_delay)
        ax.plot(t, wv, ls, color=col, linewidth=2, label=lbl)

    ax.axvline(0,           color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(2*t_delay,   color="gray", linestyle=":", linewidth=1.0,
               label=f"反射波到達  t = 2·TD = {2*t_delay:.0f} ns")

    ax.set_xlabel("時間 (ns)")
    ax.set_ylabel("送信端電圧（正規化）")
    ax.set_title("③ TDR 送信端波形（ステップ入力）     TD = 伝播遅延時間")
    ax.set_xlim(-0.3, 5.0)
    ax.set_ylim(-0.45, 2.35)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="center right")

    ax.annotate("入射ステップ到達\n（送信端で観測開始）",
                xy=(0.05, 1.0), xytext=(0.4, 1.6),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
                fontsize=8, color="steelblue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue",
                          edgecolor="steelblue", alpha=0.92))
    ax.annotate("反射波が重畳\n→ 終端条件を判定できる",
                xy=(2*t_delay+0.1, 2.0), xytext=(2.9, 2.1),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=8, color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose",
                          edgecolor="red", alpha=0.92))
    ax.text(0.99, 0.03,
            "TDR（時間領域反射率測定）の読み方\n"
            "  反射波の符号  ＋ → ZL > Z0（高インピーダンス）\n"
            "              − → ZL < Z0（低インピーダンス）\n"
            "  反射波の大きさ → |Γ| が大きいほど整合が悪い\n"
            "  反射到達時刻  → 2·TD = 2 × 線路長 / 伝播速度",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.95))


# ─────────────────────────────────────────────────────────────
# ④ 定在波パターンと VSWR
# ─────────────────────────────────────────────────────────────
def plot_standing_wave(ax):
    """
    |V(d)| = sqrt(1 + |Γ|² + 2|Γ| cos(4π d/λ))
    d : 負荷からの距離（波長 λ 単位）
    """
    d = np.linspace(0, 2, 1000)

    cases = [
        (1.0, "red",         "|Γ| = 1.0  （開放 / 短絡）"),
        (0.5, "darkorange",  "|Γ| = 0.5"),
        (0.2, "royalblue",   "|Γ| = 0.2  （実用目標 VSWR ≤ 1.5）"),
        (0.0, "green",       "|Γ| = 0.0  （完全整合）"),
    ]
    for gamma_abs, col, lbl in cases:
        if gamma_abs >= 1.0:
            vswr_str = "VSWR = ∞"
        else:
            vswr_val = (1 + gamma_abs) / (1 - gamma_abs)
            vswr_str = f"VSWR = {vswr_val:.1f}"

        V = np.sqrt(1 + gamma_abs**2 + 2 * gamma_abs * np.cos(4 * np.pi * d))
        ax.plot(d, V, "-", color=col, linewidth=2, label=f"{lbl}   {vswr_str}")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("負荷からの距離（波長 λ 単位）")
    ax.set_ylabel("電圧振幅（正規化）")
    ax.set_title("④ 定在波パターンと VSWR\n"
                 "   VSWR = (1 + |Γ|) / (1 − |Γ|)     Vmax = 1 + |Γ|,  Vmin = 1 − |Γ|")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.4)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    ax.annotate("整合時：定在波なし\n（振幅が一定）",
                xy=(0.5, 1.0), xytext=(0.65, 0.42),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                fontsize=8, color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                          edgecolor="green", alpha=0.92))
    ax.annotate("|Γ|=1：腹 Vmax=2  /  節 Vmin=0",
                xy=(0.0, 2.0), xytext=(0.25, 1.82),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=8, color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose",
                          edgecolor="red", alpha=0.92))
    ax.text(0.99, 0.97,
            "VSWR = 1       → 完全整合（反射ゼロ）\n"
            "VSWR = 1.5   → |Γ| = 0.20  ← 実用上の目標値\n"
            "VSWR = 2       → |Γ| = 0.33,  RL = 9.5 dB\n"
            "VSWR → ∞   → 全反射（開放 or 短絡）",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.95))


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(10, 22))
    fig.suptitle(
        "伝送線路の反射 (Reflection) 解説\n"
        "反射係数 Γ・リターンロス・TDR・定在波（VSWR）",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                  edgecolor="purple", alpha=0.85),
    )

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.80,
                           top=0.93, bottom=0.03,
                           height_ratios=[1.3, 1.3, 1.3, 1.3])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    draw_reflection_concept(ax0)
    plot_gamma_vs_zl(ax1)
    plot_tdr(ax2, t_delay=1.0)
    plot_standing_wave(ax3)

    plt.savefig("reflection_guide.png", dpi=150, bbox_inches="tight")
    print("Saved: reflection_guide.png")


if __name__ == "__main__":
    main()

