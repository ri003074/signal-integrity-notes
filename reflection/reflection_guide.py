"""伝送線路反射の入門ガイド図を作るスクリプト。"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
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
    ax.text(8.05, 2.5, "負荷\n$Z_L \\ne Z_0$",
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
    ax.text(4.4, 1.05, "反射波  $V_r = \\Gamma \\cdot V_i$  （インピーダンス不整合で発生）",
            color="red", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="mistyrose",
                      edgecolor="red", alpha=0.8))

    # Γ 数式
    ax.text(5.0, 4.9,
            r"反射係数  $\Gamma = (Z_L - Z_0)/(Z_L + Z_0)$    $[-1 \leq \Gamma \leq 1]$",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                      edgecolor="purple", alpha=0.9))

    # 代表的な3ケース
    for x, label, color in [
        (1.8,  "短絡端  $Z_L=0$\n$\\Gamma=-1$\n全反射・逆位相",    "purple"),
        (5.0,  "整合終端  $Z_L=Z_0$\n$\\Gamma=0$\n反射なし",        "darkgreen"),
        (8.2,  "開放端  $Z_L\\to\\infty$\n$\\Gamma=+1$\n全反射・同位相", "red"),
    ]:
        ax.text(x, 0.08, label, ha="center", va="bottom", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9))


# ─────────────────────────────────────────────────────────────
# ② 反射係数 Gamma vs ZL/Z0
# ─────────────────────────────────────────────────────────────
def plot_gamma_vs_zl(ax):
    ratio = np.logspace(-2, 2, 1000)          # ZL/Z0 (0.01 〜 100)
    gamma = (ratio - 1) / (ratio + 1)

    ax.semilogx(ratio, gamma, "-", color="royalblue", linewidth=2.5,
                label=r"反射係数 $\Gamma$")

    # 特徴点アノテーション
    for zl_r, lbl, col, dx_mul, dy in [
        (0.01, r"短絡\n$Z_L \approx 0, \Gamma=-1$",    "purple",    0.25, +0.30),
        (1.0,  r"整合\n$Z_L=Z_0, \Gamma=0$",           "darkgreen", 3.0,  +0.28),
        (100,  r"開放\n$Z_L \to \infty, \Gamma=+1$", "red",        0.3,  -0.45),
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
    ax.set_ylabel(r"反射係数 $\Gamma$", color="royalblue")
    ax.set_title(r"② 負荷比 $Z_L/Z_0$ と反射係数 $\Gamma$")
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    ax.text(0.01, 0.97,
            "$\\Gamma > 0$ : $Z_L > Z_0$（上向き反射）\n"
            "$\\Gamma = 0$ : $Z_L = Z_0$（反射なし）\n"
            "$\\Gamma < 0$ : $Z_L < Z_0$（下向き反射）",
            transform=ax.transAxes, fontsize=9, ha="left", va="top",
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
        (+1.0, r"$\Gamma=+1$  開放端", "red", "-"),
        (0.0, r"$\Gamma=0$  整合端", "green", "-"),
        (-1.0, r"$\Gamma=-1$  短絡端", "purple", "-"),
    ]
    for gamma_val, lbl, col, ls in cases:
        wv = step(0) + gamma_val * step(2 * t_delay)
        ax.plot(t, wv, ls, color=col, linewidth=2, label=lbl)

    ax.axvline(0,           color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(2*t_delay,   color="gray", linestyle=":", linewidth=1.0,
               label=rf"反射波到達  $t = 2T_D = {2*t_delay:.0f}$ ns")

    ax.set_xlabel("時間 (ns)")
    ax.set_ylabel("送信端電圧（正規化）")
    ax.set_title("③ TDR の基本（送信端で観測）")
    ax.set_xlim(-0.3, 5.0)
    ax.set_ylim(-0.45, 2.35)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9, loc="center right")

    ax.annotate("入射ステップ到達\n（送信端で観測開始）",
                xy=(0.05, 1.0), xytext=(0.4, 1.6),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
                fontsize=9, color="steelblue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue",
                          edgecolor="steelblue", alpha=0.92))
    ax.annotate("反射波が重畳\n→ 終端条件を判定できる",
                xy=(2*t_delay+0.1, 2.0), xytext=(2.9, 2.1),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose",
                          edgecolor="red", alpha=0.92))
    ax.text(0.99, 0.03,
            r"読み方: 反射が上なら $Z_L>Z_0$、下なら $Z_L<Z_0$。" + "\n"
            r"段差が出る時刻 $2T_D$ から距離を見積もれる。",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
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
        (1.0, "red", "|Γ| = 1.0"),
        (0.5, "darkorange", "|Γ| = 0.5"),
        (0.0, "green", "|Γ| = 0.0"),
    ]
    for gamma_abs, col, lbl in cases:
        if gamma_abs >= 1.0:
            vswr_str = "VSWR = $\\infty$"
        else:
            vswr_val = (1 + gamma_abs) / (1 - gamma_abs)
            vswr_str = f"VSWR = {vswr_val:.1f}"

        V = np.sqrt(1 + gamma_abs**2 + 2 * gamma_abs * np.cos(4 * np.pi * d))
        ax.plot(d, V, "-", color=col, linewidth=2, label=f"{lbl}   {vswr_str}")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("負荷からの距離（波長 λ 単位）")
    ax.set_ylabel("電圧振幅（正規化）")
    ax.set_title(r"④ 定在波の基本（$|\Gamma|$ が大きいほど振れが大きい）")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.4)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

    ax.annotate("整合時：定在波なし\n（振幅が一定）",
                xy=(0.5, 1.0), xytext=(0.65, 0.42),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                fontsize=9, color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                          edgecolor="green", alpha=0.92))
    ax.annotate(r"$|\Gamma|=1$: 腹 $V_{max}=2$ / 節 $V_{min}=0$",
                xy=(0.0, 2.0), xytext=(0.25, 1.82),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose",
                          edgecolor="red", alpha=0.92))
    ax.text(0.99, 0.97,
            r"$VSWR = (1+|\Gamma|)/(1-|\Gamma|)$" + "\n"
            r"$|\Gamma|=0$ で $VSWR=1$（整合）。",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.95))


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "伝送線路反射の入門ガイド（基本4図）",
        fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                  edgecolor="purple", alpha=0.85),
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25,
                           top=0.88, bottom=0.08, left=0.05, right=0.98)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    draw_reflection_concept(ax0)
    plot_gamma_vs_zl(ax1)
    plot_tdr(ax2, t_delay=1.0)
    plot_standing_wave(ax3)

    out_path = Path(__file__).with_name("reflection_guide.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

