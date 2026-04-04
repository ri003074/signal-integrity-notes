"""
積分と微分の関係
PDF（ヒストグラム）と CDF（Fail Count）の数学的関係
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
from scipy.stats import norm as sp_norm
import japanize_matplotlib  # noqa: F401


def main():
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "積分と微分：PDF（ヒストグラム）と CDF（Fail Count）の関係\n"
        "積分と微分は互いに逆の操作！",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                  edgecolor="purple", alpha=0.85),
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.30, wspace=0.35,
                           top=0.88, bottom=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ─────────────────────────────────────────────────────────
    # 左：上→下が積分、下→上が微分
    # ─────────────────────────────────────────────────────────
    TRUE_SIGMA = 0.15
    TRUE_CENTER = 0.5
    x = np.linspace(0, 1, 200)
    x_fine = np.linspace(0, 1, 500)

    # PDF（ヒストグラム）
    pdf_vals = 100.0 * sp_norm.pdf(x, loc=TRUE_CENTER, scale=TRUE_SIGMA)
    pdf_fine = 100.0 * sp_norm.pdf(x_fine, loc=TRUE_CENTER, scale=TRUE_SIGMA)

    # CDF（Fail Count）
    cdf_vals = 100.0 * 0.5 * erfc((x - TRUE_CENTER) / (TRUE_SIGMA * np.sqrt(2)))
    cdf_fine = 100.0 * 0.5 * erfc((x_fine - TRUE_CENTER) / (TRUE_SIGMA * np.sqrt(2)))

    # 左グラフ：PDF と CDF を重ねて表示
    ax_left_pdf = ax_left
    ax_left_cdf = ax_left.twinx()

    # PDF（ヒストグラム、上側）
    bars = ax_left_pdf.bar(x, pdf_vals, width=(x[1]-x[0])*0.9,
                           color="steelblue", alpha=0.6, label="PDF（ヒストグラム）")
    ax_left_pdf.plot(x_fine, pdf_fine, "-", color="darkblue", linewidth=2)

    # CDF（下側、別軸）
    ax_left_cdf.plot(x_fine, cdf_fine, "-", color="red", linewidth=3,
                     label="CDF（Fail Count）")
    ax_left_cdf.fill_between(x_fine, 0, cdf_fine, alpha=0.2, color="red")

    ax_left_pdf.set_xlabel("パラメータ x", fontsize=11)
    ax_left_pdf.set_ylabel("PDF の値（棒の高さ）", fontsize=11, color="steelblue")
    ax_left_cdf.set_ylabel("CDF の値（Fail Count %）", fontsize=11, color="red")
    ax_left_pdf.set_title("上段：PDF（ヒストグラム）\n下段：CDF（Fail Count）",
                          fontsize=11, fontweight="bold")
    ax_left_pdf.grid(True, alpha=0.3)

    # 矢印と説明
    ax_left_pdf.annotate("", xy=(0.15, 120), xytext=(0.15, 30),
                         arrowprops=dict(arrowstyle="<->", color="purple", lw=3))
    ax_left_pdf.text(0.02, 80, "積分\n∫", fontsize=14, color="purple", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                               edgecolor="orange"))
    ax_left_pdf.text(0.25, 80, "微分\nd/dx", fontsize=14, color="green", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                               edgecolor="darkgreen"))

    # ─────────────────────────────────────────────────────────
    # 右：数学的関係式と説明
    # ─────────────────────────────────────────────────────────
    ax_right.axis("off")
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)

    explanation = """
【積分と微分の関係】

┌────────────────────────────────────┐
│  PDF ⇄ CDF の数学的関係             │
└────────────────────────────────────┘

① 積分：PDF → CDF
  CDF(x) = ∫[x→∞] PDF(t) dt
  
  意味：x から右側全部の面積を足す
       = 「x以上のジッタが出る確率」
       = Fail Count

② 微分：CDF → PDF
  PDF(x) = − d(CDF)/dx
  
  意味：CDF の接線の傾き
       = その点でのジッタの密度
       = ヒストグラムの棒の高さ

③ 関係性
  微分と積分は逆操作
  
  PDF を積分 → CDF  （足し上げる）
  CDF を微分 → PDF  （切り取る）
  
  例：
    CDF(0.5) = 50%  ← x=0.5以上が50%
         ↓ 微分（傾きを見る）
    PDF(0.5) = 最大 ← そこが最も急峻

【コードの関係式】
  
  failcount = CDF
  
  pdf_numerical = −np.gradient(failcount)
               = − d(CDF)/dx
               = PDF
  
  ← これで正しい！（微分で OK）
"""

    ax_right.text(0.05, 0.95, explanation,
                  transform=ax_right.transAxes, fontsize=10,
                  ha="left", va="top",
                  bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                            edgecolor="steelblue", alpha=0.97, linewidth=2))

    plt.savefig("integration_differentiation_relation.png", dpi=150, bbox_inches="tight")
    print("Saved: integration_differentiation_relation.png")
    plt.show()


if __name__ == "__main__":
    main()

