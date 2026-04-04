"""
Fail Count を微分するとヒストグラムになる理由
累積分布関数(CDF)から確率密度関数(PDF)への変換
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
from scipy.stats import norm as sp_norm
import japanize_matplotlib  # noqa: F401


def main():
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        "Fail Count を微分するとヒストグラムになる理由\n"
        "CDF（累積分布関数）→ 微分 → PDF（確率密度関数）",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                  edgecolor="purple", alpha=0.85),
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                           top=0.90, bottom=0.08)
    ax_cdf = fig.add_subplot(gs[0, :])
    ax_pdf = fig.add_subplot(gs[1, 0])
    ax_slope = fig.add_subplot(gs[1, 1])

    # データ準備
    TRUE_SIGMA = 0.15
    TRUE_CENTER = 0.5
    x = np.linspace(0, 1, 100)

    # ① CDF（Fail Count）のグラフ
    failcount_float = 100.0 * 0.5 * erfc((x - TRUE_CENTER) / (TRUE_SIGMA * np.sqrt(2)))

    ax_cdf.plot(x, failcount_float, "-", color="royalblue", linewidth=3,
                label="Fail Count(x) = CDF  (累積確率)")
    ax_cdf.fill_between(x, 0, failcount_float, alpha=0.2, color="royalblue")

    # 3点での接線を表示（傾き = PDF値）
    for x_pt, col in [(0.2, "orange"), (0.5, "red"), (0.8, "green")]:
        idx = np.argmin(np.abs(x - x_pt))
        y_pt = failcount_float[idx]

        # 数値微分で傾きを計算
        slope = -np.gradient(failcount_float)[idx]

        # 接線
        dx_line = 0.08
        x_line = np.array([x_pt - dx_line, x_pt + dx_line])
        y_line = y_pt + slope * np.array([-dx_line, dx_line])
        ax_cdf.plot(x_line, y_line, "--", color=col, linewidth=2.5, alpha=0.8)

        ax_cdf.plot(x_pt, y_pt, "o", color=col, ms=10, zorder=5)

        # アノテーション
        pdf_analytical = 100.0 * sp_norm.pdf(x_pt, loc=TRUE_CENTER, scale=TRUE_SIGMA)
        ax_cdf.annotate(
            f"傾き = {slope:.0f}\n≈ PDF({x_pt:.1f})\n≈ {pdf_analytical:.0f}",
            xy=(x_pt, y_pt),
            xytext=(x_pt, y_pt + 15),
            fontsize=8, color=col, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=col, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5)
        )

    ax_cdf.set_xlabel("x （ジッタ大きさ）", fontsize=10)
    ax_cdf.set_ylabel("Fail Count %", fontsize=10)
    ax_cdf.set_title("① Fail Count = CDF（累積分布関数）\n"
                     "各点での接線の傾き = その点でのPDF値 = ヒストグラムの棒の高さ",
                     fontsize=11)
    ax_cdf.grid(True, alpha=0.4)
    ax_cdf.legend(fontsize=10, loc="upper right")

    ax_cdf.text(0.99, 0.05,
                "数学的関係式：\n"
                "Fail Count(x) = ∫[x→∞] PDF(t) dt   ← 積分 = 累積\n"
                "d(Fail Count)/dx = −PDF(x)          ← 微分 = CDF → PDF",
                transform=ax_cdf.transAxes, fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="orange", alpha=0.95))

    # ② PDF（ヒストグラム）
    x_fine = np.linspace(0, 1, 300)
    pdf_analytical = 100.0 * sp_norm.pdf(x_fine, loc=TRUE_CENTER, scale=TRUE_SIGMA)

    # 数値微分で求めたPDF
    pdf_numerical = -np.gradient(failcount_float, x)
    pdf_numerical = np.clip(pdf_numerical, 0, None)

    ax_pdf.bar(x, pdf_numerical, width=(x[1]-x[0])*0.9,
               color="steelblue", alpha=0.6, label="数値微分: −d(Fail Count)/dx")
    ax_pdf.plot(x_fine, pdf_analytical, "-", color="red", linewidth=2,
                label="解析的PDF")

    ax_pdf.set_xlabel("x", fontsize=10)
    ax_pdf.set_ylabel("確率密度", fontsize=10)
    ax_pdf.set_title("② ヒストグラム = PDF（確率密度関数）\n"
                     "棒の高さ = ジッタがそこに存在する確率",
                     fontsize=10)
    ax_pdf.grid(True, alpha=0.4, axis="y")
    ax_pdf.legend(fontsize=9)

    ax_pdf.annotate(
        "ピークが高い\n= ここにジッタが密集\n= Fail Countが急峻",
        xy=(TRUE_CENTER, np.max(pdf_numerical)*0.95),
        xytext=(TRUE_CENTER + 0.2, np.max(pdf_numerical)*0.6),
        fontsize=8, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                  edgecolor="darkgreen", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5)
    )

    # ③ 微分の幾何学的意味
    ax_slope.text(0.5, 0.9, "微分（Derivative）の幾何学的意味",
                  transform=ax_slope.transAxes, fontsize=11, ha="center",
                  fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan",
                            edgecolor="steelblue"))

    explanation = """
① CDF（Fail Count） と は
  ・「x 以上のジッタが発生する確率」の累積
  ・単調に減少する S 字曲線
  ・例：x=0.5 で Fail Count=50%
       → ジッタが 0.5 以上になる確率が 50%

② PDF（ヒストグラム） と は
  ・「各位置でのジッタの密度」
  ・CDF を微分したもの
  ・例：x=0.5 で PDF が大きい
       → そこにジッタが密集している

③ 微分 = 接線の傾き
  ・Fail Count が急峻な部分
       → 傾きが大きい
       → PDF が大きい
       → ヒストグラムの棒が高い
  
  ・Fail Count が平坦な部分
       → 傾きが小さい
       → PDF が小さい
       → ヒストグラムの棒が低い

④ 物理的解釈
  ・急峻 = 少しの変化（Δx）で大きく変わる
  ・= ジッタがその範囲に密集している
  ・= 波形がその位置でよく失敗する
"""

    ax_slope.text(0.05, 0.82, explanation,
                  transform=ax_slope.transAxes, fontsize=9,
                  ha="left", va="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                            edgecolor="steelblue", alpha=0.95))

    ax_slope.axis("off")

    plt.savefig("why_derivative_to_histogram.png", dpi=150, bbox_inches="tight")
    print("Saved: why_derivative_to_histogram.png")
    plt.show()


if __name__ == "__main__":
    main()

