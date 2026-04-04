"""
CDF（累積分布関数）= Fail Count（失敗数）
なぜ同じなのか？直感的解説
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.special import erfc
from scipy.stats import norm as sp_norm
import japanize_matplotlib  # noqa: F401


def main():
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        "CDF（累積分布関数）= Fail Count（失敗数）\n"
        "なぜ同じなのか？具体例で理解する",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender",
                  edgecolor="purple", alpha=0.85),
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                           top=0.90, bottom=0.05)
    ax_concept = fig.add_subplot(gs[0, 0])
    ax_jitter = fig.add_subplot(gs[0, 1])
    ax_failcount = fig.add_subplot(gs[1, 0])
    ax_explanation = fig.add_subplot(gs[1, 1])

    # ─────────────────────────────────────────────────────────
    # ① 概念図：測定シーン
    # ─────────────────────────────────────────────────────────
    ax_concept.set_xlim(0, 10)
    ax_concept.set_ylim(0, 10)
    ax_concept.axis("off")
    ax_concept.set_title("① 測定のシーン", fontsize=11, fontweight="bold")

    # テスター
    ax_concept.add_patch(mpatches.FancyBboxPatch(
        (0.2, 4), 1.5, 2,
        boxstyle="round,pad=0.1",
        facecolor="lightblue", edgecolor="steelblue", linewidth=2,
    ))
    ax_concept.text(1.0, 5.2, "テスター\nパラメータを\n変化させながら\n何度も測定",
                    ha="center", va="center", fontsize=8, fontweight="bold")

    # DUT
    ax_concept.add_patch(mpatches.FancyBboxPatch(
        (3.5, 3.5), 2, 3,
        boxstyle="round,pad=0.1",
        facecolor="lightyellow", edgecolor="black", linewidth=2,
    ))
    ax_concept.text(4.5, 5.2, "DUT\n被測定\nチップ",
                    ha="center", va="center", fontsize=9, fontweight="bold")

    # 矢印
    ax_concept.annotate("", xy=(3.4, 5), xytext=(1.7, 5),
                        arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
    ax_concept.text(2.55, 5.5, "タイミング\nマージン変更\n−0.5 → +0.5",
                    ha="center", fontsize=8, color="purple", fontweight="bold")

    # 失敗/成功のアイコン
    positions = [(6.0, 7.5), (7.0, 8.0), (8.0, 6.5), (9.0, 7.8)]
    for x, y in positions:
        ax_concept.add_patch(plt.Circle((x, y), 0.25, color="red", alpha=0.7))
        ax_concept.text(x, y, "✕", ha="center", va="center",
                        fontsize=14, color="white", fontweight="bold")
        ax_concept.text(x, y-0.6, "失敗", ha="center", fontsize=7, color="red")

    positions = [(6.5, 5.0), (7.5, 5.5), (8.5, 4.8), (9.5, 5.2)]
    for x, y in positions:
        ax_concept.add_patch(plt.Circle((x, y), 0.25, color="green", alpha=0.7))
        ax_concept.text(x, y, "○", ha="center", va="center",
                        fontsize=14, color="white", fontweight="bold")
        ax_concept.text(x, y-0.6, "成功", ha="center", fontsize=7, color="green")

    ax_concept.text(7.5, 3.0, "同じパラメータで1000回測定\n→ 失敗回数をカウント",
                    ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                              edgecolor="orange", alpha=0.95))

    # ─────────────────────────────────────────────────────────
    # ② ジッタの分布
    # ─────────────────────────────────────────────────────────
    TRUE_SIGMA = 0.15
    TRUE_CENTER = 0.5
    x_fine = np.linspace(0, 1, 300)

    # ジッタの PDF（正規分布）
    jitter_pdf = sp_norm.pdf(x_fine, loc=TRUE_CENTER, scale=TRUE_SIGMA)

    ax_jitter.fill_between(x_fine, 0, jitter_pdf, alpha=0.4, color="steelblue",
                           label="ジッタの分布（正規分布）")
    ax_jitter.plot(x_fine, jitter_pdf, "-", color="steelblue", linewidth=2)

    # 閾値ラインをいくつか引く
    for x_thresh, col in [(0.2, "orange"), (0.5, "red"), (0.8, "green")]:
        ax_jitter.axvline(x_thresh, color=col, linestyle="--", linewidth=2, alpha=0.7)
        # その右側を塗りつぶし
        mask = x_fine >= x_thresh
        ax_jitter.fill_between(x_fine[mask], 0, jitter_pdf[mask], alpha=0.3, color=col)
        ax_jitter.text(x_thresh, np.max(jitter_pdf)*0.05, f"閾値\n{x_thresh:.1f}",
                       ha="center", fontsize=8, color=col, fontweight="bold")

    ax_jitter.set_xlabel("タイミング（ジッタ大きさ）", fontsize=10)
    ax_jitter.set_ylabel("確率密度", fontsize=10)
    ax_jitter.set_title("② ジッタの分布（母集団）", fontsize=11, fontweight="bold")
    ax_jitter.grid(True, alpha=0.3, axis="y")
    ax_jitter.legend(fontsize=9)
    ax_jitter.text(0.99, 0.97,
                   "赤線より右側の面積\n= Fail Countが発生する\n確率（CDF）",
                   transform=ax_jitter.transAxes, fontsize=8,
                   ha="right", va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose",
                             edgecolor="red", alpha=0.9))

    # ─────────────────────────────────────────────────────────
    # ③ Fail Count グラフ
    # ─────────────────────────────────────────────────────────
    x = np.linspace(0, 1, 100)
    failcount = 100.0 * 0.5 * erfc((x - TRUE_CENTER) / (TRUE_SIGMA * np.sqrt(2)))

    ax_failcount.plot(x, failcount, "-", color="royalblue", linewidth=3,
                      label="Fail Count")
    ax_failcount.fill_between(x, 0, failcount, alpha=0.2, color="royalblue")

    # 3点でのマーカーと説明
    for x_pt, col, fail_pct in [(0.2, "orange", 2), (0.5, "red", 50), (0.8, "green", 98)]:
        idx = np.argmin(np.abs(x - x_pt))
        y_pt = failcount[idx]
        ax_failcount.plot(x_pt, y_pt, "o", color=col, ms=12, zorder=5)

        # 説明
        ax_failcount.annotate(
            f"パラメータ = {x_pt:.1f}\n失敗 {fail_pct}%\n= CDF {fail_pct}%",
            xy=(x_pt, y_pt),
            xytext=(x_pt + 0.15, y_pt - 15),
            fontsize=8, color=col,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=col, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5)
        )

    ax_failcount.set_xlabel("パラメータ（タイミングマージン）", fontsize=10)
    ax_failcount.set_ylabel("Fail Count %", fontsize=10)
    ax_failcount.set_title("③ 実測定の結果：Fail Count", fontsize=11, fontweight="bold")
    ax_failcount.grid(True, alpha=0.4)
    ax_failcount.legend(fontsize=9)

    # ─────────────────────────────────────────────────────────
    # ④ 説明テキスト
    # ─────────────────────────────────────────────────────────
    ax_explanation.axis("off")
    ax_explanation.set_xlim(0, 10)
    ax_explanation.set_ylim(0, 10)

    explanation_text = """
【CDF と Fail Count が同じ理由】

◆ Fail Count （実測値）とは
  ・「このパラメータで何回テストしたら何回失敗するか」
  ・例：x=0.5 の時、1000回テストして500回失敗
       → Fail Count = 50%
  ・つまり「このパラメータで失敗する確率 = 50%」

◆ CDF （理論値）とは
  ・「ジッタがこの値以上になる確率」
  ・例：ジッタの分布から計算すると
       x=0.5 より大きいジッタが出る確率 = 50%
  ・つまり「ここで波形が失敗する確率 = 50%」

◆ なぜ同じか？
  Fail Count = CDF

  理由：ジッタが大きくなると波形が失敗する
       → Fail Count が増える
       → CDF（大きいジッタの確率）も増える
       ↓
       完全に同じ情報！

◆ 物理的な対応関係
  ・パラメータ値が小さい
    → ジッタの余裕あり
    → 失敗しにくい（Fail Count ↓ / CDF ↓）
  
  ・パラメータ値が大きい
    → ジッタの余裕なし
    → 失敗しやすい（Fail Count ↑ / CDF ↑）
"""

    ax_explanation.text(0.05, 0.95, explanation_text,
                        transform=ax_explanation.transAxes, fontsize=9,
                        ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                                  edgecolor="orange", alpha=0.95))

    plt.savefig("cdf_equals_failcount.png", dpi=150, bbox_inches="tight")
    print("Saved: cdf_equals_failcount.png")
    plt.show()


if __name__ == "__main__":
    main()

