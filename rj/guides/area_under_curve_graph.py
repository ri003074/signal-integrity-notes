"""
Graph visualization: Area under curve = Integration

Shows:
1) Velocity graph with rectangles under the curve
2) Histogram with right-side area (FailCount)
3) Relationship between differentiation and integration
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        "グラフの下の面積 = 積分の意味\n"
        "微分と積分は逆操作",
        fontsize=13,
        fontweight="bold",
    )

    # =========================================================================
    # 左図：速度グラフ（左から足す）
    # =========================================================================
    ax = axes[0]

    # データ
    t = np.array([0, 1, 2, 3, 4])
    v = np.array([0, 10, 10, 5, 0])

    # 長方形を描画（面積を視覚化）
    colors = ["lightcoral", "steelblue", "steelblue", "lightgreen"]
    widths = [0.9, 0.9, 0.9, 0.9]
    cumulative_area = 0

    for i in range(len(v) - 1):
        area = v[i] * 1  # 各区間の面積
        cumulative_area += area
        rect = Rectangle(
            (t[i], 0),
            widths[i],
            v[i],
            facecolor=colors[i],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.add_patch(rect)
        # 各長方形の面積をラベル表示
        if v[i] > 0:
            ax.text(
                t[i] + 0.45,
                v[i] / 2,
                f"{area}\nm",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # グラフの装飾
    ax.plot(t, v, "k-", linewidth=2, marker="o", markersize=6, label="速度 v(t)")
    ax.set_xlabel("時刻 t (秒)", fontsize=11)
    ax.set_ylabel("速度 v (m/s)", fontsize=11)
    ax.set_title(
        "① 速度グラフ\n(左から足す)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 長方形の面積の合計
    ax.text(
        0.98,
        0.97,
        f"グラフの下の面積\n= 0 + 10 + 10 + 5\n= 25 m\n\n積分: ∫v dt = 25 m",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95),
        fontweight="bold",
    )

    # =========================================================================
    # 中央図：ヒストグラム（右から足す）
    # =========================================================================
    ax = axes[1]

    # データ
    x = np.array([1, 2, 3])
    h = np.array([20, 50, 30])

    # FailCount（右から足す）
    F = np.array([100, 80, 30])

    # ヒストグラムを描画
    bar_width = 0.6
    bars = ax.bar(x, h, width=bar_width, color="steelblue", edgecolor="black", linewidth=1.5, alpha=0.6)

    # 各棒の上に数値を表示
    for i, (xi, hi) in enumerate(zip(x, h)):
        ax.text(xi, hi + 2, f"{hi}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 右側面積をハイライト（各xでの右側）
    highlight_colors = [(1.0, 0.0, 0.0, 0.2), (1.0, 0.4, 0.4, 0.2), (1.0, 0.8, 0.8, 0.2)]
    for i in range(len(x)):
        # x[i] から右側の面積を半透明で表示
        for j in range(i, len(x)):
            rect = Rectangle(
                (x[j] - bar_width / 2, 0),
                bar_width,
                h[j],
                facecolor=highlight_colors[i],
                edgecolor="red",
                linewidth=1 if i == 0 else 0.5,
                alpha=1.0,
            )
            ax.add_patch(rect)

    # FailCount の値を右側に表示
    ax.text(
        3.5,
        45,
        f"F(x=1) = 20+50+30\n     = 100個",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=(1.0, 0.0, 0.0, 0.1), edgecolor="red"),
        fontweight="bold",
    )
    ax.text(
        3.5,
        20,
        f"F(x=2) = 50+30\n     = 80個",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=(1.0, 0.4, 0.4, 0.1), edgecolor="red"),
        fontweight="bold",
    )

    ax.set_xlabel("パラメータ x", fontsize=11)
    ax.set_ylabel("個数", fontsize=11)
    ax.set_title(
        "② ヒストグラム\n(右から足す)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0.2, 4.5)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3, axis="y")

    # 説明テキスト
    ax.text(
        0.02,
        0.97,
        "FailCount(x)\n= 『x 以上』の個数\n= グラフの右側面積\n\n" + "∫[x→∞] h(t)dt\n= 右から足す",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95),
        fontweight="bold",
    )

    # =========================================================================
    # 右図：微分と積分の関係
    # =========================================================================
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    explanation = """
【グラフの下の面積 = 積分】

（ここが本質！）

━━━━━━━━━━━━━━━━━━━━━━

① 速度グラフ 
   v(t) = [0, 10, 10, 5]
   
   ↓ グラフを長方形に切る
   
   面積 = (0×1) + (10×1) 
        + (10×1) + (5×1)
        = 25 m
   
   ↓ これが積分の定義
   
   ∫v dt = 25 m  ✓


━━━━━━━━━━━━━━━━━━━━━━

② ヒストグラム
   h(x) = [20, 50, 30]
   
   ↓ 「x 以上」を右から足す
   
   F(x=1) = 20+50+30 = 100
   F(x=2) = 50+30 = 80
   F(x=3) = 30
   
   ↓ これも面積を足す操作
   
   ∫[x→∞] h(t)dt  ✓


━━━━━━━━━━━━━━━━━━━━━━

③ 微分と積分
   
   v(t) ← 微分 ← 位置 x(t)
   h(x) ← 微分 ← FailCount F(x)
   
   位置 x → 積分 → v(t)
   FailCount F → 積分 → 
                新しい量


━━━━━━━━━━━━━━━━━━━━━━

結論：
「グラフの下の面積」は
積分の定義そのもの！
"""

    ax.text(
        0.05,
        0.95,
        explanation,
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="steelblue", linewidth=2, alpha=0.98),
    )

    out = "area_under_curve_graph.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

