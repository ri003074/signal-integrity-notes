"""
What does the area under FailCount mean?

Visualizes:
1) Histogram → FailCount (right integration)
2) FailCount → Area (left integration)
3) Explanation of what the stacked area means
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
        font_names = ['Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'Noto Sans CJK JP', 'Segoe UI']
        for font_name in font_names:
            try:
                matplotlib.rcParams['font.family'] = font_name
                break
            except Exception:
                continue
    except Exception:
        pass

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9


def main() -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    
    fig.suptitle(
        "Integral of FailCount: What is the area?\n"
        "Flow: h(x) -> F(x) -> A(x)",
        fontsize=14,
        fontweight="bold",
    )

    # Data
    x_bins = np.array([1, 2, 3, 4, 5])
    h = np.array([10, 30, 50, 20, 5])
    F = np.array([np.sum(h[i:]) for i in range(len(h))])
    A = np.cumsum(F)

    # =========================================================================
    # 上左：ヒストグラム h(x)
    # =========================================================================
    ax_h = plt.subplot(2, 3, 1)
    
    bar_width = 0.6
    ax_h.bar(x_bins, h, width=bar_width, color="steelblue", edgecolor="black", linewidth=1.5, alpha=0.7)
    
    for xi, hi in zip(x_bins, h):
        ax_h.text(xi, hi + 1.5, f"{int(hi)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax_h.set_xlabel("x", fontsize=10)
    ax_h.set_ylabel("Count h(x)", fontsize=10)
    ax_h.set_title("1) Histogram\nh(x) = [10, 30, 50, 20, 5]", fontsize=10, fontweight="bold")
    ax_h.set_ylim(0, 60)
    ax_h.grid(True, alpha=0.3, axis="y")
    
    ax_h.text(0.5, 0.05, "Data distribution", transform=ax_h.transAxes, 
              ha="center", fontsize=9, color="steelblue", fontweight="bold")

    # =========================================================================
    # 上中：FailCount F(x)
    # =========================================================================
    ax_f = plt.subplot(2, 3, 2)
    
    x_plot = np.linspace(0.5, 5.5, 1000)
    # Step function for FailCount
    f_plot = np.zeros_like(x_plot)
    for i, xi in enumerate(x_bins):
        f_plot[x_plot >= xi] = F[i]
    
    ax_f.plot(x_plot, f_plot, "r-", linewidth=3, label="F(x): FailCount")
    ax_f.fill_between(x_plot, 0, f_plot, alpha=0.15, color="red")
    
    for xi, Fi in zip(x_bins, F):
        ax_f.plot(xi, Fi, "ro", markersize=8)
        ax_f.text(xi, Fi + 3, f"{int(Fi)}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="red")
    
    ax_f.set_xlabel("x", fontsize=10)
    ax_f.set_ylabel("Count (cumulative)", fontsize=10)
    ax_f.set_title("2) FailCount\nF(x) = integral[x to +inf] h(t)dt", fontsize=10, fontweight="bold")
    ax_f.set_ylim(0, 130)
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(fontsize=9)
    
    ax_f.text(0.5, 0.05, "Sum h(x) from right = Count >= x", transform=ax_f.transAxes, 
              ha="center", fontsize=8, color="red", fontweight="bold")

    # =========================================================================
    # 上右：面積説明図（FailCountの下の面積）
    # =========================================================================
    ax_area = plt.subplot(2, 3, 3)
    
    # Plot FailCount
    ax_area.plot(x_plot, f_plot, "g-", linewidth=3, label="F(x)")
    
    # Shade the area under curve
    ax_area.fill_between(x_plot, 0, f_plot, alpha=0.3, color="green", label="Area A(x)")
    
    # Mark small rectangles to show integration
    for i in range(len(x_bins) - 1):
        rect_width = 1.0
        rect_height = F[i]
        rect = Rectangle((x_bins[i], 0), rect_width, rect_height, 
                         facecolor="lightgreen", edgecolor="darkgreen", linewidth=1.5, alpha=0.5)
        ax_area.add_patch(rect)
        # Area of each rectangle
        area_val = rect_width * rect_height
        ax_area.text(x_bins[i] + 0.5, rect_height / 2, f"{int(area_val)}", 
                    ha="center", va="center", fontsize=8, fontweight="bold")
    
    ax_area.set_xlabel("x", fontsize=10)
    ax_area.set_ylabel("F(x)", fontsize=10)
    ax_area.set_title("3) Area under FailCount\nA(x) = integral F(u)du", fontsize=10, fontweight="bold")
    ax_area.set_ylim(0, 130)
    ax_area.grid(True, alpha=0.3)
    ax_area.legend(fontsize=9)

    # =========================================================================
    # 下左：数値表（説明用）
    # =========================================================================
    ax_table = plt.subplot(2, 3, 4)
    ax_table.axis("off")
    
    table_text = """
CALCULATION EXAMPLE

Histogram h(x):
  x=1: 10
  x=2: 30
  x=3: 50
  x=4: 20
  x=5: 5

        [sum from right]

FailCount F(x):
  F(1) = 10+30+50+20+5 = 115
  F(2) = 30+50+20+5 = 105
  F(3) = 50+20+5 = 75
  F(4) = 20+5 = 25
  F(5) = 5

        [sum from left]

Area A(x):
  A(1) = 115
  A(2) = 115 + 105 = 220
  A(3) = 220 + 75 = 295
  A(4) = 295 + 25 = 320
  A(5) = 320 + 5 = 325
"""
    
    ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes,
                 fontsize=8, ha="left", va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.95))

    # =========================================================================
    # 下中：グラフの流れ
    # =========================================================================
    ax_flow = plt.subplot(2, 3, 5)
    ax_flow.axis("off")
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 10)
    
    flow_text = """
DIFFERENTIATION & INTEGRATION

h(x) Histogram
     ^ differentiate
     v integrate (from right)
F(x) FailCount
     ^ differentiate
     v integrate (from left)
A(x) Cumulative area

━━━━━━━━━━━━━━━━━

KEY POINTS:

✓ h(x) --(integrate)-> F(x)
✓ F(x) --(diff)-> h(x)

✓ F(x) --(integrate)-> A(x)
✓ A(x) --(diff)-> F(x)

✗ But A(x) does NOT
  differentiate to h(x)

━━━━━━━━━━━━━━━━━

A(x) means:
"Area under FailCount curve"
= Sum of F(x) values

This is correct mathematically
but NOT used in practice.

━━━━━━━━━━━━━━━━━

In practice, use:
  h(x): histogram
  F(x): FailCount (risk)
"""
    
    ax_flow.text(0.05, 0.95, flow_text, transform=ax_flow.transAxes,
                fontsize=8, ha="left", va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", edgecolor="steelblue", 
                         linewidth=2, alpha=0.95))

    # =========================================================================
    # 下右：実用面での意味
    # =========================================================================
    ax_meaning = plt.subplot(2, 3, 6)
    ax_meaning.axis("off")
    ax_meaning.set_xlim(0, 10)
    ax_meaning.set_ylim(0, 10)
    
    meaning_text = """
PRACTICAL MEANING OF A(x)

Q: What does A(x) represent?

A(x) = integral of FailCount

This is "area under curve"
Sum of all F(x) values

━━━━━━━━━━━━━━━━━

IN PRACTICE:

① h(x) = Histogram
     "Where is the data?"
     
② F(x) = FailCount
     "Risk level >= x"
     "Main metric used"
     
③ A(x) = Integral of F(x)
     "No practical use"
     "Math exercise only"


━━━━━━━━━━━━━━━━━

WHY INTEGRATE F(x)?

Q: Why integrate FailCount
   at all?

A: Normally we DON'T!

It's just to understand
the relationship between:
  differentiation
  integration

━━━━━━━━━━━━━━━━━

SUMMARY:

FailCount area A(x)
= integral of F(x)
= sum of all F values

Mathematically correct.
Practically not useful.

REAL USE:
  h(x): histogram
  F(x): FailCount ← MAIN
"""
    
    ax_meaning.text(0.05, 0.95, meaning_text, transform=ax_meaning.transAxes,
                   fontsize=8, ha="left", va="top", family="monospace",
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="mistyrose", alpha=0.95))

    out = "failcount_integral_meaning.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

