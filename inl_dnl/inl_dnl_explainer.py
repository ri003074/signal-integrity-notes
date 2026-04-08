"""
INL / DNL explanation program.

Creates a simple DAC transfer example and visualizes:
1) ideal code step
2) measured code step
3) DNL per code
4) INL accumulation
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Japanese font fallback
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    try:
        for font_name in ["Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "Segoe UI"]:
            matplotlib.rcParams["font.family"] = font_name
            break
    except Exception:
        pass

matplotlib.rcParams["axes.unicode_minus"] = False


def main() -> None:
    # Example: 3-bit DAC (codes 0..7 -> 7 steps)
    codes = np.arange(8)
    ideal_lsb = 1.0

    # Step widths between adjacent codes (7 values)
    # Ideal is 1.0 for all. Measured has slight distortion.
    measured_steps = np.array([0.95, 1.05, 1.12, 0.88, 1.02, 0.98, 1.00])
    dnl = measured_steps / ideal_lsb - 1.0

    # Transition levels from code 0 baseline
    ideal_levels = np.insert(np.cumsum(np.full(7, ideal_lsb)), 0, 0.0)
    measured_levels = np.insert(np.cumsum(measured_steps), 0, 0.0)
    inl = (measured_levels - ideal_levels) / ideal_lsb

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "INL / DNL の基礎\nDNL: 各ステップ幅の誤差, INL: 累積したズレ",
        fontsize=14,
        fontweight="bold",
    )

    # (1) Transfer curve
    ax0 = axes[0, 0]
    ax0.step(codes, ideal_levels, where="post", linewidth=2.5, label="Ideal", color="black")
    ax0.step(codes, measured_levels, where="post", linewidth=2.5, label="Measured", color="tab:blue")
    ax0.set_title("1) 伝達特性（理想 vs 実測）", fontweight="bold")
    ax0.set_xlabel("Code")
    ax0.set_ylabel("Output [LSB]")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # (2) DNL bar
    ax1 = axes[0, 1]
    step_index = np.arange(1, 8)
    bars = ax1.bar(step_index, dnl, color="tab:orange", edgecolor="black", alpha=0.8)
    ax1.axhline(0.0, color="black", linewidth=1.2)
    ax1.set_title("2) DNL（各コードのステップ誤差）", fontweight="bold")
    ax1.set_xlabel("Transition (code k -> k+1)")
    ax1.set_ylabel("DNL [LSB]")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, dnl):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + (0.02 if val >= 0 else -0.05), f"{val:+.2f}",
                 ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    # (3) INL line
    ax2 = axes[1, 0]
    ax2.plot(codes, inl, "o-", linewidth=2.5, color="tab:red")
    ax2.axhline(0.0, color="black", linewidth=1.2)
    ax2.set_title("3) INL（理想直線からの累積ズレ）", fontweight="bold")
    ax2.set_xlabel("Code")
    ax2.set_ylabel("INL [LSB]")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(codes, inl):
        ax2.text(x, y + (0.08 if y >= 0 else -0.1), f"{y:+.2f}", ha="center",
                 va="bottom" if y >= 0 else "top", fontsize=9)

    # (4) Text explanation
    ax3 = axes[1, 1]
    ax3.axis("off")
    explain = (
        "INL / DNL の意味\n"
        "=================\n\n"
        "DNL (Differential Non-Linearity)\n"
        "  DNL[k] = (実ステップ幅[k] / 理想LSB) - 1\n"
        "  -> 1つ1つのコード幅が理想からどれだけズレたか\n\n"
        "INL (Integral Non-Linearity)\n"
        "  INL[k] = (実際レベル[k] - 理想レベル[k]) / 理想LSB\n"
        "  -> これまでの誤差が積み上がって、全体でどれだけズレたか\n\n"
        "関係\n"
        "  INL は DNL を累積したもの（概念的に）\n"
        "  DNL が + 側に続くと INL は上方向へドリフトする\n\n"
        "設計での見方\n"
        "  - DNL: コード抜けや局所的な段差異常の確認\n"
        "  - INL: 全体直線性の確認"
    )
    ax3.text(
        0.02,
        0.98,
        explain,
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.95),
    )

    out = Path(__file__).with_name("inl_dnl_explanation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("INL / DNL summary:")
    print(f"  DNL range: {np.min(dnl):+.2f} to {np.max(dnl):+.2f} LSB")
    print(f"  INL range: {np.min(inl):+.2f} to {np.max(inl):+.2f} LSB")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
