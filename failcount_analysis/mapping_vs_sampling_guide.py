from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Continuous linear mapping from x:[0, 10] to y:[700, 705]
def continuous_mapping(x: np.ndarray) -> np.ndarray:
    return 700.0 + 0.5 * x


# Quantize y to 1 mV grid using half-up rule (e.g., 702.5 -> 703)
def quantize_to_1mv(y: np.ndarray) -> np.ndarray:
    return np.floor((y + 0.5) / 1.0) * 1.0


def main() -> None:
    x = np.arange(0.0, 10.0 + 1.0, 1.0)
    y_cont = continuous_mapping(x)
    y_quant = quantize_to_1mv(y_cont)

    x_target = 5.0
    y_cont_target = continuous_mapping(np.array([x_target]))[0]  # 702.5
    y_quant_target = quantize_to_1mv(np.array([y_cont_target]))[0]  # 703.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Continuous mapping vs 1mV sampled coordinates", fontsize=13, fontweight="bold")

    # Left: concept zoom around x=0..10
    ax0 = axes[0]
    x_zoom = np.linspace(0, 10, 200)
    y_zoom = continuous_mapping(x_zoom)
    ax0.plot(
        x_zoom, y_zoom, color="tab:blue", linewidth=2.2, label="Continuous mapping: y = 700 + 0.5x"
    )

    y_grid = np.arange(700, 706, 1)
    for yv in y_grid:
        ax0.axhline(yv, color="lightgray", linewidth=0.9, linestyle="--", zorder=0)

    ax0.scatter(
        [x_target],
        [y_cont_target],
        s=80,
        color="tab:blue",
        zorder=5,
        label="x=5 -> y=702.5 (continuous)",
    )
    ax0.scatter(
        [x_target],
        [y_quant_target],
        s=80,
        color="tab:orange",
        zorder=5,
        label="x=5 -> y=703 (1mV sampled)",
    )

    ax0.vlines(x_target, y_cont_target, y_quant_target, color="tab:red", linewidth=2)
    ax0.text(
        x_target + 0.3,
        (y_cont_target + y_quant_target) / 2,
        "quantization\n+0.5 mV",
        color="tab:red",
        fontsize=9,
        va="center",
    )

    ax0.set_title("At x=5: 702.5 (continuous) vs 703 (sampled)")
    ax0.set_xlabel("X [mV]")
    ax0.set_ylabel("Y [mV]")
    ax0.set_xlim(0, 10)
    ax0.set_ylim(699.5, 705.5)
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    # Right: full range comparison
    ax1 = axes[1]
    ax1.plot(x, y_cont, color="tab:blue", linewidth=2.0, label="Continuous mapping")
    ax1.step(
        x, y_quant, where="post", color="tab:orange", linewidth=1.8, label="1mV sampled/quantized"
    )

    ax1.scatter([x_target], [y_cont_target], s=55, color="tab:blue", zorder=5)
    ax1.scatter([x_target], [y_quant_target], s=55, color="tab:orange", zorder=5)
    ax1.annotate(
        "same x=5, different y\ncontinuous: 702.5\nsampled: 703",
        xy=(x_target, y_quant_target),
        xytext=(6.0, 704.6),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.95),
    )

    ax1.set_title("Reason: Y is measured on a 1mV grid")
    ax1.set_xlabel("X [mV]")
    ax1.set_ylabel("Y [mV]")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(699.5, 705.5)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

