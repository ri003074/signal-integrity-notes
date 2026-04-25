from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


V_HIGH = 0.5
FREQ_HZ = 500e6
T_START_NS = 0.0
T_END_NS = 40.0
BURST_ON_NS = 8.0
BURST_OFF_NS = 28.0
N_POINTS = 8000


def main() -> None:
    t_ns = np.linspace(T_START_NS, T_END_NS, N_POINTS)
    t_s = t_ns * 1e-9

    burst_gate = ((t_ns >= BURST_ON_NS) & (t_ns <= BURST_OFF_NS)).astype(float)
    toggle = (np.sin(2.0 * np.pi * FREQ_HZ * t_s) >= 0.0).astype(float)

    v_out = V_HIGH * burst_gate * toggle

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(t_ns, v_out, drawstyle="steps-post", linewidth=2.2, color="#1565C0", label="Burst toggle")

    ax.axvspan(BURST_ON_NS, BURST_OFF_NS, color="#FFF3E0", alpha=0.45, zorder=0, label="Burst window")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(V_HIGH, color="#2E7D32", linestyle="--", linewidth=1.2, label="High = 0.5 V")

    ax.set_title("Burst Toggle Waveform (0 V / 0.5 V)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.set_xlim(T_START_NS, T_END_NS)
    ax.set_ylim(-0.05, 0.58)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)

    ax.text(
        BURST_ON_NS + 0.5,
        0.53,
        f"Burst ON: {BURST_ON_NS:.0f} ns to {BURST_OFF_NS:.0f} ns",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#FB8C00", alpha=0.9),
    )

    out_path = Path(__file__).with_name("burst_toggle.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
