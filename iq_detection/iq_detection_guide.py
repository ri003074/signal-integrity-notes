"""Create separate IQ detection guides for two-sine and single-sine cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_phase_shifted_sines(
    duration_s: float = 2e-3,
    sample_rate_hz: float = 1e6,
    tone_hz: float = 2e3,
    amplitude: float = 1.0,
    phase_shift_rad: float = np.pi / 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate two sine waves with fixed phase shift and their instantaneous phase."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    sample_count = int(duration_s * sample_rate_hz)
    if sample_count < 100:
        raise ValueError("sample_count is too small. Increase duration or sample_rate.")
    if tone_hz <= 0:
        raise ValueError("tone_hz must be positive")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")

    t = np.arange(sample_count, dtype=float) / sample_rate_hz

    phase = 2.0 * np.pi * tone_hz * t
    i_signal = amplitude * np.sin(phase)
    q_signal = amplitude * np.sin(phase + phase_shift_rad)

    return t, i_signal, q_signal, phase


def plot_iq_detection_guide(
    t: np.ndarray,
    i_signal: np.ndarray,
    q_signal: np.ndarray,
    phase: np.ndarray,
    phase_shift_rad: float,
) -> plt.Figure:
    """Backward-compatible wrapper for the two-sine figure."""
    return plot_iq_two_sines_guide(t, i_signal, q_signal, phase, phase_shift_rad)


def _compute_marker_indices(phase: np.ndarray, sample_count: int) -> tuple[np.ndarray, int]:
    phase_step = float(phase[1] - phase[0])
    samples_per_period = max(1, int(round(2.0 * np.pi / phase_step)))
    marker_indices = np.array(
        [0, samples_per_period // 4, samples_per_period // 2, (3 * samples_per_period) // 4],
        dtype=int,
    )
    marker_indices = np.clip(marker_indices, 0, sample_count - 1)
    return marker_indices, samples_per_period


def _plot_time_domain_panel(
    ax: plt.Axes,
    t: np.ndarray,
    i_signal: np.ndarray,
    q_signal: np.ndarray,
    phase_shift_rad: float,
    marker_indices: np.ndarray,
    marker_labels: list[str],
    marker_colors: list[str],
    title: str,
) -> None:
    zoom_count = min(len(t), 300)
    zoom = slice(0, zoom_count)
    ax.plot(t[zoom] * 1e3, i_signal[zoom], color="tab:orange", linewidth=1.8, label="I(t) = A*sin(wt)")
    ax.plot(
        t[zoom] * 1e3,
        q_signal[zoom],
        color="tab:green",
        linewidth=1.8,
        label=f"Q(t) = A*sin(wt + {np.degrees(phase_shift_rad):.0f} deg)",
    )
    y_offsets = [0.12, -0.18, 0.12, -0.18]
    for idx, label, color, y_offset in zip(marker_indices, marker_labels, marker_colors, y_offsets):
        x_ms = t[idx] * 1e3
        ax.scatter(x_ms, i_signal[idx], color=color, s=45, zorder=5)
        ax.scatter(x_ms, q_signal[idx], color=color, s=45, marker="s", zorder=5)
        ax.text(x_ms, i_signal[idx] + y_offset, label, color=color, fontsize=9, fontweight="bold")

    ax.set_title(title)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def plot_iq_two_sines_guide(
    t: np.ndarray,
    i_signal: np.ndarray,
    q_signal: np.ndarray,
    phase: np.ndarray,
    phase_shift_rad: float,
) -> plt.Figure:
    """Two-sine case: IQ circle is visible."""
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)

    marker_labels = ["P1", "P2", "P3", "P4"]
    marker_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    marker_indices, _ = _compute_marker_indices(phase, len(t))

    _plot_time_domain_panel(
        axes[0],
        t,
        i_signal,
        q_signal,
        phase_shift_rad,
        marker_indices,
        marker_labels,
        marker_colors,
        "1) Two sine inputs with phase shift",
    )

    decim = max(1, len(t) // 2500)
    axes[1].plot(i_signal[::decim], q_signal[::decim], color="purple", linewidth=1.8, label="(I(t), Q(t))")
    text_offsets = [(0.06, 0.06), (0.06, -0.09), (-0.14, -0.09), (-0.14, 0.06)]
    for idx, label, color, (dx, dy) in zip(marker_indices, marker_labels, marker_colors, text_offsets):
        axes[1].scatter(i_signal[idx], q_signal[idx], color=color, s=55, zorder=6)
        axes[1].text(i_signal[idx] + dx, q_signal[idx] + dy, label, color=color, fontsize=9, fontweight="bold")

    # atan2 is the angle of the IQ vector from origin.
    atan_demo_idx = int(np.argmin(np.abs(np.mod(np.degrees(phase), 360.0) - 35.0)))
    demo_i = float(i_signal[atan_demo_idx])
    demo_q = float(q_signal[atan_demo_idx])
    demo_deg = float(np.degrees(np.arctan2(demo_q, demo_i)))
    axes[1].annotate(
        "",
        xy=(demo_i, demo_q),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
    )
    axes[1].text(
        demo_i * 0.55,
        demo_q * 0.55,
        "theta = atan2(Q, I)",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    axes[1].text(
        demo_i + 0.08,
        demo_q + 0.05,
        f"(I, Q)=({demo_i:.2f}, {demo_q:.2f})\nangle ~ {demo_deg:.1f} deg",
        fontsize=8,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffde7", edgecolor="#fbc02d", alpha=0.9),
    )

    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].axvline(0.0, color="gray", linewidth=0.8)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("2) IQ plane mapping: circle from two shifted sines")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)

    max_abs = float(max(np.max(np.abs(i_signal)), np.max(np.abs(q_signal))))
    axes[1].set_xlim(-1.15 * max_abs, 1.15 * max_abs)
    axes[1].set_ylim(-1.15 * max_abs, 1.15 * max_abs)

    fig.suptitle("Two-Sine IQ Detection: Phase Is Observable", fontsize=14, fontweight="bold")
    return fig


def plot_iq_single_sine_guide(
    t: np.ndarray,
    i_signal: np.ndarray,
    phase: np.ndarray,
) -> plt.Figure:
    """Single-sine case: IQ trajectory collapses to a line."""
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)

    marker_labels = ["P1", "P2", "P3", "P4"]
    marker_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    marker_indices, _ = _compute_marker_indices(phase, len(t))
    q_single = np.zeros_like(i_signal)

    _plot_time_domain_panel(
        axes[0],
        t,
        i_signal,
        q_single,
        0.0,
        marker_indices,
        marker_labels,
        marker_colors,
        "1) Single sine input only (Q is fixed to 0)",
    )

    decim = max(1, len(t) // 2500)
    axes[1].plot(i_signal[::decim], q_single[::decim], color="dimgray", linewidth=2.0, linestyle="--", label="(I(t), 0)")
    text_offsets = [(0.06, 0.06), (0.06, -0.09), (-0.14, -0.09), (-0.14, 0.06)]
    for idx, label, color, (dx, dy) in zip(marker_indices, marker_labels, marker_colors, text_offsets):
        axes[1].scatter(i_signal[idx], q_single[idx], color=color, s=55, zorder=6)
        axes[1].text(i_signal[idx] + dx, q_single[idx] + dy, label, color=color, fontsize=9, fontweight="bold")

    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].axvline(0.0, color="gray", linewidth=0.8)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("2) IQ plane mapping: line when only one sine is used")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].text(
        0.01,
        0.97,
        "Q is always zero, so the point cannot rotate in IQ plane.\n"
        "Result: phase information is lost (ambiguous).",
        transform=axes[1].transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    max_abs = float(np.max(np.abs(i_signal)))
    axes[1].set_xlim(-1.15 * max_abs, 1.15 * max_abs)
    axes[1].set_ylim(-1.15 * max_abs, 1.15 * max_abs)

    fig.suptitle("Single-Sine Case: Phase Is Not Fully Observable", fontsize=14, fontweight="bold")
    return fig


def plot_iq_stability_guide(
    t: np.ndarray,
    i_signal: np.ndarray,
    q_signal: np.ndarray,
    phase: np.ndarray,
) -> plt.Figure:
    """Explain why IQ phase estimation is more stable than single-sine estimation."""
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12), constrained_layout=True)

    phase_deg = np.degrees(phase)
    one_period_mask = phase <= (2.0 * np.pi)
    phase_one_period = phase_deg[one_period_mask]
    i_one_period = i_signal[one_period_mask]
    q_one_period = q_signal[one_period_mask]

    sample_points_deg = np.array([0.0, 90.0, 180.0, 270.0])
    sample_indices = [int(np.argmin(np.abs(phase_one_period - deg))) for deg in sample_points_deg]
    marker_labels = ["P1", "P2", "P3", "P4"]
    marker_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    # 1) Single-sine weakness: same voltage noise, very different phase ambiguity.
    ax0 = axes[0]
    ax0.plot(phase_one_period, i_one_period, color="tab:orange", linewidth=2.0, label="Single signal: sin(phase)")
    noise_amp = 0.18
    inspect_points = [30.0, 90.0]
    inspect_colors = ["#8e24aa", "#e53935"]
    inspect_texts = [
        "Near zero crossing:\nsmall voltage noise -> small phase error",
        "Near peak:\nsame voltage noise -> large phase ambiguity",
    ]
    for deg, color, text in zip(inspect_points, inspect_colors, inspect_texts):
        idx = int(np.argmin(np.abs(phase_one_period - deg)))
        x_val = float(phase_one_period[idx])
        y_val = float(i_one_period[idx])
        ax0.scatter(x_val, y_val, color=color, s=55, zorder=5)
        ax0.vlines(x_val, y_val - noise_amp, y_val + noise_amp, color=color, linewidth=3, alpha=0.8)
        ax0.text(x_val + 10, y_val + 0.08, text, color=color, fontsize=9, va="center")

    ax0.set_title("1) One sine only: phase stability depends on where you sample")
    ax0.set_xlabel("Phase [deg]")
    ax0.set_ylabel("Amplitude")
    ax0.set_xlim(0, 360)
    ax0.set_ylim(-1.25, 1.25)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower left", fontsize=9)

    # 2) Complementarity of I/Q: one is steep when the other is flat.
    ax1 = axes[1]
    ax1.plot(phase_one_period, i_one_period, color="tab:orange", linewidth=2.0, label="I = sin(phase)")
    ax1.plot(phase_one_period, q_one_period, color="tab:green", linewidth=2.0, label="Q = cos(phase)")
    for idx, label, color in zip(sample_indices, marker_labels, marker_colors):
        x_val = float(phase_one_period[idx])
        y_i = float(i_one_period[idx])
        y_q = float(q_one_period[idx])
        ax1.scatter(x_val, y_i, color=color, s=45, zorder=5)
        ax1.scatter(x_val, y_q, color=color, s=45, marker="s", zorder=5)
        ax1.text(x_val + 6, y_i + 0.1, label, color=color, fontsize=9, fontweight="bold")

    ax1.text(96, 0.88, "I is flat here,\nbut Q changes fast", color="tab:green", fontsize=9)
    ax1.text(4, -1.08, "Q is flat here,\nbut I changes fast", color="tab:orange", fontsize=9)
    ax1.text(186, 0.88, "Again, one weakens\nwhile the other helps", color="black", fontsize=9)
    ax1.set_title("2) IQ complementarity: one channel helps when the other becomes insensitive")
    ax1.set_xlabel("Phase [deg]")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-1.25, 1.25)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

    # 3) Normalized phase sensitivity comparison.
    ax2 = axes[2]
    cos_abs = np.abs(np.cos(np.radians(phase_one_period)))
    single_sensitivity = 1.0 / np.maximum(cos_abs, 0.08)
    single_sensitivity = np.clip(single_sensitivity, 0.0, 6.0)
    iq_sensitivity = np.ones_like(single_sensitivity)
    ax2.plot(
        phase_one_period,
        single_sensitivity,
        color="crimson",
        linewidth=2.0,
        label="Single sine: normalized phase sensitivity",
    )
    ax2.plot(
        phase_one_period,
        iq_sensitivity,
        color="navy",
        linewidth=2.2,
        label="IQ atan2(Q, I): nearly uniform sensitivity",
    )
    ax2.fill_between(
        phase_one_period,
        iq_sensitivity,
        single_sensitivity,
        where=single_sensitivity >= iq_sensitivity,
        color="#ffcdd2",
        alpha=0.45,
    )
    ax2.text(88, 5.45, "Single-sine gets unstable near peaks", color="crimson", fontsize=9)
    ax2.text(192, 1.12, "IQ stays much more uniform", color="navy", fontsize=9)
    ax2.set_title("3) Why IQ is more stable: phase error does not blow up at specific phases")
    ax2.set_xlabel("Phase [deg]")
    ax2.set_ylabel("Relative sensitivity / phase error")
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 6.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("Why IQ Detection Is More Stable", fontsize=15, fontweight="bold")
    return fig


def main() -> None:
    phase_shift_rad = np.pi / 2.0
    t, i_signal, q_signal, phase = generate_phase_shifted_sines(phase_shift_rad=phase_shift_rad)

    two_sines_fig = plot_iq_two_sines_guide(t, i_signal, q_signal, phase, phase_shift_rad)
    two_sines_path = Path(__file__).with_name("iq_detection_two_sines_guide.png")
    two_sines_fig.savefig(two_sines_path, dpi=150, bbox_inches="tight")
    plt.close(two_sines_fig)
    print(f"Saved: {two_sines_path}")

    single_sine_fig = plot_iq_single_sine_guide(t, i_signal, phase)
    single_sine_path = Path(__file__).with_name("iq_detection_single_sine_guide.png")
    single_sine_fig.savefig(single_sine_path, dpi=150, bbox_inches="tight")
    plt.close(single_sine_fig)
    print(f"Saved: {single_sine_path}")

    stability_fig = plot_iq_stability_guide(t, i_signal, q_signal, phase)
    stability_path = Path(__file__).with_name("iq_detection_stability_guide.png")
    stability_fig.savefig(stability_path, dpi=150, bbox_inches="tight")
    plt.close(stability_fig)
    print(f"Saved: {stability_path}")


if __name__ == "__main__":
    main()

