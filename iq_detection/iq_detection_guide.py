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

    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].axvline(0.0, color="gray", linewidth=0.8)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("2) IQ plane mapping: circle from two shifted sines")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

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


if __name__ == "__main__":
    main()

