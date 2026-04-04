"""Generate and visualize an ideal NRZ waveform and a jittered waveform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class JitterWaveformData:
    bits: np.ndarray
    levels: np.ndarray
    time: np.ndarray
    ideal_edges: np.ndarray
    jittered_edges: np.ndarray
    jitter_offsets: np.ndarray
    ideal_waveform: np.ndarray
    jittered_waveform: np.ndarray


@dataclass
class SineEdgeOverlapData:
    time: np.ndarray
    ideal_trace: np.ndarray
    jittered_traces: np.ndarray
    crossing_times: np.ndarray


def generate_bit_pattern(n_bits: int, seed: int) -> np.ndarray:
    """Generate a reproducible bit sequence with enough transitions to visualize jitter."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=int)

    transitions = np.count_nonzero(np.diff(bits))
    if transitions < max(3, n_bits // 4):
        bits = np.arange(n_bits, dtype=int) % 2

    return bits


def bits_to_levels(bits: np.ndarray) -> np.ndarray:
    """Convert 0/1 bits to NRZ levels -1/+1."""
    return np.where(bits > 0, 1.0, -1.0)


def sample_piecewise_constant(
    time: np.ndarray,
    edges: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Sample a piecewise-constant waveform defined by interval edges and levels."""
    interval_index = np.searchsorted(edges[1:], time, side="right")
    interval_index = np.clip(interval_index, 0, len(levels) - 1)
    return levels[interval_index]


def generate_jittered_waveform(
    n_bits: int = 64,
    ui: float = 1.0,
    samples_per_ui: int = 200,
    jitter_sigma_ui: float = 0.04,
    seed: int = 7,
    jitter_clip_ui: float = 0.35,
) -> JitterWaveformData:
    """Generate ideal and jittered NRZ waveforms.

    Args:
        n_bits: Number of unit intervals (bits).
        ui: Unit interval duration.
        samples_per_ui: Time samples per UI for plotting.
        jitter_sigma_ui: RMS jitter in UI units.
        seed: Random seed for reproducibility.
        jitter_clip_ui: Maximum absolute jitter in UI units.
    """
    if n_bits < 4:
        raise ValueError("n_bits must be at least 4.")
    if samples_per_ui < 8:
        raise ValueError("samples_per_ui must be at least 8.")
    if ui <= 0:
        raise ValueError("ui must be positive.")
    if jitter_sigma_ui < 0:
        raise ValueError("jitter_sigma_ui must be non-negative.")
    if jitter_clip_ui < 0 or jitter_clip_ui >= 0.5:
        raise ValueError("jitter_clip_ui must be in [0, 0.5).")

    bits = generate_bit_pattern(n_bits=n_bits, seed=seed)
    levels = bits_to_levels(bits)

    ideal_edges = np.arange(n_bits + 1, dtype=float) * ui
    time = np.linspace(0.0, n_bits * ui, n_bits * samples_per_ui, endpoint=False)

    rng = np.random.default_rng(seed + 1)
    jitter_offsets = rng.normal(loc=0.0, scale=jitter_sigma_ui * ui, size=n_bits - 1)
    jitter_offsets = np.clip(jitter_offsets, -jitter_clip_ui * ui, jitter_clip_ui * ui)

    jittered_edges = ideal_edges.copy()
    jittered_edges[1:-1] += jitter_offsets

    ideal_waveform = sample_piecewise_constant(time, ideal_edges, levels)
    jittered_waveform = sample_piecewise_constant(time, jittered_edges, levels)

    return JitterWaveformData(
        bits=bits,
        levels=levels,
        time=time,
        ideal_edges=ideal_edges,
        jittered_edges=jittered_edges,
        jitter_offsets=jitter_offsets,
        ideal_waveform=ideal_waveform,
        jittered_waveform=jittered_waveform,
    )


def generate_sine_edge_overlap(
    n_traces: int = 80,
    period_ui: float = 1.0,
    samples_per_ui: int = 600,
    jitter_sigma_ui: float = 0.04,
    seed: int = 17,
    window_ui: float = 0.70,
    jitter_clip_ui: float = 0.25,
) -> SineEdgeOverlapData:
    """Generate many sine traces whose zero-crossing times are jittered.

    This models repeated acquisitions of the same edge. When overlaid,
    multiple edges appear piled up around one nominal timing.
    """
    if n_traces < 2:
        raise ValueError("n_traces must be at least 2.")
    if period_ui <= 0:
        raise ValueError("period_ui must be positive.")
    if samples_per_ui < 50:
        raise ValueError("samples_per_ui must be at least 50.")
    if window_ui <= 0:
        raise ValueError("window_ui must be positive.")
    if jitter_sigma_ui < 0:
        raise ValueError("jitter_sigma_ui must be non-negative.")

    half_window = window_ui / 2.0
    time = np.linspace(-half_window, half_window, int(samples_per_ui * window_ui))

    ideal_trace = np.sin(2.0 * np.pi * time / period_ui)

    rng = np.random.default_rng(seed)
    crossing_times = rng.normal(loc=0.0, scale=jitter_sigma_ui, size=n_traces)
    crossing_times = np.clip(crossing_times, -jitter_clip_ui, jitter_clip_ui)

    jittered_traces = np.array(
        [np.sin(2.0 * np.pi * (time - crossing) / period_ui) for crossing in crossing_times],
        dtype=float,
    )

    return SineEdgeOverlapData(
        time=time,
        ideal_trace=ideal_trace,
        jittered_traces=jittered_traces,
        crossing_times=crossing_times,
    )


def plot_jitter_waveform(
    data: JitterWaveformData,
    sine_data: SineEdgeOverlapData,
    zoom_ui_start: float = 8.0,
    zoom_ui_width: float = 16.0,
) -> plt.Figure:
    """Create a figure comparing ideal and jittered waveforms."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), constrained_layout=True)

    zoom_start = zoom_ui_start
    zoom_end = zoom_ui_start + zoom_ui_width
    mask = (data.time >= zoom_start) & (data.time <= zoom_end)

    # 1) Ideal vs jittered waveform in a zoomed region.
    axes[0].step(
        data.time[mask],
        data.ideal_waveform[mask],
        where="post",
        linewidth=2.0,
        color="gray",
        label="Ideal waveform",
    )
    axes[0].step(
        data.time[mask],
        data.jittered_waveform[mask],
        where="post",
        linewidth=1.8,
        color="royalblue",
        alpha=0.9,
        label="Jittered waveform",
    )
    axes[0].set_title("1) Ideal waveform vs jittered waveform")
    axes[0].set_ylabel("Level")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

    # 2) Edge timing error per transition.
    transition_index = np.arange(1, len(data.ideal_edges) - 1)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].plot(
        transition_index,
        data.jitter_offsets,
        "o-",
        color="tomato",
        linewidth=1.4,
        markersize=4,
    )
    axes[1].set_title("2) Edge timing error")
    axes[1].set_xlabel("Transition index")
    axes[1].set_ylabel("Timing error [UI]")
    axes[1].grid(True, alpha=0.3)

    # 3) Histogram of jitter offsets.
    axes[2].hist(
        data.jitter_offsets,
        bins=16,
        color="slateblue",
        alpha=0.75,
        edgecolor="white",
    )
    sigma_ui = float(np.std(data.jitter_offsets))
    axes[2].set_title(f"3) Jitter histogram (measured sigma ≈ {sigma_ui:.4f} UI)")
    axes[2].set_xlabel("Timing error [UI]")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3, axis="y")

    # 4) Overlapped sine edges around one nominal crossing time.
    for trace in sine_data.jittered_traces:
        axes[3].plot(
            sine_data.time,
            trace,
            color="royalblue",
            alpha=0.10,
            linewidth=1.0,
        )
    axes[3].plot(
        sine_data.time,
        sine_data.ideal_trace,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Ideal sine edge",
    )
    axes[3].axvline(0.0, color="gray", linestyle=":", linewidth=1.2, label="Ideal crossing")
    axes[3].scatter(
        sine_data.crossing_times,
        np.zeros_like(sine_data.crossing_times),
        s=16,
        color="tomato",
        alpha=0.70,
        label="Jittered crossing times",
        zorder=3,
    )
    axes[3].set_title("4) Overlapped sine edges at one nominal timing")
    axes[3].set_xlabel("Time [UI]")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8, loc="upper left")

    fig.suptitle("Random jitter visualization with Matplotlib", fontsize=13, fontweight="bold")
    return fig


def main() -> None:
    data = generate_jittered_waveform(
        n_bits=64,
        ui=1.0,
        samples_per_ui=200,
        jitter_sigma_ui=0.04,
        seed=7,
        jitter_clip_ui=0.35,
    )
    sine_data = generate_sine_edge_overlap(
        n_traces=80,
        period_ui=1.0,
        samples_per_ui=600,
        jitter_sigma_ui=0.04,
        seed=17,
        window_ui=0.70,
        jitter_clip_ui=0.25,
    )
    fig = plot_jitter_waveform(data, sine_data)

    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
