"""Show how overlapped sine-edge timing data becomes Fail Count and a histogram."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SineEdgeData:
    time: np.ndarray
    traces: np.ndarray
    ideal_trace: np.ndarray
    crossing_times: np.ndarray


@dataclass
class FailCountAnalysis:
    sample_time: np.ndarray
    fail_count: np.ndarray
    hist_centers: np.ndarray
    hist_density: np.ndarray
    hist_edges: np.ndarray


def generate_sine_edge_data(
    n_traces: int = 120,
    period_ui: float = 1.0,
    samples_in_window: int = 700,
    jitter_sigma_ui: float = 0.04,
    jitter_clip_ui: float = 0.20,
    window_ui: float = 0.80,
    seed: int = 21,
) -> SineEdgeData:
    """Generate many sine edges with jittered zero-crossing time."""
    if n_traces < 4:
        raise ValueError("n_traces must be at least 4.")
    if period_ui <= 0:
        raise ValueError("period_ui must be positive.")
    if samples_in_window < 100:
        raise ValueError("samples_in_window must be at least 100.")
    if jitter_sigma_ui < 0:
        raise ValueError("jitter_sigma_ui must be non-negative.")
    if jitter_clip_ui < 0:
        raise ValueError("jitter_clip_ui must be non-negative.")
    if window_ui <= 0:
        raise ValueError("window_ui must be positive.")

    half_window = window_ui / 2.0
    time = np.linspace(-half_window, half_window, samples_in_window)
    ideal_trace = np.sin(2.0 * np.pi * time / period_ui)

    rng = np.random.default_rng(seed)
    crossing_times = rng.normal(loc=0.0, scale=jitter_sigma_ui, size=n_traces)
    crossing_times = np.clip(crossing_times, -jitter_clip_ui, jitter_clip_ui)

    traces = np.array(
        [np.sin(2.0 * np.pi * (time - crossing) / period_ui) for crossing in crossing_times],
        dtype=float,
    )

    return SineEdgeData(
        time=time,
        traces=traces,
        ideal_trace=ideal_trace,
        crossing_times=crossing_times,
    )


def analyze_failcount_from_crossings(
    crossing_times: np.ndarray,
    sample_time: np.ndarray,
    histogram_bins: int = 24,
) -> FailCountAnalysis:
    """Convert crossing-time spread to Fail Count and histogram.

    Definition used here:
        Fail Count(t) = number of traces whose crossing time is later than t.

    For a rising edge sampled at threshold 0, this is the number of waveforms
    that have not crossed yet at time t. As t increases, Fail Count decreases.
    """
    if crossing_times.ndim != 1:
        raise ValueError("crossing_times must be 1D.")
    if sample_time.ndim != 1:
        raise ValueError("sample_time must be 1D.")
    if crossing_times.size < 4:
        raise ValueError("At least 4 crossing times are required.")
    if np.any(np.diff(sample_time) <= 0):
        raise ValueError("sample_time must be strictly increasing.")

    fail_count = np.sum(crossing_times[:, None] > sample_time[None, :], axis=0).astype(float)
    hist_density, hist_edges = np.histogram(
        crossing_times,
        bins=histogram_bins,
        range=(sample_time.min(), sample_time.max()),
        density=False,
    )
    bin_width = hist_edges[1] - hist_edges[0]
    hist_density = hist_density.astype(float) / bin_width
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    return FailCountAnalysis(
        sample_time=sample_time,
        fail_count=fail_count,
        hist_centers=hist_centers,
        hist_density=hist_density,
        hist_edges=hist_edges,
    )


def plot_sine_edges_to_failcount(
    edge_data: SineEdgeData,
    analysis: FailCountAnalysis,
) -> plt.Figure:
    """Plot the full chain: overlapped edges -> Fail Count -> histogram."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)

    # 1) Overlapped sine edges.
    for trace in edge_data.traces:
        axes[0].plot(edge_data.time, trace, color="royalblue", alpha=0.10, linewidth=1.0)
    axes[0].plot(
        edge_data.time,
        edge_data.ideal_trace,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Ideal sine edge",
    )
    axes[0].axhline(0.0, color="gray", linestyle=":", linewidth=1.0, label="Threshold = 0")
    axes[0].axvline(0.0, color="gray", linestyle="--", linewidth=1.0, label="Ideal crossing")
    axes[0].scatter(
        edge_data.crossing_times,
        np.zeros_like(edge_data.crossing_times),
        color="tomato",
        s=18,
        alpha=0.75,
        zorder=3,
        label="Crossing times",
    )
    axes[0].set_title("1) Overlapped sine edges around one nominal crossing")
    axes[0].set_xlabel("Time [UI]")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    # 2) Fail Count = CCDF of crossing times.
    axes[1].plot(
        analysis.sample_time,
        analysis.fail_count,
        color="darkorange",
        linewidth=2.2,
        label="Fail Count(t) = count(crossing_time > t)",
    )
    axes[1].set_title("2) Fail Count generated from the overlapped-edge data")
    axes[1].set_xlabel("Sample timing t [UI]")
    axes[1].set_ylabel("Fail Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")

    # 3) Histogram of crossing times.
    bin_width = analysis.hist_edges[1] - analysis.hist_edges[0]
    axes[2].bar(
        analysis.hist_centers,
        analysis.hist_density,
        width=bin_width * 0.90,
        color="lightgray",
        alpha=0.85,
        edgecolor="white",
        label="Histogram density",
    )
    axes[2].set_title("3) Histogram of crossing times")
    axes[2].set_xlabel("Time [UI]")
    axes[2].set_ylabel("Count density [count/UI]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Overlapped sine edges -> Fail Count -> histogram",
        fontsize=13,
        fontweight="bold",
    )
    return fig


def main() -> None:
    edge_data = generate_sine_edge_data(
        n_traces=120,
        period_ui=1.0,
        samples_in_window=700,
        jitter_sigma_ui=0.04,
        jitter_clip_ui=0.20,
        window_ui=0.80,
        seed=21,
    )
    analysis = analyze_failcount_from_crossings(
        crossing_times=edge_data.crossing_times,
        sample_time=edge_data.time,
        histogram_bins=24,
    )
    fig = plot_sine_edges_to_failcount(edge_data, analysis)

    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
