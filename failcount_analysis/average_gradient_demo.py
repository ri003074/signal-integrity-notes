from typing import Tuple

import numpy as np
import pandas as pd

try:
    from failcount_analysis.calc_core import (
        AVERAGE_ROW_LABEL,
        compute_average_index_per_code,
        compute_inner_gradient_norm,
    )
    from failcount_analysis.sample_data import build_sample_df
except ModuleNotFoundError:
    from calc_core import (
        AVERAGE_ROW_LABEL,
        compute_average_index_per_code,
        compute_inner_gradient_norm,
    )
    from sample_data import build_sample_df


AVERAGE_PNG = "average_fail_count.png"
GRADIENT_PNG = "fcd_gradient_heatmap.png"


def _print_dataframes(source_df: pd.DataFrame, avg_df: pd.DataFrame, grad_df: pd.DataFrame) -> None:
    print("Original DataFrame:")
    print(source_df)
    print("\nAveraged DataFrame:")
    print(avg_df)
    print("\nGradient DataFrame (center area):")
    print(grad_df)


def _extract_average_series(df_avg: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(df_avg.columns, dtype=float)
    avg_values = np.asarray(df_avg.loc[AVERAGE_ROW_LABEL].values, dtype=float)
    return codes, avg_values


def _plot_average(df_avg: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    codes, avg_values = _extract_average_series(df_avg)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
    ax.plot(codes, avg_values, "o-", linewidth=2, color="tab:blue", label="Average")
    ax.set_title("FailCount Average Example")
    ax.set_xlabel("Column (code)")
    ax.set_ylabel("Average index")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(AVERAGE_PNG, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {AVERAGE_PNG}")


def _plot_gradient_heatmap(source_df: pd.DataFrame, grad_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if grad_df.empty:
        print("Gradient DataFrame is empty. Skipped heatmap output.")
        return

    grad_values = grad_df.to_numpy(dtype=float)
    failcount_max = float(source_df.to_numpy(dtype=float).max())
    cmap_white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#0000ff"])

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
    im = ax.imshow(
        grad_values,
        cmap=cmap_white_to_blue,
        vmin=0.0,
        vmax=failcount_max,
        origin="upper",
        aspect="auto",
    )
    ax.set_title("Gradient Heatmap (from Sample FailCount)")
    ax.set_xlabel("Column (code)")
    ax.set_ylabel("Index")
    ax.set_xticks(np.arange(len(grad_df.columns)))
    ax.set_xticklabels(grad_df.columns)
    ax.set_yticks(np.arange(len(grad_df.index)))
    ax.set_yticklabels(grad_df.index)

    for row_idx in range(grad_values.shape[0]):
        for col_idx in range(grad_values.shape[1]):
            text = str(int(grad_values[row_idx, col_idx]))
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ceiled gradient norm")
    fig.savefig(GRADIENT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved: {GRADIENT_PNG}")


def main() -> None:
    df = build_sample_df()
    df_avg = compute_average_index_per_code(df)
    df_grad = compute_inner_gradient_norm(df)

    _print_dataframes(df, df_avg, df_grad)
    _plot_average(df_avg)
    _plot_gradient_heatmap(df, df_grad)


if __name__ == "__main__":
    main()
