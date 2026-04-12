import numpy as np

try:
    from failcount_analysis.calc_core import AVERAGE_ROW_LABEL, compute_average_index_per_code, compute_inner_gradient_norm
    from failcount_analysis.sample_data import build_sample_df
except ModuleNotFoundError:
    from calc_core import AVERAGE_ROW_LABEL, compute_average_index_per_code, compute_inner_gradient_norm
    from sample_data import build_sample_df


OUTPUT_PNG = "averaging_gradient_guide.png"


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    df = build_sample_df()
    df_avg = compute_average_index_per_code(df)
    df_grad = compute_inner_gradient_norm(df)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    # Panel 1: original FailCount map
    base_values = df.to_numpy(dtype=float)
    base_max = float(base_values.max()) if base_values.size else 1.0
    cmap_white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#0000ff"])
    im0 = ax0.imshow(
        base_values,
        cmap=cmap_white_to_blue,
        vmin=0.0,
        vmax=base_max,
        origin="upper",
        aspect="equal",
    )
    ax0.set_box_aspect(1.0)
    ax0.set_title("1) Input FailCount")
    ax0.set_xlabel("Code")
    ax0.set_ylabel("Index")
    ax0.set_xticks(np.arange(len(df.columns)))
    ax0.set_xticklabels(df.columns)
    ax0.set_yticks(np.arange(len(df.index)))
    ax0.set_yticklabels(df.index)
    for r in range(base_values.shape[0]):
        for c in range(base_values.shape[1]):
            ax0.text(
                c,
                r,
                str(int(base_values[r, c])),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    # Panel 2: averaging result by code
    codes = np.asarray(df_avg.columns, dtype=float)
    avg_values = np.asarray(df_avg.loc[AVERAGE_ROW_LABEL].values, dtype=float)
    ax1.plot(codes, avg_values, "o-", linewidth=2, color="tab:blue")
    ax1.set_title("2) Averaging per Code")
    ax1.set_xlabel("Code")
    ax1.set_ylabel("Average index")
    ax1.set_xticks(codes)
    ax1.margins(x=0.02)
    ax1.set_box_aspect(1.0)
    ax1.grid(True, alpha=0.3)


    # Panel 3: gradient norm map from center cells
    grad_values = df_grad.to_numpy(dtype=float)
    im2 = ax2.imshow(
        grad_values,
        cmap=cmap_white_to_blue,
        vmin=0.0,
        vmax=base_max,
        origin="upper",
        aspect="equal",
    )
    ax2.set_box_aspect(1.0)
    ax2.set_title("3) Gradient Magnitude")
    ax2.set_xlabel("Inner code")
    ax2.set_ylabel("Inner index")
    ax2.set_xticks(np.arange(len(df_grad.columns)))
    ax2.set_xticklabels(df_grad.columns)
    ax2.set_yticks(np.arange(len(df_grad.index)))
    ax2.set_yticklabels(df_grad.index)
    for r in range(grad_values.shape[0]):
        for c in range(grad_values.shape[1]):
            ax2.text(
                c,
                r,
                str(int(grad_values[r, c])),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )


    fig.suptitle("FailCount averaging and gradient overview", fontsize=13)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
