import numpy as np
import pandas as pd

try:
    from failcount_average.fcd import calculate_gradient
except ModuleNotFoundError:
    from fcd import calculate_gradient


def apply_averaging(df):
    # Calculate the average of each row and add it as a new column

    df_avg = pd.DataFrame()
    length = len(df.index)
    max_fail = df.max().max() * length
    index_min = df.index.min()
    index_max = df.index.max()
    for col in df.columns:
        sum_val = df[col].sum()
        avg = (1 - sum_val / max_fail) * (index_max - index_min) + index_min
        df_avg[col] = [avg]

    df_avg.index = ["Average"]
    return df_avg


def build_sample_df() -> pd.DataFrame:
    data = [
        [100, 100, 100, 100, 100, 0],
        [100, 100, 100, 100, 20, 0],
        [100, 100, 100, 40, 0, 0],
        [100, 100, 60, 0, 0, 0],
        [100, 80, 0, 0, 0, 0],
        [100, 0, 0, 0, 0, 0],
    ]
    index = [60, 50, 40, 30, 20, 10]
    columns = [0, 10, 20, 30, 40, 50]
    return pd.DataFrame(data, index=index, columns=columns)


def main():
    df = build_sample_df()

    df_avg = apply_averaging(df)
    df_grad = calculate_gradient(df)

    print("Original DataFrame:")
    print(df)
    print("\nAveraged DataFrame:")
    print(df_avg)
    print("\nGradient DataFrame (center area):")
    print(df_grad)

    # --- plot: average only ---
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    codes = np.array(df_avg.columns, dtype=float)
    avg_values = df_avg.loc["Average"].values
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8), constrained_layout=True)
    ax.plot(codes, avg_values, "o-", linewidth=2, color="tab:blue", label="Average")
    ax.set_title("FailCount Average Example")
    ax.set_xlabel("Column (code)")
    ax.set_ylabel("Average index")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig("average_fail_count.png", dpi=150)
    plt.close(fig)
    print("\nSaved: average_fail_count.png")

    # --- plot: gradient heatmap ---
    if not df_grad.empty:
        grad_values = df_grad.to_numpy(dtype=float)
        failcount_max = float(df.to_numpy(dtype=float).max())
        fig2, ax2 = plt.subplots(1, 1, figsize=(6.5, 5), constrained_layout=True)
        cmap_white_to_blue = LinearSegmentedColormap.from_list(
            "white_to_blue", ["#ffffff", "#0000ff"]
        )
        im = ax2.imshow(
            grad_values,
            cmap=cmap_white_to_blue,
            vmin=0.0,
            vmax=failcount_max,
            origin="upper",
            aspect="auto",
        )
        ax2.set_title("Gradient Heatmap (from Sample FailCount)")
        ax2.set_xlabel("Column (code)")
        ax2.set_ylabel("Index")
        ax2.set_xticks(np.arange(len(df_grad.columns)))
        ax2.set_xticklabels(df_grad.columns)
        ax2.set_yticks(np.arange(len(df_grad.index)))
        ax2.set_yticklabels(df_grad.index)

        # セル中央に値を表示して勾配強度を読み取りやすくする
        for r in range(grad_values.shape[0]):
            for c in range(grad_values.shape[1]):
                ax2.text(c, r, str(int(grad_values[r, c])), ha="center", va="center", color="white", fontsize=9)

        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label("Ceiled gradient norm")

        fig2.savefig("fcd_gradient_heatmap.png", dpi=150)
        plt.close(fig2)
        print("Saved: fcd_gradient_heatmap.png")
    else:
        print("Gradient DataFrame is empty. Skipped heatmap output.")


if __name__ == "__main__":
    main()
