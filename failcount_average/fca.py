import numpy as np
import pandas as pd


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

    print("Original DataFrame:")
    print(df)
    print("\nAveraged DataFrame:")
    print(df_avg)

    # --- plot: average only ---
    import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    main()
