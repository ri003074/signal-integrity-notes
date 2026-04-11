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


def calc_inl_dnl(avg_values):
    """
    Calculate INL and DNL from a series of average crossing values.

    Parameters
    ----------
    avg_values : array-like
        Measured average crossing positions for each code (e.g. df_avg.iloc[0].values).

    Returns
    -------
    dnl : np.ndarray  shape (n-1,)
        DNL[k] = (avg[k+1] - avg[k]) / ideal_step - 1  [LSB]
    inl : np.ndarray  shape (n,)
        INL[k] = (avg[k] - ideal[k]) / ideal_step  [LSB]
    ideal_step : float
        Ideal step size (LSB) computed from end-point straight line.
    """
    avg = np.asarray(avg_values, dtype=float)
    n = len(avg)

    # End-point straight line defines the ideal
    ideal_step = (avg[-1] - avg[0]) / (n - 1)
    ideal = avg[0] + np.arange(n) * ideal_step

    dnl = np.diff(avg) / ideal_step - 1.0  # shape (n-1,)
    inl = (avg - ideal) / ideal_step        # shape (n,)

    return dnl, inl, ideal_step


def main():
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
    df = pd.DataFrame(data, index=index, columns=columns)

    df_avg = apply_averaging(df)

    print("Original DataFrame:")
    print(df)
    print("\nAveraged DataFrame:")
    print(df_avg)

    # --- INL / DNL ---
    codes = np.array(df_avg.columns, dtype=float)
    avg_values = df_avg.loc["Average"].values
    dnl, inl, ideal_step = calc_inl_dnl(avg_values)

    print(f"\nIdeal step (LSB) = {ideal_step:.4f}")
    print("\nDNL [LSB]:")
    for k, v in enumerate(dnl):
        print(f"  code {int(codes[k])} -> {int(codes[k+1])}: {v:+.4f}")
    print("\nINL [LSB]:")
    for k, v in enumerate(inl):
        print(f"  code {int(codes[k])}: {v:+.4f}")

    # --- plots ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    fig.suptitle("FailCount Average / DNL / INL", fontsize=13, fontweight="bold")

    # 1) Average crossing values
    ax0 = axes[0]
    n = len(codes)
    ideal_line = avg_values[0] + np.arange(n) * ideal_step
    ax0.plot(codes, avg_values, "o-", linewidth=2, color="tab:blue", label="Measured avg")
    ax0.plot(codes, ideal_line, "--", linewidth=1.5, color="gray", label="Ideal (end-point line)")
    ax0.set_title("Average Crossing Position")
    ax0.set_xlabel("Column (code)")
    ax0.set_ylabel("Avg index value")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # 1 LSB arrow annotation on the first gap
    x_arrow = codes[0] - (codes[1] - codes[0]) * 0.6
    y0_arrow = ideal_line[0]
    y1_arrow = ideal_line[1]
    ax0.annotate(
        "",
        xy=(x_arrow, y1_arrow),
        xytext=(x_arrow, y0_arrow),
        arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.8),
    )
    ax0.text(
        x_arrow - (codes[1] - codes[0]) * 0.05,
        (y0_arrow + y1_arrow) / 2,
        f"1 LSB\n= {ideal_step:.2f}",
        ha="right",
        va="center",
        fontsize=9,
        color="tab:green",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="tab:green", alpha=0.8),
    )

    # 1 LSB calculation formula text box
    lsb_formula = (
        "1 LSB の計算方法\n"
        "─────────────────────────────\n"
        "1 LSB = (avg[last] - avg[first]) / (N - 1)\n"
        f"      = ({avg_values[-1]:.2f} - {avg_values[0]:.2f}) / ({n} - 1)\n"
        f"      = {avg_values[-1] - avg_values[0]:.2f} / {n - 1}\n"
        f"      = {ideal_step:.4f}"
    )
    ax0.text(
        0.98, 0.97,
        lsb_formula,
        transform=ax0.transAxes,
        ha="right", va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="tab:green", alpha=0.92),
    )

    # 2) DNL
    ax1 = axes[1]
    step_codes = (codes[:-1] + codes[1:]) / 2  # midpoint labels
    bars = ax1.bar(step_codes, dnl, width=codes[1] - codes[0] * 0.7,
                   color="tab:orange", edgecolor="black", alpha=0.8)
    ax1.axhline(0, color="black", linewidth=1.2)
    ax1.axhline(1.0, color="tab:green", linewidth=1.2, linestyle="--", label="+1 LSB limit")
    ax1.axhline(-1.0, color="tab:red", linewidth=1.2, linestyle="--", label="-1 LSB limit")
    ax1.set_title(f"DNL  [measured step / ideal step - 1]   (1 LSB = {ideal_step:.2f})")
    ax1.set_xlabel("Column (code)")
    ax1.set_ylabel("DNL [LSB]")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, dnl):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.02 if val >= 0 else -0.05),
                 f"{val:+.3f}", ha="center",
                 va="bottom" if val >= 0 else "top", fontsize=9)

    # 3) INL
    ax2 = axes[2]
    ax2.plot(codes, inl, "o-", linewidth=2, color="tab:red")
    ax2.axhline(0, color="black", linewidth=1.2)
    ax2.axhline(1.0, color="tab:green", linewidth=1.2, linestyle="--", label="+1 LSB limit")
    ax2.axhline(-1.0, color="tab:red", linewidth=1.2, linestyle="--", alpha=0.5, label="-1 LSB limit")
    ax2.set_title(f"INL  [(measured - ideal) / ideal step]   (1 LSB = {ideal_step:.2f})")
    ax2.set_xlabel("Column (code)")
    ax2.set_ylabel("INL [LSB]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    for x, y in zip(codes, inl):
        ax2.text(x, y + (0.05 if y >= 0 else -0.08),
                 f"{y:+.3f}", ha="center",
                 va="bottom" if y >= 0 else "top", fontsize=9)

    fig.savefig("average_fail_count.png", dpi=150)
    plt.close(fig)
    print("\nSaved: average_fail_count.png")


if __name__ == "__main__":
    main()
