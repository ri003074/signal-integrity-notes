import numpy as np

try:
    from failcount_analysis.calc_core import AVERAGE_ROW_LABEL, apply_averaging
    from failcount_analysis.sample_data import build_sample_df
except ModuleNotFoundError:
    from calc_core import AVERAGE_ROW_LABEL, apply_averaging
    from sample_data import build_sample_df


def calc_inl_dnl(avg_values):
    avg = np.asarray(avg_values, dtype=float)
    n = len(avg)

    ideal_step = (avg[-1] - avg[0]) / (n - 1)
    ideal = avg[0] + np.arange(n) * ideal_step

    dnl = np.diff(avg) / ideal_step - 1.0
    inl = (avg - ideal) / ideal_step
    return dnl, inl, ideal_step, ideal


def main() -> None:
    df = build_sample_df()
    df_avg = apply_averaging(df)

    codes = np.array(df_avg.columns, dtype=float)
    avg_values = df_avg.loc[AVERAGE_ROW_LABEL].values
    dnl, inl, ideal_step, ideal = calc_inl_dnl(avg_values)

    print(f"Ideal step (LSB) = {ideal_step:.4f}")
    print("DNL [LSB]:")
    for k, v in enumerate(dnl):
        print(f"  code {int(codes[k])} -> {int(codes[k + 1])}: {v:+.4f}")
    print("INL [LSB]:")
    for k, v in enumerate(inl):
        print(f"  code {int(codes[k])}: {v:+.4f}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    fig.suptitle("INL / DNL from FailCount Average", fontsize=13, fontweight="bold")

    ax0 = axes[0]
    ax0.plot(codes, avg_values, "o-", linewidth=2, color="tab:blue", label="Measured avg")
    ax0.plot(codes, ideal, "--", linewidth=1.5, color="gray", label="Ideal (end-point line)")
    ax0.set_title("Average Crossing Position")
    ax0.set_xlabel("Column (code)")
    ax0.set_ylabel("Average index")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1 = axes[1]
    step_codes = (codes[:-1] + codes[1:]) / 2
    bars = ax1.bar(step_codes, dnl, color="tab:orange", edgecolor="black", alpha=0.8)
    ax1.axhline(0, color="black", linewidth=1.2)
    ax1.set_title(f"DNL (1 LSB = {ideal_step:.2f})")
    ax1.set_xlabel("Column (code)")
    ax1.set_ylabel("DNL [LSB]")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, dnl):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.02 if val >= 0 else -0.05),
            f"{val:+.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    ax2 = axes[2]
    ax2.plot(codes, inl, "o-", linewidth=2, color="tab:red")
    ax2.axhline(0, color="black", linewidth=1.2)
    ax2.set_title(f"INL (1 LSB = {ideal_step:.2f})")
    ax2.set_xlabel("Column (code)")
    ax2.set_ylabel("INL [LSB]")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(codes, inl):
        ax2.text(
            x,
            y + (0.05 if y >= 0 else -0.08),
            f"{y:+.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
        )

    fig.savefig("inl_dnl_from_failcount_average.png", dpi=150)
    plt.close(fig)
    print("Saved: inl_dnl_from_failcount_average.png")


if __name__ == "__main__":
    main()


