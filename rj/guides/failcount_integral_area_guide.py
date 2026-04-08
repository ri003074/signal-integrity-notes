"""
FailCount integral area guide.

This figure explains three layers with one x-axis:
1) Histogram/PDF h(x)
2) FailCount F(x) = integral from x to +inf of h(t) dt
3) Area of FailCount A(x0) = integral from xmin to x0 of F(u) du
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    # Keep script runnable even if japanese font helper is not installed.
    pass


def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def normal_ccdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 - erf(z))


def main() -> None:
    x = np.linspace(-4.0, 4.0, 801)
    dx = x[1] - x[0]

    h = normal_pdf(x)
    f = normal_ccdf(x)
    a = np.cumsum(f) * dx  # A(x) = integral of FailCount from left edge to x

    x0 = 0.8
    i0 = int(np.argmin(np.abs(x - x0)))
    x0 = x[i0]

    f0 = f[i0]
    a0 = a[i0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle("FailCount and Integral Area", fontsize=13, fontweight="bold")

    # 1) PDF and right-tail area
    ax = axes[0]
    ax.plot(x, h, color="tab:blue", lw=2, label="h(x): histogram/PDF")
    ax.fill_between(x[i0:], 0.0, h[i0:], color="tab:blue", alpha=0.25, label="tail area")
    ax.axvline(x0, color="gray", ls="--", lw=1.2)
    ax.set_title("1) Tail area in PDF")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.03,
        0.97,
        f"F(x0) = integral_x0^inf h(t)dt\n(x0={x0:.2f}, F(x0)={f0:.3f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # 2) FailCount curve
    ax = axes[1]
    ax.plot(x, f, color="tab:red", lw=2.2, label="F(x): FailCount (CCDF)")
    ax.plot([x0], [f0], "o", color="black", ms=5)
    ax.axvline(x0, color="gray", ls="--", lw=1.2)
    ax.axhline(f0, color="gray", ls=":", lw=1.2)
    ax.set_title("2) FailCount is already an area")
    ax.set_xlabel("x")
    ax.set_ylabel("FailCount")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.03,
        0.97,
        "Point value F(x0) equals\nright-side area of panel 1",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # 3) Integral of FailCount
    ax = axes[2]
    ax.plot(x, a, color="tab:green", lw=2.2, label="A(x)=integral F(u)du")
    ax.axvline(x0, color="gray", ls="--", lw=1.2)
    ax.fill_between(x[: i0 + 1], 0.0, f[: i0 + 1], color="tab:red", alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(x, f, color="tab:red", lw=1.2, alpha=0.75)
    ax2.set_ylabel("F(x) reference", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    ax.set_title("3) Integrating FailCount")
    ax.set_xlabel("x")
    ax.set_ylabel("A(x)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(
        0.03,
        0.97,
        f"A(x0)=integral_xmin^x0 F(u)du\n(x0={x0:.2f}, A(x0)={a0:.3f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    out = "failcount_integral_area_guide.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

