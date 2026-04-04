from typing import Optional, Tuple

import numpy as np
from scipy.special import erfc, ndtri
from scipy.stats import linregress


def q_function(x: np.ndarray) -> np.ndarray:
    """
    Q function: probability that a standard normal variable exceeds x.

        Q(x) = P(X > x),  X ~ N(0, 1)
             = 0.5 * erfc(x / sqrt(2))

    Relationship with erfc:
        Q(x)     = 0.5 * erfc(x / sqrt(2))
        erfc(x)  = 2  * Q(x * sqrt(2))

    Args:
        x: Input value(s). Can be scalar or ndarray.

    Returns:
        Q(x) values in range (0, 1).
    """
    return 0.5 * erfc(x / np.sqrt(2))


def q_inv(ber: np.ndarray) -> np.ndarray:
    """
    Inverse Q function: Q^{-1}(BER)

        Q^{-1}(p) = sqrt(2) * erfinv(1 - 2p)
                  = -ndtri(p)   (ndtri = inverse of normal CDF)

    Args:
        ber: BER values in (0, 1). Values of 0 or 1 are clipped to avoid ±inf.

    Returns:
        Q^{-1}(BER) values.
    """
    ber = np.clip(ber, 1e-15, 1 - 1e-15)
    return -ndtri(ber)  # ndtri(p) = inverse CDF of N(0,1)


def extract_rj(
    x: np.ndarray,
    failcount: np.ndarray,
    max_fail: int = 100,
    ber_range: Tuple[float, float] = (0.05, 0.95),
) -> dict:
    """
    Extract Random Jitter (RJ) sigma from a Fail Count vs. x curve.

    Method:
        BER(x) = Q((x - mu) / sigma_RJ)
        Q^{-1}(BER) = (x - mu) / sigma_RJ   ← linear in x
        slope = 1 / sigma_RJ  →  sigma_RJ = 1 / slope

    Args:
        x:          x-axis values (e.g. timing margin, normalized 0..1).
        failcount:  Fail count values (integers or floats).
        max_fail:   Maximum fail count (used to normalize to BER).
        ber_range:  BER range used for linear fitting (exclude near 0 and 1
                    where Q^{-1} diverges).

    Returns:
        dict with keys:
            sigma_rj   : RJ sigma (same unit as x)
            mu         : Center of transition (x where BER=50%)
            slope      : Fitted slope (= 1/sigma_rj)
            r_squared  : Goodness of fit
            x_fit      : x values used for fitting
            q_inv_fit  : Q^{-1}(BER) values used for fitting
            q_inv_line : Fitted line values
    """
    ber = np.asarray(failcount, dtype=float) / max_fail

    # Keep only the valid fitting range
    mask = (ber > ber_range[0]) & (ber < ber_range[1])
    x_fit = x[mask]
    q_fit = q_inv(ber[mask])

    if len(x_fit) < 3:
        raise ValueError("Not enough data points in ber_range for fitting.")

    # Linear regression: Q^{-1}(BER) = slope * x + intercept
    result = linregress(x_fit, q_fit)
    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2

    sigma_rj = 1.0 / slope          # RJ sigma
    mu = -intercept / slope          # x where Q^{-1}(BER) = 0, i.e. BER = 50%

    return {
        "sigma_rj": sigma_rj,
        "mu": mu,
        "slope": slope,
        "r_squared": r_squared,
        "x_fit": x_fit,
        "q_inv_fit": q_fit,
        "q_inv_line": slope * x_fit + intercept,
    }


def generate_normal_failcount(
    n_rows: int = 21,
    n_cols: int = 21,
    mean: float = 50.0,
    sigma: float = 20.0,
    max_fail: int = 100,
    min_fail: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate fail count data following a normal distribution.

    Each cell value is drawn from N(mean, sigma), then clipped to [min_fail, max_fail].

    Args:
        n_rows:   Number of rows (e.g. voltage steps).
        n_cols:   Number of columns (e.g. phase steps).
        mean:     Center of the normal distribution.
        sigma:    Standard deviation.
        max_fail: Upper limit (default 100).
        min_fail: Lower limit (default 0).
        seed:     Random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_rows, n_cols), dtype int.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=mean, scale=sigma, size=(n_rows, n_cols))
    data = np.clip(data, min_fail, max_fail)
    return data.astype(int)


def generate_normal_failcount_1d(
    n: int = 21,
    center: float = 0.5,
    sigma: float = 0.15,
    max_fail: int = 100,
    min_fail: int = 0,
    noise_sigma: float = 2.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 1D fail count data that smoothly decreases from max to min.

    Uses the complementary CDF of the normal distribution (erfc curve),
    which naturally produces a smooth S-shaped descent from max_fail to min_fail.
    Typical BER-like fail count profile.

    Args:
        n:           Number of data points.
        center:      Normalized position (0.0-1.0) where fail count = 50%.
        sigma:       Steepness of the transition (smaller = steeper).
        max_fail:    Upper limit (default 100).
        min_fail:    Lower limit (default 0).
        noise_sigma: Standard deviation of additive noise (0 = no noise).
        seed:        Random seed for reproducibility.

    Returns:
        np.ndarray of shape (n,), dtype int.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, n)

    # Complementary CDF: smoothly descends from ~1 to ~0
    base = 0.5 * erfc((x - center) / (sigma * np.sqrt(2)))

    # Scale to [min_fail, max_fail]
    data = min_fail + base * (max_fail - min_fail)

    # Add noise
    if noise_sigma > 0:
        data = data + rng.normal(0, noise_sigma, size=n)

    data = np.clip(data, min_fail, max_fail)
    return data.astype(int)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import japanize_matplotlib  # noqa: F401  日本語フォント自動設定
    from scipy.stats import norm as sp_norm


    # --- Generate Fail Count data (with noise) ---
    TRUE_SIGMA = 0.15
    TRUE_CENTER = 0.5
    n = 50

    x = np.linspace(0.0, 1.0, n)
    failcount = generate_normal_failcount_1d(
        n=n, center=TRUE_CENTER, sigma=TRUE_SIGMA,
        max_fail=100, min_fail=0, noise_sigma=2.0, seed=7,
    )

    # --- Extract RJ ---
    rj = extract_rj(x, failcount, max_fail=100, ber_range=(0.05, 0.95))

    print("=== Random Jitter Extraction ===")
    print(f"  True  sigma : {TRUE_SIGMA:.4f}")
    print(f"  True  center: {TRUE_CENTER:.4f}")
    print(f"  Estimated sigma_RJ : {rj['sigma_rj']:.4f}")
    print(f"  Estimated mu       : {rj['mu']:.4f}")
    print(f"  R²                 : {rj['r_squared']:.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 16))

    x_fine = np.linspace(0, 1, 300)
    fit_curve = 100 * q_function((x_fine - rj["mu"]) / rj["sigma_rj"])
    true_curve = 100 * q_function((x_fine - TRUE_CENTER) / TRUE_SIGMA)

    # Fail Count の float 版を計算して数値微分
    failcount_float = 100.0 * 0.5 * erfc((x - TRUE_CENTER) / (TRUE_SIGMA * np.sqrt(2)))
    pdf_numerical = -np.gradient(failcount_float, x)
    pdf_numerical = np.clip(pdf_numerical, 0, None)

    pdf_analytical = 100.0 * sp_norm.pdf(x_fine, loc=TRUE_CENTER, scale=TRUE_SIGMA)
    pdf_fitted     = 100.0 * sp_norm.pdf(x_fine, loc=rj["mu"],    scale=rj["sigma_rj"])

    peak_h = 100.0 * sp_norm.pdf(TRUE_CENTER, loc=TRUE_CENTER, scale=TRUE_SIGMA)

    # 全体フロータイトル
    fig.suptitle(
        "解析フロー:  ① Histogram  ←  −d/dx  ←  ② Fail Count  →  Q⁻¹(BER)  →  ③ Q-scale  →  σ_RJ",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", edgecolor="purple", alpha=0.85),
    )

    # ─── ① Histogram ───────────────────────────────────────
    axes[0].bar(x, pdf_numerical, width=(x[1] - x[0]) * 0.9,
                color="steelblue", alpha=0.6, label="−d(Fail Count)/dx  [numerical]")
    axes[0].plot(x_fine, pdf_analytical, "--", color="gray", linewidth=1.5,
                 label=f"True PDF  σ={TRUE_SIGMA}, μ={TRUE_CENTER}")
    axes[0].plot(x_fine, pdf_fitted, "-", color="tomato", linewidth=2,
                 label=f"Fitted PDF  σ_RJ={rj['sigma_rj']:.3f}, μ={rj['mu']:.3f}")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title("① Jitter Histogram  ( = −d(② Fail Count)/dx )")
    axes[0].axvline(TRUE_CENTER, color="gray", linestyle="--", linewidth=0.8)
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    # ① アノテーション：ピーク
    axes[0].annotate(
        f"ピーク = μ = {TRUE_CENTER}\n最もジッタが多い位置\n（②の最も急な部分に対応）",
        xy=(TRUE_CENTER, peak_h * 0.95),
        xytext=(TRUE_CENTER + 0.18, peak_h * 0.62),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        fontsize=8, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="darkgreen", alpha=0.92),
    )
    # ① アノテーション：左端の低い棒
    axes[0].annotate(
        "低い棒 = ジッタほぼなし\n（②Fail Countが\n平坦な部分に対応）",
        xy=(0.07, peak_h * 0.015),
        xytext=(0.02, peak_h * 0.27),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        fontsize=8, color="steelblue",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", alpha=0.92),
    )
    # ① 説明ボックス
    axes[0].text(
        0.99, 0.97,
        "棒の高さ ∝ Fail Countの傾きの急さ\n= ジッタがそこに存在する確率密度",
        transform=axes[0].transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange", alpha=0.95),
    )

    # ─── ② Fail Count ──────────────────────────────────────
    axes[1].plot(x, failcount, "o", color="steelblue", ms=5, label="Fail Count (measured)")
    axes[1].plot(x_fine, fit_curve, "-", color="tomato", linewidth=2,
                 label=f"Fitted  σ_RJ={rj['sigma_rj']:.3f}, μ={rj['mu']:.3f}")
    axes[1].plot(x_fine, true_curve, "--", color="gray", linewidth=1.5,
                 label=f"True    σ={TRUE_SIGMA}, μ={TRUE_CENTER}")

    # 中心での接線（傾きを可視化）
    tang_slope = -100.0 * sp_norm.pdf(TRUE_CENTER, loc=TRUE_CENTER, scale=TRUE_SIGMA)
    dx_tang = 0.065
    x_tang = np.array([TRUE_CENTER - dx_tang, TRUE_CENTER + dx_tang])
    y_tang = 50.0 + tang_slope * np.array([-dx_tang, dx_tang])
    axes[1].plot(x_tang, y_tang, "-", color="darkgreen", linewidth=2.5,
                 label=f"Tangent at μ  (slope ≈ {tang_slope:.0f})")

    axes[1].set_ylabel("Fail Count")
    axes[1].set_title("② Fail Count  (Complementary CDF)    Fail Count(x) = P(jitter > x)")
    axes[1].set_ylim(-5, 105)
    axes[1].axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    axes[1].axvline(TRUE_CENTER, color="gray", linestyle="--", linewidth=0.8,
                    label=f"center={TRUE_CENTER}")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    # ② アノテーション：左端 平坦
    axes[1].annotate(
        "平坦 → 傾き ≈ 0\n= ここにジッタほぼなし\n（波形がほぼ通らない）",
        xy=(0.06, 97),
        xytext=(0.10, 68),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        fontsize=8, color="steelblue",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", alpha=0.92),
    )
    # ② アノテーション：中心 急峻
    axes[1].annotate(
        "急峻 → 傾き最大\n= ここにジッタが密集\n（①のピークに対応）",
        xy=(TRUE_CENTER, 50),
        xytext=(TRUE_CENTER + 0.16, 74),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        fontsize=8, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="darkgreen", alpha=0.92),
    )
    # ② アノテーション：右端 平坦
    axes[1].annotate(
        "平坦 → 傾き ≈ 0\n= ここにジッタほぼなし",
        xy=(0.87, 3),
        xytext=(0.60, 22),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        fontsize=8, color="steelblue",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", edgecolor="steelblue", alpha=0.92),
    )
    # ② 数式ボックス
    axes[1].text(
        0.99, 0.04,
        "Fail Count(x) = max_fail × Q((x−μ)/σ_RJ)\n= max_fail × 0.5 × erfc((x−μ)/(σ√2))",
        transform=axes[1].transAxes, fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange", alpha=0.95),
    )

    # ─── ③ Q-scale ─────────────────────────────────────────
    axes[2].plot(rj["x_fit"], rj["q_inv_fit"], "o", color="steelblue", ms=5,
                 label="Q⁻¹(BER)  measured")
    axes[2].plot(rj["x_fit"], rj["q_inv_line"], "-", color="tomato", linewidth=2,
                 label=f"Linear fit  slope=1/σ_RJ={1/rj['sigma_rj']:.2f},  R²={rj['r_squared']:.4f}")
    axes[2].set_xlabel("x (normalized)")
    axes[2].set_ylabel("Q⁻¹(BER)")
    axes[2].set_title("③ Q-scale plot    Q⁻¹(BER) = (1/σ_RJ)×x − μ/σ_RJ   →   slope = 1/σ_RJ")
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8, label="y=0 : BER=50% line")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)

    # ③ アノテーション：傾き
    ann_i = len(rj["x_fit"]) * 3 // 4
    ann_x = rj["x_fit"][ann_i]
    ann_y = rj["q_inv_line"][ann_i]
    axes[2].annotate(
        f"傾き = 1/σ_RJ = {1/rj['sigma_rj']:.2f}\n→ σ_RJ = {rj['sigma_rj']:.3f}",
        xy=(ann_x, ann_y),
        xytext=(ann_x - 0.18, ann_y + 0.38),
        arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5),
        fontsize=8, color="tomato",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", edgecolor="tomato", alpha=0.92),
    )
    # ③ アノテーション：μ（y=0 交点）
    axes[2].annotate(
        f"y=0 との交点 = μ = {rj['mu']:.3f}\n（Fail Countが50%になるx）",
        xy=(rj["mu"], 0.0),
        xytext=(rj["mu"] + 0.08, 0.55),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        fontsize=8, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="darkgreen", alpha=0.92),
    )
    # ③ 説明ボックス
    axes[2].text(
        0.01, 0.97,
        "直線 = 純粋な RJ（ガウシアン）\n非直線 = DJ混在の可能性",
        transform=axes[2].transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange", alpha=0.95),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.5)
    plt.show()


