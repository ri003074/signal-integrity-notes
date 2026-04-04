"""Streamlit app: plot Fail Count, derivative histogram, and RJ from Q-transform."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.special import erfc

try:
    from rj.rj_app_core import analyze_failcount, parse_csv_content, parse_numeric_series
except ModuleNotFoundError:
    from rj_app_core import analyze_failcount, parse_csv_content, parse_numeric_series


st.set_page_config(page_title="Fail Count to RJ", layout="wide")
st.title("Fail Count -> Derivative -> Q-transform -> RJ")
st.caption("Input Fail Count data and optionally x values to estimate Random Jitter (RJ).")

with st.sidebar:
    st.header("Settings")
    max_fail = st.number_input("max_fail", min_value=1, value=100, step=1)
    ber_low = st.slider("ber_range low", min_value=0.001, max_value=0.49, value=0.05, step=0.001)
    ber_high = st.slider("ber_range high", min_value=0.51, max_value=0.999, value=0.95, step=0.001)
    input_mode = st.radio(
        "Input mode",
        options=["Sample", "Paste values", "Upload CSV"],
        index=0,
    )

failcount = None
x_vals = None

if input_mode == "Sample":
    n_points = st.slider("Sample points", min_value=20, max_value=200, value=50, step=5)
    center = st.slider("Sample center", min_value=0.1, max_value=0.9, value=0.5, step=0.01)
    sigma = st.slider("Sample sigma", min_value=0.03, max_value=0.30, value=0.15, step=0.01)
    noise = st.slider("Sample noise", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    seed = st.number_input("Sample seed", min_value=0, value=7, step=1)

    rng = np.random.default_rng(seed)
    x_vals = np.linspace(0.0, 1.0, n_points)
    failcount = max_fail * 0.5 * erfc((x_vals - center) / (sigma * np.sqrt(2)))
    failcount = np.clip(failcount + rng.normal(0, noise, size=n_points), 0, max_fail)

elif input_mode == "Paste values":
    st.markdown("Paste Fail Count values (comma/space/newline separated).")
    fail_text = st.text_area("Fail Count", value="100,96,90,75,55,35,18,8,2,0", height=120)
    x_text = st.text_area("x values (optional)", value="", height=80)

    if fail_text.strip():
        try:
            failcount = parse_numeric_series(fail_text)
            if x_text.strip():
                x_vals = parse_numeric_series(x_text)
        except ValueError as exc:
            st.error(str(exc))

else:
    st.markdown("Upload CSV with either:")
    st.markdown("- columns: `x,failcount` (header), or")
    st.markdown("- one column (failcount only), or")
    st.markdown("- two columns without header (`x,failcount`).")
    uploaded = st.file_uploader("CSV file", type=["csv", "txt"])
    if uploaded is not None:
        try:
            text = uploaded.read().decode("utf-8")
            failcount, x_vals = parse_csv_content(text)
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")

if failcount is None:
    st.info("Provide input data to run analysis.")
    st.stop()

result = analyze_failcount(
    failcount=failcount,
    x=x_vals,
    max_fail=max_fail,
    ber_range=(ber_low, ber_high),
)

if result.warning:
    st.warning(result.warning)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Points", f"{result.x.size}")
with col2:
    st.metric("BER min", f"{result.ber.min():.4f}")
with col3:
    st.metric("BER max", f"{result.ber.max():.4f}")
with col4:
    st.metric("RJ RMS sigma", "-" if result.rms_sigma is None else f"{result.rms_sigma:.5f}")
with col5:
    st.metric("RJ RMS mu", "-" if result.rms_mu is None else f"{result.rms_mu:.5f}")

if result.rj is not None:
    st.success(
        f"RJ (Q-fit): sigma_rj={result.rj['sigma_rj']:.5f}, "
        f"mu={result.rj['mu']:.5f}, R^2={result.rj['r_squared']:.5f}"
    )
else:
    st.error("RJ fit failed. Try adjusting ber_range or data quality.")

fig, axes = plt.subplots(3, 1, figsize=(10, 13))

# 1) Fail Count and BER
ax = axes[0]
ax.plot(result.x, result.failcount, "o-", color="steelblue", label="Fail Count")
ax.set_ylabel("Fail Count")
ax.grid(True, alpha=0.35)
ax.set_title("1) Input Fail Count")
ax2 = ax.twinx()
ax2.plot(result.x, result.ber, "--", color="darkorange", label="BER")
ax2.set_ylabel("BER")

# 2) Derivative
axes[1].bar(
    result.x,
    result.pdf_clipped,
    width=(result.x[1] - result.x[0]) * 0.85,
    color="slateblue",
    alpha=0.65,
    label="-d(Fail Count)/dx (clipped >=0)",
)
axes[1].plot(result.x, result.pdf_raw, "-", color="gray", linewidth=1.2, label="raw derivative")
axes[1].set_title("2) Numerical Derivative (Histogram-equivalent)")
axes[1].set_ylabel("Density-like value")
axes[1].grid(True, alpha=0.35)
axes[1].legend(fontsize=8)

# 3) Q-transform and RJ fit
if result.rj is not None:
    axes[2].plot(result.rj["x_fit"], result.rj["q_inv_fit"], "o", color="teal", label="Q^-1(BER)")
    axes[2].plot(result.rj["x_fit"], result.rj["q_inv_line"], "-", color="tomato", label="linear fit")
    axes[2].set_title("3) Q-transform and RJ linear fit")
    axes[2].set_ylabel("Q^-1(BER)")
    axes[2].grid(True, alpha=0.35)
    axes[2].legend(fontsize=8)
else:
    axes[2].plot(result.x, result.ber, "o-", color="gray")
    axes[2].set_title("3) Q-transform unavailable (fit failed)")
    axes[2].set_ylabel("BER")
    axes[2].grid(True, alpha=0.35)

axes[2].set_xlabel("x")
plt.tight_layout()
st.pyplot(fig)

with st.expander("Show processed arrays"):
    st.write(
        {
            "x": result.x.tolist(),
            "failcount": result.failcount.tolist(),
            "ber": result.ber.tolist(),
            "rms_mu": result.rms_mu,
            "rms_sigma": result.rms_sigma,
        }
    )
