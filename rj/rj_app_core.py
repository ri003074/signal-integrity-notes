"""Core logic for the Fail Count to RJ Streamlit app."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from rj.ber import extract_rj
except ModuleNotFoundError:
    from ber import extract_rj


@dataclass
class AnalysisResult:
    x: np.ndarray
    failcount: np.ndarray
    ber: np.ndarray
    pdf_raw: np.ndarray
    pdf_clipped: np.ndarray
    rj: Optional[dict]
    rms_mu: Optional[float] = None
    rms_sigma: Optional[float] = None
    warning: Optional[str] = None


def parse_numeric_series(text: str) -> np.ndarray:
    """Parse numbers split by comma/space/newline into a 1D float array."""
    tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    if not tokens:
        raise ValueError("No numeric values were found.")
    try:
        values = np.array([float(t) for t in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError("Input contains non-numeric values.") from exc
    return values


def parse_csv_content(content: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Parse CSV text. Supports either failcount-only or x,failcount columns."""
    reader = csv.reader(io.StringIO(content))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("CSV is empty.")

    header = [c.strip().lower() for c in rows[0]]
    has_named_columns = "failcount" in header

    if has_named_columns:
        dict_reader = csv.DictReader(io.StringIO(content))
        fail_vals = []
        x_vals = []
        has_x = False
        for row in dict_reader:
            if row is None:
                continue
            fail_raw = (row.get("failcount") or "").strip()
            if not fail_raw:
                continue
            fail_vals.append(float(fail_raw))
            x_raw = (row.get("x") or "").strip()
            if x_raw:
                has_x = True
                x_vals.append(float(x_raw))
            elif has_x:
                raise ValueError("CSV has mixed x values: some rows are missing x.")

        if not fail_vals:
            raise ValueError("No failcount values found in CSV.")
        fail = np.array(fail_vals, dtype=float)
        x = np.array(x_vals, dtype=float) if has_x else None
        return fail, x

    # No header: allow 1-column (failcount) or 2-column (x,failcount)
    parsed = []
    for row in rows:
        nums = [c.strip() for c in row if c.strip()]
        if len(nums) not in (1, 2):
            raise ValueError("CSV without header must have 1 or 2 columns.")
        parsed.append([float(v) for v in nums])

    arr = np.array(parsed, dtype=float)
    if arr.shape[1] == 1:
        return arr[:, 0], None
    return arr[:, 1], arr[:, 0]


def validate_and_prepare(
    failcount: np.ndarray,
    x: Optional[np.ndarray],
    max_fail: float,
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Validate arrays and return prepared x/failcount and optional warning."""
    failcount = np.asarray(failcount, dtype=float)
    if failcount.ndim != 1:
        raise ValueError("Fail Count must be a 1D array.")
    if failcount.size < 5:
        raise ValueError("At least 5 data points are required.")

    warning = None
    clipped = np.clip(failcount, 0, max_fail)
    if not np.allclose(clipped, failcount):
        warning = "Fail Count values were clipped to [0, max_fail]."
        failcount = clipped

    if x is None:
        x = np.linspace(0.0, 1.0, failcount.size)
    else:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != failcount.size:
            raise ValueError("x must be 1D and have the same length as Fail Count.")

    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    return x, failcount, warning


def analyze_failcount(
    failcount: np.ndarray,
    x: Optional[np.ndarray],
    max_fail: float,
    ber_range: tuple[float, float],
) -> AnalysisResult:
    """Compute BER, numerical derivative, and optional RJ fit result."""
    x, failcount, warning = validate_and_prepare(failcount, x, max_fail)

    ber = failcount / max_fail
    pdf_raw = -np.gradient(failcount, x)
    pdf_clipped = np.clip(pdf_raw, 0, None)

    rms_mu = None
    rms_sigma = None
    area = float(np.sum(pdf_clipped))
    if area > 0:
        w = pdf_clipped / area
        rms_mu = float(np.sum(w * x))
        rms_sigma = float(np.sqrt(np.sum(w * (x - rms_mu) ** 2)))

    rj_result = None
    try:
        rj_result = extract_rj(x, failcount, max_fail=int(max_fail), ber_range=ber_range)
    except Exception:
        rj_result = None

    return AnalysisResult(
        x=x,
        failcount=failcount,
        ber=ber,
        pdf_raw=pdf_raw,
        pdf_clipped=pdf_clipped,
        rj=rj_result,
        rms_mu=rms_mu,
        rms_sigma=rms_sigma,
        warning=warning,
    )
