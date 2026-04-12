import numpy as np
import pandas as pd

AVERAGE_ROW_LABEL = "Average"


def apply_averaging(df: pd.DataFrame) -> pd.DataFrame:
    """Return 1-row DataFrame containing average index for each column."""
    df_avg = pd.DataFrame()
    index_count = len(df.index)
    max_fail_per_cell = float(df.max().max())
    max_fail_total = max_fail_per_cell * index_count
    index_min = float(df.index.min())
    index_max = float(df.index.max())

    for col in df.columns:
        col_sum = float(df[col].sum())
        avg_index = (1 - col_sum / max_fail_total) * (index_max - index_min) + index_min
        df_avg[col] = [avg_index]

    df_avg.index = [AVERAGE_ROW_LABEL]
    return df_avg


def _empty_inner_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame aligned to the inner area (excluding borders)."""
    return pd.DataFrame(columns=df.columns[1:-1], index=df.index[1:-1])


def calculate_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ceiled gradient norm on inner cells using central difference."""
    if df.empty or df.shape[0] < 3 or df.shape[1] < 3:
        return _empty_inner_frame(df)

    data = df.to_numpy(dtype=float)
    vertical = (data[:-2, 1:-1] - data[2:, 1:-1]) / 2.0
    horizontal = (data[1:-1, 2:] - data[1:-1, :-2]) / 2.0
    gradient_norm = np.sqrt(vertical**2 + horizontal**2)
    ceiled_gradient = np.ceil(gradient_norm).astype(int)

    return pd.DataFrame(ceiled_gradient, index=df.index[1:-1], columns=df.columns[1:-1])

