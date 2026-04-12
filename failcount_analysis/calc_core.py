import numpy as np
import pandas as pd

AVERAGE_ROW_LABEL = "Average"


def compute_average_index_per_code(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-row DataFrame of average index for each code column."""
    averaged_df = pd.DataFrame()
    row_count = len(df.index)
    max_fail_value = float(df.max().max())
    max_possible_fail_sum = max_fail_value * row_count
    min_index_value = float(df.index.min())
    max_index_value = float(df.index.max())

    for column_label in df.columns:
        column_fail_sum = float(df[column_label].sum())
        averaged_index = (1 - column_fail_sum / max_possible_fail_sum) * (max_index_value - min_index_value) + min_index_value
        averaged_df[column_label] = [averaged_index]

    averaged_df.index = [AVERAGE_ROW_LABEL]
    return averaged_df


def _empty_inner_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame aligned to the inner area (excluding borders)."""
    return pd.DataFrame(columns=df.columns[1:-1], index=df.index[1:-1])


def compute_inner_gradient_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ceiled gradient norm on inner cells using central difference."""
    if df.empty or df.shape[0] < 3 or df.shape[1] < 3:
        return _empty_inner_frame(df)

    values = df.to_numpy(dtype=float)
    vertical_diff = (values[:-2, 1:-1] - values[2:, 1:-1]) / 2.0
    horizontal_diff = (values[1:-1, 2:] - values[1:-1, :-2]) / 2.0
    gradient_magnitude = np.sqrt(vertical_diff**2 + horizontal_diff**2)
    ceiled_gradient_magnitude = np.ceil(gradient_magnitude).astype(int)

    return pd.DataFrame(ceiled_gradient_magnitude, index=df.index[1:-1], columns=df.columns[1:-1])


def apply_averaging(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for compute_average_index_per_code."""
    return compute_average_index_per_code(df)


def calculate_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for compute_inner_gradient_norm."""
    return compute_inner_gradient_norm(df)


