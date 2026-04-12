import numpy as np
import pandas as pd


def calculate_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """中心部分の勾配ノルムを計算して切り上げ整数で返す。境界行列は除外される。"""
    if df.empty or df.shape[0] < 3 or df.shape[1] < 3:
        return pd.DataFrame(columns=df.columns[1:-1], index=df.index[1:-1])

    arr = df.astype(float).to_numpy()
    # 中央部のみ計算: (up-down)/2 と (right-left)/2
    vert = (arr[:-2, 1:-1] - arr[2:, 1:-1]) / 2.0
    horz = (arr[1:-1, 2:] - arr[1:-1, :-2]) / 2.0
    grad = np.sqrt(vert**2 + horz**2)
    result = np.ceil(grad).astype(int)

    return pd.DataFrame(result, index=df.index[1:-1], columns=df.columns[1:-1])