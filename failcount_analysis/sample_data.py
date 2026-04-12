import pandas as pd


def build_sample_df() -> pd.DataFrame:
    """Create sample FailCount table used by demos."""
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


