"""Backward-compatible entry point for legacy fca usage."""

try:
    from failcount_average.average_gradient_demo import main
    from failcount_average.calc_core import AVERAGE_ROW_LABEL, apply_averaging
    from failcount_average.sample_data import build_sample_df
except ModuleNotFoundError:
    from average_gradient_demo import main
    from calc_core import AVERAGE_ROW_LABEL, apply_averaging
    from sample_data import build_sample_df


if __name__ == "__main__":
    main()
