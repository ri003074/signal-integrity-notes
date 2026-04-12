"""Backward-compatible entry point for legacy fca usage."""

try:
    import failcount_analysis.average_gradient_demo as average_gradient_demo
    from failcount_analysis.calc_core import AVERAGE_ROW_LABEL, apply_averaging
    from failcount_analysis.sample_data import build_sample_df
except ModuleNotFoundError:
    import average_gradient_demo
    from calc_core import AVERAGE_ROW_LABEL, apply_averaging
    from sample_data import build_sample_df


if __name__ == "__main__":
    average_gradient_demo.main()

