"""Backward-compatible export for legacy fcd usage."""

try:
    from failcount_average.calc_core import calculate_gradient
except ModuleNotFoundError:
    from calc_core import calculate_gradient
