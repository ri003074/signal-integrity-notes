"""Backward-compatible entry point for legacy fca_inl_dnl usage."""

try:
    from failcount_analysis.inl_dnl_demo import calc_inl_dnl, main
except ModuleNotFoundError:
    from inl_dnl_demo import calc_inl_dnl, main


if __name__ == "__main__":
    main()


