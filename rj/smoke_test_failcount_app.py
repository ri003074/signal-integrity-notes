"""Smoke test for the Fail Count -> RJ analysis core."""

import numpy as np

from rj.rj_app_core import analyze_failcount


def main() -> None:
    x = np.linspace(0.0, 1.0, 50)
    failcount = np.array(
        [100, 100, 100, 99, 99, 98, 97, 95, 93, 90,
         86, 82, 77, 71, 65, 58, 51, 44, 37, 31,
         25, 20, 15, 11, 8, 6, 4, 3, 2, 1,
         1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=float,
    )

    result = analyze_failcount(
        failcount=failcount,
        x=x,
        max_fail=100,
        ber_range=(0.05, 0.95),
    )

    print(f"points={result.x.size}")
    print(f"ber_min={result.ber.min():.4f}")
    print(f"ber_max={result.ber.max():.4f}")

    if result.rj is None:
        raise RuntimeError("RJ fit failed in smoke test.")

    print(f"sigma_rj={result.rj['sigma_rj']:.5f}")
    print(f"mu={result.rj['mu']:.5f}")
    print(f"r_squared={result.rj['r_squared']:.5f}")

    if result.rms_sigma is None or result.rms_mu is None:
        raise RuntimeError("RMS RJ calculation failed in smoke test.")

    print(f"rms_sigma={result.rms_sigma:.5f}")
    print(f"rms_mu={result.rms_mu:.5f}")


if __name__ == "__main__":
    main()

