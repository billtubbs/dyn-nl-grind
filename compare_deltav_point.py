"""Temporary check: compare Python model f(x,u) and h(x,u) against a single
snapshot of inputs/states/outputs read off the DeltaV NLGRIND_ODE block, to
see whether the DeltaV divergence is a calc discrepancy or just an artifact
of DeltaV's lower-precision Euler integration.

Run:
    python compare_deltav_point.py
"""

import numpy as np

from model import build_grinding_circuit_model

# ── Values transcribed from the DeltaV NLGRIND_ODE block screenshot ────────
x = np.array(
    [
        14.3058,  # X_MW
        6.91106,  # X_MS
        2.11922,  # X_MR
        5.64887,  # X_MF
        119.298,  # X_SW
        29.0445,  # X_SS
        24.4024,  # X_SF
    ]
)

u = np.array(
    [
        1191.0,  # U_MFO
        0.572,  # U_RMIW
        0.768,  # U_PHIC
        870.0,  # U_SFW
        2142.68,  # U_CFF
    ]
)

deltav_rhs = {
    "water_volume": 4.02475,  # D_X_MW
    "solids_volume": 2.54341,  # D_X_MS
    "rock_volume": 0.78006,  # D_X_MR
    "fines_volume": -0.265747,  # D_X_MF
    "sump_water_volume": 57.8048,  # D_X_SW
    "sump_solids_volume": 20.5587,  # D_X_SS
    "sump_fines_volume": 7.23651,  # D_X_SF
}

deltav_y = {
    "charge_fill_fraction": 0.237264,  # Y_JT
    "mill_power": 15.0854,  # Y_PMILL
    "sump_level": 42.8984,  # Y_SLEV
    "sump_density": 1.43075,  # Y_RHO
    "product_size": 76.9123,  # Y_PSE
}


def main():
    model = build_grinding_circuit_model()

    rhs = np.asarray(model.f(0.0, x, u)).ravel()
    y = np.asarray(model.h(0.0, x, u)).ravel()

    print("State derivatives: Python vs DeltaV")
    print(
        f"  {'name':<20}{'python':>14}{'deltav':>14}{'abs diff':>14}{'rel diff %':>14}"
    )
    for name, py_val in zip(model.state_names, rhs):
        dv_val = deltav_rhs[name]
        abs_diff = py_val - dv_val
        rel_diff = 100.0 * abs_diff / dv_val if dv_val != 0 else float("nan")
        print(
            f"  {name:<20}{py_val:>14.6g}{dv_val:>14.6g}"
            f"{abs_diff:>14.6g}{rel_diff:>14.4g}"
        )

    print("\nOutputs: Python vs DeltaV")
    print(
        f"  {'name':<20}{'python':>14}{'deltav':>14}{'abs diff':>14}{'rel diff %':>14}"
    )
    for name, py_val in zip(model.output_names, y):
        dv_val = deltav_y[name]
        abs_diff = py_val - dv_val
        rel_diff = 100.0 * abs_diff / dv_val if dv_val != 0 else float("nan")
        print(
            f"  {name:<20}{py_val:>14.6g}{dv_val:>14.6g}"
            f"{abs_diff:>14.6g}{rel_diff:>14.4g}"
        )


if __name__ == "__main__":
    main()
