"""Temporary check: does the DeltaV SLEV_CTRL PID's actual configuration
(SP=10, PV_SCALE=[0,100], OUT_SCALE=[0,4742.85], direct acting, GAIN=1,
BIAS=582.654) explain convergence to a different steady state than the
Python model's controller (cff_max ~4150, level_min=10, zero bias)?

DeltaV direct-acting P-only law, in engineering units:
    u_CFF = (OUT_HI - OUT_LO)/100 * (y_SLEV - SP) + BIAS + OUT_LO

We simulate the closed-loop plant with this exact DeltaV law and compare the
resulting steady state to:
  - the DeltaV snapshot state itself (from the screenshot), and
  - the correct (Python) NOP.

Run:
    python compare_deltav_bias.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from model import (
    INPUTS_NOP,
    OUTPUTS_NOP,
    STATE_NAMES,
    SUMP_LEVEL_MAX,
    SUMP_LEVEL_MIN,
    SUMP_VOLUME,
    build_grinding_circuit_model,
)

# DeltaV snapshot state (screenshot)
x0 = np.array(
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

# 4 exogenous inputs (cyclone_feed_flow is computed by the controller below)
u4 = np.array(
    [
        1191.0,  # U_MFO
        0.572,  # U_RMIW
        0.768,  # U_PHIC
        870.0,  # U_SFW
    ]
)

# ── Python's intended controller law ────────────────────────────────────
level_min = SUMP_LEVEL_MIN
level_max = SUMP_LEVEL_MAX
u_cff_ss = INPUTS_NOP["cyclone_feed_flow"]
level_ss = OUTPUTS_NOP["sump_level"]
cff_max_py = u_cff_ss * (level_max - level_min) / (level_ss - level_min)
Kp_py = cff_max_py / (level_max - level_min)

# ── DeltaV SLEV_CTRL block's actual configuration ───────────────────────
DV_SP = 10.0  # %
DV_OUT_LO = 0.0  # m3/h
DV_OUT_HI = 4742.85  # m3/h
DV_GAIN = 1.0  # dimensionless
DV_BIAS = 582.654  # m3/h -- should be 0


def main():
    model = build_grinding_circuit_model()

    def controller_python(x):
        sump_level = 100.0 * (x[4] + x[5]) / SUMP_VOLUME
        return np.clip(Kp_py * (sump_level - level_min), 0.0, cff_max_py)

    def controller_deltav(x):
        sump_level = 100.0 * (x[4] + x[5]) / SUMP_VOLUME
        out = (
            DV_GAIN * (DV_OUT_HI - DV_OUT_LO) / 100.0 * (sump_level - DV_SP)
            + DV_BIAS
            + DV_OUT_LO
        )
        return np.clip(out, DV_OUT_LO, DV_OUT_HI)

    def rhs(t, x, controller):
        cff = controller(x)
        u = np.concatenate([u4, [cff]])
        return np.asarray(model.f(t, x, u)).ravel()

    def steady_state(x_guess, controller, t_horizon=200.0):
        sol = solve_ivp(
            rhs,
            [0.0, t_horizon],
            x_guess,
            args=(controller,),
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        x_ss = fsolve(
            lambda x: rhs(0.0, x, controller), sol.y[:, -1], full_output=False
        )
        resid = np.max(np.abs(rhs(0.0, x_ss, controller)))
        return x_ss, resid

    x_ss_python, resid_py = steady_state(x0, controller_python)
    x_ss_deltav, resid_dv = steady_state(x0, controller_deltav)

    def report(x, resid, controller, label):
        cff = controller(x)
        y = np.asarray(model.h(0.0, x, np.concatenate([u4, [cff]]))).ravel()
        sump_level = y[2]
        print(f"\n{label} steady state (max |residual| = {resid:.2g}):")
        for name, val in zip(STATE_NAMES, x):
            print(f"  {name:<20}{val:>14.6g}")
        print(f"  {'cyclone_feed_flow':<20}{cff:>14.6g}")
        print(f"  {'sump_level (%)':<20}{sump_level:>14.6g}")

    report(x_ss_python, resid_py, controller_python, "Python controller law")
    report(x_ss_deltav, resid_dv, controller_deltav, "DeltaV controller config")

    print(
        "\nDeltaV snapshot state (from screenshot, mid-transient toward its SS):"
    )
    for name, val in zip(STATE_NAMES, x0):
        print(f"  {name:<20}{val:>14.6g}")
    print(f"  {'sump_level (%)':<20}{42.8984:>14.6g}  (Y_SLEV from screenshot)")

    print(
        "\nDeviation of DeltaV-law steady state from DeltaV snapshot state (abs):"
    )
    for name, val, dv in zip(STATE_NAMES, x_ss_deltav, x0):
        print(f"  {name:<20}{val - dv:>+14.6g}")

    print("\nDeviation of DeltaV-law steady state from Python NOP (abs):")
    for name, val, nop in zip(STATE_NAMES, x_ss_deltav, x_ss_python):
        print(f"  {name:<20}{val - nop:>+14.6g}")


if __name__ == "__main__":
    main()
