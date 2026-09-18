"""Temporary check: does DeltaV's 1-second explicit-Euler step (and possibly
float32 precision) explain convergence to a different steady state than the
Python/CasADi model predicts?

Holding u_CFF fixed at the DeltaV snapshot value (2142.68 m3/h) has no real
steady state: it is well below the ~2900 m3/h needed to balance mill
discharge + sump_feed_water, so the sump volume grows without bound in the
model itself, independent of integrator. So instead we use the sump-level
P-controlled closed-loop model (u_CFF computed internally from sump level),
which does have a genuine attracting steady state, and compare integrators
starting from the DeltaV snapshot state:

  1. "Truth"   — accurate variable-step integration (scipy solve_ivp, RK45,
                 float64) run out to steady state, refined with fsolve.
  2. "Euler64" — explicit Euler, dt=1 s, float64 (matches DeltaV's step size
                 exactly, but not its precision).
  3. "Euler32" — explicit Euler, dt=1 s, float32 (matches DeltaV's step size
                 and approximates its lower precision).

If Euler64/Euler32 converge to the same point as "truth", the DeltaV
divergence is not explained by step size or precision alone. If Euler32 (or
even Euler64) settles somewhere clearly different, that supports the
integration-method/precision theory.

Run:
    python compare_deltav_integration.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from model import STATE_NAMES, build_grinding_circuit_model_with_sump_control

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

# Closed-loop model inputs: feed_ore_rate, water_ore_ratio,
# critical_speed_fraction, sump_feed_water (cyclone_feed_flow is controlled
# internally by the sump-level P-controller, so it is dropped here).
u = np.array(
    [
        1191.0,  # U_MFO
        0.572,  # U_RMIW
        0.768,  # U_PHIC
        870.0,  # U_SFW
    ]
)

T_HOURS = 60.0  # horizon assumed long enough to reach steady state
DT_SEC = 1.0  # DeltaV's execution period
DT_H = DT_SEC / 3600.0


def main():
    model = build_grinding_circuit_model_with_sump_control()

    def rhs(t, x):
        return np.asarray(model.f(t, x, u)).ravel()

    # 1. Accurate reference trajectory / steady state
    sol = solve_ivp(
        rhs, [0.0, T_HOURS], x0, method="RK45", rtol=1e-10, atol=1e-12
    )
    x_truth = sol.y[:, -1]
    x_ss_solver = fsolve(lambda x: rhs(0.0, x), x_truth, full_output=False)
    max_resid = np.max(np.abs(rhs(0.0, x_ss_solver)))

    # 2. Explicit Euler, dt = 1 s, float64
    n_steps = int(round(T_HOURS / DT_H))
    x_euler64 = x0.astype(np.float64).copy()
    for _ in range(n_steps):
        x_euler64 = x_euler64 + DT_H * rhs(0.0, x_euler64)

    # 3. Explicit Euler, dt = 1 s, float32 (state, rhs, and dt all float32)
    x_euler32 = x0.astype(np.float32).copy()
    u32 = u.astype(np.float32)
    dt32 = np.float32(DT_H)

    def rhs32(x):
        # model.f is built on float64 CasADi internals; cast the result back
        # to float32 each step to mimic a float32 execution environment.
        return (
            np.asarray(model.f(0.0, x.astype(np.float64), u32))
            .ravel()
            .astype(np.float32)
        )

    for _ in range(n_steps):
        x_euler32 = x_euler32 + dt32 * rhs32(x_euler32)

    print(
        f"Sump-level-controlled model, simulated {T_HOURS:g} h "
        "from DeltaV snapshot:"
    )
    print(f"  (accurate steady-state max |residual| = {max_resid:.3g})\n")
    header = f"  {'state':<20}{'truth (SS)':>16}{'euler64':>16}{'euler32':>16}"
    print(header)
    for i, name in enumerate(STATE_NAMES):
        print(
            f"  {name:<20}{x_ss_solver[i]:>16.6g}{x_euler64[i]:>16.6g}"
            f"{x_euler32[i]:>16.6g}"
        )

    print("\nMax abs deviation from accurate steady state:")
    print(f"  euler64: {np.max(np.abs(x_euler64 - x_ss_solver)):.6g}")
    print(f"  euler32: {np.max(np.abs(x_euler32 - x_ss_solver)):.6g}")

    print("\nMax relative deviation from accurate steady state (%):")
    rel64 = 100.0 * np.abs(x_euler64 - x_ss_solver) / np.abs(x_ss_solver)
    rel32 = 100.0 * np.abs(x_euler32 - x_ss_solver) / np.abs(x_ss_solver)
    print(f"  euler64: {np.max(rel64):.6g}")
    print(f"  euler32: {np.max(rel32):.6g}")

    print(
        "\nMax abs deviation between euler64 and euler32 (precision-only effect):"
    )
    print(f"  {np.max(np.abs(x_euler64 - x_euler32.astype(np.float64))):.6g}")


if __name__ == "__main__":
    main()
