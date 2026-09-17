"""Compute the grinding circuit steady state with sump level control."""

import numpy as np
from cas_models.continuous_time.simulate import make_steady_state_solver

from model import (
    INPUTS_NOP,
    STATES_NOP,
    build_grinding_circuit_model_with_sump_control,
)


def main():
    # The controller computes cyclone_feed_flow from the sump level, so it is
    # intentionally absent from u_nom and appears as a monitored output.
    model = build_grinding_circuit_model_with_sump_control()
    x_guess = np.array([STATES_NOP[name] for name in model.state_names])
    u_nom = np.array([INPUTS_NOP[name] for name in model.input_names])
    rhs_guess = np.asarray(model.f(0.0, x_guess, u_nom)).ravel()

    solve_steady_state = make_steady_state_solver(model)
    x_ss, y_ss = solve_steady_state(x_guess, u_nom, {})
    rhs_ss = np.asarray(model.f(0.0, x_ss, u_nom)).ravel()

    print("Nominal exogenous inputs:")
    for name, value in zip(model.input_names, u_nom):
        print(f"  {name:<25} {value:.12g}")

    print("\nODE residuals at stored NOP:")
    for name, value in zip(model.state_names, rhs_guess):
        print(f"  d_{name}_dt: {value:+.12g}")

    print("\nComputed steady-state states:")
    for name, value, guess in zip(model.state_names, x_ss, x_guess):
        print(
            f"  {name:<25} {value:.12g}"
            f"  (stored NOP: {guess:.12g}, delta: {value - guess:+.6g})"
        )

    controlled_input_name = "cyclone_feed_flow"
    controlled_input_index = model.output_names.index(controlled_input_name)

    print("\nComputed steady-state process outputs:")
    for name, value in zip(model.output_names, y_ss):
        if name == controlled_input_name:
            continue
        print(f"  {name:<25} {value:.12g}")

    print("\nSump level controller action at steady state:")
    print(f"  {controlled_input_name:<25} {y_ss[controlled_input_index]:.12g}")

    print("\nODE residuals at computed steady state:")
    for name, value in zip(model.state_names, rhs_ss):
        print(f"  d_{name}_dt: {value:+.12g}")
    print(f"  max_abs_residual: {np.max(np.abs(rhs_ss)):.12g}")


if __name__ == "__main__":
    main()
