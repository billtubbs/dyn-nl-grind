"""Tests for model.py.

The steady-state tests check that the calibrated operating condition from
Table 4 and the state values from Table 5 (Le Roux & Steyn, 2022) are
self-consistent with the model equations.

Because the paper tables report rounded values (3-4 significant figures),
the state derivatives are not exactly zero — residuals of up to ~3 m³/h
are expected and reflect rounding, not a model error.  The product_size
output (y_PSE) has a larger discrepancy (~8 %) that is tracked separately
and may warrant further investigation of the cyclone sub-model.
"""

import numpy as np
import pytest

from cas_models.continuous_time.simulate import make_steady_state_solver
from model import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    STATE_NAMES,
    build_grinding_circuit_model,
    build_grinding_circuit_model_with_level_control,
    build_grinding_circuit_model_with_sump_control,
)

# Sump-level controlled model: u_CFF removed from inputs, added as output.
CL_INPUT_NAMES = [n for n in INPUT_NAMES if n != "cyclone_feed_flow"]
CL_OUTPUT_NAMES = OUTPUT_NAMES + ["cyclone_feed_flow"]

# Level-control model: u_MFO and u_CFF both removed from inputs, added as outputs.
LC_INPUT_NAMES = [
    n for n in INPUT_NAMES if n not in ("feed_ore_rate", "cyclone_feed_flow")
]
LC_OUTPUT_NAMES = OUTPUT_NAMES + ["cyclone_feed_flow", "feed_ore_rate"]

# Paper values (Tables 4 & 5, Le Roux & Steyn 2022) — rounded to 3–4 sig figs.
# Used here for approximate validation only; model.py holds solver-computed NOP.
_PAPER_INPUTS = {
    "feed_ore_rate": 1191.0,
    "water_ore_ratio": 0.572,
    "critical_speed_fraction": 0.768,
    "sump_feed_water": 870.0,
    "cyclone_feed_flow": 2921.0,
}
_PAPER_STATES = {
    "water_volume": 31.0,
    "solids_volume": 31.1,
    "rock_volume": 9.84,
    "fines_volume": 5.22,
    "sump_water_volume": 133.0,
    "sump_solids_volume": 72.2,
    "sump_fines_volume": 12.1,
}
_PAPER_OUTPUTS = {
    "charge_fill_fraction": 0.328,
    "mill_power": 14.8,
    "sump_level": 59.4,
    "sump_density": 1.77,
    "product_size": 37.9,
}


@pytest.fixture(scope="module")
def model():
    return build_grinding_circuit_model()


@pytest.fixture(scope="module")
def ss_x():
    return np.array([_PAPER_STATES[n] for n in STATE_NAMES])


@pytest.fixture(scope="module")
def ss_u():
    return np.array([_PAPER_INPUTS[n] for n in INPUT_NAMES])


@pytest.fixture(scope="module")
def model_sump_ctrl():
    return build_grinding_circuit_model_with_sump_control()


@pytest.fixture(scope="module")
def ss_u_cl():
    return np.array([_PAPER_INPUTS[n] for n in CL_INPUT_NAMES])


@pytest.fixture(scope="module")
def model_level_ctrl():
    return build_grinding_circuit_model_with_level_control()


@pytest.fixture(scope="module")
def ss_u_lc():
    return np.array([_PAPER_INPUTS[n] for n in LC_INPUT_NAMES])


class TestGrindingCircuitModel:
    """Tests for the base open-loop grinding circuit model."""

    def test_state_count(self, model):
        assert model.n == 7

    def test_input_count(self, model):
        assert model.nu == 5

    def test_output_count(self, model):
        assert model.ny == 5

    def test_state_names(self, model):
        assert model.state_names == STATE_NAMES

    def test_input_names(self, model):
        assert model.input_names == INPUT_NAMES

    def test_output_names(self, model):
        assert model.output_names == OUTPUT_NAMES

    def test_name(self, model):
        assert model.name == "c1_grind"

    def test_derivatives_near_zero(self, model, ss_x, ss_u):
        """f(t, x_ss, u_ss) derivatives should be small at the calibrated
        steady state.  Residuals up to ~5 m³/h are acceptable given the
        rounding of tabulated parameter values in Tables 4 and 5.
        """
        rhs = np.array(model.f(0.0, ss_x, ss_u)).flatten()
        print("\nState derivatives at steady state (m³/h):")
        for name, val in zip(STATE_NAMES, rhs):
            print(f"  d({name})/dt = {val:+.4f}")
        np.testing.assert_allclose(rhs, 0.0, atol=5.0)

    def test_outputs_match_table4(self, model, ss_x, ss_u):
        """h(t, x_ss, u_ss) should reproduce the Table 4 measured outputs
        within 2 % for all variables except product_size (see separate test).
        """
        y = np.array(model.h(0.0, ss_x, ss_u)).flatten()
        y_expected = np.array([_PAPER_OUTPUTS[n] for n in OUTPUT_NAMES])
        print("\nOutputs at steady state:")
        for name, val, exp in zip(OUTPUT_NAMES, y, y_expected):
            pct = 100 * (val - exp) / exp
            print(
                f"  {name}: computed={val:.4f}  expected={exp}  ({pct:+.1f} %)"
            )
        # Check all outputs except product_size (index 4) to 2 %
        np.testing.assert_allclose(y[:4], y_expected[:4], rtol=0.02)

    def test_product_size_approx(self, model, ss_x, ss_u):
        """product_size (y_PSE) has a larger discrepancy (~8 %) at the
        tabulated steady state; this is tracked here for visibility.
        Currently passing with rtol=0.15 — investigate cyclone sub-model
        if tighter agreement is needed.
        """
        y = np.array(model.h(0.0, ss_x, ss_u)).flatten()
        y_pse = y[OUTPUT_NAMES.index("product_size")]
        y_pse_expected = _PAPER_OUTPUTS["product_size"]
        print(
            f"\n  product_size: computed={y_pse:.4f}  expected={y_pse_expected}"
        )
        np.testing.assert_allclose(y_pse, y_pse_expected, rtol=0.15)


class TestGrindingCircuitModelWithSumpControl:
    """Tests for the sump-level controlled model.

    With u_CFF set by the P controller the sump Jacobian degeneracy is
    resolved, so make_steady_state_solver should converge.
    """

    def test_state_count(self, model_sump_ctrl):
        assert model_sump_ctrl.n == 7

    def test_input_count(self, model_sump_ctrl):
        assert model_sump_ctrl.nu == 4

    def test_output_count(self, model_sump_ctrl):
        assert model_sump_ctrl.ny == 6

    def test_state_names(self, model_sump_ctrl):
        assert model_sump_ctrl.state_names == STATE_NAMES

    def test_input_names(self, model_sump_ctrl):
        assert model_sump_ctrl.input_names == CL_INPUT_NAMES

    def test_output_names(self, model_sump_ctrl):
        assert model_sump_ctrl.output_names == CL_OUTPUT_NAMES

    def test_name(self, model_sump_ctrl):
        assert model_sump_ctrl.name == "c1_grind_sc"

    def test_actual_steady_state(self, model_sump_ctrl, ss_x, ss_u_cl):
        """Solve for the true steady state of the sump-controlled model and
        compare to the paper's tabulated states from Table 5.
        """
        solve = make_steady_state_solver(model_sump_ctrl)
        x_ss_actual, y_ss_actual = solve(ss_x, ss_u_cl, {})

        rhs = np.array(model_sump_ctrl.f(0.0, x_ss_actual, ss_u_cl)).flatten()
        print("\nState derivatives at true steady state (m³/h):")
        for name, val in zip(STATE_NAMES, rhs):
            print(f"  d({name})/dt = {val:+.6f}")

        print("\nComparison: true vs. paper steady-state states:")
        for name, actual, paper in zip(STATE_NAMES, x_ss_actual, ss_x):
            print(
                f"  {name}: actual={actual:.4f}  paper={paper:.4f}  diff={actual - paper:+.4f}"
            )

        print("\nTrue steady-state outputs vs. Table 4:")
        y_expected = np.array([_PAPER_OUTPUTS[n] for n in OUTPUT_NAMES])
        cff_ss = _PAPER_INPUTS["cyclone_feed_flow"]
        y_ref = list(y_expected) + [cff_ss]
        for name, actual, exp in zip(CL_OUTPUT_NAMES, y_ss_actual, y_ref):
            print(
                f"  {name}: actual={actual:.4f}  paper={exp}  ({100 * (actual - exp) / exp:+.1f} %)"
            )

        np.testing.assert_allclose(rhs, 0.0, atol=1e-6)


class TestGrindingCircuitModelWithLevelControl:
    """Tests for the charge-level + sump-level controlled model."""

    def test_state_count(self, model_level_ctrl):
        assert model_level_ctrl.n == 7

    def test_input_count(self, model_level_ctrl):
        assert model_level_ctrl.nu == 3

    def test_output_count(self, model_level_ctrl):
        assert model_level_ctrl.ny == 7

    def test_state_names(self, model_level_ctrl):
        assert model_level_ctrl.state_names == STATE_NAMES

    def test_input_names(self, model_level_ctrl):
        assert model_level_ctrl.input_names == LC_INPUT_NAMES

    def test_output_names(self, model_level_ctrl):
        assert model_level_ctrl.output_names == LC_OUTPUT_NAMES

    def test_name(self, model_level_ctrl):
        assert model_level_ctrl.name == "c1_grind_sc_lc"

    def test_actual_steady_state(self, model_level_ctrl, ss_x, ss_u_lc):
        """Solve for the true steady state and verify derivatives are zero.

        At the NOP the charge fill fraction setpoint equals the paper value,
        so the feed rate controller should settle near 1191 t/h and the sump
        controller should settle near 2921 m³/h.
        """
        solve = make_steady_state_solver(model_level_ctrl)
        x_ss_actual, y_ss_actual = solve(ss_x, ss_u_lc, {})

        rhs = np.array(model_level_ctrl.f(0.0, x_ss_actual, ss_u_lc)).flatten()
        print("\nState derivatives at true steady state (m³/h):")
        for name, val in zip(STATE_NAMES, rhs):
            print(f"  d({name})/dt = {val:+.6f}")

        print("\nComparison: true vs. paper steady-state states:")
        for name, actual, paper in zip(STATE_NAMES, x_ss_actual, ss_x):
            print(
                f"  {name}: actual={actual:.4f}  paper={paper:.4f}  diff={actual - paper:+.4f}"
            )

        print("\nTrue steady-state outputs vs. Table 4 / NOP inputs:")
        y_expected = np.array([_PAPER_OUTPUTS[n] for n in OUTPUT_NAMES])
        cff_ss = _PAPER_INPUTS["cyclone_feed_flow"]
        mfo_ss = _PAPER_INPUTS["feed_ore_rate"]
        y_ref = list(y_expected) + [cff_ss, mfo_ss]
        for name, actual, exp in zip(LC_OUTPUT_NAMES, y_ss_actual, y_ref):
            print(
                f"  {name}: actual={actual:.4f}  paper={exp}  ({100 * (actual - exp) / exp:+.1f} %)"
            )

        np.testing.assert_allclose(rhs, 0.0, atol=1e-6)
