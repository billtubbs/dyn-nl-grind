"""Grinding circuit simulation model.

Implements the nonlinear dynamic model described in:
    J.D. le Roux and C.W. Steyn, "Validation of a dynamic non-linear
    grinding circuit model for process control," Minerals Engineering
    187 (2022) 107780.

State vector  x (n=7): water_volume, solids_volume, rock_volume, fines_volume,
                        sump_water_volume, sump_solids_volume, sump_fines_volume
Input vector  u (nu=5): feed_ore_rate, water_ore_ratio, critical_speed_fraction,
                         sump_feed_water, cyclone_feed_flow
Output vector y (ny=5): charge_fill_fraction, mill_power, sump_level,
                         sump_density, product_size

f(·) implements equations (1) and (8); h(·) implements equations (5), (6),
(10), (11), and (16), as stated in Section 2.3 of the paper.

Note: rho_balls, rho_charge, discharge_solids_fraction, and charge_voidage
appear only in eq. (7) for mill charge density, which is not part of the
dynamic model and are therefore omitted here.
"""

import casadi as cas
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.param_utils import make_symbolic_vars_from_kwargs

N_STATES = 7
N_INPUTS = 5
N_OUTPUTS = 5

STATE_NAMES = [
    "water_volume",  # x_mw: mill water (m³)
    "solids_volume",  # x_ms: mill solids (m³)
    "rock_volume",  # x_mr: mill rocks (m³)
    "fines_volume",  # x_mf: mill fines (m³)
    "sump_water_volume",  # x_sw: sump water (m³)
    "sump_solids_volume",  # x_ss: sump solids (m³)
    "sump_fines_volume",  # x_sf: sump fines (m³)
]

INPUT_NAMES = [
    "feed_ore_rate",  # u_MFO (t/h)
    "water_ore_ratio",  # u_rMIW (-)
    "critical_speed_fraction",  # u_phic (-)
    "sump_feed_water",  # u_SFW (m³/h)
    "cyclone_feed_flow",  # u_CFF (m³/h)
]

OUTPUT_NAMES = [
    "charge_fill_fraction",  # y_JT (-)
    "mill_power",  # y_Pmill (MW)
    "sump_level",  # y_SLEV (%)
    "sump_density",  # y_rho (t/m³)
    "product_size",  # y_PSE (%)
]

# Normal operating point (NOP) — inputs from Table 4 (Le Roux & Steyn, 2022).
# States and outputs are solver-computed from these inputs; the paper's rounded
# values (Tables 4 & 5) are kept in tests/test_model.py for approx. validation.
INPUTS_NOP = {
    "feed_ore_rate": 1191.0,  # u_MFO  (t/h)
    "water_ore_ratio": 0.572,  # u_rMIW (-)
    "critical_speed_fraction": 0.768,  # u_phic (-)
    "sump_feed_water": 870.0,  # u_SFW  (m³/h)
    "cyclone_feed_flow": 2921.0,  # u_CFF  (m³/h)
}

STATES_NOP = {
    "water_volume": 30.789,  # x_mw  (m³)
    "solids_volume": 30.723,  # x_ms  (m³)
    "rock_volume": 9.6708,  # x_mr  (m³)
    "fines_volume": 5.2661,  # x_mf  (m³)
    "sump_water_volume": 132.87,  # x_sw  (m³)
    "sump_solids_volume": 71.641,  # x_ss  (m³)
    "sump_fines_volume": 12.280,  # x_sf  (m³)
}

OUTPUTS_NOP = {
    "charge_fill_fraction": 0.3257,  # y_JT   (-)
    "mill_power": 14.85,  # y_Pmill (MW)
    "sump_level": 59.27,  # y_SLEV  (%)
    "sump_density": 1.771,  # y_rho   (t/m³)
    "product_size": 35.51,  # y_PSE   (%)
}

# ── Default model parameters (Tables 4 & 5, Le Roux & Steyn 2022) ────────────
# Densities
RHO_ORE = 3.2  # ρ_o (t/m³)
RHO_WATER = 1.0  # ρ_w (t/m³)
# Feed ore composition
ALPHA_FINES = 0.10  # α_f: fines mass fraction in feed
ALPHA_ROCKS = 0.50  # α_r: rocks mass fraction in feed
# Mill
BALL_VOLUME = 105.0  # x_mb: steel ball volume (m³)        — Table 5
DELTA_SOLIDS = 0.0911  # δ_s: power param, solids fraction    — Table 5
DELTA_VOLUME = 0.0911  # δ_v: power param, mill fill volume   — Table 5
DISCHARGE_RATE = 114.7  # d_q (h⁻¹): mill discharge rate       — Table 5
EPSILON_ZERO = 0.60  # ε₀: max solids fraction at zero flow  — paper text
FILL_FRACTION_MAX_POWER = (
    0.23  # J_TPmax: fill fraction at max power  — Table 4
)
K_FINES_PRODUCTION = 15.0e-3  # K_FP (MWh/t): fines production        — Table 5
K_FINES_PRODUCTION_JT = 20.0  # K_FPjt: fines-production fill sens.   — Table 5
K_ROCK_CONSUMPTION = 5.97e-3  # K_RC (MWh/t): rock consumption        — Table 5
MILL_VOLUME = 540.9  # v_mill (m³)                           — Table 4
PHI_NORM = 0.70  # φ_N: rheology normalisation factor    — Table 5
POWER_MAX = 19.7  # P_max (MW): maximum mill power draw   — Table 4
# Sump
SUMP_VOLUME = 345.8  # v_sump (m³)                           — Table 4
# Cyclone
CYCLONE_ALPHA_UNDERFLOW = (
    0.119  # α_su                                  — Table 5
)
CYCLONE_C1 = 0.70  # C₁                                    — paper text
CYCLONE_C2 = 0.70  # C₂                                    — paper text
CYCLONE_C3 = 4  # C₃ (integer)                          — Table 5
CYCLONE_EPSILON_C = 2528.0  # ε_c (m³/h)                            — Table 5

# ── Controller defaults ───────────────────────────────────────────────────────
SUMP_LEVEL_MIN = 10.0  # % — sump level at which CFF pump is off
SUMP_LEVEL_MAX = 80.0  # % — sump level at which CFF pump is at maximum
MFO_MIN = 0.0  # t/h — minimum ore feed rate
MFO_MAX = 1500.0  # t/h — maximum ore feed rate

# Charge-level P-controller plant gain — average of ±50 t/h step tests
CHARGE_LEVEL_PLANT_GAIN = sum([-0.02576 / -50, 0.02300 / 50]) / 2


def build_grinding_circuit_model(
    rho_ore=RHO_ORE,
    rho_water=RHO_WATER,
    alpha_fines=ALPHA_FINES,
    alpha_rocks=ALPHA_ROCKS,
    ball_volume=BALL_VOLUME,
    delta_solids=DELTA_SOLIDS,
    delta_volume=DELTA_VOLUME,
    discharge_rate=DISCHARGE_RATE,
    epsilon_zero=EPSILON_ZERO,
    fill_fraction_max_power=FILL_FRACTION_MAX_POWER,
    k_fines_production=K_FINES_PRODUCTION,
    k_fines_production_jt=K_FINES_PRODUCTION_JT,
    k_rock_consumption=K_ROCK_CONSUMPTION,
    mill_volume=MILL_VOLUME,
    phi_norm=PHI_NORM,
    power_max=POWER_MAX,
    sump_volume=SUMP_VOLUME,
    cyclone_alpha_underflow=CYCLONE_ALPHA_UNDERFLOW,
    cyclone_c1=CYCLONE_C1,
    cyclone_c2=CYCLONE_C2,
    cyclone_c3=CYCLONE_C3,
    cyclone_epsilon_c=CYCLONE_EPSILON_C,
    name="c1_grind",
):
    """Build a continuous-time state-space model of a grinding mill circuit.

    Default parameter values come from Tables 4 and 5 of the paper.
    Pass None for any parameter to leave it as a free CasADi SX symbolic
    variable (useful for parameter estimation).

    Returns:
        StateSpaceModelCT with n=7, nu=5, ny=5.
    """
    params = make_symbolic_vars_from_kwargs(
        rho_ore=rho_ore,
        rho_water=rho_water,
        alpha_fines=alpha_fines,
        alpha_rocks=alpha_rocks,
        ball_volume=ball_volume,
        delta_solids=delta_solids,
        delta_volume=delta_volume,
        discharge_rate=discharge_rate,
        epsilon_zero=epsilon_zero,
        fill_fraction_max_power=fill_fraction_max_power,
        k_fines_production=k_fines_production,
        k_fines_production_jt=k_fines_production_jt,
        k_rock_consumption=k_rock_consumption,
        mill_volume=mill_volume,
        phi_norm=phi_norm,
        power_max=power_max,
        sump_volume=sump_volume,
        cyclone_alpha_underflow=cyclone_alpha_underflow,
        cyclone_c1=cyclone_c1,
        cyclone_c2=cyclone_c2,
        cyclone_c3=cyclone_c3,
        cyclone_epsilon_c=cyclone_epsilon_c,
    )

    rho_ore = params["rho_ore"]
    rho_water = params["rho_water"]
    alpha_fines = params["alpha_fines"]
    alpha_rocks = params["alpha_rocks"]
    ball_volume = params["ball_volume"]
    delta_solids = params["delta_solids"]
    delta_volume = params["delta_volume"]
    discharge_rate = params["discharge_rate"]
    epsilon_zero = params["epsilon_zero"]
    fill_fraction_max_power = params["fill_fraction_max_power"]
    k_fines_production = params["k_fines_production"]
    k_fines_production_jt = params["k_fines_production_jt"]
    k_rock_consumption = params["k_rock_consumption"]
    mill_volume = params["mill_volume"]
    phi_norm = params["phi_norm"]
    power_max = params["power_max"]
    sump_volume = params["sump_volume"]
    cyclone_alpha_underflow = params["cyclone_alpha_underflow"]
    cyclone_c1 = params["cyclone_c1"]
    cyclone_c2 = params["cyclone_c2"]
    cyclone_c3 = params["cyclone_c3"]
    cyclone_epsilon_c = params["cyclone_epsilon_c"]

    # Symbolic time, state, and input vectors
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", N_STATES)
    u = cas.SX.sym("u", N_INPUTS)

    # Unpack states
    x_mw = x[0]  # mill water volume (m³)
    x_ms = x[1]  # mill solids volume (m³)
    x_mr = x[2]  # mill rock volume (m³)
    x_mf = x[3]  # mill fines volume (m³)
    x_sw = x[4]  # sump water volume (m³)
    x_ss = x[5]  # sump solids volume (m³)
    x_sf = x[6]  # sump fines volume (m³)

    # Unpack inputs
    u_MFO = u[0]  # mill feed ore rate (t/h)
    u_rMIW = u[1]  # mill water-to-ore ratio (-)
    u_phic = u[2]  # fraction of critical mill speed (-)
    u_SFW = u[3]  # sump feed water (m³/h)
    u_CFF = u[4]  # cyclone feed flow (m³/h)

    # ------------------------------------------------------------------
    # Mill intermediate variables
    # ------------------------------------------------------------------

    x_mb = ball_volume

    # Mill charge fraction, eq. (5)
    y_JT = (x_mw + x_ms + x_mr + x_mb) / mill_volume

    # Rheology factor φ, eq. (3)
    # φ = sqrt(1 - (ε₀⁻¹ - 1) x_ms/x_mw)  for x_ms/x_mw ≤ ε₀/(1-ε₀), else 0
    eps_slope = 1.0 / epsilon_zero - 1.0  # ε₀⁻¹ - 1
    eps_threshold = 1.0 / eps_slope  # (ε₀⁻¹ - 1)⁻¹ = ε₀/(1-ε₀)
    ratio_ms_mw = x_ms / x_mw
    phi = cas.if_else(
        ratio_ms_mw <= eps_threshold,
        cas.sqrt(1.0 - eps_slope * ratio_ms_mw),
        0.0,
    )

    # Mill power draw, eq. (6)
    y_Pmill = (
        power_max
        * u_phic
        * (
            1.0
            - delta_volume * (y_JT / fill_fraction_max_power - 1.0) ** 2
            - delta_solids * (phi / phi_norm - 1.0) ** 2
        )
    )

    # Mill discharge flow-rates, eq. (2)
    mill_slurry = x_ms + x_mw
    Q_mwo = phi * discharge_rate * x_mw * (x_mw / mill_slurry)  # eq. (2a)
    Q_mso = phi * discharge_rate * x_mw * (x_ms / mill_slurry)  # eq. (2b)
    Q_mfo = phi * discharge_rate * x_mw * (x_mf / mill_slurry)  # eq. (2c)

    # Rock consumption and fines production, eq. (4)
    Q_RC = (x_mr * y_Pmill) / (rho_ore * k_rock_consumption * (x_mr + x_ms))
    Q_FP = y_Pmill / (
        rho_ore
        * k_fines_production
        * (1.0 + k_fines_production_jt * (y_JT - fill_fraction_max_power))
    )

    # ------------------------------------------------------------------
    # Sump intermediate variables
    # ------------------------------------------------------------------

    # Sump discharge flow-rates, eq. (9)
    sump_slurry = x_sw + x_ss
    Q_swo = u_CFF * (x_sw / sump_slurry)  # eq. (9a)
    Q_sso = u_CFF * (x_ss / sump_slurry)  # eq. (9b)
    Q_sfo = u_CFF * (x_sf / sump_slurry)  # eq. (9c)

    # ------------------------------------------------------------------
    # Cyclone intermediate variables
    # ------------------------------------------------------------------

    # Feed fractions (sec. 3.2, step 2.8)
    F_i = Q_sso / u_CFF  # fraction solids in cyclone feed
    P_i = Q_sfo / Q_sso  # fraction fines in the solids feed

    # Coarse underflow, eq. (12)
    Q_ccu = (
        (Q_sso - Q_sfo)
        * (1.0 - cyclone_c1 * cas.exp(-u_CFF / cyclone_epsilon_c))
        * (1.0 - (F_i / cyclone_c2) ** cyclone_c3)
        * (1.0 - P_i**cyclone_c3)
    )

    # Fraction of solids in underflow, eq. (14)
    F_u = cyclone_c2 - (cyclone_c2 - F_i) * cas.exp(
        -Q_ccu / (cyclone_alpha_underflow * cyclone_epsilon_c)
    )

    # Cyclone underflow flow-rates, eq. (15)
    underflow_denom = F_u * Q_swo + F_u * Q_sfo - Q_sfo
    Q_cwu = Q_swo * Q_ccu * (1.0 - F_u) / underflow_denom  # eq. (15a)
    Q_cfu = Q_sfo * Q_ccu * (1.0 - F_u) / underflow_denom  # eq. (15b)
    Q_csu = Q_ccu + Q_cfu  # eq. (15c)

    # ------------------------------------------------------------------
    # State equations f(t, x, u, p) — equations (1) and (8)
    # ------------------------------------------------------------------

    dx_mw = u_rMIW * u_MFO / rho_water - Q_mwo + Q_cwu  # eq. (1a)
    dx_ms = (
        (1.0 - alpha_rocks) * u_MFO / rho_ore - Q_mso + Q_csu + Q_RC
    )  # eq. (1b)
    dx_mr = alpha_rocks * u_MFO / rho_ore - Q_RC  # eq. (1c)
    dx_mf = alpha_fines * u_MFO / rho_ore - Q_mfo + Q_cfu + Q_FP  # eq. (1d)

    dx_sw = Q_mwo - Q_swo + u_SFW  # eq. (8a)
    dx_ss = Q_mso - Q_sso  # eq. (8b)
    dx_sf = Q_mfo - Q_sfo  # eq. (8c)

    rhs = cas.vertcat(dx_mw, dx_ms, dx_mr, dx_mf, dx_sw, dx_ss, dx_sf)
    assert rhs.shape == (N_STATES, 1)

    # ------------------------------------------------------------------
    # Output equations h(t, x, u, p) — equations (5), (6), (10), (11), (16)
    # ------------------------------------------------------------------

    # Sump level, eq. (10)
    y_SLEV = 100.0 * (x_ss + x_sw) / sump_volume

    # Sump discharge density, eq. (11)
    y_rho = (rho_water * Q_swo + rho_ore * Q_sso) / (Q_swo + Q_sso)

    # Product particle size, eq. (16)
    Q_cfo = Q_sfo - Q_cfu  # cyclone fines overflow
    Q_cso = Q_sso - Q_ccu  # cyclone solids overflow
    y_PSE = 100.0 * (Q_cfo / Q_cso)

    y = cas.vertcat(y_JT, y_Pmill, y_SLEV, y_rho, y_PSE)
    assert y.shape == (N_OUTPUTS, 1)

    # ------------------------------------------------------------------
    # Build CasADi Function objects
    # ------------------------------------------------------------------

    # Only CasADi SX variables appear as explicit function arguments;
    # concrete Python scalars are folded directly into the expressions.
    sym_params = {k: v for k, v in params.items() if isinstance(v, cas.SX)}

    f_func = cas.Function(
        "f",
        [t, x, u, *sym_params.values()],
        [rhs],
        ["t", "x", "u", *sym_params.keys()],
        ["rhs"],
    )
    h_func = cas.Function(
        "h",
        [t, x, u, *sym_params.values()],
        [y],
        ["t", "x", "u", *sym_params.keys()],
        ["y"],
    )

    return StateSpaceModelCT(
        f_func,
        h_func,
        n=N_STATES,
        nu=N_INPUTS,
        ny=N_OUTPUTS,
        params=sym_params,
        name=name,
        state_names=STATE_NAMES,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
    )


def build_grinding_circuit_model_with_sump_control(
    level_min=SUMP_LEVEL_MIN,
    level_max=SUMP_LEVEL_MAX,
    cff_max=None,
    sump_volume=SUMP_VOLUME,
    name="c1_grind_sc",
    **model_kwargs,
):
    """Build a grinding circuit model with proportional sump level control.

    The cyclone feed pump (u_CFF) is no longer a free input — it is computed
    internally from a P controller that maps sump level linearly from
    ``level_min`` (pump off) to ``level_max`` (pump at maximum), saturating
    at both ends:

        u_CFF = clip(cff_max * (y_SLEV - level_min) / (level_max - level_min),
                     0, cff_max)

    ``cff_max`` defaults to the value that gives the paper's steady-state
    pump flow (2921 m³/h) at the paper's steady-state sump level (59.4 %).

    Parameters
    ----------
    level_min : float
        Sump level (%) at which u_CFF = 0.  Default SUMP_LEVEL_MIN.
    level_max : float
        Sump level (%) at which u_CFF = cff_max.  Default SUMP_LEVEL_MAX.
    cff_max : float or None
        Maximum cyclone feed flow (m³/h).  If None, derived from the paper's
        steady-state inputs and outputs.
    sump_volume : float
        Sump volume (m³); must match the value used in the base model.
    **model_kwargs
        Forwarded verbatim to ``build_grinding_circuit_model``.

    Returns
    -------
    StateSpaceModelCT
        n=7 states, nu=4 inputs (feed_ore_rate, water_ore_ratio,
        critical_speed_fraction, sump_feed_water), ny=6 outputs (original 5
        plus cyclone_feed_flow so the controller action can be monitored).
    """
    if cff_max is None:
        u_cff_ss = INPUTS_NOP["cyclone_feed_flow"]
        level_ss = OUTPUTS_NOP["sump_level"]
        cff_max = u_cff_ss * (level_max - level_min) / (level_ss - level_min)

    base = build_grinding_circuit_model(
        sump_volume=sump_volume, **model_kwargs
    )

    # Closed-loop model (sump level controller): u_CFF is internal, shown as output
    CL_INPUT_NAMES = INPUT_NAMES.copy()
    CL_INPUT_NAMES.remove("cyclone_feed_flow")
    CL_OUTPUT_NAMES = OUTPUT_NAMES + ["cyclone_feed_flow"]

    N_CL_INPUTS = len(CL_INPUT_NAMES)  # 4 cyclone_feed_flow is now controlled
    N_CL_OUTPUTS = len(CL_OUTPUT_NAMES)  # 6 cyclone_feed_flow added

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", N_STATES)
    u = cas.SX.sym("u", N_CL_INPUTS)

    # Sump level (%) from state — same formula as base model eq. (10)
    x_sw = x[4]  # sump water volume (m³)
    x_ss = x[5]  # sump solids volume (m³)
    y_SLEV_ctrl = 100.0 * (x_sw + x_ss) / sump_volume

    # P controller with saturation
    Kp = cff_max / (level_max - level_min)
    u_CFF = cas.fmin(
        float(cff_max), cas.fmax(0.0, Kp * (y_SLEV_ctrl - level_min))
    )

    # Full 5-element input vector for the base model
    u_full = cas.vertcat(u, u_CFF)
    assert u_full.shape == (N_INPUTS, 1)

    # Compose with base model (calling a CasADi Function with SX args returns SX)
    sym_params = base.params
    rhs = base.f(t, x, u_full, *sym_params.values())
    y_base = base.h(t, x, u_full, *sym_params.values())
    y_cl = cas.vertcat(y_base, u_CFF)
    assert y_cl.shape == (N_CL_OUTPUTS, 1)

    f_func = cas.Function(
        "f_cl",
        [t, x, u, *sym_params.values()],
        [rhs],
        ["t", "x", "u", *sym_params.keys()],
        ["rhs"],
    )
    h_func = cas.Function(
        "h_cl",
        [t, x, u, *sym_params.values()],
        [y_cl],
        ["t", "x", "u", *sym_params.keys()],
        ["y"],
    )

    return StateSpaceModelCT(
        f_func,
        h_func,
        n=N_STATES,
        nu=N_CL_INPUTS,
        ny=N_CL_OUTPUTS,
        params=sym_params,
        name=name,
        state_names=STATE_NAMES,
        input_names=CL_INPUT_NAMES,
        output_names=CL_OUTPUT_NAMES,
    )


def build_grinding_circuit_model_with_level_control(
    charge_fill_fraction_sp=OUTPUTS_NOP["charge_fill_fraction"],
    mfo_nop=INPUTS_NOP["feed_ore_rate"],
    mfo_min=MFO_MIN,
    mfo_max=MFO_MAX,
    sump_level_min=SUMP_LEVEL_MIN,
    sump_level_max=SUMP_LEVEL_MAX,
    cff_max=None,
    sump_volume=SUMP_VOLUME,
    ball_volume=BALL_VOLUME,
    mill_volume=MILL_VOLUME,
    name="c1_grind_sc_lc",
    **model_kwargs,
):
    """Build a grinding circuit model with charge level and sump level control.

    Both the ore feed rate (u_MFO) and the cyclone feed pump (u_CFF) are
    computed internally from P controllers.

    Charge level controller — u_MFO is driven by the charge fill fraction
    error, biased at the NOP feed rate:

        u_MFO = clip(mfo_nop + K_c * (charge_fill_fraction_sp - y_JT),
                     mfo_min, mfo_max)

    where K_c is derived from step-test plant gain data.

    Sump level controller — u_CFF is mapped proportionally from sump level,
    saturating at both ends:

        u_CFF = clip(cff_max * (y_SLEV - sump_level_min)
                              / (sump_level_max - sump_level_min),
                     0, cff_max)

    ``cff_max`` defaults to the value that gives the paper's steady-state
    pump flow (2921 m³/h) at the paper's steady-state sump level (59.4 %).

    Parameters
    ----------
    charge_fill_fraction_sp : float
        Charge fill fraction setpoint (-).  Default OUTPUTS_NOP["charge_fill_fraction"].
    mfo_nop : float
        Feed rate bias at NOP (t/h).  Default INPUTS_NOP["feed_ore_rate"].
    mfo_min : float
        Minimum ore feed rate (t/h).  Default MFO_MIN.
    mfo_max : float
        Maximum ore feed rate (t/h).  Default MFO_MAX.
    sump_level_min : float
        Sump level (%) at which u_CFF = 0.  Default SUMP_LEVEL_MIN.
    sump_level_max : float
        Sump level (%) at which u_CFF = cff_max.  Default SUMP_LEVEL_MAX.
    cff_max : float or None
        Maximum cyclone feed flow (m³/h).  If None, derived from the paper's
        steady-state inputs and outputs.
    sump_volume : float
        Sump volume (m³); must match the value used in the base model.
    ball_volume : float
        Volume of steel balls in mill (m³).  Default BALL_VOLUME.
    mill_volume : float
        Mill volume (m³).  Default MILL_VOLUME.
    **model_kwargs
        Forwarded verbatim to ``build_grinding_circuit_model``.

    Returns
    -------
    StateSpaceModelCT
        n=7 states, nu=3 inputs (water_ore_ratio, critical_speed_fraction,
        sump_feed_water), ny=7 outputs (original 5 plus cyclone_feed_flow
        and feed_ore_rate so both controller actions can be monitored).
    """

    K_c_mfo = 1 / CHARGE_LEVEL_PLANT_GAIN

    # Plant time constant
    # T1 = 64  # mins approx. from step tests
    # TODO: do we need a PI controller here? If so, we need to add an integrator state
    # T_i = T1

    # Sump level controller parameters
    if cff_max is None:
        u_cff_ss = INPUTS_NOP["cyclone_feed_flow"]
        level_ss = OUTPUTS_NOP["sump_level"]
        cff_max = (
            u_cff_ss
            * (sump_level_max - sump_level_min)
            / (level_ss - sump_level_min)
        )

    Kc_sump = cff_max / (sump_level_max - sump_level_min)

    base = build_grinding_circuit_model(
        sump_volume=sump_volume,
        ball_volume=ball_volume,
        mill_volume=mill_volume,
        **model_kwargs,
    )

    # Closed-loop model (sump level controller): u_CFF is internal, shown as output
    CL_INPUT_NAMES = INPUT_NAMES.copy()
    CL_INPUT_NAMES.remove("cyclone_feed_flow")
    CL_INPUT_NAMES.remove("feed_ore_rate")
    CL_OUTPUT_NAMES = OUTPUT_NAMES + ["cyclone_feed_flow", "feed_ore_rate"]

    # Cyclone_feed_flow and feed_ore_rate are now controlled
    N_CL_INPUTS = len(CL_INPUT_NAMES)  # 3
    N_CL_OUTPUTS = len(CL_OUTPUT_NAMES)  # 7

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", N_STATES)
    u = cas.SX.sym("u", N_CL_INPUTS)

    # Sump level (%) from state — same formula as base model eq. (10)
    x_sw = x[4]  # sump water volume (m³)
    x_ss = x[5]  # sump solids volume (m³)
    y_SLEV_ctrl = 100.0 * (x_sw + x_ss) / sump_volume

    # Sump level P controller with saturation
    u_CFF = cas.fmin(
        float(cff_max), cas.fmax(0.0, Kc_sump * (y_SLEV_ctrl - sump_level_min))
    )

    # Charge level from state — same formula as base model eq. (5)
    x_mw = x[0]
    x_ms = x[1]
    x_mr = x[2]
    ball_volume = base.params.get("ball_volume", ball_volume)
    mill_volume = base.params.get("mill_volume", mill_volume)
    y_JT = (x_mw + x_ms + x_mr + ball_volume) / mill_volume

    # Charge level P Controller with saturation
    u_mfo = cas.fmin(
        float(mfo_max),
        cas.fmax(
            float(mfo_min),
            mfo_nop + K_c_mfo * (charge_fill_fraction_sp - y_JT),
        ),
    )

    # Full 5-element input vector for the base model
    u_full = cas.vertcat(u_mfo, u, u_CFF)
    assert u_full.shape == (N_INPUTS, 1)

    # Compose with base model (calling a CasADi Function with SX args returns SX)
    sym_params = base.params
    rhs = base.f(t, x, u_full, *sym_params.values())
    y_base = base.h(t, x, u_full, *sym_params.values())

    # Append additional output variables for monitoring controller actions
    y_cl = cas.vertcat(y_base, u_CFF, u_mfo)
    assert y_cl.shape == (N_CL_OUTPUTS, 1)

    f_func = cas.Function(
        "f_cl",
        [t, x, u, *sym_params.values()],
        [rhs],
        ["t", "x", "u", *sym_params.keys()],
        ["rhs"],
    )
    h_func = cas.Function(
        "h_cl",
        [t, x, u, *sym_params.values()],
        [y_cl],
        ["t", "x", "u", *sym_params.keys()],
        ["y"],
    )

    return StateSpaceModelCT(
        f_func,
        h_func,
        n=N_STATES,
        nu=N_CL_INPUTS,
        ny=N_CL_OUTPUTS,
        params=sym_params,
        name=name,
        state_names=STATE_NAMES,
        input_names=CL_INPUT_NAMES,
        output_names=CL_OUTPUT_NAMES,
    )
