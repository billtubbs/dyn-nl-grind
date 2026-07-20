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

# Names for the sump-level controlled model (u_CFF removed from inputs, added
# as a monitored output).
CL_INPUT_NAMES = [n for n in INPUT_NAMES if n != "cyclone_feed_flow"]
CL_OUTPUT_NAMES = OUTPUT_NAMES + ["cyclone_feed_flow"]

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
PHI_BETA = (
    20.0  # β: softplus sharpness for φ smoothing  (larger → closer to paper)
)
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


def calculate_mill_power(
    u_phic,
    y_JT,
    phi,
    power_max=POWER_MAX,
    fill_fraction_max_power=FILL_FRACTION_MAX_POWER,
    delta_volume=DELTA_VOLUME,
    phi_norm=PHI_NORM,
    delta_solids=DELTA_SOLIDS,
):
    """Mill power draw, eq. (6)."""
    y_Pmill = (
        power_max
        * u_phic
        * (
            1.0
            - delta_volume * (y_JT / fill_fraction_max_power - 1.0) ** 2
            - delta_solids * (phi / phi_norm - 1.0) ** 2
        )
    )
    return y_Pmill


def calculate_rheology_factor(
    x_ms,
    x_mw,
    epsilon_zero=EPSILON_ZERO,
):
    """Rheology factor φ — original paper eq. (3).

    φ = sqrt(1 - ε_slope · r)  for r ≤ ε₀/(1-ε₀), else 0
    where r = x_ms/x_mw and ε_slope = ε₀⁻¹ - 1.

    Note: uses cas.if_else, which introduces a discontinuity at the
    threshold.  Use calculate_rheology_factor_smoothed for gradient-based
    optimisation or when using stiff ODE integrators.
    """
    eps_slope = 1.0 / epsilon_zero - 1.0  # ε₀⁻¹ - 1
    eps_threshold = 1.0 / eps_slope  # ε₀ / (1 - ε₀)
    ratio_ms_mw = x_ms / x_mw
    return cas.if_else(
        ratio_ms_mw <= eps_threshold,
        cas.sqrt(1.0 - eps_slope * ratio_ms_mw),
        0.0,
    )


def calculate_rheology_factor_smoothed(
    x_ms,
    x_mw_p,
    epsilon_zero=EPSILON_ZERO,
    phi_beta=PHI_BETA,
):
    """Rheology factor φ — softplus-smoothed version of eq. (3).

    Paper: φ = sqrt(max(0, 1 - ε_slope·r))  where r = x_ms/x_mw.
    Softplus replaces max(0, x) with log(1+exp(β·x))/β, giving a C∞
    function with no conditional branches (safe for any CasADi integrator).
    β=phi_beta controls sharpness: larger β → closer to paper formula.
    """
    eps_slope = 1.0 / epsilon_zero - 1.0  # ε₀⁻¹ - 1
    ratio_ms_mw = x_ms / x_mw_p
    _sp_arg = phi_beta * (1.0 - eps_slope * ratio_ms_mw)
    # Numerically stable softplus: log(1+exp(x)) = max(x,0) + log(1+exp(-|x|))
    # Avoids exp overflow when x_mw is transiently negative in RK4 stages.
    _softplus = cas.fmax(_sp_arg, 0.0) + cas.log(
        1.0 + cas.exp(-cas.fabs(_sp_arg))
    )
    return cas.sqrt(_softplus / phi_beta)


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
    phi_beta=PHI_BETA,
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

    State vector x (n=7)
    --------------------
    x[0]  water_volume          mill water volume            (m³)   [x_mw]
    x[1]  solids_volume         mill solids volume           (m³)   [x_ms]
    x[2]  rock_volume           mill rock volume             (m³)   [x_mr]
    x[3]  fines_volume          mill fines volume            (m³)   [x_mf]
    x[4]  sump_water_volume     sump water volume            (m³)   [x_sw]
    x[5]  sump_solids_volume    sump solids volume           (m³)   [x_ss]
    x[6]  sump_fines_volume     sump fines volume            (m³)   [x_sf]

    Input vector u (nu=5)
    ---------------------
    u[0]  feed_ore_rate         mill feed ore rate           (t/h)  [u_MFO]
    u[1]  water_ore_ratio       mill water-to-ore ratio      (-)    [u_rMIW]
    u[2]  critical_speed_fraction  fraction of critical speed  (-)  [u_phic]
    u[3]  sump_feed_water       sump feed water flow         (m³/h) [u_SFW]
    u[4]  cyclone_feed_flow     cyclone feed pump flow       (m³/h) [u_CFF]

    Output vector y (ny=5)
    ----------------------
    y[0]  charge_fill_fraction  mill charge fill fraction    (-)    [y_JT]
    y[1]  mill_power            mill power draw              (MW)   [y_Pmill]
    y[2]  sump_level            sump level                   (%)    [y_SLEV]
    y[3]  sump_density          sump discharge density       (t/m³) [y_rho]
    y[4]  product_size          product particle size (PSE)  (%)    [y_PSE]

    Returns
    -------
    StateSpaceModelCT
        Continuous-time state-space model with n=7, nu=5, ny=5.
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
        phi_beta=phi_beta,
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
    phi_beta = params["phi_beta"]
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
    water_volume = x[0]
    solids_volume = x[1]
    rock_volume = x[2]
    fines_volume = x[3]
    sump_water_volume = x[4]
    sump_solids_volume = x[5]
    sump_fines_volume = x[6]

    # Unpack inputs
    feed_ore_rate = u[0]
    water_ore_ratio = u[1]
    critical_speed_fraction = u[2]
    sump_feed_water = u[3]
    cyclone_feed_flow = u[4]

    # Volume floors — prevent division-by-zero when the integrator drives a
    # state transiently negative.  1e-6 m³ ≪ all physical steady-state volumes
    # (~5–130 m³), so these guards are inactive during normal operation.
    _EPS = 1e-6
    water_volume_p = cas.fmax(water_volume, _EPS)
    sump_solids_volume_p = cas.fmax(sump_solids_volume, _EPS)
    mill_slurry_p = cas.fmax(solids_volume + water_volume, _EPS)
    sump_slurry_p = cas.fmax(sump_water_volume + sump_solids_volume, _EPS)

    # ------------------------------------------------------------------
    # Mill intermediate variables
    # ------------------------------------------------------------------

    # Mill charge fraction, eq. (5)
    charge_fill_fraction = (
        water_volume + solids_volume + rock_volume + ball_volume
    ) / mill_volume

    # Rheology factor φ, eq. (3)
    phi = calculate_rheology_factor_smoothed(
        solids_volume,
        water_volume_p,
        epsilon_zero=epsilon_zero,
        phi_beta=phi_beta,
    )

    # Mill power draw, eq. (6)
    mill_power = calculate_mill_power(
        critical_speed_fraction,
        charge_fill_fraction,
        phi,
        power_max=power_max,
        fill_fraction_max_power=fill_fraction_max_power,
        delta_volume=delta_volume,
        phi_norm=phi_norm,
        delta_solids=delta_solids,
    )

    # Mill discharge flow-rates, eq. (2a-c)
    Q_mwo = phi * discharge_rate * water_volume**2 / mill_slurry_p
    Q_mso = phi * discharge_rate * water_volume * solids_volume / mill_slurry_p
    Q_mfo = phi * discharge_rate * water_volume * fines_volume / mill_slurry_p

    # Rock consumption and fines production, eq. (4)
    Q_RC = (rock_volume * mill_power) / (
        rho_ore
        * k_rock_consumption
        * cas.fmax(rock_volume + solids_volume, _EPS)
    )
    Q_FP = mill_power / (
        rho_ore
        * k_fines_production
        * (
            1.0
            + k_fines_production_jt
            * (charge_fill_fraction - fill_fraction_max_power)
        )
    )

    # ------------------------------------------------------------------
    # Sump intermediate variables
    # ------------------------------------------------------------------

    # Sump discharge flow-rates, eq. (9a-c)
    Q_swo = cyclone_feed_flow * sump_water_volume / sump_slurry_p
    Q_sso = cyclone_feed_flow * sump_solids_volume / sump_slurry_p
    Q_sfo = cyclone_feed_flow * sump_fines_volume / sump_slurry_p

    # ------------------------------------------------------------------
    # Cyclone intermediate variables
    # ------------------------------------------------------------------

    # Feed fractions (sec. 3.2, step 2.8)
    F_i = sump_solids_volume / sump_slurry_p  # solids fraction in cyclone feed
    P_i = sump_fines_volume / sump_solids_volume_p  # fines fraction in solids

    # Coarse underflow, eq. (12)
    Q_ccu = (
        (Q_sso - Q_sfo)
        * (1.0 - cyclone_c1 * cas.exp(-cyclone_feed_flow / cyclone_epsilon_c))
        * (1.0 - (F_i / cyclone_c2) ** cyclone_c3)
        * (1.0 - P_i**cyclone_c3)
    )

    # Fraction of solids in underflow, eq. (14)
    F_u = cyclone_c2 - (cyclone_c2 - F_i) * cas.exp(
        -Q_ccu / (cyclone_alpha_underflow * cyclone_epsilon_c)
    )

    # Cyclone underflow flow-rates, eq. (15)
    # Clip the fraction to [0, 1] so Q_cwu ≤ Q_swo and Q_cfu ≤ Q_sfo always,
    # preventing Inf when underflow_denom approaches zero transiently.
    _uf_denom = F_u * Q_swo + F_u * Q_sfo - Q_sfo
    _uf_frac = cas.fmin(
        1.0, cas.fmax(0.0, Q_ccu * (1.0 - F_u) / cas.fmax(_uf_denom, _EPS))
    )
    Q_cwu = Q_swo * _uf_frac  # eq. (15a)
    Q_cfu = Q_sfo * _uf_frac  # eq. (15b)
    Q_csu = Q_ccu + Q_cfu  # eq. (15c)

    # ------------------------------------------------------------------
    # State equations f(t, x, u, p) — equations (1) and (8)
    # ------------------------------------------------------------------

    d_water_volume = (  # eq. (1a)
        water_ore_ratio * feed_ore_rate / rho_water - Q_mwo + Q_cwu
    )
    d_solids_volume = (  # eq. (1b)
        (1.0 - alpha_rocks) * feed_ore_rate / rho_ore - Q_mso + Q_csu + Q_RC
    )
    d_rock_volume = alpha_rocks * feed_ore_rate / rho_ore - Q_RC  # eq. (1c)
    d_fines_volume = (  # eq. (1d)
        alpha_fines * feed_ore_rate / rho_ore - Q_mfo + Q_cfu + Q_FP
    )
    d_sump_water_volume = Q_mwo - Q_swo + sump_feed_water  # eq. (8a)
    d_sump_solids_volume = Q_mso - Q_sso  # eq. (8b)
    d_sump_fines_volume = Q_mfo - Q_sfo  # eq. (8c)

    rhs = cas.vertcat(
        d_water_volume,
        d_solids_volume,
        d_rock_volume,
        d_fines_volume,
        d_sump_water_volume,
        d_sump_solids_volume,
        d_sump_fines_volume,
    )
    assert rhs.shape == (N_STATES, 1)

    # ------------------------------------------------------------------
    # Output equations h(t, x, u, p) — equations (5), (6), (10), (11), (16)
    # ------------------------------------------------------------------

    # Sump level, eq. (10)
    sump_level = 100.0 * (sump_solids_volume + sump_water_volume) / sump_volume

    # Sump discharge density, eq. (11)
    sump_density = (rho_water * Q_swo + rho_ore * Q_sso) / cas.fmax(
        Q_swo + Q_sso, _EPS
    )

    # Product particle size, eq. (16)
    Q_cfo = Q_sfo - Q_cfu  # cyclone fines overflow
    Q_cso = Q_sso - Q_ccu  # cyclone solids overflow
    product_size = 100.0 * (Q_cfo / cas.fmax(Q_cso, _EPS))

    y = cas.vertcat(
        charge_fill_fraction,
        mill_power,
        sump_level,
        sump_density,
        product_size,
    )
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

    State vector x (n=7)
    --------------------
    x[0]  water_volume          mill water volume            (m³)   [x_mw]
    x[1]  solids_volume         mill solids volume           (m³)   [x_ms]
    x[2]  rock_volume           mill rock volume             (m³)   [x_mr]
    x[3]  fines_volume          mill fines volume            (m³)   [x_mf]
    x[4]  sump_water_volume     sump water volume            (m³)   [x_sw]
    x[5]  sump_solids_volume    sump solids volume           (m³)   [x_ss]
    x[6]  sump_fines_volume     sump fines volume            (m³)   [x_sf]

    Input vector u (nu=4)
    ---------------------
    u[0]  feed_ore_rate         mill feed ore rate           (t/h)  [u_MFO]
    u[1]  water_ore_ratio       mill water-to-ore ratio      (-)    [u_rMIW]
    u[2]  critical_speed_fraction  fraction of critical speed  (-)  [u_phic]
    u[3]  sump_feed_water       sump feed water flow         (m³/h) [u_SFW]

    Output vector y (ny=6)
    ----------------------
    y[0]  charge_fill_fraction  mill charge fill fraction    (-)    [y_JT]
    y[1]  mill_power            mill power draw              (MW)   [y_Pmill]
    y[2]  sump_level            sump level                   (%)    [y_SLEV]
    y[3]  sump_density          sump discharge density       (t/m³) [y_rho]
    y[4]  product_size          product particle size (PSE)  (%)    [y_PSE]
    y[5]  cyclone_feed_flow     cyclone feed pump flow (controlled)  (m³/h) [u_CFF]

    Returns
    -------
    StateSpaceModelCT
        Continuous-time state-space model with n=7, nu=4, ny=6.
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
    sump_water_volume = x[4]
    sump_solids_volume = x[5]
    sump_level_ctrl = (
        100.0 * (sump_water_volume + sump_solids_volume) / sump_volume
    )

    # P controller with saturation
    Kp = cff_max / (level_max - level_min)
    cyclone_feed_flow = cas.fmin(
        float(cff_max), cas.fmax(0.0, Kp * (sump_level_ctrl - level_min))
    )

    # Full 5-element input vector for the base model
    u_full = cas.vertcat(u, cyclone_feed_flow)
    assert u_full.shape == (N_INPUTS, 1)

    # Compose with base model (calling a CasADi Function with SX args returns SX)
    sym_params = base.params
    rhs = base.f(t, x, u_full, *sym_params.values())
    y_base = base.h(t, x, u_full, *sym_params.values())
    y_cl = cas.vertcat(y_base, cyclone_feed_flow)
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

    State vector x (n=7)
    --------------------
    x[0]  water_volume          mill water volume            (m³)   [x_mw]
    x[1]  solids_volume         mill solids volume           (m³)   [x_ms]
    x[2]  rock_volume           mill rock volume             (m³)   [x_mr]
    x[3]  fines_volume          mill fines volume            (m³)   [x_mf]
    x[4]  sump_water_volume     sump water volume            (m³)   [x_sw]
    x[5]  sump_solids_volume    sump solids volume           (m³)   [x_ss]
    x[6]  sump_fines_volume     sump fines volume            (m³)   [x_sf]

    Input vector u (nu=3)
    ---------------------
    u[0]  water_ore_ratio       mill water-to-ore ratio      (-)    [u_rMIW]
    u[1]  critical_speed_fraction  fraction of critical speed  (-)  [u_phic]
    u[2]  sump_feed_water       sump feed water flow         (m³/h) [u_SFW]

    Output vector y (ny=7)
    ----------------------
    y[0]  charge_fill_fraction  mill charge fill fraction    (-)    [y_JT]
    y[1]  mill_power            mill power draw              (MW)   [y_Pmill]
    y[2]  sump_level            sump level                   (%)    [y_SLEV]
    y[3]  sump_density          sump discharge density       (t/m³) [y_rho]
    y[4]  product_size          product particle size (PSE)  (%)    [y_PSE]
    y[5]  cyclone_feed_flow     cyclone feed pump flow (controlled)  (m³/h) [u_CFF]
    y[6]  feed_ore_rate         mill feed ore rate (controlled)      (t/h)  [u_MFO]

    Returns
    -------
    StateSpaceModelCT
        Continuous-time state-space model with n=7, nu=3, ny=7.
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
    sump_water_volume = x[4]
    sump_solids_volume = x[5]
    sump_level_ctrl = (
        100.0 * (sump_water_volume + sump_solids_volume) / sump_volume
    )

    # Sump level P controller with saturation
    cyclone_feed_flow = cas.fmin(
        float(cff_max),
        cas.fmax(0.0, Kc_sump * (sump_level_ctrl - sump_level_min)),
    )

    # Charge level from state — same formula as base model eq. (5)
    water_volume = x[0]
    solids_volume = x[1]
    rock_volume = x[2]
    ball_volume = base.params.get("ball_volume", ball_volume)
    mill_volume = base.params.get("mill_volume", mill_volume)
    charge_fill_fraction = (
        water_volume + solids_volume + rock_volume + ball_volume
    ) / mill_volume

    # Charge level P controller with saturation
    feed_ore_rate = cas.fmin(
        float(mfo_max),
        cas.fmax(
            float(mfo_min),
            mfo_nop
            + K_c_mfo * (charge_fill_fraction_sp - charge_fill_fraction),
        ),
    )

    # Full 5-element input vector for the base model
    u_full = cas.vertcat(feed_ore_rate, u, cyclone_feed_flow)
    assert u_full.shape == (N_INPUTS, 1)

    # Compose with base model (calling a CasADi Function with SX args returns SX)
    sym_params = base.params
    rhs = base.f(t, x, u_full, *sym_params.values())
    y_base = base.h(t, x, u_full, *sym_params.values())

    # Append additional output variables for monitoring controller actions
    y_cl = cas.vertcat(y_base, cyclone_feed_flow, feed_ore_rate)
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
