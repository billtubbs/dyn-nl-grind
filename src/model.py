import casadi as cas

from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.param_utils import make_symbolic_vars_from_kwargs


_default_parameters = {
    "rho_balls": 7.85,
    "rho_charge": None,  # Calculated output
    "rho_ore": 2.8,
    "rho_water": 1.0,
    "alpha_fines": 0.1,
    "alpha_rocks": 0.1,
    "delta_solid_fraction": 0.5,
    "delta_fill_volume": 0.5,
    "discharge_rate": 0.5,
    "epsilon_zero": 0.6,
    "porosity": 0.4,
    "phi_normalization": 1.0,
    "ball_fill_fraction": 0.3,
    "fill_fraction_max_power": 0.5,
    "k_fines_production": 0.1,
    "k_fpjt": 0.5,
    "k_rock_consumption": 0.1,
    "p_max": 4.0,
    "discharge_solids_content": 0.6,
    "charge_voidage": 0.4,
    "mill_volume": 100,
    "sump_volume": 200,
    "cyclone_alpha_underflow": 0.5,
    "cyclone_c1": 0.7,
    "cyclone_c2": 0.7,
    "cyclone_c3": 3,
    "cyclone_epsilon_c": 0.1,
    "x_mb": 0.0,
}


def build_mill_model(**parameter_values) -> StateSpaceModelCT:
    """Build the grinding circuit state-space model from le Roux and Steyn (2022)."""
    params = make_symbolic_vars_from_kwargs(
        **{**_default_parameters, **parameter_values}
    )

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 7, 1)
    u = cas.SX.sym("u", 5, 1)

    x_mw, x_ms, x_mr, x_mf, x_sw, x_ss, x_sf = [x[i] for i in range(7)]
    u_MFO, u_rMIW, u_phi_c, u_SFW, u_CFF = [u[i] for i in range(5)]

    y_JT = (x_mw + x_ms + x_mr + params["x_mb"]) / params["mill_volume"]

    phi_condition = x_mw / x_ms <= params["epsilon_zero"] - 1
    phi_term = 1 - (params["epsilon_zero"] - 1 - x_ms / x_mw) ** (-1)
    phi = cas.if_else(phi_condition, phi_term, 0)

    q_denominator = x_ms + x_mw
    q_mwo = params["discharge_rate"] * phi * x_mw / q_denominator
    q_mso = params["discharge_rate"] * phi * x_mw * x_ms / q_denominator
    q_mfo = params["discharge_rate"] * phi * x_mw * x_mf / q_denominator

    term_fill = (1 - params["delta_fill_volume"]) / (1 / y_JT - 1)
    term_viscosity = (
        params["delta_solid_fraction"]
        * phi
        / (params["fill_fraction_max_power"] * params["phi_normalization"])
    )
    y_Pmill = (
        params["p_max"] * u_phi_c * cas.power(term_fill - term_viscosity, 2)
    )

    q_RC = (
        x_mr
        * y_Pmill
        / (params["rho_ore"] * params["k_rock_consumption"] * (x_mr + x_ms))
    )
    q_FP = y_Pmill / (
        params["rho_ore"]
        * params["k_fines_production"]
        * (1 + params["k_fpjt"] * y_JT - params["fill_fraction_max_power"])
    )

    rho_charge = (
        params["rho_ore"]
        * (
            1
            - params["porosity"]
            + params["porosity"]
            * params["charge_voidage"]
            * params["discharge_solids_content"]
        )
        + params["ball_fill_fraction"]
        * (params["rho_balls"] - params["rho_ore"])
        * (1 - params["porosity"])
        + params["porosity"]
        * params["charge_voidage"]
        * (1 - params["discharge_solids_content"])
    ) / y_JT

    q_swo = u_CFF * x_sw / (x_sw + x_ss)
    q_sso = u_CFF * x_ss / (x_sw + x_ss)
    q_sfo = u_CFF * x_sf / (x_sw + x_ss)

    Fi = q_sso / (q_swo + q_sso)
    Pi = q_sfo / q_sso
    q_ccu = q_sso - q_sfo * (
        1
        - params["cyclone_c1"]
        * cas.exp(
            -u_CFF
            / params["cyclone_epsilon_c"]
            * (Fi**3 / (1 - (1 - Pi ** params["cyclone_c3"]) ** (-1)))
        )
    )

    Fu = params["cyclone_c2"] - (params["cyclone_c2"] - Fi) * cas.exp(
        -q_ccu
        / (params["cyclone_alpha_underflow"] * params["cyclone_epsilon_c"])
    )

    q_cwu = q_swo * (q_ccu - Fu * q_ccu) / (Fu * q_swo + Fu * q_sfo - q_sfo)
    q_cfu = q_sfo * (q_ccu - Fu * q_ccu) / (Fu * q_swo + Fu * q_sfo - q_sfo)
    q_csu = q_ccu + q_cfu

    q_cwo = q_swo - q_cwu
    q_cso = q_sso - q_csu
    q_cfo = q_sfo - q_cfu

    y_SLEV = 100 * (x_ss + x_sw) / params["sump_volume"]
    y_rho = (params["rho_water"] * q_swo + params["rho_ore"] * q_sso) / (
        q_swo + q_sso
    )
    y_PSE = 100 * q_cfo / q_cso

    dx_mw = (
        (1 - params["alpha_rocks"] - params["alpha_fines"])
        / params["rho_ore"]
        * u_MFO
        - q_mwo
        + q_cwu
    )
    dx_ms = (
        (1 - params["alpha_rocks"]) / params["rho_ore"] * u_MFO
        - q_mso
        + q_csu
        + q_RC
    )
    dx_mr = params["alpha_rocks"] / params["rho_ore"] * u_MFO - q_RC
    dx_mf = (
        params["alpha_fines"] / params["rho_ore"] * u_MFO
        - q_mfo
        + q_cfu
        + q_FP
    )
    dx_sw = q_mwo - q_swo + u_SFW
    dx_ss = q_mso - q_sso
    dx_sf = q_mfo - q_sfo

    rhs = cas.vertcat(dx_mw, dx_ms, dx_mr, dx_mf, dx_sw, dx_ss, dx_sf)
    y = cas.vertcat(y_JT, y_Pmill, y_SLEV, y_rho, y_PSE)

    param_list = [params[name] for name in params]
    param_names = list(params.keys())

    f = cas.Function(
        "f",
        [t, x, u, *param_list],
        [rhs],
        ["t", "x", "u", *param_names],
        ["rhs"],
    )
    h = cas.Function(
        "h",
        [t, x, u, *param_list],
        [y],
        ["t", "x", "u", *param_names],
        ["y"],
    )

    return StateSpaceModelCT(
        f,
        h,
        n=7,
        nu=5,
        ny=5,
        params=params,
        name="grinding_circuit",
        input_names=[
            "u_MFO",
            "u_rMIW",
            "u_phi_c",
            "u_SFW",
            "u_CFF",
        ],
        state_names=[
            "x_mw",
            "x_ms",
            "x_mr",
            "x_mf",
            "x_sw",
            "x_ss",
            "x_sf",
        ],
        output_names=["y_JT", "y_Pmill", "y_SLEV", "y_rho", "y_PSE"],
    )
