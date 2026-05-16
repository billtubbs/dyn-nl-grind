"""Steady-state input-output analysis for casadi-models.

Generic utilities for computing and plotting steady-state I/O characteristics
of any continuous-time model built with casadi-models.

Functions
---------
compute_ss_sweeps
    Sweep each input over a range, collecting steady-state outputs via
    warm-start Newton continuation from the nominal operating point.
plot_main
    Full n_outputs × n_inputs grid of steady-state gain curves with axes.
plot_abstract
    Compact version without tick marks, for qualitative analysis.

Run as a script to generate the steady-state I/O characteristics of the
grinding circuit model defined in src/model.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _bisect_boundary(
    ss_solver, x0_stable, u_nom, j, stable_val, unstable_val, param_vals, tol
):
    """Bisect to find the input stability boundary between stable_val and unstable_val.

    Works in either direction (stable_val may be above or below unstable_val).
    Returns the last stable input value found — a conservative estimate of the
    true boundary.  Warm-starts from x0_stable throughout so Newton stays near
    convergence.
    """
    x0 = x0_stable.copy()
    while abs(unstable_val - stable_val) > tol:
        mid = 0.5 * (stable_val + unstable_val)
        u = u_nom.copy()
        u[j] = mid
        try:
            x_ss, _ = ss_solver(x0, u, param_vals)
            stable_val = mid
            x0 = x_ss
        except RuntimeError:
            unstable_val = mid
    return stable_val


def compute_ss_sweeps(
    ss_solver,
    x0_nom,
    u_nom,
    input_names,
    output_names,
    output_idx,
    input_sweeps,
    all_input_names=None,
    param_vals=None,
    find_boundary=False,
    boundary_tol=None,
):
    """Compute steady-state outputs over independent input sweeps.

    For each input in input_names, holds all other inputs at nominal and sweeps
    that input over input_sweeps[input_name], collecting steady-state output values.
    Uses warm-start Newton continuation outward from the nominal point in
    both directions.

    Parameters
    ----------
    ss_solver : callable
        Steady-state solver, signature ``(x0, u, param_vals) -> (x_ss, y_ss)``.
        Typically from ``cas_models.continuous_time.simulate.make_steady_state_solver``.
    x0_nom : array-like, shape (n,)
        Nominal state vector used as the initial guess for the Newton solver.
    u_nom : array-like, shape (nu,)
        Nominal input vector.
    input_names : list of str
        Names of inputs to sweep.
    output_names : list of str
        Names of outputs to collect.
    output_idx : dict
        Mapping ``{output_name: index}`` in the y_ss vector.
    input_sweeps : dict
        ``{input_name: sorted_array_of_values}`` for each input; arrays should
        include or straddle the nominal value so warm-starting works.
    all_input_names : list of str, optional
        Ordered list of all model input names matching u_nom.  Used to
        locate each input's position in u_nom.  If None, input_names is assumed
        to match u_nom in order and length.
    param_vals : dict, optional
        Parameter values forwarded to ss_solver.  Defaults to ``{}``.
    find_boundary : bool, optional
        If True, when the solver fails during a sweep, bisect between the
        last stable and first unstable grid points to refine the stability
        boundary estimate.  Grid points beyond the boundary remain NaN.
        Returns ``(results, boundaries)`` instead of just ``results``.
        Default False.
    boundary_tol : float, optional
        Absolute tolerance on the input value for bisection stopping criterion.
        Defaults to 1/10 of the grid step size, giving roughly 4 bisection
        iterations per boundary.  Only used when ``find_boundary=True``.

    Returns
    -------
    results : dict
        ``{input_name: {output_name: np.ndarray}}`` of steady-state values.
    boundaries : dict
        Only returned when ``find_boundary=True``.
        ``{input_name: {"upper": float or None, "lower": float or None}}``
        where each value is the last stable input value found by bisection, or
        None if no boundary was encountered in that direction.
    """
    if param_vals is None:
        param_vals = {}
    u_nom = np.asarray(u_nom, dtype=float)

    if all_input_names is not None:
        input_u_idx = {
            name: all_input_names.index(name) for name in input_names
        }
    else:
        input_u_idx = {name: j for j, name in enumerate(input_names)}

    print("\nComputing steady-state sweeps...")
    results = {}
    boundaries = {}
    for input_name in input_names:
        j = input_u_idx[input_name]
        input_vals = np.asarray(input_sweeps[input_name])
        n_pts = len(input_vals)
        output_vals = {out: np.full(n_pts, np.nan) for out in output_names}
        nom_idx = int(np.searchsorted(input_vals, u_nom[j]))
        upper_boundary = None
        lower_boundary = None

        # Default tolerance: 1/10 of the grid step size (~4 bisection iterations)
        if find_boundary and boundary_tol is None:
            step = (input_vals[-1] - input_vals[0]) / (n_pts - 1)
            tol = step / 10.0
        else:
            tol = boundary_tol

        x0 = np.array(x0_nom, dtype=float)
        for k in range(nom_idx, n_pts):
            u = u_nom.copy()
            u[j] = input_vals[k]
            try:
                x_ss, y_ss = ss_solver(x0, u, param_vals)
            except RuntimeError:
                if find_boundary and k > nom_idx:
                    upper_boundary = _bisect_boundary(
                        ss_solver,
                        x0,
                        u_nom,
                        j,
                        stable_val=input_vals[k - 1],
                        unstable_val=input_vals[k],
                        param_vals=param_vals,
                        tol=tol,
                    )
                break
            for out in output_names:
                output_vals[out][k] = y_ss[output_idx[out]]
            x0 = x_ss

        x0 = np.array(x0_nom, dtype=float)
        for k in range(nom_idx - 1, -1, -1):
            u = u_nom.copy()
            u[j] = input_vals[k]
            try:
                x_ss, y_ss = ss_solver(x0, u, param_vals)
            except RuntimeError:
                if find_boundary and k < nom_idx - 1:
                    lower_boundary = _bisect_boundary(
                        ss_solver,
                        x0,
                        u_nom,
                        j,
                        stable_val=input_vals[k + 1],
                        unstable_val=input_vals[k],
                        param_vals=param_vals,
                        tol=tol,
                    )
                break
            for out in output_names:
                output_vals[out][k] = y_ss[output_idx[out]]
            x0 = x_ss

        results[input_name] = output_vals
        boundaries[input_name] = {
            "upper": upper_boundary,
            "lower": lower_boundary,
        }
        print(f"  {input_name}: {n_pts} points")

    if find_boundary:
        return results, boundaries
    return results


def _axis_label(name, info):
    """Multi-line axis label: name, symbol, and units."""
    if info and name in info:
        d = info[name]
        label = d.get("label", name)
        symbol = d.get("symbol", "")
        units = d.get("units", "")
        parts = [label]
        if symbol:
            parts.append(f"({symbol})")
        if units:
            parts.append(f"[{units}]")
        return "\n".join(parts)
    return name


def _abstract_label(name, info):
    """Short axis label: name and symbol only, no units."""
    if info and name in info:
        d = info[name]
        label = d.get("label", name)
        symbol = d.get("symbol", "")
        return f"{label}\n({symbol})" if symbol else label
    return name


def plot_main(
    results,
    input_names,
    output_names,
    input_sweeps,
    u_nom_dict,
    y_nom_dict,
    input_info=None,
    output_info=None,
    boundaries=None,
    title=None,
    save_path=None,
):
    """Steady-state I/O matrix plot with full axis labels.

    Produces an n_outputs × n_inputs grid where each cell shows the steady-state
    value of one output as a function of one input (all others held at nominal).
    Dashed crosshairs mark the nominal operating point.

    Parameters
    ----------
    results : dict
        Output of :func:`compute_ss_sweeps`.
    input_names : list of str
        Input names — one column per input.
    output_names : list of str
        Output names — one row per output.
    input_sweeps : dict
        ``{input_name: array_of_values}`` — the sweep ranges.
    u_nom_dict : dict
        ``{input_name: nominal_value}`` for each input.
    y_nom_dict : dict
        ``{output_name: nominal_value}`` for each output.
    input_info : dict, optional
        ``{input_name: {"label", "symbol", "units"}}`` for x-axis labels.
    output_info : dict, optional
        ``{output_name: {"label", "symbol", "units"}}`` for y-axis labels.
    boundaries : dict, optional
        ``{input_name: {"upper": float or None, "lower": float or None}}`` as
        returned by :func:`compute_ss_sweeps` with ``find_boundary=True``.
        Where present, vertical red dashed lines mark the stability boundary.
    title : str, optional
        Figure suptitle.  Defaults to ``'Steady-state I/O characteristics'``.
    save_path : str or Path, optional
        If provided, saves the figure at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_outputs = len(output_names)
    n_inputs = len(input_names)

    fig, axs = plt.subplots(
        n_outputs,
        n_inputs,
        figsize=(1.25 + 2 * n_inputs, 1 + 1.5 * n_outputs),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    if n_outputs == 1:
        axs = axs[np.newaxis, :]
    if n_inputs == 1:
        axs = axs[:, np.newaxis]

    for col, input_name in enumerate(input_names):
        input_vals = np.asarray(input_sweeps[input_name])
        nom_input = u_nom_dict[input_name]
        input_bounds = (boundaries or {}).get(input_name, {})

        for row, output_name in enumerate(output_names):
            ax = axs[row, col]
            ax.plot(input_vals, results[input_name][output_name], color="C0")
            ax.axvline(
                nom_input, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axhline(
                y_nom_dict[output_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.plot(nom_input, y_nom_dict[output_name], "ko", markersize=4)
            for bound in (
                input_bounds.get("upper"),
                input_bounds.get("lower"),
            ):
                if bound is not None:
                    ax.axvline(
                        bound,
                        color="r",
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.7,
                    )
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(
                    _axis_label(output_name, output_info), fontsize=8
                )
            if row == n_outputs - 1:
                ax.set_xlabel(_axis_label(input_name, input_info), fontsize=7)

    fig.suptitle(title or "Steady-state I/O characteristics", fontsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    return fig


def plot_abstract(
    results,
    input_names,
    output_names,
    input_sweeps,
    u_nom_dict,
    y_nom_dict,
    input_info=None,
    output_info=None,
    boundaries=None,
    title=None,
    save_path=None,
    subplot_size=(1.5, 1.5),
):
    """Compact steady-state I/O matrix plot without tick marks.

    Same layout as :func:`plot_main` but with tick marks and numeric labels
    suppressed — suitable for qualitative input-output analysis and publication.

    Parameters
    ----------
    results : dict
        Output of :func:`compute_ss_sweeps`.
    input_names : list of str
        Input names — one column per input.
    output_names : list of str
        Output names — one row per output.
    input_sweeps : dict
        ``{input_name: array_of_values}`` — the sweep ranges.
    u_nom_dict : dict
        ``{input_name: nominal_value}`` for each input.
    y_nom_dict : dict
        ``{output_name: nominal_value}`` for each output.
    input_info : dict, optional
        ``{input_name: {"label", "symbol", "units"}}`` for axis labels.
    output_info : dict, optional
        ``{output_name: {"label", "symbol", "units"}}`` for axis labels.
    boundaries : dict, optional
        ``{input_name: {"upper": float or None, "lower": float or None}}`` as
        returned by :func:`compute_ss_sweeps` with ``find_boundary=True``.
        Where present, vertical red dashed lines mark the stability boundary.
    title : str, optional
        Figure suptitle.
    save_path : str or Path, optional
        If provided, saves the figure at 150 dpi.
    subplot_size : tuple of float, optional
        ``(width, height)`` in inches per subplot cell.  Default ``(1.75, 1.75)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_outputs = len(output_names)
    n_inputs = len(input_names)

    fig, axs = plt.subplots(
        n_outputs,
        n_inputs,
        figsize=(subplot_size[0] * n_inputs, subplot_size[1] * n_outputs),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    if n_outputs == 1:
        axs = axs[np.newaxis, :]
    if n_inputs == 1:
        axs = axs[:, np.newaxis]

    for col, input_name in enumerate(input_names):
        input_vals = np.asarray(input_sweeps[input_name])
        nom_input = u_nom_dict[input_name]
        input_bounds = (boundaries or {}).get(input_name, {})

        for row, output_name in enumerate(output_names):
            ax = axs[row, col]
            ax.plot(input_vals, results[input_name][output_name], color="C0")
            ax.axvline(
                nom_input, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axhline(
                y_nom_dict[output_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.plot(nom_input, y_nom_dict[output_name], "ko", markersize=3)
            for bound in (
                input_bounds.get("upper"),
                input_bounds.get("lower"),
            ):
                if bound is not None:
                    ax.axvline(
                        bound,
                        color="r",
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.7,
                    )
            ax.margins(0.12)
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            ax.grid(False)

            if col == 0:
                ax.set_ylabel(_abstract_label(output_name, output_info))
            if row == n_outputs - 1:
                ax.set_xlabel(_abstract_label(input_name, input_info))

    fig.tight_layout(h_pad=0, w_pad=0, rect=[0, 0, 1, 0.94])
    fig.suptitle(title or "Steady-state I/O characteristics")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    return fig


if __name__ == "__main__":
    import argparse

    from cas_models.continuous_time.simulate import make_steady_state_solver
    from model import (
        INPUTS_NOP,
        STATES_NOP,
        build_grinding_circuit_model_with_level_control,
        build_grinding_circuit_model_with_sump_control,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="c1_grind_sc",
        choices=["c1_grind_sc", "c1_grind_sc_lc"],
    )
    args = parser.parse_args()

    PLOT_DIR = Path("plots")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Variable metadata for axis labels ────────────────────────────────
    INPUT_INFO = {
        "feed_ore_rate": {
            "label": "Feed ore rate",
            "symbol": r"$u_{MFO}$",
            "units": "t/h",
        },
        "water_ore_ratio": {
            "label": "Water/ore ratio",
            "symbol": r"$u_{rMIW}$",
            "units": "-",
        },
        "critical_speed_fraction": {
            "label": "Critical speed frac.",
            "symbol": r"$u_{\phi_c}$",
            "units": "-",
        },
        "sump_feed_water": {
            "label": "Sump feed water",
            "symbol": r"$u_{SFW}$",
            "units": "m³/h",
        },
    }
    OUTPUT_INFO = {
        "charge_fill_fraction": {
            "label": "Charge fill frac.",
            "symbol": r"$y_{JT}$",
            "units": "-",
        },
        "mill_power": {
            "label": "Mill power",
            "symbol": r"$y_{P}$",
            "units": "MW",
        },
        "sump_level": {
            "label": "Sump level",
            "symbol": r"$y_{SLEV}$",
            "units": "%",
        },
        "sump_density": {
            "label": "Sump density",
            "symbol": r"$y_{\rho}$",
            "units": "t/m³",
        },
        "product_size": {
            "label": "Product size (PSE)",
            "symbol": r"$y_{PSE}$",
            "units": "%",
        },
        "cyclone_feed_flow": {
            "label": "Cyclone feed flow",
            "symbol": r"$u_{CFF}$",
            "units": "m³/h",
        },
        "feed_ore_rate": {
            "label": "Feed ore rate",
            "symbol": r"$u_{MFO}$",
            "units": "t/h",
        },
    }

    # ── Build model and steady-state solver ──────────────────────────────
    MODEL_BUILDERS = {
        "c1_grind_sc": build_grinding_circuit_model_with_sump_control,
        "c1_grind_sc_lc": build_grinding_circuit_model_with_level_control,
    }
    print("Building model and compiling steady-state solver...")
    model = MODEL_BUILDERS[args.model]()
    ss_solver = make_steady_state_solver(model)
    print(f"  Model: {model.name}, n={model.n}, nu={model.nu}, ny={model.ny}")

    INPUT_NAMES = model.input_names
    OUTPUT_NAMES = model.output_names

    # ── Compute actual nominal steady state from paper's inputs ──────────
    u_nom = np.array([INPUTS_NOP[n] for n in INPUT_NAMES])
    x0_guess = np.array([STATES_NOP[n] for n in model.state_names])
    param_vals = {}  # all parameters are concrete numerics in the default model

    x_ss_nom, y_ss_nom = ss_solver(x0_guess, u_nom, param_vals)
    u_nom_dict = dict(zip(INPUT_NAMES, u_nom))
    y_nom_dict = dict(zip(OUTPUT_NAMES, y_ss_nom))

    print("\nNominal steady state:")
    for name, val in zip(OUTPUT_NAMES, y_ss_nom):
        print(f"  {name:<25} {val:.4g}")

    # ── Input sweep ranges: ±20% of nominal, 31 points each ─────────────
    SWEEP_FRACTION = 0.20
    N_SWEEP = 31
    INPUT_SWEEPS = {
        name: np.linspace(
            u_nom_dict[name] * (1.0 - SWEEP_FRACTION),
            u_nom_dict[name] * (1.0 + SWEEP_FRACTION),
            N_SWEEP,
        )
        for name in INPUT_NAMES
    }

    output_idx = {name: i for i, name in enumerate(OUTPUT_NAMES)}

    # ── Compute steady-state sweeps ───────────────────────────────────────
    results, boundaries = compute_ss_sweeps(
        ss_solver,
        x_ss_nom,
        u_nom,
        INPUT_NAMES,
        OUTPUT_NAMES,
        output_idx,
        INPUT_SWEEPS,
        all_input_names=INPUT_NAMES,
        param_vals=param_vals,
        find_boundary=True,
    )

    print("\nStability boundaries:")
    for input_name, bounds in boundaries.items():
        lo, hi = bounds["lower"], bounds["upper"]
        parts = []
        if lo is not None:
            parts.append(f"lower={lo:.4g}")
        if hi is not None:
            parts.append(f"upper={hi:.4g}")
        print(
            f"  {input_name:<25} {', '.join(parts) if parts else 'none found'}"
        )

    # ── Plots ─────────────────────────────────────────────────────────────
    PLOT_TITLE = (
        f"Grinding circuit – Steady-state I/O characteristics\n({model.name})"
    )

    plot_main(
        results,
        INPUT_NAMES,
        OUTPUT_NAMES,
        INPUT_SWEEPS,
        u_nom_dict,
        y_nom_dict,
        input_info=INPUT_INFO,
        output_info=OUTPUT_INFO,
        boundaries=boundaries,
        title=PLOT_TITLE,
        save_path=PLOT_DIR / f"ss_io_resp_{model.name}.png",
    )
    print(f"\nSaved {PLOT_DIR}/ss_io_resp_{model.name}.png")

    plot_abstract(
        results,
        INPUT_NAMES,
        OUTPUT_NAMES,
        INPUT_SWEEPS,
        u_nom_dict,
        y_nom_dict,
        input_info=INPUT_INFO,
        output_info=OUTPUT_INFO,
        boundaries=boundaries,
        title=PLOT_TITLE,
        save_path=PLOT_DIR / f"ss_io_abstract_{model.name}.png",
    )
    print(f"Saved {PLOT_DIR}/ss_io_abstract_{model.name}.png")

    plt.show()
