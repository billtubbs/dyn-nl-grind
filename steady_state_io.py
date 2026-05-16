"""Steady-state input-output analysis for casadi-models.

Generic utilities for computing and plotting steady-state I/O characteristics
of any continuous-time model built with casadi-models.

Functions
---------
compute_ss_sweeps
    Sweep each input over a range, collecting steady-state outputs via
    warm-start Newton continuation from the nominal operating point.
plot_main
    Full n_cvs × n_mvs grid of steady-state gain curves with axes.
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
    """Bisect to find the MV stability boundary between stable_val and unstable_val.

    Works in either direction (stable_val may be above or below unstable_val).
    Returns the last stable MV value found — a conservative estimate of the
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
    mv_names,
    cv_names,
    output_idx,
    mv_sweeps,
    input_names=None,
    param_vals=None,
    find_boundary=False,
    boundary_tol=None,
):
    """Compute steady-state outputs over independent input sweeps.

    For each MV in mv_names, holds all other inputs at nominal and sweeps
    that input over mv_sweeps[mv_name], collecting steady-state CV values.
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
    mv_names : list of str
        Names of inputs to sweep (manipulated variables).
    cv_names : list of str
        Names of outputs to collect (controlled variables).
    output_idx : dict
        Mapping ``{output_name: index}`` in the y_ss vector.
    mv_sweeps : dict
        ``{mv_name: sorted_array_of_values}`` for each MV; arrays should
        include or straddle the nominal value so warm-starting works.
    input_names : list of str, optional
        Ordered list of all model input names matching u_nom.  Used to
        locate each MV's position in u_nom.  If None, mv_names is assumed
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
        Absolute tolerance on the MV value for bisection stopping criterion.
        Defaults to 1/10 of the grid step size, giving roughly 4 bisection
        iterations per boundary.  Only used when ``find_boundary=True``.

    Returns
    -------
    results : dict
        ``{mv_name: {cv_name: np.ndarray}}`` of steady-state values.
    boundaries : dict
        Only returned when ``find_boundary=True``.
        ``{mv_name: {"upper": float or None, "lower": float or None}}``
        where each value is the last stable MV value found by bisection, or
        None if no boundary was encountered in that direction.
    """
    if param_vals is None:
        param_vals = {}
    u_nom = np.asarray(u_nom, dtype=float)

    if input_names is not None:
        mv_u_idx = {name: input_names.index(name) for name in mv_names}
    else:
        mv_u_idx = {name: j for j, name in enumerate(mv_names)}

    print("\nComputing steady-state sweeps...")
    results = {}
    boundaries = {}
    for mv_name in mv_names:
        j = mv_u_idx[mv_name]
        mv_vals = np.asarray(mv_sweeps[mv_name])
        n_pts = len(mv_vals)
        cv_vals = {cv: np.full(n_pts, np.nan) for cv in cv_names}
        nom_idx = int(np.searchsorted(mv_vals, u_nom[j]))
        upper_boundary = None
        lower_boundary = None

        # Default tolerance: 1/10 of the grid step size (~4 bisection iterations)
        if find_boundary and boundary_tol is None:
            step = (mv_vals[-1] - mv_vals[0]) / (n_pts - 1)
            tol = step / 10.0
        else:
            tol = boundary_tol

        x0 = np.array(x0_nom, dtype=float)
        for k in range(nom_idx, n_pts):
            u = u_nom.copy()
            u[j] = mv_vals[k]
            try:
                x_ss, y_ss = ss_solver(x0, u, param_vals)
            except RuntimeError:
                if find_boundary and k > nom_idx:
                    upper_boundary = _bisect_boundary(
                        ss_solver, x0, u_nom, j,
                        stable_val=mv_vals[k - 1], unstable_val=mv_vals[k],
                        param_vals=param_vals, tol=tol,
                    )
                break
            for cv in cv_names:
                cv_vals[cv][k] = y_ss[output_idx[cv]]
            x0 = x_ss

        x0 = np.array(x0_nom, dtype=float)
        for k in range(nom_idx - 1, -1, -1):
            u = u_nom.copy()
            u[j] = mv_vals[k]
            try:
                x_ss, y_ss = ss_solver(x0, u, param_vals)
            except RuntimeError:
                if find_boundary and k < nom_idx - 1:
                    lower_boundary = _bisect_boundary(
                        ss_solver, x0, u_nom, j,
                        stable_val=mv_vals[k + 1], unstable_val=mv_vals[k],
                        param_vals=param_vals, tol=tol,
                    )
                break
            for cv in cv_names:
                cv_vals[cv][k] = y_ss[output_idx[cv]]
            x0 = x_ss

        results[mv_name] = cv_vals
        boundaries[mv_name] = {"upper": upper_boundary, "lower": lower_boundary}
        print(f"  {mv_name}: {n_pts} points")

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
    mv_names,
    cv_names,
    mv_sweeps,
    u_nom_dict,
    y_nom_dict,
    input_info=None,
    output_info=None,
    boundaries=None,
    title=None,
    save_path=None,
):
    """Steady-state I/O matrix plot with full axis labels.

    Produces an n_cvs × n_mvs grid where each cell shows the steady-state
    value of one CV as a function of one MV (all others held at nominal).
    Dashed crosshairs mark the nominal operating point.

    Parameters
    ----------
    results : dict
        Output of :func:`compute_ss_sweeps`.
    mv_names : list of str
        Input names — one column per MV.
    cv_names : list of str
        Output names — one row per CV.
    mv_sweeps : dict
        ``{mv_name: array_of_values}`` — the sweep ranges.
    u_nom_dict : dict
        ``{mv_name: nominal_value}`` for each MV.
    y_nom_dict : dict
        ``{cv_name: nominal_value}`` for each CV.
    input_info : dict, optional
        ``{mv_name: {"label", "symbol", "units"}}`` for x-axis labels.
    output_info : dict, optional
        ``{cv_name: {"label", "symbol", "units"}}`` for y-axis labels.
    boundaries : dict, optional
        ``{mv_name: {"upper": float or None, "lower": float or None}}`` as
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
    n_cvs = len(cv_names)
    n_mvs = len(mv_names)

    fig, axs = plt.subplots(
        n_cvs,
        n_mvs,
        figsize=(3.0 * n_mvs, 2.5 * n_cvs),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    if n_cvs == 1:
        axs = axs[np.newaxis, :]
    if n_mvs == 1:
        axs = axs[:, np.newaxis]

    for col, mv_name in enumerate(mv_names):
        mv_vals = np.asarray(mv_sweeps[mv_name])
        nom_mv = u_nom_dict[mv_name]
        mv_bounds = (boundaries or {}).get(mv_name, {})

        for row, cv_name in enumerate(cv_names):
            ax = axs[row, col]
            ax.plot(mv_vals, results[mv_name][cv_name], color="C0")
            ax.axvline(nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.axhline(
                y_nom_dict[cv_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.plot(nom_mv, y_nom_dict[cv_name], "ko", markersize=4)
            for bound in (mv_bounds.get("upper"), mv_bounds.get("lower")):
                if bound is not None:
                    ax.axvline(bound, color="r", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(_axis_label(cv_name, output_info), fontsize=8)
            if row == n_cvs - 1:
                ax.set_xlabel(_axis_label(mv_name, input_info), fontsize=7)

    fig.suptitle(title or "Steady-state I/O characteristics", fontsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    return fig


def plot_abstract(
    results,
    mv_names,
    cv_names,
    mv_sweeps,
    u_nom_dict,
    y_nom_dict,
    input_info=None,
    output_info=None,
    boundaries=None,
    title=None,
    save_path=None,
    subplot_size=(1.75, 1.75),
):
    """Compact steady-state I/O matrix plot without tick marks.

    Same layout as :func:`plot_main` but with tick marks and numeric labels
    suppressed — suitable for qualitative input-output analysis and publication.

    Parameters
    ----------
    results : dict
        Output of :func:`compute_ss_sweeps`.
    mv_names : list of str
        Input names — one column per MV.
    cv_names : list of str
        Output names — one row per CV.
    mv_sweeps : dict
        ``{mv_name: array_of_values}`` — the sweep ranges.
    u_nom_dict : dict
        ``{mv_name: nominal_value}`` for each MV.
    y_nom_dict : dict
        ``{cv_name: nominal_value}`` for each CV.
    input_info : dict, optional
        ``{mv_name: {"label", "symbol", "units"}}`` for axis labels.
    output_info : dict, optional
        ``{cv_name: {"label", "symbol", "units"}}`` for axis labels.
    boundaries : dict, optional
        ``{mv_name: {"upper": float or None, "lower": float or None}}`` as
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
    n_cvs = len(cv_names)
    n_mvs = len(mv_names)

    fig, axs = plt.subplots(
        n_cvs,
        n_mvs,
        figsize=(subplot_size[0] * n_mvs, subplot_size[1] * n_cvs),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    if n_cvs == 1:
        axs = axs[np.newaxis, :]
    if n_mvs == 1:
        axs = axs[:, np.newaxis]

    for col, mv_name in enumerate(mv_names):
        mv_vals = np.asarray(mv_sweeps[mv_name])
        nom_mv = u_nom_dict[mv_name]
        mv_bounds = (boundaries or {}).get(mv_name, {})

        for row, cv_name in enumerate(cv_names):
            ax = axs[row, col]
            ax.plot(mv_vals, results[mv_name][cv_name], color="C0")
            ax.axvline(nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.axhline(
                y_nom_dict[cv_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.plot(nom_mv, y_nom_dict[cv_name], "ko", markersize=3)
            for bound in (mv_bounds.get("upper"), mv_bounds.get("lower")):
                if bound is not None:
                    ax.axvline(bound, color="r", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.margins(0.12)
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            ax.grid(False)

            if col == 0:
                ax.set_ylabel(_abstract_label(cv_name, output_info))
            if row == n_cvs - 1:
                ax.set_xlabel(_abstract_label(mv_name, input_info))

    fig.tight_layout(h_pad=0, w_pad=0, rect=[0, 0, 1, 0.94])
    fig.suptitle(title or "Steady-state I/O characteristics")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    return fig


if __name__ == "__main__":
    from cas_models.continuous_time.simulate import make_steady_state_solver
    from model import (
        STEADY_STATE_INPUTS,
        STEADY_STATE_STATES,
        build_grinding_circuit_model_with_sump_control,
    )

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
    }

    # ── Build model and steady-state solver ──────────────────────────────
    print("Building model and compiling steady-state solver...")
    model = build_grinding_circuit_model_with_sump_control()
    ss_solver = make_steady_state_solver(model)
    print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")

    MV_NAMES = model.input_names   # 4 free inputs (cyclone_feed_flow is internal)
    CV_NAMES = model.output_names  # 6 outputs (5 process + cyclone_feed_flow)

    # ── Compute actual nominal steady state from paper's inputs ──────────
    u_nom = np.array([STEADY_STATE_INPUTS[n] for n in MV_NAMES])
    x0_guess = np.array([STEADY_STATE_STATES[n] for n in model.state_names])
    param_vals = {}  # all parameters are concrete numerics in the default model

    x_ss_nom, y_ss_nom = ss_solver(x0_guess, u_nom, param_vals)
    u_nom_dict = dict(zip(MV_NAMES, u_nom))
    y_nom_dict = dict(zip(CV_NAMES, y_ss_nom))

    print("\nNominal steady state:")
    for name, val in zip(CV_NAMES, y_ss_nom):
        print(f"  {name:<25} {val:.4g}")

    # ── Input sweep ranges: ±10 % of nominal, 31 points each ─────────────
    SWEEP_FRACTION = 0.20
    N_SWEEP = 31
    MV_SWEEPS = {
        name: np.linspace(
            u_nom_dict[name] * (1.0 - SWEEP_FRACTION),
            u_nom_dict[name] * (1.0 + SWEEP_FRACTION),
            N_SWEEP,
        )
        for name in MV_NAMES
    }

    output_idx = {name: i for i, name in enumerate(CV_NAMES)}

    # ── Compute steady-state sweeps ───────────────────────────────────────
    results, boundaries = compute_ss_sweeps(
        ss_solver,
        x_ss_nom,
        u_nom,
        MV_NAMES,
        CV_NAMES,
        output_idx,
        MV_SWEEPS,
        input_names=MV_NAMES,
        param_vals=param_vals,
        find_boundary=True,
    )

    print("\nStability boundaries:")
    for mv_name, bounds in boundaries.items():
        lo, hi = bounds["lower"], bounds["upper"]
        parts = []
        if lo is not None:
            parts.append(f"lower={lo:.4g}")
        if hi is not None:
            parts.append(f"upper={hi:.4g}")
        print(f"  {mv_name:<25} {', '.join(parts) if parts else 'none found'}")

    # ── Plots ─────────────────────────────────────────────────────────────
    PLOT_TITLE = "Grinding circuit – Steady-state I/O characteristics"

    plot_main(
        results,
        MV_NAMES,
        CV_NAMES,
        MV_SWEEPS,
        u_nom_dict,
        y_nom_dict,
        input_info=INPUT_INFO,
        output_info=OUTPUT_INFO,
        boundaries=boundaries,
        title=PLOT_TITLE,
        save_path=PLOT_DIR / "ss_io_main.png",
    )
    print(f"\nSaved {PLOT_DIR}/ss_io_main.png")

    plot_abstract(
        results,
        MV_NAMES,
        CV_NAMES,
        MV_SWEEPS,
        u_nom_dict,
        y_nom_dict,
        input_info=INPUT_INFO,
        output_info=OUTPUT_INFO,
        boundaries=boundaries,
        title=PLOT_TITLE,
        save_path=PLOT_DIR / "ss_io_abstract.png",
    )
    print(f"Saved {PLOT_DIR}/ss_io_abstract.png")

    plt.show()
