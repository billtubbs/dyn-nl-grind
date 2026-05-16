"""Step response input-output analysis for casadi-models.

Generic utilities for computing and plotting multi-level step response I/O
characteristics of any continuous-time model built with casadi-models.

For each input, several step sizes are applied from the same steady-state
initial condition and the resulting output deviations are plotted as separate
coloured lines on a shared matrix grid.

Functions
---------
compute_step_responses
    Run one simulation per (input, step size) from the computed SS, returning
    output deviation arrays.
plot_main
    Full n_outputs × n_inputs grid of step response time series with axes and legend.
plot_abstract
    Compact version without tick marks, for qualitative I/O analysis.

Run as a script to generate step response plots for the grinding circuit model
defined in src/model.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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


def _step_colors(step_fractions, cmap=None):
    """Map step fractions to colours using a diverging colormap.

    Fractions are normalised symmetrically so the midpoint (0) maps to the
    centre of the colormap.
    """
    if cmap is None:
        cmap = plt.cm.coolwarm
    max_abs = max(abs(f) for f in step_fractions)
    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    return [cmap(norm(f)) for f in step_fractions]


def compute_step_responses(
    sim_func,
    t_eval,
    x_ss,
    y_ss,
    u_nom,
    step_sizes,
    output_names,
    step_time_idx,
    param_vals=None,
    all_output_names=None,
):
    """Compute multi-level step response deviations for each input.

    For each input, the provided step magnitudes are applied one at a time from
    the same steady-state initial condition and the output deviations are recorded.

    Parameters
    ----------
    sim_func : callable
        Multi-step simulation function, signature
        ``(t_eval, U, x0, *params) -> (X, Y)``.
        From ``cas_models.continuous_time.simulate
        .make_n_step_simulation_function_from_model``.
    t_eval : array-like, shape (n_steps+1,)
        Time vector.
    x_ss : array-like, shape (n,)
        Steady-state state vector used as initial condition for all runs.
    y_ss : array-like, shape (ny,)
        Steady-state output vector; output deviations are computed relative to this.
    u_nom : array-like, shape (nu,)
        Nominal input vector; the baseline around which steps are applied.
        Must be ordered consistently with ``step_sizes``.
    step_sizes : dict
        ``{input_name: [Δu_1, Δu_2, ...]}`` mapping each input to an ordered
        list of absolute step magnitudes (same length for all inputs).  The dict
        key order determines which column of ``u_nom`` each input maps to.
    output_names : list of str
        Output names to collect in the results.
    step_time_idx : int
        Time-step index at which the step is applied.
    param_vals : dict, optional
        Parameter values forwarded to sim_func.  Defaults to ``{}``.
    all_output_names : list of str, optional
        Ordered list of all model output names (matching Y columns).  Used to
        locate output_names in Y.  If None, output_names is assumed to match
        in order.

    Returns
    -------
    dict
        ``{input_name: {"step_fractions": ndarray, "step_sizes": ndarray,
        output_name: ndarray shape (n_steps+1, n_fracs)}}``
        where each output entry contains deviations from y_ss, and
        ``step_fractions`` is back-computed as ``step_sizes / u_nom[j]``.
    """
    if param_vals is None:
        param_vals = {}
    u_nom = np.asarray(u_nom, dtype=float)
    x_ss = np.asarray(x_ss, dtype=float)
    y_ss = np.asarray(y_ss, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    n_steps = len(t_eval) - 1
    input_names = list(step_sizes.keys())
    n_fracs = len(next(iter(step_sizes.values())))

    if all_output_names is None:
        output_idx = {name: i for i, name in enumerate(output_names)}
    else:
        output_idx = {name: i for i, name in enumerate(all_output_names)}

    print("\nComputing step responses...")
    step_data = {}
    for j, input_name in enumerate(input_names):
        step_sizes_arr = np.asarray(step_sizes[input_name], dtype=float)
        step_fracs = step_sizes_arr / u_nom[j]
        output_arrays = {
            out: np.full((n_steps + 1, n_fracs), np.nan) for out in output_names
        }

        print(f"  {input_name}:")
        for k, (frac, step_size) in enumerate(zip(step_fracs, step_sizes_arr)):
            U = np.tile(u_nom, (n_steps, 1))
            U[step_time_idx:, j] += step_size
            try:
                _, y_cas = sim_func(t_eval, U, x_ss, *param_vals.values())
                Y = np.array(y_cas)
                for output_name in output_names:
                    output_arrays[output_name][:, k] = (
                        Y[:, output_idx[output_name]]
                        - y_ss[output_idx[output_name]]
                    )
                peak = max(
                    np.nanmax(np.abs(output_arrays[out][:, k]))
                    for out in output_names
                )
                print(
                    f"    {frac:+.0%}  (Δu={step_size:+.3g})  max|Δy|={peak:.4g}"
                )
            except RuntimeError:
                print(
                    f"    {frac:+.0%}  (Δu={step_size:+.3g})  simulation failed"
                )

        step_data[input_name] = {
            "step_fractions": step_fracs,
            "step_sizes": step_sizes_arr,
            **output_arrays,
        }

    return step_data


def _compute_ylim(data_arrays, clip_quantile):
    """Return symmetric y-axis half-width from finite values across arrays.

    Returns the ``clip_quantile`` percentile of absolute finite values, or
    ``None`` if there are no finite values or the result is zero.
    """
    if not data_arrays:
        return None
    finite_vals = np.concatenate(
        [a.ravel() for a in data_arrays if a is not None]
    )
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if len(finite_vals) == 0:
        return None
    limit = np.percentile(np.abs(finite_vals), clip_quantile * 100)
    return float(limit) if limit > 0 else None


def plot_main(
    step_data,
    input_names,
    output_names,
    t_eval,
    step_time_idx,
    input_info=None,
    output_info=None,
    cmap=None,
    clip_quantile=0.99,
    ylim_from=None,
    title=None,
    save_path=None,
):
    """Step response matrix plot with full axis labels.

    Each column corresponds to one input; each row to one output.  Multiple
    lines per cell show the response to different step magnitudes, coloured
    from blue (most negative) to red (most positive) via a diverging colormap.

    Parameters
    ----------
    step_data : dict
        Output of :func:`compute_step_responses`.
    input_names : list of str
        Input names — one column per input.
    output_names : list of str
        Output names — one row per output.
    t_eval : array-like
        Time vector.
    step_time_idx : int
        Index into t_eval where the step occurs.
    input_info : dict, optional
        ``{input_name: {"label", "symbol", "units"}}`` for column titles.
    output_info : dict, optional
        ``{output_name: {"label", "symbol", "units"}}`` for y-axis labels.
    cmap : matplotlib colormap, optional
        Diverging colormap for step sizes.  Defaults to ``plt.cm.coolwarm``.
    clip_quantile : float, optional
        Percentile (0–1) of finite absolute values used to set y-axis limits,
        preventing unstable runs from collapsing the scale.  Default 0.99.
    ylim_from : dict, optional
        ``{output_name: [input_name, ...]}`` mapping.  For each output row,
        only the listed input columns' data is used to compute the shared
        y-axis range.  If an output is not in the dict (or ``ylim_from`` is
        ``None``), all input columns contribute.
    title : str, optional
        Figure suptitle.
    save_path : str or Path, optional
        If provided, saves the figure at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cmap is None:
        cmap = plt.cm.coolwarm
    n_outputs = len(output_names)
    n_inputs = len(input_names)
    t = np.asarray(t_eval)

    # Use fractions from first input (same for all)
    step_fractions = step_data[input_names[0]]["step_fractions"]
    colors = _step_colors(step_fractions, cmap)

    fig, axs = plt.subplots(
        n_outputs,
        n_inputs,
        figsize=(1.25 + 2.0 * n_inputs, 1 + 1.5 * n_outputs),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    if n_outputs == 1:
        axs = axs[np.newaxis, :]
    if n_inputs == 1:
        axs = axs[:, np.newaxis]

    for col, input_name in enumerate(input_names):
        input_entry = step_data.get(input_name, {})

        for row, output_name in enumerate(output_names):
            ax = axs[row, col]
            output_matrix = input_entry.get(output_name)  # (n_steps+1, n_fracs)

            if output_matrix is not None:
                for k, color in enumerate(colors):
                    label = f"{step_fractions[k]:+.0%}"
                    ax.plot(t, output_matrix[:, k], color=color, label=label)

            ax.axhline(
                0.0, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axvline(
                t[step_time_idx],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(_axis_label(output_name, output_info), fontsize=8)
            if row == 0:
                ax.set_title(_axis_label(input_name, input_info), fontsize=8)
            if row == n_outputs - 1:
                ax.set_xlabel("Time (h)", fontsize=8)

    # Set y-axis limits per output row from governing input columns only.
    # Setting on axs[row, 0] propagates to all shared axes in the row.
    if clip_quantile is not None:
        ylim_from_ = ylim_from or {}
        for row, output_name in enumerate(output_names):
            governing = ylim_from_.get(output_name, input_names)
            arrays = [step_data.get(inp, {}).get(output_name) for inp in governing]
            limit = _compute_ylim(
                [a for a in arrays if a is not None], clip_quantile
            )
            if limit is not None:
                axs[row, 0].set_ylim(-limit * 1.05, limit * 1.05)

    # Legend from the first cell
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=7,
        framealpha=0.8,
        title="step",
        title_fontsize=7,
    )
    fig.suptitle(title or "Step response I/O characteristics", fontsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    return fig


def plot_abstract(
    step_data,
    input_names,
    output_names,
    t_eval,
    step_time_idx,
    input_info=None,
    output_info=None,
    cmap=None,
    clip_quantile=0.99,
    ylim_from=None,
    title=None,
    save_path=None,
    subplot_size=(1.5, 1.5),
):
    """Compact step response matrix plot without tick marks.

    Same layout as :func:`plot_main` but with tick marks and numeric labels
    suppressed — suitable for qualitative input-output analysis and publication.
    The input identity appears as the x-axis label on the bottom row.

    Parameters
    ----------
    step_data : dict
        Output of :func:`compute_step_responses`.
    input_names : list of str
        Input names — one column per input.
    output_names : list of str
        Output names — one row per output.
    t_eval : array-like
        Time vector.
    step_time_idx : int
        Index into t_eval where the step occurs.
    input_info : dict, optional
        ``{input_name: {"label", "symbol", "units"}}`` for x-axis labels.
    output_info : dict, optional
        ``{output_name: {"label", "symbol", "units"}}`` for y-axis labels.
    cmap : matplotlib colormap, optional
        Diverging colormap.  Defaults to ``plt.cm.coolwarm``.
    clip_quantile : float, optional
        Percentile (0–1) for y-axis clipping.  Default 0.99.
    ylim_from : dict, optional
        ``{output_name: [input_name, ...]}`` mapping.  Only the listed input
        columns' data determines the shared y-axis range for that output row.
        Missing keys (or ``None``) default to all input columns.
    title : str, optional
        Figure suptitle.
    save_path : str or Path, optional
        If provided, saves the figure at 150 dpi.
    subplot_size : tuple of float, optional
        ``(width, height)`` in inches per subplot cell.  Default ``(1.5, 1.5)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cmap is None:
        cmap = plt.cm.coolwarm
    n_outputs = len(output_names)
    n_inputs = len(input_names)
    t = np.asarray(t_eval)

    step_fractions = step_data[input_names[0]]["step_fractions"]
    colors = _step_colors(step_fractions, cmap)

    fig, axs = plt.subplots(
        n_outputs,
        n_inputs,
        figsize=(subplot_size[0] * n_inputs, subplot_size[1] * n_outputs),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    if n_outputs == 1:
        axs = axs[np.newaxis, :]
    if n_inputs == 1:
        axs = axs[:, np.newaxis]

    for col, input_name in enumerate(input_names):
        input_entry = step_data.get(input_name, {})

        for row, output_name in enumerate(output_names):
            ax = axs[row, col]
            output_matrix = input_entry.get(output_name)

            if output_matrix is not None:
                for k, color in enumerate(colors):
                    ax.plot(t, output_matrix[:, k], color=color)

            ax.axhline(
                0.0, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axvline(
                t[step_time_idx],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.margins(0.06)
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            ax.grid(False)

            if col == 0:
                ax.set_ylabel(_abstract_label(output_name, output_info))
            if row == n_outputs - 1:
                ax.set_xlabel(_abstract_label(input_name, input_info))

    # Set y-axis limits per output row from governing input columns only.
    if clip_quantile is not None:
        ylim_from_ = ylim_from or {}
        for row, output_name in enumerate(output_names):
            governing = ylim_from_.get(output_name, input_names)
            arrays = [step_data.get(inp, {}).get(output_name) for inp in governing]
            limit = _compute_ylim(
                [a for a in arrays if a is not None], clip_quantile
            )
            if limit is not None:
                axs[row, 0].set_ylim(-limit * 1.05, limit * 1.05)

    fig.tight_layout(h_pad=0, w_pad=0, rect=[0, 0, 1, 0.94])
    fig.suptitle(title or "Step response I/O characteristics")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    return fig


if __name__ == "__main__":
    from cas_models.continuous_time.simulate import (
        make_n_step_simulation_function_from_model,
        make_steady_state_solver,
    )
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

    # ── Build model ───────────────────────────────────────────────────────
    print("Building model...")
    model = build_grinding_circuit_model_with_sump_control()
    print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")

    INPUT_NAMES = model.input_names
    OUTPUT_NAMES = model.output_names

    # ── Compute actual nominal steady state ───────────────────────────────
    ss_solver = make_steady_state_solver(model)
    u_nom = np.array([STEADY_STATE_INPUTS[n] for n in INPUT_NAMES])
    x0_guess = np.array([STEADY_STATE_STATES[n] for n in model.state_names])
    param_vals = {}

    x_ss, y_ss = ss_solver(x0_guess, u_nom, param_vals)
    print("\nNominal steady state:")
    for name, val in zip(OUTPUT_NAMES, y_ss):
        print(f"  {name:<25} {val:.4g}")

    # ── Simulation parameters ─────────────────────────────────────────────
    DT_H = 1.0 / 60.0  # 1-minute steps
    T_SIM_H = 6.0
    N_STEPS = int(T_SIM_H / DT_H)

    STEP_TIME_H = 1.0
    STEP_TIME_IDX = int(STEP_TIME_H / DT_H)

    t_eval = np.linspace(0.0, T_SIM_H, N_STEPS + 1)

    print(
        f"\nBuilding {N_STEPS}-step simulation "
        f"(dt={DT_H * 60:.0f} min, T={T_SIM_H:.0f} h)..."
    )
    sim_func = make_n_step_simulation_function_from_model(
        model, dt=DT_H, nT=N_STEPS
    )

    # ── Step sizes ────────────────────────────────────────────────────────
    STEP_FRACTIONS = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]
    STEP_SIZES = {
        inp: list(u_nom[j] * np.asarray(STEP_FRACTIONS))
        for j, inp in enumerate(INPUT_NAMES)
    }

    # ── Y-axis governing columns ──────────────────────────────────────────
    # Maps each output row to the input columns whose data sets the y-axis range.
    # None (or a missing key) means all input columns contribute.  Example:
    #   "mill_power": ["critical_speed_fraction"]
    YLIM_FROM = None

    # ── Compute step responses ────────────────────────────────────────────
    step_data = compute_step_responses(
        sim_func,
        t_eval,
        x_ss,
        y_ss,
        u_nom,
        STEP_SIZES,
        OUTPUT_NAMES,
        step_time_idx=STEP_TIME_IDX,
        param_vals=param_vals,
        all_output_names=OUTPUT_NAMES,
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    fracs_str = ", ".join(f"{f:+.0%}" for f in STEP_FRACTIONS)
    PLOT_TITLE = (
        f"Grinding Circuit – Step Responses ({fracs_str}, "
        f"step at {STEP_TIME_H:.0f} h)"
    )

    plot_main(
        step_data,
        INPUT_NAMES,
        OUTPUT_NAMES,
        t_eval,
        STEP_TIME_IDX,
        input_info=INPUT_INFO,
        output_info=OUTPUT_INFO,
        ylim_from=YLIM_FROM,
        title=PLOT_TITLE,
        save_path=PLOT_DIR / "step_io_main.png",
    )
    print(f"\nSaved {PLOT_DIR}/step_io_main.png")

    PLOT_TITLE = (
        f"Grinding Circuit – Step Responses "
        f"({STEP_FRACTIONS[0]:+.0%} to {STEP_FRACTIONS[-1]:+.0%})"
    )

    plot_abstract(
        step_data,
        INPUT_NAMES,
        OUTPUT_NAMES,
        t_eval,
        STEP_TIME_IDX,
        input_info=INPUT_INFO,
        output_info=OUTPUT_INFO,
        ylim_from=YLIM_FROM,
        title=PLOT_TITLE,
        save_path=PLOT_DIR / "step_io_abstract.png",
    )
    print(f"Saved {PLOT_DIR}/step_io_abstract.png")

    plt.show()
