"""Time constant analysis: charge fill fraction response to feed rate steps.

For each of 6 feed rate step changes, computes the first-order time constant
defined as the time for the output to reach 62.3% of the steady-state change.
Large negative steps exhibit oscillatory behaviour and are flagged accordingly.

Run:
    python feed_step_time_constants.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cas_models.continuous_time.simulate import (
    make_sim_step_function_integrator_fixed_dt,
)
from model import (
    build_grinding_circuit_model_with_sump_control,
    INPUTS_NOP,
    STATES_NOP,
)


def plot_step_responses(t_eval, rows, y_traces, feed_nom):
    """Plot charge fill fraction step responses for all feed rate steps."""
    t_rel = t_eval - T_STEP_H
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    y0_nop = rows[0][2]

    for ax, (dU, feed, y0_val, y_ss, delta, _, converged, tau), y in zip(
        axes_flat, rows, y_traces
    ):
        color = "C0" if dU > 0 else "C1"
        ax.axhline(
            y0_nop, color="grey", linestyle="--", linewidth=0.8, label="NOP"
        )
        ax.axvline(0, color="k", linestyle=":", linewidth=0.8)
        ax.plot(t_rel, y, color=color, linewidth=1.0)

        if not np.isnan(tau):
            thresh = y0_val + TAU_FRAC * delta
            ax.scatter(
                [tau / 60.0],
                [thresh],
                marker="x",
                s=60,
                color="k",
                zorder=5,
                label=f"\u03c4 = {tau:.0f} min",
            )
            ax.legend(fontsize=7, loc="best")
            ax.legend(fontsize=7, loc="best")

        ax.set_title(f"\u0394feed = {dU:+.0f} t/h  (feed = {feed:.0f} t/h)", fontsize=8)
        ax.set_ylabel(OUTPUT_NAME.replace("_", " "), fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("Time after step (h)", fontsize=8)

    fig.suptitle(
        f"Charge fill fraction step responses \u2014 feed rate steps from NOP ({feed_nom:.0f} t/h)\n"
        f"(dt = {DT_SEC:.0f} s,  T = {T_SIM_H:.0f} h)",
        fontsize=10,
    )
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feed_step_time_constants.png"), dpi=150)
    print(f"\nSaved {PLOT_DIR}/feed_step_time_constants.png")
    plt.show()


def build_sim_results(model, t_eval, sim_data):
    """Build MultiIndex DataFrames from simulation data."""
    col_index = pd.MultiIndex.from_tuples(
        [("inputs", n) for n in model.input_names]
        + [("states", n) for n in model.state_names]
        + [("outputs", n) for n in model.output_names],
        names=["category", "variable"],
    )
    dfs = []
    for dU, (U_arr, X, Y) in zip(STEP_SIZES, sim_data):
        U_full = np.vstack([U_arr, U_arr[-1:]])
        row_index = pd.Index(t_eval - T_STEP_H, name="time_rel_h")
        data = np.hstack([U_full, X, Y])
        dfs.append(pd.DataFrame(data, index=row_index, columns=col_index))

    return dfs


PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DT_SEC = 60.0
DT_H = DT_SEC / 3600.0
T_SIM_H = 20.0
N_STEPS = int(T_SIM_H / DT_H)
T_STEP_H = 0.5  # step applied at 30 min
STEP_IDX = int(T_STEP_H / DT_H)

STEP_SIZES = [-150, -100, -50, 50, 100, 150]  # t/h
TAU_FRAC = 0.623
CONVERGE_TOL = 0.005  # max std of last 2 h as fraction of |delta|

print("Building model...")
model = build_grinding_circuit_model_with_sump_control()

INPUT_NAME = "feed_ore_rate"
OUTPUT_NAME = "charge_fill_fraction"
FEED_IDX = model.input_names.index(INPUT_NAME)
OUTPUT_IDX = model.output_names.index(OUTPUT_NAME)

# Options for 'RK' solver:
# integrator_opts = {
#     'number_of_finite_elements': 1,  # Sub-steps per interval
# }
SOLVER = "cvodes"
INTEGRATOR_OPTS = {
    "max_num_steps": 1000,  # default 500 — raise for stiff dynamics near boundaries
    "reltol": 1e-5,          # slightly looser than default 1e-6 for speed
}

sim_step = make_sim_step_function_integrator_fixed_dt(
    model.f, model.n, model.nu, DT_H,
    params=model.params, solver=SOLVER, integrator_opts=INTEGRATOR_OPTS,
    name="F_step",
)
_param_vals = list(model.params.values())  # empty for default (concrete) parameters

STATE_MIN = 0.01  # m³ — stop simulation if any volume drops below this


def run_sim(t_eval, U, x0):
    """Run simulation step by step in Python.

    Stops early and prints a warning if any state volume drops below STATE_MIN,
    leaving the remainder of X and Y filled with NaN.
    """
    X = np.full((len(t_eval), model.n), np.nan)
    Y = np.full((len(t_eval), model.ny), np.nan)
    x = x0.copy()
    X[0] = x
    Y[0] = np.array(model.h(t_eval[0], x, U[0], *_param_vals)).flatten()
    for k in range(len(t_eval) - 1):
        x_next = np.array(sim_step(t_eval[k], x, U[k], *_param_vals)).flatten()
        below = [model.state_names[i] for i in np.where(x_next < STATE_MIN)[0]]
        if below:
            print(
                f"  STOP at step {k + 1} "
                f"(t_rel={t_eval[k + 1] - T_STEP_H:+.3f} h): "
                f"{below} below {STATE_MIN} m³"
            )
            break
        x = x_next
        X[k + 1] = x
        Y[k + 1] = np.array(
            model.h(t_eval[k + 1], x, U[min(k + 1, len(U) - 1)], *_param_vals)
        ).flatten()
    return X, Y

x0 = np.array([STATES_NOP[n] for n in model.state_names])
u0 = np.array([INPUTS_NOP[n] for n in model.input_names])
feed_nom = u0[FEED_IDX]
t_eval = np.linspace(0.0, T_SIM_H, N_STEPS + 1)

print(
    f"Running {len(STEP_SIZES)} simulations "
    f"(T={T_SIM_H:.0f} h, dt={DT_SEC:.0f} s, step at t={T_STEP_H * 60:.0f} min)...\n"
)

rows = []
y_traces = []
sim_data = []  # list of (U_arr, X, Y) for each step

for dU in STEP_SIZES:
    U = np.tile(u0, (N_STEPS, 1))
    U[STEP_IDX:, FEED_IDX] += dU

    X, Y = run_sim(t_eval, U, x0)
    y = Y[:, OUTPUT_IDX]
    y_traces.append(y)
    sim_data.append((U.copy(), X, Y))

    y0_val = y[STEP_IDX]
    y_ss = y[-1]
    delta = y_ss - y0_val
    thresh = y0_val + TAU_FRAC * delta

    # Convergence check: std of last 2 h should be small relative to total change
    last_2h = y[int(-2.0 / DT_H) :]
    converged = np.std(last_2h) < CONVERGE_TOL * abs(delta)

    # Time delay: response should start at STEP_IDX+1 (no dead time in model)
    has_delay = abs(y[STEP_IDX + 1] - y0_val) < 1e-6 * abs(delta)

    # Time constant via linear interpolation (only for converged cases)
    tau_min = np.nan
    if converged:
        y_post = y[STEP_IDX:]
        t_post = t_eval[STEP_IDX:]
        cross = np.where(np.sign(y_post - thresh) == np.sign(delta))[0]
        if len(cross) > 0 and cross[0] > 0:
            i = cross[0]
            alpha = (thresh - y_post[i - 1]) / (y_post[i] - y_post[i - 1])
            t_tau = t_post[i - 1] + alpha * (t_post[i] - t_post[i - 1])
            tau_min = (t_tau - T_STEP_H) * 60.0

    rows.append(
        (dU, feed_nom + dU, y0_val, y_ss, delta, has_delay, converged, tau_min)
    )

# --- Delay check ---
if any(r[5] for r in rows):
    print("WARNING: time delay detected in one or more step responses.")
else:
    print(
        f"Time delay: none (response starts within first {DT_SEC:.0f} s after step).\n"
    )

# --- Table ---
W = (11, 11, 9, 9, 9, 12, 10)
hdr = (
    f"{'Step (t/h)':>{W[0]}}  {'Feed (t/h)':>{W[1]}}"
    f"  {'y_0':>{W[2]}}  {'y_ss':>{W[3]}}  {'Δy':>{W[4]}}"
    f"  {'Converged':>{W[5]}}  {'τ (min)':>{W[6]}}"
)
sep = "-" * len(hdr)
print(hdr)
print(sep)
for dU, feed, y0_val, y_ss, delta, _, converged, tau in rows:
    tau_str = f"{tau:.2f}" if not np.isnan(tau) else "N/A"
    print(
        f"{dU:>+{W[0]}.0f}  {feed:>{W[1]}.0f}"
        f"  {y0_val:>{W[2]}.5f}  {y_ss:>{W[3]}.5f}  {delta:>{W[4]}.5f}"
        f"  {'yes' if converged else 'no (oscillating)':>{W[5]}}  {tau_str:>{W[6]}}"
    )
print(sep)
tau_vals = [r[7] for r in rows if not np.isnan(r[7])]
n_before = len(hdr) - W[-1] - 2
print(
    f"{'Average τ (min) [converged cases]':>{n_before}}  {np.mean(tau_vals):>{W[-1]}.2f}"
)

# --- Plots ---
plot_step_responses(t_eval, rows, y_traces, feed_nom)

# --- Build simulation results DataFrame ---
dfs = build_sim_results(model, t_eval, sim_data)
sim_results = pd.concat(dfs, keys=STEP_SIZES, names=["feed_step_th", "time_rel_h"])

csv_path = os.path.join(RESULTS_DIR, "feed_step_sim_results.csv")
sim_results.to_csv(csv_path)
print(f"Saved {csv_path}  ({sim_results.shape[0]} rows x {sim_results.shape[1]} cols)")

for i, df in enumerate(dfs, start=1):
    csv_path_i = os.path.join(RESULTS_DIR, f"feed_step_sim_results_{i}.csv")
    df.to_csv(csv_path_i)
    print(f"Saved {csv_path_i}")
