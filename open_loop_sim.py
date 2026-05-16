"""Simulation of the grinding circuit with sump level control, starting from
the paper's tabulated steady-state conditions (Tables 4 & 5, Le Roux & Steyn
2022).

Run:
    python open_loop_sim.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from cas_models.continuous_time.simulate import (
    make_n_step_simulation_function_from_model,
)
from model import (
    CL_INPUT_NAMES,
    CL_OUTPUT_NAMES,
    STATE_NAMES,
    STEADY_STATE_INPUTS,
    STEADY_STATE_STATES,
    build_grinding_circuit_model_with_sump_control,
)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# Simulation parameters
DT_SEC = 60.0  # sample time (seconds)
DT_H = DT_SEC / 3600.0  # sample time (hours)
T_SIM_H = 18.0  # simulation duration (hours)
N_STEPS = int(T_SIM_H / DT_H)  # number of steps

# Step change: feed_ore_rate 1191 → 1100 t/h at 60 min
T_STEP_H = 120.0 / 60.0
STEP_IDX = int(T_STEP_H / DT_H)
FEED_STEP_VALUE = 1100.0  # t/h

# Initial conditions and constant inputs from Tables 4 & 5
x0 = np.array([STEADY_STATE_STATES[n] for n in STATE_NAMES])
u0 = np.array([STEADY_STATE_INPUTS[n] for n in CL_INPUT_NAMES])

print("Building model with sump level control...")
model = build_grinding_circuit_model_with_sump_control()

print(
    f"Building {N_STEPS}-step simulation (dt={DT_H:.5f} h = {DT_SEC:.0f} s)..."
)
sim = make_n_step_simulation_function_from_model(model, dt=DT_H, nT=N_STEPS)

# Inputs: constant before step, +5 % feed_ore_rate from 30 min onward
t_eval = np.linspace(0.0, T_SIM_H, N_STEPS + 1)
U = np.tile(u0, (N_STEPS, 1))  # shape (N_STEPS, nu=4)
feed_idx = CL_INPUT_NAMES.index("feed_ore_rate")
U[STEP_IDX:, feed_idx] = FEED_STEP_VALUE

print(f"Running simulation ({T_SIM_H:.0f} h)...")
X_cas, Y_cas = sim(t_eval, U, x0)
X = np.array(X_cas)  # (N_STEPS+1, n)
Y = np.array(Y_cas)  # (N_STEPS+1, ny)

# ── State plots ───────────────────────────────────────────────────────────────
state_labels = [
    "Mill water ($x_{mw}$)",
    "Mill solids ($x_{ms}$)",
    "Mill rocks ($x_{mr}$)",
    "Mill fines ($x_{mf}$)",
    "Sump water ($x_{sw}$)",
    "Sump solids ($x_{ss}$)",
    "Sump fines ($x_{sf}$)",
]

fig, axes = plt.subplots(4, 2, figsize=(8, 7.5), sharex=True)
axes = axes.flatten()

for i, (ax, label) in enumerate(zip(axes, state_labels)):
    ax.plot(t_eval, X[:, i], color=f"C{i}")
    ax.axhline(
        x0[i], color="grey", linestyle="--", linewidth=0.8, label="Paper SS"
    )
    ax.axvline(T_STEP_H, color="k", linestyle=":", linewidth=0.8)
    ax.set_title(label, fontsize=9)
    ax.set_ylabel("m³")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

axes[-1].set_visible(False)
for ax in axes[-2:]:
    ax.set_xlabel("Time (h)")

fig.suptitle(
    "Grinding circuit with sump level control — states\n"
    f"(dt = {DT_SEC:.0f} s, T = {T_SIM_H:.0f} h, "
    f"feed_ore_rate {u0[feed_idx]:.0f}→{FEED_STEP_VALUE:.0f} t/h at {T_STEP_H * 60:.0f} min)",
    fontsize=11,
)
fig.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "open_loop_states.png"), dpi=150)
print(f"Saved {PLOT_DIR}/open_loop_states.png")

# ── Output plots (incl. controller action) ────────────────────────────────────
output_units = ["-", "MW", "%", "t/m³", "%", "m³/h"]
input_units = ["t/h", "-", "-", "m³/h"]

fig2, axes2 = plt.subplots(3, 2, figsize=(8, 5.85), sharex=True)
axes2 = axes2.flatten()

for i, (ax, name, unit) in enumerate(
    zip(axes2, CL_OUTPUT_NAMES, output_units)
):
    ax.plot(t_eval, Y[:, i], color=f"C{i}")
    ax.axvline(T_STEP_H, color="k", linestyle=":", linewidth=0.8)
    ax.set_title(name, fontsize=9)
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)

for ax in axes2[-2:]:
    ax.set_xlabel("Time (h)")

fig2.suptitle(
    "Grinding circuit with sump level control — outputs\n"
    f"(dt = {DT_SEC:.0f} s, T = {T_SIM_H:.0f} h, "
    f"feed_ore_rate {u0[feed_idx]:.0f}→{FEED_STEP_VALUE:.0f} t/h at {T_STEP_H * 60:.0f} min)",
    fontsize=11,
)
fig2.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "open_loop_outputs.png"), dpi=150)
print(f"Saved {PLOT_DIR}/open_loop_outputs.png")

# ── Summary table: inputs, states, outputs at t=0 vs t=12h ───────────────────
_W = (25, 5, 10, 10, 10)
_HDR = (
    f"{'Variable':<{_W[0]}} {'Unit':>{_W[1]}}"
    f" {'t=0':>{_W[2]}} {'t=12h':>{_W[3]}} {'change':>{_W[4]}}"
)
_SEP = "-" * len(_HDR)


def _print_section(title, names, units, vals_0, vals_f):
    print(f"\n{title}")
    print(_HDR)
    print(_SEP)
    for name, unit, v0, vf in zip(names, units, vals_0, vals_f):
        print(
            f"{name:<{_W[0]}} {unit:>{_W[1]}}"
            f" {v0:>{_W[2]}.3f} {vf:>{_W[3]}.3f} {vf - v0:>+{_W[4]}.3f}"
        )


_print_section("Inputs:", CL_INPUT_NAMES, input_units, U[0], U[-1])
_print_section("States:", STATE_NAMES, ["m³"] * len(STATE_NAMES), x0, X[-1])
_print_section("Outputs:", CL_OUTPUT_NAMES, output_units, Y[0], Y[-1])

plt.show()
