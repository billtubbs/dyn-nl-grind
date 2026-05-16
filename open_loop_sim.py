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


# Derive charge_porosity from x_mb = 105 m³ (Table 5): x_mb = (1-ε_p)*J_B*v_mill
_X_MB_SS = 105.0
CHARGE_POROSITY_SS = 1.0 - _X_MB_SS / (0.30 * 540.9)

# Simulation parameters
DT_SEC = 60.0  # sample time (seconds)
DT_H = DT_SEC / 3600.0  # sample time (hours)
T_SIM_H = 6.0  # simulation duration (hours)
N_STEPS = int(T_SIM_H / DT_H)  # number of steps

# Initial conditions and constant inputs from Tables 4 & 5
x0 = np.array([STEADY_STATE_STATES[n] for n in STATE_NAMES])
u0 = np.array([STEADY_STATE_INPUTS[n] for n in CL_INPUT_NAMES])

print("Building model with sump level control...")
model = build_grinding_circuit_model_with_sump_control(
    charge_porosity=CHARGE_POROSITY_SS
)

print(
    f"Building {N_STEPS}-step simulation (dt={DT_H:.5f} h = {DT_SEC:.0f} s)..."
)
sim = make_n_step_simulation_function_from_model(model, dt=DT_H, nT=N_STEPS)

# Constant inputs (4 free inputs; u_CFF handled by controller)
t_eval = np.linspace(0.0, T_SIM_H, N_STEPS + 1)
U = np.tile(u0, (N_STEPS, 1))  # shape (N_STEPS, nu=4)

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

fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i, (ax, label) in enumerate(zip(axes, state_labels)):
    ax.plot(t_eval, X[:, i], color=f"C{i}")
    ax.axhline(
        x0[i], color="grey", linestyle="--", linewidth=0.8, label="Paper SS"
    )
    ax.set_ylabel(f"{label}\n(m³)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

axes[-1].set_visible(False)
for ax in axes[-2:]:
    ax.set_xlabel("Time (h)")

fig.suptitle(
    "Grinding circuit with sump level control — states\n"
    f"(dt = {DT_SEC:.0f} s, T = {T_SIM_H:.0f} h, constant mill inputs)",
    fontsize=11,
)
fig.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "open_loop_states.png"), dpi=150)
print(f"Saved {PLOT_DIR}/open_loop_states.png")

# ── Output plots (incl. controller action) ────────────────────────────────────
output_units = ["-", "MW", "%", "t/m³", "%", "m³/h"]

fig2, axes2 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
axes2 = axes2.flatten()

for i, (ax, name, unit) in enumerate(
    zip(axes2, CL_OUTPUT_NAMES, output_units)
):
    ax.plot(t_eval, Y[:, i], color=f"C{i}")
    ax.set_ylabel(f"{name}\n({unit})")
    ax.grid(True, alpha=0.3)

for ax in axes2[-2:]:
    ax.set_xlabel("Time (h)")

fig2.suptitle(
    "Grinding circuit with sump level control — outputs\n"
    f"(dt = {DT_SEC:.0f} s, T = {T_SIM_H:.0f} h, constant mill inputs)",
    fontsize=11,
)
fig2.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "open_loop_outputs.png"), dpi=150)
print(f"Saved {PLOT_DIR}/open_loop_outputs.png")

plt.show()

# Print final state vs initial
print("\nState at t=0 vs t=6h:")
print(f"{'State':<25} {'t=0':>10} {'t=6h':>10} {'change':>10}")
print("-" * 57)
for name, x_init, x_final in zip(STATE_NAMES, x0, X[-1]):
    print(
        f"{name:<25} {x_init:>10.3f} {x_final:>10.3f} {x_final - x_init:>+10.3f}"
    )
