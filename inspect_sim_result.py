"""Inspect a single step-response simulation result CSV.

Load, slice to a time window, and plot selected variables.
Edit PLOT_VARS and TIME_RANGE below to change what is shown.

Run:
    python inspect_sim_result.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_FILE = "results/feed_step_sim_results_2.csv"

TIME_RANGE = (9.5, 10.5)  # (start, end) hours relative to step

PLOT_VARS = {
    "inputs": ["feed_ore_rate", "water_ore_ratio"],
    "states": ["water_volume", "solids_volume", "rock_volume", "fines_volume"],
    "outputs": ["charge_fill_fraction", "mill_power", "sump_level"],
}

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0)
df.index = df.index.astype(float)
df.index.name = "time_rel_h"

# ── Slice to time range ────────────────────────────────────────────────────────
t0, t1 = TIME_RANGE
df_slice = df.loc[(df.index >= t0) & (df.index <= t1)]

# ── Plot ───────────────────────────────────────────────────────────────────────
var_list = [
    (cat, var)
    for cat, vars_ in PLOT_VARS.items()
    for var in vars_
]
n = len(var_list)
fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
if n == 1:
    axes = [axes]

for ax, (cat, var) in zip(axes, var_list):
    ax.plot(df_slice.index, df_slice[(cat, var)], linewidth=1.0)
    ax.set_ylabel(var.replace("_", " "), fontsize=8)
    ax.set_title(f"[{cat}]  {var}", fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time relative to step (h)", fontsize=9)
fig.tight_layout()
plt.show()

# ── Mill power intermediate variables (eq. 6) ──────────────────────────────────
# Constants from the model (Tables 4 & 5, Le Roux & Steyn 2022)
EPSILON_ZERO = 0.60          # ε₀: max solids fraction at zero flow
PHI_NORM = 0.70              # φ_N: rheology normalisation factor
POWER_MAX = 19.7             # P_max (MW)
DELTA_VOLUME = 0.0911        # δ_v: power param, mill fill volume
DELTA_SOLIDS = 0.0911        # δ_s: power param, solids fraction
FILL_FRACTION_MAX_POWER = 0.23  # J_TPmax: fill fraction at max power
BALL_VOLUME = 105.0          # x_mb (m³)
MILL_VOLUME = 540.9          # v_mill (m³)

x_mw = df_slice[("states", "water_volume")]
x_ms = df_slice[("states", "solids_volume")]
x_mr = df_slice[("states", "rock_volume")]
u_phic = df_slice[("inputs", "critical_speed_fraction")]

eps_slope = 1.0 / EPSILON_ZERO - 1.0        # ε₀⁻¹ - 1
eps_threshold = 1.0 / eps_slope              # ε₀ / (1 - ε₀)  = 1.5
ratio_ms_mw = x_ms / x_mw

phi = np.where(
    ratio_ms_mw <= eps_threshold,
    np.sqrt(np.maximum(0.0, 1.0 - eps_slope * ratio_ms_mw)),
    0.0,
)

y_JT = (x_mw + x_ms + x_mr + BALL_VOLUME) / MILL_VOLUME

term_volume = DELTA_VOLUME * (y_JT / FILL_FRACTION_MAX_POWER - 1.0) ** 2
term_solids = DELTA_SOLIDS * (phi / PHI_NORM - 1.0) ** 2
y_Pmill_calc = POWER_MAX * u_phic * (1.0 - term_volume - term_solids)

power_table = pd.DataFrame(
    {
        "x_ms/x_mw": ratio_ms_mw.values,
        f"threshold({eps_threshold:.2f})": eps_threshold,
        "phi": phi,
        "y_JT": y_JT.values,
        "term_volume": term_volume.values,
        "term_solids": term_solids,
        "Pmill_calc": y_Pmill_calc,
        "Pmill_csv": df_slice[("outputs", "mill_power")].values,
    },
    index=df_slice.index,
)

print(f"\nMill power intermediates over t ∈ [{t0}, {t1}] h  "
      f"(ε₀/(1-ε₀) threshold = {eps_threshold:.3f}):\n")
print(power_table.round(4).to_string())
