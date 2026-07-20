"""Plot mill power draw as a function of charge fill fraction.

Implements eq. (6) from Le Roux & Steyn (2022):

    P_mill = P_max · φ_c · (1 - δ_v·(J_T/J_TPmax - 1)² - δ_s·(φ/φ_N - 1)²)

The charge composition (water:solids:rocks ratio) is held fixed at the NOP
values; only the total charge volume is varied to sweep J_T.  Five curves are
plotted at different critical-speed fractions φ_c, including the NOP value.

Run:
    python plot_power_model.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Model parameters (Tables 4 & 5, Le Roux & Steyn 2022) ─────────────────────
POWER_MAX = 19.7  # P_max (MW)
DELTA_VOLUME = 0.0911  # δ_v
DELTA_SOLIDS = 0.0911  # δ_s
FILL_FRACTION_MAX_POWER = 0.23  # J_TPmax: fill fraction at peak power
PHI_NORM = 0.70  # φ_N
EPSILON_ZERO = 0.60  # ε₀
PHI_BETA = 20.0  # softplus sharpness

# NOP state values used to fix the charge composition ratio
X_MW_NOP = 30.789  # water volume (m³)
X_MS_NOP = 30.723  # solids volume (m³)
X_MR_NOP = 9.6708  # rock volume (m³)
X_MB = 105.0  # ball volume (m³) — constant
MILL_VOLUME = 540.9  # v_mill (m³)

# NOP speed and fill level
PHI_C_NOP = 0.768
Y_JT_NOP = (X_MW_NOP + X_MS_NOP + X_MR_NOP + X_MB) / MILL_VOLUME

# ── Critical-speed fractions to plot ──────────────────────────────────────────
PHI_C_VALUES = [0.60, 0.70, PHI_C_NOP, 0.85, 0.90]

# ── Derived: phi at NOP composition (fixed for all fill levels) ───────────────
eps_slope = 1.0 / EPSILON_ZERO - 1.0
ratio_nop = X_MS_NOP / X_MW_NOP  # x_ms/x_mw at NOP
# softplus phi (eq. 3, smoothed)
_arg = PHI_BETA * (1.0 - eps_slope * ratio_nop)
_sp = np.where(
    _arg > 0, _arg + np.log1p(np.exp(-_arg)), np.log1p(np.exp(_arg))
)
phi_nop = np.sqrt(_sp / PHI_BETA)

# Constant solids-penalty term (fixed composition → fixed phi)
solids_penalty = DELTA_SOLIDS * (phi_nop / PHI_NORM - 1.0) ** 2

# ── Fill-level sweep ──────────────────────────────────────────────────────────
y_JT = np.linspace(0.15, 0.60, 500)


def mill_power(y_jt, phi_c):
    """Mill power (MW) at given fill fraction and critical-speed fraction."""
    return (
        POWER_MAX
        * phi_c
        * (
            1.0
            - DELTA_VOLUME * (y_jt / FILL_FRACTION_MAX_POWER - 1.0) ** 2
            - solids_penalty
        )
    )


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(PHI_C_VALUES)))

for phi_c, color in zip(PHI_C_VALUES, colors):
    P = mill_power(y_JT, phi_c)
    label = (
        rf"$\varphi_c = {phi_c:.3f}$ (NOP)"
        if phi_c == PHI_C_NOP
        else rf"$\varphi_c = {phi_c:.2f}$"
    )
    ls = "-" if phi_c == PHI_C_NOP else "--"
    ax.plot(
        y_JT,
        P,
        color=color,
        linewidth=1.6 if phi_c == PHI_C_NOP else 1.2,
        linestyle=ls,
        label=label,
    )

    # Mark each curve's peak (always at J_TPmax)
    P_peak = mill_power(FILL_FRACTION_MAX_POWER, phi_c)
    ax.scatter(
        [FILL_FRACTION_MAX_POWER],
        [P_peak],
        marker="^",
        s=35,
        color=color,
        zorder=4,
    )

# NOP operating point
P_nop = mill_power(Y_JT_NOP, PHI_C_NOP)
ax.scatter(
    [Y_JT_NOP],
    [P_nop],
    marker="o",
    s=60,
    color="k",
    zorder=5,
    label=f"NOP  ($J_T={Y_JT_NOP:.4f}$, $P={P_nop:.2f}$ MW)",
)

# Reference lines
ax.axvline(
    FILL_FRACTION_MAX_POWER,
    color="grey",
    linewidth=0.9,
    linestyle=":",
    label=rf"$J_{{TPmax}} = {FILL_FRACTION_MAX_POWER}$",
)
ax.axvline(Y_JT_NOP, color="k", linewidth=0.7, linestyle=":")

ax.set_xlabel(
    r"Charge fill fraction  $J_T = (x_{mw}+x_{ms}+x_{mr}+x_{mb})\,/\,v_{mill}$",
    fontsize=10,
)
ax.set_ylabel("Mill power  $P_{mill}$  (MW)", fontsize=10)
ax.set_title(
    "Mill power vs. fill fraction — eq. (6), Le Roux & Steyn (2022)\n"
    r"(NOP charge composition: $x_{ms}/x_{mw}=$"
    f"{ratio_nop:.3f}"
    r", $\varphi=$"
    f"{phi_nop:.3f})",
    fontsize=9,
)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.15, 0.60)

fig.tight_layout()
plt.savefig("plots/power_model.png", dpi=150)
print("Saved plots/power_model.png")
plt.show()
