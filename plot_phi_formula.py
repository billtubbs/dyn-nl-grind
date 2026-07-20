"""Plot the mill rheology factor phi as a function of x_ms / x_mw.

Compares the original paper formula (eq. 3) with a smooth softplus
approximation that is C-infinity everywhere and avoids the derivative
singularity at the threshold.

Smooth formula:
    phi_smooth(r) = sqrt(softplus_beta(1 - eps_slope * r) / beta)
                  = sqrt(log(1 + exp(beta * (1 - eps_slope * r))) / beta)

Tunable parameter
-----------------
BETA : controls the sharpness of the approximation.
       Large beta  → closely follows the paper formula (sharper kink).
       Small beta  → smoother transition but departs more from the paper.

Run:
    python plot_phi_formula.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Model parameters (Table 5, Le Roux & Steyn 2022)
EPSILON_ZERO = 0.60  # ε₀: max solids fraction at zero flow

# ── Softplus smoothing parameter ───────────────────────────────────────────────
BETA_VALUES = [20, 10, 5]  # larger = closer to paper formula

# ── Derived constants ──────────────────────────────────────────────────────────
eps_slope = 1.0 / EPSILON_ZERO - 1.0  # ε₀⁻¹ - 1  ≈ 0.667
eps_threshold = 1.0 / eps_slope  # ε₀ / (1 - ε₀) = 1.5

ratio = np.linspace(0.0, 2.5, 2000)

# Paper formula
phi_paper = np.where(
    ratio <= eps_threshold,
    np.sqrt(np.maximum(0.0, 1.0 - eps_slope * ratio)),
    0.0,
)


def phi_softplus(r, beta):
    """Smooth C-inf approximation: sqrt(softplus_beta(1 - eps_slope*r) / beta)."""
    arg = beta * (1.0 - eps_slope * r)
    # numerically stable: log(1+exp(x)) = x + log(1+exp(-x)) for x>0
    sp = np.where(arg > 0, arg + np.log1p(np.exp(-arg)), np.log1p(np.exp(arg)))
    return np.sqrt(sp / beta)


fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(
    ratio,
    phi_paper,
    color="C0",
    linewidth=2.0,
    label=r"$\varphi$ paper (eq. 3)",
    zorder=3,
)

colors = ["C1", "C2", "C3"]
for beta, color in zip(BETA_VALUES, colors):
    ax.plot(
        ratio,
        phi_softplus(ratio, beta),
        color=color,
        linewidth=1.4,
        linestyle="--",
        label=rf"softplus  $\beta={beta}$",
    )

# Mark threshold
ax.axvline(eps_threshold, color="grey", linewidth=0.9, linestyle="--")
ax.text(
    eps_threshold + 0.04,
    0.65,
    rf"threshold = {eps_threshold:.1f}",
    fontsize=8,
    color="grey",
)

# Mark NOP
ratio_nop = 30.723 / 30.789
phi_nop = np.sqrt(1.0 - eps_slope * ratio_nop)
ax.scatter(
    [ratio_nop],
    [phi_nop],
    marker="o",
    s=50,
    color="k",
    zorder=5,
    label=f"NOP  (ratio={ratio_nop:.3f}, φ={phi_nop:.3f})",
)

ax.set_xlabel(
    r"$x_{ms} / x_{mw}$  (solids-to-water volume ratio)", fontsize=10
)
ax.set_ylabel(r"$\varphi$  (rheology factor)", fontsize=10)
ax.set_title(
    r"Mill rheology factor $\varphi$ — paper eq. (3) vs. softplus smoothing"
    "\n"
    r"$\varphi_\mathrm{smooth}(r)=\sqrt{\frac{1}{\beta}\ln\!\left(1+e^{\,\beta(1-\varepsilon_\mathrm{slope}\,r)}\right)}$",
    fontsize=9,
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2.5)
ax.set_ylim(-0.05, 1.1)

fig.tight_layout()
plt.savefig("plots/rheo_curve.png", dpi=150)
print("Saved plots/rheo_curve.png")
plt.show()
