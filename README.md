# Grinding Circuit Simulation Model

Python implementation of the nonlinear dynamic grinding mill circuit model from:

> J.D. le Roux and C.W. Steyn, "Validation of a dynamic non-linear grinding circuit
> model for process control," *Minerals Engineering* 187 (2022) 107780.
> https://doi.org/10.1016/j.mineng.2022.107780

The model is built with [CasADi](https://web.casadi.org/) and wrapped via the
[casadi-models](https://github.com/billtubbs/casadi-models) library, giving
symbolic state-space functions `f(t, x, u)` and `h(t, x, u)` that can be
differentiated, compiled, and used directly in optimisation or estimation.

## License

This project is released under the [GNU General Public License v3.0](LICENSE).

---

## Installation

**Requirements:** Python ≥ 3.11, plus the packages listed in `pyproject.toml`.

```bash
git clone <repo-url>
cd dyn-nl-grind
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`.[dev]` installs the project in editable mode and adds `pytest` for running
the test suite.

---

## Quick Start

### Construct the model and evaluate at a known operating point

```python
import numpy as np
from model import build_grinding_circuit_model, STEADY_STATE_STATES, STEADY_STATE_INPUTS

# charge_porosity is calibrated from Table 5: x_mb = (1 - ε_p) * J_B * v_mill = 105 m³
charge_porosity = 1.0 - 105.0 / (0.30 * 540.9)

model = build_grinding_circuit_model(charge_porosity=charge_porosity)
# model.n=7 states, model.nu=5 inputs, model.ny=5 outputs

x0 = np.array([STEADY_STATE_STATES[n] for n in model.state_names])
u0 = np.array([STEADY_STATE_INPUTS[n] for n in model.input_names])

dxdt = np.array(model.f(0.0, x0, u0)).flatten()  # state derivatives (m³/h)
y0   = np.array(model.h(0.0, x0, u0)).flatten()   # outputs at this operating point
```

### Simulate with sump level control

The open-loop model (constant `u_CFF`) is unstable — the sump drains or
overflows without level control.  `build_grinding_circuit_model_with_sump_control`
embeds a proportional controller that manipulates the cyclone feed pump (`u_CFF`)
to maintain the sump level between configurable limits (default 5 %–80 %).

```python
import numpy as np
from cas_models.continuous_time.simulate import make_n_step_simulation_function_from_model
from model import (
    build_grinding_circuit_model_with_sump_control,
    STEADY_STATE_STATES, STEADY_STATE_INPUTS,
)

charge_porosity = 1.0 - 105.0 / (0.30 * 540.9)
model = build_grinding_circuit_model_with_sump_control(charge_porosity=charge_porosity)
# model.nu=4 inputs (u_CFF is now internal), model.ny=6 outputs (adds u_CFF as monitor)

dt      = 60 / 3600        # 1-minute sample time in hours
n_steps = int(6.0 / dt)    # 6-hour simulation
sim = make_n_step_simulation_function_from_model(model, dt=dt, nT=n_steps)

x0     = np.array([STEADY_STATE_STATES[n] for n in model.state_names])
u0     = np.array([STEADY_STATE_INPUTS[n] for n in model.input_names])
t_eval = np.linspace(0.0, 6.0, n_steps + 1)   # hours
U      = np.tile(u0, (n_steps, 1))             # constant inputs, shape (n_steps, 4)

X, Y = sim(t_eval, U, x0)
X = np.array(X)   # state trajectory, shape (n_steps+1, 7)
Y = np.array(Y)   # output trajectory, shape (n_steps+1, 6)
```

See [`open_loop_sim.py`](open_loop_sim.py) for a complete simulation script with plots.

---

## Model Documentation

> All equations and parameter definitions below follow Section 2.2 of the paper.

## Mill Model

The mill circuit is modelled with an adapted version of the continuous time
phenomenological non-linear population balance model of Le Roux et al. (2013).
Four volume states describe the mill: water ($x_{mw}$), solids ($x_{ms}$),
rocks ($x_{mr}$), and fines ($x_{mf}$), all in m³.

### Mill Mass Balances (Equations 1a–1d)

$$\frac{d}{dt}x_{mw} = \frac{u_{rMIW}\,u_{MFO}}{\rho_w} - Q_{mwo} + Q_{cwu} \tag{1a}$$

$$\frac{d}{dt}x_{ms} = \frac{(1-\alpha_r)\,u_{MFO}}{\rho_o} - Q_{mso} + Q_{csu} + Q_{RC} \tag{1b}$$

$$\frac{d}{dt}x_{mr} = \frac{\alpha_r\,u_{MFO}}{\rho_o} - Q_{RC} \tag{1c}$$

$$\frac{d}{dt}x_{mf} = \frac{\alpha_f\,u_{MFO}}{\rho_o} - Q_{mfo} + Q_{cfu} + Q_{FP} \tag{1d}$$

where $\alpha_f$ and $\alpha_r$ are the fines and rocks mass fractions in the feed ore
$u_{MFO}$; $\rho_o$ and $\rho_w$ are the ore and water density; $Q_{mwo}$, $Q_{mso}$,
$Q_{mfo}$ are the mill discharge flow-rates of water, solids and fines; $Q_{cwu}$,
$Q_{csu}$, $Q_{cfu}$ are the cyclone underflow rates of water, solids and fines;
$Q_{RC}$ is the rock consumption rate; and $Q_{FP}$ is the fines production rate.

### Mill Discharge Flow-Rates (Equations 2a–2c)

$$Q_{mwo} = \varphi\,d_q\,x_{mw}\!\left(\frac{x_{mw}}{x_{ms}+x_{mw}}\right) \tag{2a}$$

$$Q_{mso} = \varphi\,d_q\,x_{mw}\!\left(\frac{x_{ms}}{x_{ms}+x_{mw}}\right) \tag{2b}$$

$$Q_{mfo} = \varphi\,d_q\,x_{mw}\!\left(\frac{x_{mf}}{x_{ms}+x_{mw}}\right) \tag{2c}$$

where $d_q$ (h⁻¹) is the discharge rate, a fitting parameter for the discharge
mechanism.

### Rheology Factor (Equation 3)

The rheology factor $\varphi$ is an empirically defined function that incorporates the
effect of the fluidity and density of the slurry on mill performance:

$$\varphi = \begin{cases}
\sqrt{1 - \left(\varepsilon_0^{-1}-1\right)\dfrac{x_{ms}}{x_{mw}}}, &
\dfrac{x_{ms}}{x_{mw}} \le \left(\varepsilon_0^{-1}-1\right)^{-1} \\[8pt]
0, & \dfrac{x_{ms}}{x_{mw}} > \left(\varepsilon_0^{-1}-1\right)^{-1}
\end{cases} \tag{3}$$

where $\varepsilon_0 = 0.60$ is the approximate maximum fraction of solids by volume in
the slurry at zero slurry flow. The slurry is pure water ($\varphi = 1$) when
$x_{ms}/x_{mw} = 0$ and becomes a non-flowing mud ($\varphi = 0$) when
$x_{ms}/x_{mw} = 1.5$.

### Rock Consumption and Fines Production (Equations 4a–4b)

$$Q_{RC} = \frac{x_{mr}\,y_{Pmill}}{\rho_o\,K_{RC}\,(x_{mr}+x_{ms})} \tag{4a}$$

$$Q_{FP} = \frac{y_{Pmill}}{\rho_o\,K_{FP}\!\left(1 + K_{FP_{JT}}\!\left(y_{JT}-J_{TP_{max}}\right)\right)} \tag{4b}$$

where $K_{RC}$ (MWh/t) is the rock consumption factor, $K_{FP}$ (MWh/t) is the fines
production factor, and $K_{FP_{JT}}$ is the fractional change in fines production per
unit change in fractional mill filling.

### Mill Charge Fraction (Equation 5)

$$y_{JT} = \frac{x_{mw} + x_{ms} + x_{mr} + x_{mb}}{v_{mill}} \tag{5}$$

where $v_{mill}$ (m³) is the total internal volume of the mill and $x_{mb}$ is the
(constant) volume of steel balls in the mill.

### Mill Power Draw (Equation 6)

$$y_{Pmill} = P_{max}\,u_{\varphi_c}\!\left[
  1 - \delta_v\!\left(\frac{y_{JT}}{J_{TP_{max}}}-1\right)^{\!2}
    - \delta_s\!\left(\frac{\varphi}{\varphi_N}-1\right)^{\!2}
\right] \tag{6}$$

where $\delta_v$ is the power parameter for mill fill volume, $\delta_s$ is the power
parameter for fraction solids in the slurry, $\varphi_N$ is the rheology
normalisation factor, $J_{TP_{max}}$ is the fractional fill at maximum power draw,
and $P_{max}$ (MW) is the maximum mill power draw.

### Mill Charge Density (Equation 7)

$$\rho_{mc} = \rho_o(1-\varepsilon_p+\varepsilon_p US)
  + \frac{J_B}{y_{JT}}(\rho_b-\rho_o)(1-\varepsilon_p)
  + \varepsilon_p U(1-S) \tag{7}$$

where $\varepsilon_p$ is the porosity of the mill charge, $J_B$ is the fraction of the
mill filled with steel balls, $U$ is the voidage in the mill charge, and $S$ is the
mill discharge volumetric solids content.

---

## Sump Model

Three volume states describe the sump: water ($x_{sw}$), solids ($x_{ss}$), and fines
($x_{sf}$), all in m³. Rocks and balls do not exit through the mill discharge mechanism
and so do not form part of the sump balance.

### Sump Mass Balances (Equations 8a–8c)

$$\frac{d}{dt}x_{sw} = Q_{mwo} - Q_{swo} + u_{SFW} \tag{8a}$$

$$\frac{d}{dt}x_{ss} = Q_{mso} - Q_{sso} \tag{8b}$$

$$\frac{d}{dt}x_{sf} = Q_{mfo} - Q_{sfo} \tag{8c}$$

where $u_{SFW}$ (m³/h) is the sump feed water and the sump is assumed to be
fully mixed.

### Sump Discharge Flow-Rates (Equations 9a–9c)

The sump discharges to the cyclone cluster via a variable speed pump at total flow
rate $u_{CFF}$:

$$Q_{swo} = u_{CFF}\!\left(\frac{x_{sw}}{x_{sw}+x_{ss}}\right) \tag{9a}$$

$$Q_{sso} = u_{CFF}\!\left(\frac{x_{ss}}{x_{sw}+x_{ss}}\right) \tag{9b}$$

$$Q_{sfo} = u_{CFF}\!\left(\frac{x_{sf}}{x_{sw}+x_{ss}}\right) \tag{9c}$$

### Sump Metrics (Equations 10–11)

$$y_{SLEV} = 100\,\frac{x_{ss}+x_{sw}}{v_{sump}} \tag{10}$$

$$y_\rho = \frac{\rho_w Q_{swo} + \rho_o Q_{sso}}{Q_{swo}+Q_{sso}} \tag{11}$$

where $v_{sump}$ (m³) is the physical volume of the sump, $y_{SLEV}$ (%) is the sump
slurry fill level, and $y_\rho$ (t/m³) is the sump discharge density.

---

## Cyclone Cluster Model

The cyclone cluster is modelled as a single classifier. The aim is to calculate the
total water, solids and fines split at the cluster. Define the feed fractions:

$$F_i = \frac{Q_{sso}}{u_{CFF}}, \qquad P_i = \frac{Q_{sfo}}{Q_{sso}}$$

### Coarse Underflow (Equation 12)

$$Q_{ccu} = (Q_{sso}-Q_{sfo})\!\left(1-C_1\exp\!\left(-\frac{u_{CFF}}{\varepsilon_c}\right)\right)
  \left(1-\left(\frac{F_i}{C_2}\right)^{\!C_3}\right)\!(1-P_i^{C_3}) \tag{12}$$

where $C_1 = 0.70$ relates to the split at low flows, $C_2 = 0.70$ normalises the
fraction solids in the feed, $C_3$ is an integer adjusting the sharpness of the
dependency on $F_i$ and $P_i$, and $\varepsilon_c$ (m³/h) relates to the coarse split
at the cyclone.

### Fraction of Solids in the Underflow (Equations 13–14)

$$F_u = \frac{Q_{csu}}{Q_{csu}+Q_{cwu}} \tag{13}$$

$$F_u = C_2 - (C_2-F_i)\exp\!\left(-\frac{Q_{ccu}}{\alpha_{su}\,\varepsilon_c}\right) \tag{14}$$

where $\alpha_{su}$ is a parameter related to the fraction of solids in the cyclone
underflow.

### Cyclone Underflow Flow-Rates (Equations 15a–15c)

$$Q_{cwu} = \frac{Q_{swo}\,(Q_{ccu}-F_u Q_{ccu})}{F_u Q_{swo}+F_u Q_{sfo}-Q_{sfo}} \tag{15a}$$

$$Q_{cfu} = \frac{Q_{sfo}\,(Q_{ccu}-F_u Q_{ccu})}{F_u Q_{swo}+F_u Q_{sfo}-Q_{sfo}} \tag{15b}$$

$$Q_{csu} = Q_{ccu}+Q_{cfu} \tag{15c}$$

### Product Particle Size (Equation 16)

$$y_{PSE} = 100\!\left(\frac{Q_{cfo}}{Q_{cso}}\right) \tag{16}$$

where $Q_{cfo}$ and $Q_{cso}$ are the cyclone overflow fines and solids flow-rates,
respectively, and $y_{PSE}$ (%) is the product particle size estimate passing 75 µm.

---

## State-Space Representation (Section 2.3)

The grinding mill circuit is formulated as:

$$\frac{d}{dt}\mathbf{x} = \mathbf{f}(t,\mathbf{x},\mathbf{u},\mathbf{p}) \tag{17a}$$

$$\mathbf{y} = \mathbf{h}(t,\mathbf{x},\mathbf{u},\mathbf{p}) \tag{17b}$$

where $\mathbf{p}$ is the model parameter vector (Table 2). The dynamic function
$\mathbf{f}(\cdot)$ collects equations (1) and (8); the output function $\mathbf{h}(\cdot)$
collects equations (5), (6), (10), (11), and (16).

### State Vector

Seven volume states — four in the mill and three in the sump:

$$\mathbf{x} = \begin{bmatrix}
x_{mw} \\ x_{ms} \\ x_{mr} \\ x_{mf} \\ x_{sw} \\ x_{ss} \\ x_{sf}
\end{bmatrix}$$

| Symbol | Python name | Unit | Description |
|---|---|---|---|
| $x_{mw}$ | `water_volume` | m³ | Volume of water in the mill |
| $x_{ms}$ | `solids_volume` | m³ | Volume of solids in the mill |
| $x_{mr}$ | `rock_volume` | m³ | Volume of rocks in the mill |
| $x_{mf}$ | `fines_volume` | m³ | Volume of fines in the mill |
| $x_{sw}$ | `sump_water_volume` | m³ | Volume of water in the sump |
| $x_{ss}$ | `sump_solids_volume` | m³ | Volume of solids in the sump |
| $x_{sf}$ | `sump_fines_volume` | m³ | Volume of fines in the sump |

### Input Vector

Five manipulated variables:

$$\mathbf{u} = \begin{bmatrix}
u_{MFO} \\ u_{rMIW} \\ u_{\varphi_c} \\ u_{SFW} \\ u_{CFF}
\end{bmatrix}$$

| Symbol | Python name | Unit | Description |
|---|---|---|---|
| $u_{MFO}$ | `feed_ore_rate` | t/h | Mill feed ore |
| $u_{rMIW}$ | `water_ore_ratio` | — | Ratio of mill inlet water to feed ore |
| $u_{\varphi_c}$ | `critical_speed_fraction` | — | Fraction of critical mill speed |
| $u_{SFW}$ | `sump_feed_water` | m³/h | Sump feed water flow rate |
| $u_{CFF}$ | `cyclone_feed_flow` | m³/h | Cyclone feed flow rate |

### Output Vector

Five measured variables:

$$\mathbf{y} = \begin{bmatrix}
y_{JT} \\ y_{Pmill} \\ y_{SLEV} \\ y_\rho \\ y_{PSE}
\end{bmatrix}$$

| Symbol | Python name | Unit | Description |
|---|---|---|---|
| $y_{JT}$ | `charge_fill_fraction` | — | Fraction of mill filled with charge |
| $y_{Pmill}$ | `mill_power` | MW | Power draw of the mill |
| $y_{SLEV}$ | `sump_level` | % | Sump slurry fill level |
| $y_\rho$ | `sump_density` | t/m³ | Sump discharge density |
| $y_{PSE}$ | `product_size` | % | Product particle size estimate ($< 75\,\mu$m) |

---

## Model Parameters (Table 2)

| Symbol | Python name | Unit | Description |
|---|---|---|---|
| $\rho_b$ | `rho_balls` | t/m³ | Density of balls |
| $\rho_{mc}$ | `rho_charge` | t/m³ | Density of mill charge |
| $\rho_o$ | `rho_ore` | t/m³ | Density of ore |
| $\rho_w$ | `rho_water` | t/m³ | Density of water |
| $\alpha_f$ | `alpha_fines` | — | Mass fraction of fines in the feed ore |
| $\alpha_r$ | `alpha_rocks` | — | Mass fraction of rocks in the feed ore |
| $\delta_s$ | `delta_solids` | — | Power parameter for fraction solids in the mill |
| $\delta_v$ | `delta_volume` | — | Power parameter for volume of mill filled |
| $d_q$ | `discharge_rate` | h⁻¹ | Discharge rate |
| $\varepsilon_0$ | `epsilon_zero` | — | Maximum fraction of solids by volume in slurry at zero slurry flow |
| $\varepsilon_p$ | `charge_porosity` | — | Porosity of the mill charge |
| $\varphi_N$ | `phi_norm` | — | Rheology normalisation factor |
| $J_B$ | `ball_fill_fraction` | — | Fraction of mill filled with steel balls |
| $J_{TP_{max}}$ | `fill_fraction_max_power` | — | Fraction of mill filled at maximum power draw |
| $K_{FP}$ | `k_fines_production` | MWh/t | Fines production factor |
| $K_{FP_{JT}}$ | `k_fines_production_jt` | — | Fractional change in fines production factor per unit change in fractional mill filling |
| $K_{RC}$ | `k_rock_consumption` | MWh/t | Rock consumption factor |
| $P_{max}$ | `power_max` | MW | Maximum mill power draw |
| $S$ | `discharge_solids_fraction` | — | Mill discharge volumetric solids content |
| $U$ | `charge_voidage` | — | Voidage in the mill charge |
| $v_{mill}$ | `mill_volume` | m³ | Mill volume |
| $v_{sump}$ | `sump_volume` | m³ | Sump volume |
| $\alpha_{su}$ | `cyclone_alpha_underflow` | — | Parameter related to fraction solids in cyclone underflow |
| $C_1$ | `cyclone_c1` | — | Cyclone model constant |
| $C_2$ | `cyclone_c2` | — | Cyclone model constant |
| $C_3$ | `cyclone_c3` | — | Cyclone model constant (integer) |
| $\varepsilon_c$ | `cyclone_epsilon_c` | m³/h | Parameter related to coarse split at cyclone |

