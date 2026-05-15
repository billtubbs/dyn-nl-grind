# Grinding Circuit Simulation Model

These formulas are entirely based on the following research paper and reflect the model described in Section 2.2: Process Model. The equations and parameter definitions below are sourced directly from the paper text and notation.

Reference:
- J.D. le Roux and C.W. Steyn, Validation of a dynamic non-linear grinding circuit model for
process control, *Minerals Engineering* 187 (2022) 107780.

## Mill Model

### Mill mass balances

- $\displaystyle \frac{dx_{ms}}{dt} = \frac{1 - \alpha_r}{\rho_o} u_{MFO} - Q_{mso} + Q_{csu} + Q_{RC}$
- $\displaystyle \frac{dx_{mr}}{dt} = \frac{\alpha_r}{\rho_o} u_{MFO} - Q_{RC}$
- $\displaystyle \frac{dx_{mf}}{dt} = \frac{\alpha_f}{\rho_o} u_{MFO} - Q_{mfo} + Q_{cfu} + Q_{FP}$

### Mill discharge flow-rates

- $\displaystyle Q_{mwo} = \phi \, d_q \, \frac{x_{mw}}{x_{ms} + x_{mw}}$
- $\displaystyle Q_{mso} = \phi \, d_q \, \frac{x_{mw} \, x_{ms}}{x_{ms} + x_{mw}}$
- $\displaystyle Q_{mfo} = \phi \, d_q \, \frac{x_{mw} \, x_{mf}}{x_{ms} + x_{mw}}$

### Rheology factor

- $\displaystyle \phi = \begin{cases}
1 - \left(\varepsilon_0 - 1 - \frac{x_{ms}}{x_{mw}}\right)^{-1}, & \frac{x_{mw}}{x_{ms}} \le \varepsilon_0 - 1 \\
0, & \frac{x_{mw}}{x_{ms}} > \varepsilon_0 - 1
\end{cases}$

### Rock consumption and fines production

- $\displaystyle Q_{RC} = \frac{x_{mr} \, y_{Pmill}}{\rho_o \, K_{RC} \left(x_{mr} + x_{ms}\right)}$
- $\displaystyle Q_{FP} = \frac{y_{Pmill}}{\rho_o \, K_{FP} \left(1 + K_{FPJT} y_{JT} - J_{TPmax}\right)}$

### Mill charge fraction

- $\displaystyle y_{JT} = \frac{x_{mw} + x_{ms} + x_{mr} + x_{mb}}{v_{mill}}$

### Mill power draw

- $\displaystyle y_{Pmill} = P_{max} \, u_{\phi c} \, \left[ \frac{1 - \delta_v}{y_{JT}^{-1} - 1} - \delta_s \frac{\phi}{J_{TPmax}} \phi_N \right]^2$

### Mill charge density

- $\displaystyle \rho_{mc} = \frac{\rho_o (1 - \varepsilon_p + \varepsilon_p U S) + JB (\rho_b - \rho_o)(1 - \varepsilon_p) + \varepsilon_p U (1 - S)}{y_{JT}}$

## Sump Model

### Sump mass balances

- $\displaystyle \frac{dx_{sw}}{dt} = Q_{mwo} - Q_{swo} + u_{SFW}$
- $\displaystyle \frac{dx_{ss}}{dt} = Q_{mso} - Q_{sso}$
- $\displaystyle \frac{dx_{sf}}{dt} = Q_{mfo} - Q_{sfo}$

### Sump discharge flow-rates

- $\displaystyle Q_{swo} = u_{CFF} \, \frac{x_{sw}}{x_{sw} + x_{ss}}$
- $\displaystyle Q_{sso} = u_{CFF} \, \frac{x_{ss}}{x_{sw} + x_{ss}}$
- $\displaystyle Q_{sfo} = u_{CFF} \, \frac{x_{sf}}{x_{sw} + x_{ss}}$

### Sump metrics

- $\displaystyle y_{SLEV} = 100 \, \frac{x_{ss} + x_{sw}}{v_{sump}}$
- $\displaystyle y_{\rho} = \frac{\rho_w Q_{swo} + \rho_o Q_{sso}}{Q_{swo} + Q_{sso}}$

## Cyclone Cluster Model

### Underflow and split equations

- $\displaystyle Q_{ccu} = Q_{sso} - Q_{sfo} \left[ 1 - C_1 \exp\left(-\frac{u_{CFF}}{\varepsilon_c} \times \left(\frac{F_i^3}{1 - (1 - P_i^{C_3})^{-1}}\right)\right) \right]$

- $\displaystyle F_u = \frac{Q_{csu}}{Q_{csu} + Q_{cwu}}$

- $\displaystyle F_u = C_2 - (C_2 - F_i) \exp\left(-\frac{Q_{ccu}}{\alpha_{su} \varepsilon_c}\right)$

- $\displaystyle Q_{cwu} = \frac{Q_{swo} \left(Q_{ccu} - F_u Q_{ccu}\right)}{F_u Q_{swo} + F_u Q_{sfo} - Q_{sfo}}$
- $\displaystyle Q_{cfu} = \frac{Q_{sfo} \left(Q_{ccu} - F_u Q_{ccu}\right)}{F_u Q_{swo} + F_u Q_{sfo} - Q_{sfo}}$
- $\displaystyle Q_{csu} = Q_{ccu} + Q_{cfu}$

### Product specification

- $\displaystyle y_{PSE} = 100 \, \frac{Q_{cfo}}{Q_{cso}}$

## State-space Representation

The state-space model is defined in Section 2.3 using the mill and sump balance equations from (1) and (8), with outputs given by (5), (6), (10), (11), and (16).

### State vector

- $\displaystyle x = \begin{bmatrix} x_{mw} & x_{ms} & x_{mr} & x_{mf} & x_{sw} & x_{ss} & x_{sf} \end{bmatrix}^T$

### Input vector

- $\displaystyle u = \begin{bmatrix} u_{MFO} & u_{rMIW} & u_{\phi c} & u_{SFW} & u_{CFF} \end{bmatrix}^T$

### Output vector

- $\displaystyle y = \begin{bmatrix} y_{JT} & y_{Pmill} & y_{SLEV} & y_{\rho} & y_{PSE} \end{bmatrix}^T$

### State-space equations

- $\displaystyle \frac{dx}{dt} = f(t, x, u, p)$
- $\displaystyle y = h(t, x, u, p)$

Here, $p$ contains the model parameters listed in Table 2. The function $f(\cdot)$ collects the dynamic mill and sump mass-balance equations, and $h(\cdot)$ collects the algebraic output relationships.

## Model Parameters (Table 2)

| Parameter | Python name | Unit | Description |
|---|---|---|---|
| $\rho_b$ | `rho_balls` | t/m³ | Density of balls |
| $\rho_{mc}$ | `rho_charge` | t/m³ | Density of mill charge |
| $\rho_o$ | `rho_ore` | t/m³ | Density of ore |
| $\rho_w$ | `rho_water` | t/m³ | Density of water |
| $\alpha_f$ | `alpha_fines` | - | Mass fraction of fines in the feed ore |
| $\alpha_r$ | `alpha_rocks` | - | Mass fraction of rocks in the feed ore |
| $\delta_s$ | `delta_solid_fraction` | - | Power parameter for fraction solids in the mill |
| $\delta_v$ | `delta_fill_volume` | - | Power parameter for volume of mill filled |
| $d_q$ | `discharge_rate` | h⁻¹ | Discharge rate |
| $\varepsilon_0$ | `epsilon_zero` | - | Maximum fraction of solids by volume slurry at zero slurry flow |
| $\varepsilon_p$ | `porosity` | - | Porosity of the mill charge |
| $\phi_N$ | `phi_normalization` | - | Rheology normalisation factor |
| $J_B$ | `ball_fill_fraction` | - | Fraction of mill filled with steel balls |
| $J_{TPmax}$ | `fill_fraction_max_power` | - | Fraction of mill filled at maximum power draw |
| $K_{FP}$ | `k_fines_production` | MWh/t | Fines production factor |
| $K_{FPJT}$ | `k_fpjt` | - | Fractional change in fines production factor per change in fractional mill filling |
| $K_{RC}$ | `k_rock_consumption` | MWh/t | Rock consumption factor |
| $P_{max}$ | `p_max` | MW | Maximum mill power draw |
| $S$ | `discharge_solids_content` | - | Mill discharge volumetric solids content |
| $U$ | `charge_voidage` | - | Voidage in the mill charge |
| $v_{mill}$ | `mill_volume` | m³ | Mill volume |
| $v_{sump}$ | `sump_volume` | m³ | Sump volume |
| $\alpha_{su}$ | `cyclone_alpha_underflow` | - | Parameter related to fraction solids in cyclone underflow |
| $C_1$ | `cyclone_c1` | - | Cyclone model constant |
| $C_2$ | `cyclone_c2` | - | Cyclone model constant |
| $C_3$ | `cyclone_c3` | - | Cyclone model constant |
| $\varepsilon_c$ | `cyclone_epsilon_c` | m³/h | Parameter related to coarse split at cyclone |

## State Variables

| Symbol | Python name | Description |
|---|---|---|
| $x_{mw}$ | `water_volume` | Volume of water in the mill |
| $x_{ms}$ | `solids_volume` | Volume of solids in the mill |
| $x_{mr}$ | `rock_volume` | Volume of rocks in the mill |
| $x_{mf}$ | `fines_volume` | Volume of fines in the mill |
| $x_{sw}$ | `sump_water_volume` | Volume of water in the sump |
| $x_{ss}$ | `sump_solids_volume` | Volume of solids in the sump |
| $x_{sf}$ | `sump_fines_volume` | Volume of fines in the sump |
