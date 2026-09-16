"""Write paper-point model calculations to a readable CSV report."""

import csv
from pathlib import Path

import numpy as np

from model import (
    INPUT_NAMES,
    INPUTS_NOP,
    OUTPUT_NAMES,
    OUTPUTS_NOP,
    STATE_NAMES,
    STATES_NOP,
    build_grinding_circuit_model,
)

UNITS = {
    "feed_ore_rate": "t/h",
    "water_ore_ratio": "-",
    "critical_speed_fraction": "-",
    "sump_feed_water": "m3/h",
    "cyclone_feed_flow": "m3/h",
    "water_volume": "m3",
    "solids_volume": "m3",
    "rock_volume": "m3",
    "fines_volume": "m3",
    "sump_water_volume": "m3",
    "sump_solids_volume": "m3",
    "sump_fines_volume": "m3",
    "charge_fill_fraction": "-",
    "mill_power": "MW",
    "sump_level": "%",
    "sump_density": "t/m3",
    "product_size": "%",
}


def make_rows():
    model = build_grinding_circuit_model()
    x = np.array([STATES_NOP[name] for name in STATE_NAMES])
    u = np.array([INPUTS_NOP[name] for name in INPUT_NAMES])
    rhs = np.asarray(model.f(0.0, x, u)).ravel()
    outputs = np.asarray(model.h(0.0, x, u)).ravel()
    rows = []
    for name in INPUT_NAMES:
        rows.append(["input", name, UNITS[name], INPUTS_NOP[name], "", ""])
    for name in STATE_NAMES:
        index = STATE_NAMES.index(name)
        rows.append(["state", name, UNITS[name], STATES_NOP[name], "", ""])
        rows.append(
            ["rhs", f"d_{name}_dt", "m3/h", rhs[index], 0.0, rhs[index]]
        )
    for name, value in zip(OUTPUT_NAMES, outputs):
        paper_value = OUTPUTS_NOP[name]
        rows.append(
            [
                "output",
                name,
                UNITS[name],
                value,
                paper_value,
                value - paper_value,
            ]
        )
    return rows


def main(output_path="results/model_test_results.csv"):
    path = Path(output_path)
    with path.open("w", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            ["category", "variable", "units", "computed", "paper", "difference"]
        )
        writer.writerows(make_rows())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
