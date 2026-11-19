"""Generate per-seed results for the controlled S1 joint-SLA simulation."""

import argparse
from pathlib import Path

import pandas as pd

from optimization.joint_sla.simulation import generate_simulation_series
from optimization.joint_sla.simulation_experiment import evaluate_simulation_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-hours", type=int, choices=(48, 96, 192, 384), default=384)
    args = parser.parse_args()

    rows = []
    for seed in range(100):
        series = generate_simulation_series("S1", n_cal=args.calibration_hours, seed=seed)
        rows.extend(evaluate_simulation_series(
            series, service_levels=(0.95,), weight_settings=((0.50, 0.50),)
        ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} result rows from 100 seeds to {args.output}.")


if __name__ == "__main__":
    main()
