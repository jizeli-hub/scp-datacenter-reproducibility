# Joint SLA Provisioning: Reproduction Scripts

These scripts rerun the controlled CPU-memory simulation (S1) at the 95% service target and reproduce the four-method paired comparison in Table 3. No input dataset is needed. The real-trace and actuation experiments are outside this repository's scope.

Use Python 3.12 in a virtual environment. The commands below keep generated results outside the checkout.

```sh
python -m pip install numpy==2.3.5 pandas==2.2.3
python -B scripts/run_s1.py --output ../scp-repro-output/s1-384.csv
python -B scripts/reproduce_table3.py --source ../scp-repro-output/s1-384.csv --output-dir ../scp-repro-output
```

The simulation uses 100 seeds and 384 calibration hours by default. Use `--calibration-hours 48`, `96`, or `192` to generate other S1 runs. Table 3 uses only the 384-hour run: it restricts all four methods to the same 68 seeds based on their calibration admission statuses, then reports paired differences in shortage frequency and normalized mismatch cost with 95% bootstrap intervals. See `table3_recomputed.csv` in the output directory.
