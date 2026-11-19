"""Reproduce Table 3 from the S1 seed-level results.

Membership uses recorded calibration-only admission statuses, not outer outcomes.
This reporting-only calculation does not rerun the simulator or the controller.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ECRA = "ECRA-JSLA"
COMPARATORS = ("DirectJointConformal", "EqualResourceWise", "Fixed25CPU75Memory")
METHODS = (ECRA,) + COMPARATORS
IDENTITY = ("regime", "n_cal", "seed", "rng_seed", "service_level", "alpha_joint",
            "weight_cpu", "weight_memory", "method", "test_samples")
ELIGIBILITY = IDENTITY + ("admission_status",)
STATUS = {"SCREEN-PASS": "ADMIT", "STABILITY-WARNING": "MONITOR",
          "EVIDENCE-INSUFFICIENT": "DEFER", "ADMIT": "ADMIT",
          "MONITOR": "MONITOR", "DEFER": "DEFER"}
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPLICATES = 2000


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result.")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def primary_cell(rows):
    cell = [r for r in rows if r["regime"] == "S1" and int(r["n_cal"]) == 384
            and abs(float(r["service_level"]) - .95) < 1e-12
            and abs(float(r["weight_cpu"]) - .5) < 1e-12
            and abs(float(r["weight_memory"]) - .5) < 1e-12
            and r["method"] in METHODS]
    keys = [(int(r["seed"]), r["method"]) for r in cell]
    expected = {(s, m) for s in range(100) for m in METHODS}
    if len(keys) != 400 or set(keys) != expected:
        raise ValueError("Expected exactly one row per method and seed 0..99.")
    for r in cell:
        if r["outer_test_used_for_selection"].lower() not in ("false", "0"):
            raise ValueError("Source does not attest selection without outer-test use.")
    for seed in range(100):
        group = [r for r in cell if int(r["seed"]) == seed]
        for field in IDENTITY:
            if field not in ("seed", "method") and len({r[field] for r in group}) != 1:
                raise ValueError(f"Unpaired scientific path: seed={seed}, field={field}")
    return cell


def project_eligibility(cell):
    projected = []
    for row in cell:
        item = {key: row[key] for key in IDENTITY}
        legacy = row.get("screen_status")
        current = row.get("admission_status")
        if legacy is not None and current is not None and STATUS[legacy] != STATUS[current]:
            raise ValueError("Conflicting legacy/current status columns.")
        item["admission_status"] = STATUS[current if current is not None else legacy]
        projected.append(item)
    return projected


def common_membership(eligibility):
    """Fail closed if any outcome or other non-eligibility column enters here."""
    if any(set(row) != set(ELIGIBILITY) for row in eligibility):
        raise ValueError("Membership accepts only the explicit eligibility schema.")
    mapping = {(int(r["seed"]), r["method"]): r["admission_status"] for r in eligibility}
    if len(mapping) != 400 or len(eligibility) != 400:
        raise ValueError("Missing or duplicated eligibility row.")
    result = []
    for seed in range(100):
        statuses = {m: mapping[(seed, m)] for m in METHODS}
        excluded = [m for m in METHODS if statuses[m] != "ADMIT"]
        result.append({"seed": seed, **statuses, "common_support": not excluded,
                       "exclusion_reason": "|".join(excluded) or "NONE"})
    return result


def selected_outcomes(cell, membership):
    seeds = sorted(r["seed"] for r in membership if r["common_support"])
    if len(seeds) != 68:
        raise ValueError(f"Expected frozen 68-seed intersection, found {len(seeds)}.")
    result = []
    for r in sorted(cell, key=lambda x: (int(x["seed"]), x["method"])):
        if int(r["seed"]) not in seeds:
            continue
        violation = float(r["selected_joint_violation"])
        cost = float(r["selected_normalized_joint_cost"])
        count = violation * int(r["test_samples"])
        if (not np.isfinite([violation, cost]).all() or not 0 <= violation <= 1
                or abs(count - round(count)) > 1e-10):
            raise ValueError("Invalid selected outcome or nonintegral violation count.")
        result.append({"seed": int(r["seed"]), "rng_seed": r["rng_seed"],
                       "method": r["method"], "test_samples": int(r["test_samples"]),
                       "joint_violation_count": round(count), "joint_violation": violation,
                       "normalized_cost": cost, "alpha_cpu": r["selected_alpha_cpu"],
                       "alpha_memory": r["selected_alpha_memory"],
                       "decision_fingerprint": r["decision_fingerprint"]})
    seed_lists = [[r["seed"] for r in result if r["method"] == m] for m in METHODS]
    if any(values != seeds for values in seed_lists):
        raise ValueError("Methods do not have identical ordered seed sets.")
    return result


def summarize(outcomes):
    seed_lists = [[r["seed"] for r in outcomes if r["method"] == m] for m in METHODS]
    seeds = sorted(seed_lists[0])
    if len(seeds) != 68 or len(set(seeds)) != 68 or any(sorted(s) != seeds for s in seed_lists):
        raise ValueError("Paired comparisons require the same 68 unique seeds in each arm.")
    by_key = {(r["seed"], r["method"]): r for r in outcomes}
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 68, size=(2000, 68))
    summary, differences, draws = [], [], {}
    ecra_cost = np.array([by_key[s, ECRA]["normalized_cost"] for s in seeds])
    ecra_violation = np.array([by_key[s, ECRA]["joint_violation"] for s in seeds])
    for method in COMPARATORS:
        cost = np.array([by_key[s, method]["normalized_cost"] for s in seeds])
        violation = np.array([by_key[s, method]["joint_violation"] for s in seeds])
        dc, dv = ecra_cost - cost, 100 * (ecra_violation - violation)
        cd, vd = dc[indices].mean(axis=1), dv[indices].mean(axis=1)
        cl, ch = np.quantile(cd, [.025, .975], method="linear")
        vl, vh = np.quantile(vd, [.025, .975], method="linear")
        draws[method + "_cost"] = cd
        draws[method + "_shortage_pp"] = vd
        for s, c, v in zip(seeds, dc, dv):
            differences.append({"seed": s, "comparator": method,
                                "delta_normalized_cost": c, "delta_shortage_pp": v})
        summary.append({"comparator": method, "n_common": 68,
                        "ecra_mean_cost": ecra_cost.mean(), "comparator_mean_cost": cost.mean(),
                        "mean_cost_difference": dc.mean(), "cost_ci_lower": cl, "cost_ci_upper": ch,
                        "ecra_mean_violation": ecra_violation.mean(),
                        "comparator_mean_violation": violation.mean(),
                        "mean_violation_difference_pp": dv.mean(),
                        "violation_ci_lower_pp": vl, "violation_ci_upper_pp": vh,
                        "negative_cost_differences": int((dc < -1e-12).sum()),
                        "zero_cost_differences": int(np.isclose(dc, 0, atol=1e-12, rtol=0).sum()),
                        "positive_cost_differences": int((dc > 1e-12).sum()),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED})
    return summary, differences, draws, indices, seeds


def compare_reference(summary, reference):
    reference = {r["comparator"]: r for r in reference}
    if set(reference) != set(COMPARATORS):
        raise ValueError("Reference has missing or extra comparators.")
    maximum = 0.
    for row in summary:
        for field, value in row.items():
            if field == "comparator":
                continue
            difference = abs(float(value) - float(reference[row["comparator"]][field]))
            maximum = max(maximum, difference)
            if difference > 5e-11:
                raise ValueError(f"Reference mismatch: {row['comparator']} / {field}")
    return maximum


def run(source, output_dir, reference=None):
    cell = primary_cell(read_csv(source))
    membership = common_membership(project_eligibility(cell))
    outcomes = selected_outcomes(cell, membership)
    summary, differences, _, _, seeds = summarize(outcomes)
    if reference is not None:
        max_error = compare_reference(summary, read_csv(reference))
        print(f"Stored summary fields match within {max_error:.3g}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "common_support_membership.csv", membership)
    write_csv(output_dir / "paired_seed_differences.csv", differences)
    write_csv(output_dir / "table3_recomputed.csv", summary)
    print(f"Common support: {len(seeds)} seeds and {len(differences)} paired differences.")
    for r in summary:
        print(f"{r['comparator']}: shortage {r['mean_violation_difference_pp']:+.2f} "
              f"[{r['violation_ci_lower_pp']:+.2f}, {r['violation_ci_upper_pp']:+.2f}] pp; "
              f"cost {r['mean_cost_difference']:+.4f} "
              f"[{r['cost_ci_lower']:+.4f}, {r['cost_ci_upper']:+.4f}]")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    run(args.source, args.output_dir, args.reference)
