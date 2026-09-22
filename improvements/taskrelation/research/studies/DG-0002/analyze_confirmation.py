"""Analyze DG-0002 matched-seed confirmation runs."""

import json
import math
from pathlib import Path
import statistics
import tempfile
from datetime import datetime, timezone

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

from analyze import (
    EXPERIMENT_NAME,
    PAIRS,
    PHASES,
    REPO_ROOT,
    STUDY_DIR,
    STUDY_ID,
    TASKS,
    TRACKING_URI,
    _diagnostic_comparison,
    _gradient_record,
    _omega_record,
    _run_record,
    _validate_pairing,
)


SEEDS = (0, 1, 2, 3, 4)
METHOD_TAGS = ("wavcse-baseline", "mtrl")
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
}


def _summary(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    if len(values) == 1:
        std = 0.0
        interval = [mean, mean]
    else:
        std = statistics.stdev(values)
        critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
        half_width = critical * std / math.sqrt(len(values))
        interval = [mean - half_width, mean + half_width]
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sample_std": std,
        "t_95_interval": interval,
        "min": min(values),
        "max": max(values),
    }


def _compact_gradient(record):
    return {
        key: value
        for key, value in record.items()
        if key != "records"
    }


def _download_json(client, run_id, artifact_path, destination):
    run_destination = destination / run_id
    run_destination.mkdir(parents=True, exist_ok=True)
    return Path(client.download_artifacts(
        run_id,
        artifact_path,
        dst_path=str(run_destination),
    ))


def _outcome_deltas(baseline, mtrl):
    return {
        checkpoint: {
            metric: (
                mtrl["metrics"][checkpoint][metric]
                - baseline["metrics"][checkpoint][metric]
            )
            for metric in ("aggregate", *TASKS)
        }
        for checkpoint in ("epoch", "opt", "best")
    }


def _method_phase_summary(seed_records, method, phase, field, key):
    values = []
    for seed in SEEDS:
        gradient = seed_records[str(seed)]["gradient_diagnostics"][method]
        values.append(gradient["summary"][phase][field][key]["mean"])
    return _summary(values)


def _cross_seed_summary(seed_records):
    outcomes = {}
    for checkpoint in ("epoch", "opt", "best"):
        outcomes[checkpoint] = {}
        for metric in ("aggregate", *TASKS):
            baseline_values = [
                seed_records[str(seed)]["runs"]["baseline"]["metrics"][checkpoint][metric]
                for seed in SEEDS
            ]
            mtrl_values = [
                seed_records[str(seed)]["runs"]["mtrl"]["metrics"][checkpoint][metric]
                for seed in SEEDS
            ]
            outcomes[checkpoint][metric] = {
                "baseline": _summary(baseline_values),
                "mtrl": _summary(mtrl_values),
                "paired_delta_mtrl_minus_baseline": _summary([
                    mtrl - baseline
                    for baseline, mtrl in zip(baseline_values, mtrl_values)
                ]),
            }

    gradients = {}
    for method in ("baseline", "mtrl"):
        gradients[method] = {}
        for phase in PHASES:
            ratios = [
                seed_records[str(seed)]["gradient_diagnostics"][method]["summary"][phase][
                    "max_min_mean_norm_ratio"
                ]
                for seed in SEEDS
            ]
            er_to_smallest_other = []
            for seed in SEEDS:
                task_norms = seed_records[str(seed)]["gradient_diagnostics"][method][
                    "summary"
                ][phase]["task_norms"]
                er_to_smallest_other.append(
                    task_norms["er"]["mean"]
                    / min(task_norms["ks"]["mean"], task_norms["si"]["mean"])
                )
            gradients[method][phase] = {
                "max_min_mean_norm_ratio": _summary(ratios),
                "er_to_smallest_ks_si_mean_norm_ratio": _summary(
                    er_to_smallest_other
                ),
                "task_norm_means": {
                    task: _method_phase_summary(
                        seed_records, method, phase, "task_norms", task
                    )
                    for task in TASKS
                },
                "pairwise_cosine_means": {
                    pair: _method_phase_summary(
                        seed_records, method, phase, "pairwise_cosines", pair
                    )
                    for pair in PAIRS
                },
                "negative_conflict_frequencies": {
                    pair: _summary([
                        seed_records[str(seed)]["gradient_diagnostics"][method][
                            "summary"
                        ][phase]["pairwise_cosines"][pair]["negative_frequency"]
                        for seed in SEEDS
                    ])
                    for pair in PAIRS
                },
            }

    ratio_deltas = {}
    for phase in PHASES:
        baseline = gradients["baseline"][phase]["max_min_mean_norm_ratio"]["values"]
        mtrl = gradients["mtrl"][phase]["max_min_mean_norm_ratio"]["values"]
        ratio_deltas[phase] = _summary([
            mtrl_value - baseline_value
            for baseline_value, mtrl_value in zip(baseline, mtrl)
        ])

    omega = {
        "final_pair_values": {
            pair: _summary([
                seed_records[str(seed)]["omega"]["final_pair_values"][pair]
                for seed in SEEDS
            ])
            for pair in PAIRS
        },
        "absolute_final_pair_magnitudes": {
            pair: _summary([
                abs(seed_records[str(seed)]["omega"]["final_pair_values"][pair])
                for seed in SEEDS
            ])
            for pair in PAIRS
        },
        "off_diagonal_absolute_range": _summary([
            max(abs(value) for value in seed_records[str(seed)]["omega"][
                "final_pair_values"
            ].values())
            - min(abs(value) for value in seed_records[str(seed)]["omega"][
                "final_pair_values"
            ].values())
            for seed in SEEDS
        ]),
        "off_diagonal_range": _summary([
            seed_records[str(seed)]["omega"]["off_diagonal_range"]
            for seed in SEEDS
        ]),
    }

    checks = {
        "all_exposure_pairings_passed": all(
            seed_records[str(seed)]["pairing_check"]["passed"]
            for seed in SEEDS
        ),
        "git_commits_by_seed": {
            str(seed): {
                method: seed_records[str(seed)]["runs"][method]["git_commit"]
                for method in ("baseline", "mtrl")
            }
            for seed in SEEDS
        },
        "all_runs_same_git_commit": len({
            seed_records[str(seed)]["runs"][method]["git_commit"]
            for seed in SEEDS
            for method in ("baseline", "mtrl")
        }) == 1,
        "all_runs_have_notes": all(
            seed_records[str(seed)]["runs"][method]["standard_tags"][
                "mlflow.note.content"
            ]
            for seed in SEEDS
            for method in ("baseline", "mtrl")
        ),
        "baseline_norm_dominance_phases_by_seed": {
            str(seed): seed_records[str(seed)]["diagnostics"][
                "norm_dominance_phases"
            ]
            for seed in SEEDS
        },
        "mtrl_norm_dominance_phases_by_seed": {
            str(seed): [
                phase
                for phase in PHASES
                if seed_records[str(seed)]["gradient_diagnostics"]["mtrl"][
                    "summary"
                ][phase]["max_min_mean_norm_ratio"] >= 3.0
            ]
            for seed in SEEDS
        },
        "persistent_conflicts_by_seed": {
            str(seed): seed_records[str(seed)]["diagnostics"][
                "persistent_conflicts"
            ]
            for seed in SEEDS
        },
        "omega_magnitude_saturation_by_seed": {
            str(seed): (
                min(abs(value) for value in seed_records[str(seed)]["omega"][
                    "final_pair_values"
                ].values()) >= 0.30
                and (
                    max(abs(value) for value in seed_records[str(seed)]["omega"][
                        "final_pair_values"
                    ].values())
                    - min(abs(value) for value in seed_records[str(seed)]["omega"][
                        "final_pair_values"
                    ].values())
                ) <= 0.01
            )
            for seed in SEEDS
        },
        "uniform_positive_omega_by_seed": {
            str(seed): (
                seed_records[str(seed)]["omega"]["off_diagonal_range"] <= 0.01
                and min(seed_records[str(seed)]["omega"]["final_pair_values"].values())
                >= 0.30
            )
            for seed in SEEDS
        },
        "omega_sign_pattern_by_seed": {
            str(seed): {
                pair: (
                    1 if value > 0.0 else -1 if value < 0.0 else 0
                )
                for pair, value in seed_records[str(seed)]["omega"][
                    "final_pair_values"
                ].items()
            }
            for seed in SEEDS
        },
    }

    return {
        "outcomes": outcomes,
        "gradients": gradients,
        "paired_norm_ratio_delta_mtrl_minus_baseline": ratio_deltas,
        "omega": omega,
        "confirmation_checks": checks,
    }

def _study_decision(cross_seed):
    checks = cross_seed["confirmation_checks"]
    required_phases = {"middle", "late"}
    baseline_replicated = all(
        required_phases.issubset(phases)
        for phases in checks["baseline_norm_dominance_phases_by_seed"].values()
    )
    mtrl_failed_to_remove = all(
        required_phases.issubset(phases)
        for phases in checks["mtrl_norm_dominance_phases_by_seed"].values()
    )
    persistent_conflict_replicated = any(
        conflicts
        for conflicts in checks["persistent_conflicts_by_seed"].values()
    )
    decision = (
        "CONFIRMED"
        if (
            checks["all_exposure_pairings_passed"]
            and baseline_replicated
            and mtrl_failed_to_remove
        )
        else "INCONCLUSIVE"
    )
    return {
        "decision": decision,
        "decision_basis": {
            "baseline_norm_dominance_replicated_in_middle_and_late_all_seeds": (
                baseline_replicated
            ),
            "mtrl_failed_to_remove_norm_dominance_all_seeds": (
                mtrl_failed_to_remove
            ),
            "persistent_pairwise_conflict_replicated": (
                persistent_conflict_replicated
            ),
            "omega_magnitude_saturated_all_seeds": all(
                checks["omega_magnitude_saturation_by_seed"].values()
            ),
            "uniform_positive_omega_all_seeds": all(
                checks["uniform_positive_omega_by_seed"].values()
            ),
            "scope": (
                "Confirms an optimization-scale diagnostic under the matched "
                "smp25 protocol; does not establish causation or an ER "
                "performance effect."
            ),
        },
    }




def main():
    load_dotenv(REPO_ROOT / ".env")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Missing MLflow experiment {EXPERIMENT_NAME}")

    runs = client.search_runs(
        [experiment.experiment_id],
        f"tags.study_id = '{STUDY_ID}' and tags.stage = 'confirm'",
        max_results=20,
    )
    by_key = {}
    for run in runs:
        key = (run.data.tags.get("method"), int(run.data.tags["seed"]))
        if key in by_key:
            raise RuntimeError(f"Duplicate confirmation run for {key}")
        by_key[key] = run

    expected = {
        (method, seed)
        for method in METHOD_TAGS
        for seed in SEEDS
    }
    if set(by_key) != expected:
        raise RuntimeError(
            f"Expected confirmation runs {sorted(expected)}, found {sorted(by_key)}"
        )
    unfinished = [
        key for key, run in by_key.items() if run.info.status != "FINISHED"
    ]
    if unfinished:
        raise RuntimeError(f"Confirmation runs are not finished: {unfinished}")

    seed_records = {}
    with tempfile.TemporaryDirectory(dir=STUDY_DIR) as temporary_directory:
        destination = Path(temporary_directory)
        for seed in SEEDS:
            baseline_run = by_key[("wavcse-baseline", seed)]
            mtrl_run = by_key[("mtrl", seed)]
            baseline_gradient = _gradient_record(_download_json(
                client,
                baseline_run.info.run_id,
                "results/gradient_diagnostics.json",
                destination,
            ))
            mtrl_gradient = _gradient_record(_download_json(
                client,
                mtrl_run.info.run_id,
                "results/gradient_diagnostics.json",
                destination,
            ))
            omega = _omega_record(_download_json(
                client,
                mtrl_run.info.run_id,
                "results/omega_history.json",
                destination,
            ), mtrl_gradient)
            baseline_gradient["artifact"] = (
                f"mlflow://{baseline_run.info.run_id}/results/gradient_diagnostics.json"
            )
            mtrl_gradient["artifact"] = (
                f"mlflow://{mtrl_run.info.run_id}/results/gradient_diagnostics.json"
            )
            omega["artifact"] = (
                f"mlflow://{mtrl_run.info.run_id}/results/omega_history.json"
            )

            run_records = {
                "baseline": _run_record(client, baseline_run),
                "mtrl": _run_record(client, mtrl_run),
            }
            seed_records[str(seed)] = {
                "runs": run_records,
                "outcome_delta_mtrl_minus_baseline": _outcome_deltas(
                    run_records["baseline"], run_records["mtrl"]
                ),
                "pairing_check": _validate_pairing(
                    baseline_gradient, mtrl_gradient
                ),
                "gradient_diagnostics": {
                    "baseline": _compact_gradient(baseline_gradient),
                    "mtrl": _compact_gradient(mtrl_gradient),
                },
                "diagnostics": _diagnostic_comparison(
                    baseline_gradient, mtrl_gradient
                ),
                "omega": omega,
            }

    cross_seed = _cross_seed_summary(seed_records)
    decision = _study_decision(cross_seed)

    result = {
        "study_id": STUDY_ID,
        "title": "Exposure-controlled gradient compatibility confirmation",
        "type": "diagnostic",
        "stage": "confirm",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "protocol": {
            "task_set": "ks_si_er",
            "pooling": {"type": "smp", "parameter": 0.5},
            "layers": "all",
            "epochs": 30,
            "batch_size": 2048,
            "primary_checkpoint": "epoch",
            "independent_unit": "seed-level phase summary",
            "er_protocol": "ordinary speaker-leaky split; no ER performance claim",
        },
        "seed_records": seed_records,
        "cross_seed": cross_seed,
        **decision,
    }
    output_path = STUDY_DIR / "confirmation_result.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)

    print(json.dumps({
        "result": str(output_path),
        "run_ids": {
            f"{method}:s{seed}": run.info.run_id
            for (method, seed), run in sorted(by_key.items())
        },
        "decision": decision,
        "confirmation_checks": result["cross_seed"]["confirmation_checks"],
        "middle_ratio": {
            method: result["cross_seed"]["gradients"][method]["middle"][
                "max_min_mean_norm_ratio"
            ]
            for method in ("baseline", "mtrl")
        },
        "late_ratio": {
            method: result["cross_seed"]["gradients"][method]["late"][
                "max_min_mean_norm_ratio"
            ]
            for method in ("baseline", "mtrl")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
