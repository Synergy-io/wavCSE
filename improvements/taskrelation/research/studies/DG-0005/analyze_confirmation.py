"""Analyze DG-0005 matched-seed A0/A1 confirmation runs."""

import json
import math
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from analyze import (
    EXPERIMENT_NAME,
    METHODS,
    PHASES,
    REPO_ROOT,
    STUDY_DIR,
    STUDY_ID,
    TASKS,
    TRACKING_URI,
    _exposure_check,
    _gradient_record,
    _metric,
    _run_record,
)


SEEDS = (0, 1, 2, 3, 4)
PAIRS = ("ks_si", "ks_er", "si_er")
CONFIRMATION_COMMIT = "8032a937050d8bbd3114b172cb813a8fc7370b37"
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
        sample_std = 0.0
        interval = [mean, mean]
    else:
        sample_std = statistics.stdev(values)
        critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
        half_width = critical * sample_std / math.sqrt(len(values))
        interval = [mean - half_width, mean + half_width]
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sample_std": sample_std,
        "t_95_interval": interval,
        "min": min(values),
        "max": max(values),
    }


def _download_gradient(client, run, destination):
    run_destination = destination / run.info.run_id
    run_destination.mkdir(parents=True, exist_ok=True)
    path = Path(client.download_artifacts(
        run.info.run_id,
        "results/gradient_diagnostics.json",
        dst_path=str(run_destination),
    ))
    record = _gradient_record(path)
    record["artifact"] = (
        f"mlflow://{run.info.run_id}/results/gradient_diagnostics.json"
    )
    return record


def _full_run_record(client, run):
    record = _run_record(client, run)
    record["metrics"] = {
        checkpoint: {
            "aggregate": _metric(run, f"test_{checkpoint}_acc_all"),
            **{
                task: _metric(run, f"test_{checkpoint}_{task}_acc")
                for task in TASKS
            },
        }
        for checkpoint in ("epoch", "opt", "best")
    }
    record["runtime_device_index"] = run.data.tags.get("runtime_device_index")
    return record


def _phase_value(seed_records, seed, arm, phase, field, key):
    return seed_records[str(seed)]["gradients"][arm]["summary"][phase][field][key]


def _cross_seed_summary(seed_records):
    gradients = {}
    for arm in ("a0", "a1"):
        gradients[arm] = {}
        for phase in PHASES:
            ratios = [
                seed_records[str(seed)]["gradients"][arm]["summary"][phase][
                    "max_min_mean_norm_ratio"
                ]
                for seed in SEEDS
            ]
            gradients[arm][phase] = {
                "max_min_mean_norm_ratio": _summary(ratios),
                "er_to_smallest_ratio": _summary([
                    seed_records[str(seed)]["gradients"][arm][
                        "er_to_smallest_ratio"
                    ][phase]
                    for seed in SEEDS
                ]),
                "task_norm_means": {
                    task: _summary([
                        _phase_value(
                            seed_records, seed, arm, phase, "task_norms", task
                        )["mean"]
                        for seed in SEEDS
                    ])
                    for task in TASKS
                },
                "pairwise_cosine_means": {
                    pair: _summary([
                        _phase_value(
                            seed_records,
                            seed,
                            arm,
                            phase,
                            "pairwise_cosines",
                            pair,
                        )["mean"]
                        for seed in SEEDS
                    ])
                    for pair in PAIRS
                },
                "negative_conflict_frequencies": {
                    pair: _summary([
                        _phase_value(
                            seed_records,
                            seed,
                            arm,
                            phase,
                            "pairwise_cosines",
                            pair,
                        )["negative_frequency"]
                        for seed in SEEDS
                    ])
                    for pair in PAIRS
                },
            }

    paired_ratio_deltas = {
        phase: _summary([
            gradients["a1"][phase]["max_min_mean_norm_ratio"]["values"][index]
            - gradients["a0"][phase]["max_min_mean_norm_ratio"]["values"][index]
            for index in range(len(SEEDS))
        ])
        for phase in PHASES
    }
    paired_task_norm_deltas = {
        phase: {
            task: _summary([
                gradients["a1"][phase]["task_norm_means"][task]["values"][index]
                - gradients["a0"][phase]["task_norm_means"][task]["values"][index]
                for index in range(len(SEEDS))
            ])
            for task in TASKS
        }
        for phase in PHASES
    }

    outcomes = {}
    for checkpoint in ("epoch", "opt", "best"):
        outcomes[checkpoint] = {}
        for metric in ("aggregate", *TASKS):
            a0_values = [
                seed_records[str(seed)]["runs"]["a0"]["metrics"][checkpoint][metric]
                for seed in SEEDS
            ]
            a1_values = [
                seed_records[str(seed)]["runs"]["a1"]["metrics"][checkpoint][metric]
                for seed in SEEDS
            ]
            outcomes[checkpoint][metric] = {
                "a0": _summary(a0_values),
                "a1": _summary(a1_values),
                "paired_delta_a1_minus_a0": _summary([
                    a1 - a0 for a0, a1 in zip(a0_values, a1_values)
                ]),
            }

    train_val_gaps = {
        arm: {
            task: _summary([
                seed_records[str(seed)]["runs"][arm]["train_val_accuracy_gap"][task]
                for seed in SEEDS
            ])
            for task in TASKS
        }
        for arm in ("a0", "a1")
    }

    checks = {
        "all_exposure_checks_passed": all(
            seed_records[str(seed)]["exposure_check"]["passed"]
            for seed in SEEDS
        ),
        "all_runs_same_git_commit": len({
            seed_records[str(seed)]["runs"][arm]["git_commit"]
            for seed in SEEDS
            for arm in ("a0", "a1")
        }) == 1,
        "git_commits": sorted({
            seed_records[str(seed)]["runs"][arm]["git_commit"]
            for seed in SEEDS
            for arm in ("a0", "a1")
        }),
        "all_runs_have_notes": all(
            seed_records[str(seed)]["runs"][arm]["standard_tags"][
                "mlflow.note.content"
            ]
            for seed in SEEDS
            for arm in ("a0", "a1")
        ),
        # Gate conditions, transcribed from PLAN.md's pre-registered
        # falsification rule.
        "a0_late_ratio_at_least_3_by_seed": {
            str(seed): (
                seed_records[str(seed)]["gradients"]["a0"]["summary"]["late"][
                    "max_min_mean_norm_ratio"
                ] >= 3.0
            )
            for seed in SEEDS
        },
        "a1_late_ratio_below_3_by_seed": {
            str(seed): (
                seed_records[str(seed)]["gradients"]["a1"]["summary"]["late"][
                    "max_min_mean_norm_ratio"
                ] < 3.0
            )
            for seed in SEEDS
        },
        "a1_middle_and_late_ratio_at_least_3_by_seed": {
            str(seed): all(
                seed_records[str(seed)]["gradients"]["a1"]["summary"][phase][
                    "max_min_mean_norm_ratio"
                ] >= 3.0
                for phase in ("middle", "late")
            )
            for seed in SEEDS
        },
        # Descriptive only: reported beside the gate, never a gate itself.
        "a0_middle_ratio_at_least_3_by_seed": {
            str(seed): (
                seed_records[str(seed)]["gradients"]["a0"]["summary"]["middle"][
                    "max_min_mean_norm_ratio"
                ] >= 3.0
            )
            for seed in SEEDS
        },
        "a1_middle_ratio_below_3_by_seed": {
            str(seed): (
                seed_records[str(seed)]["gradients"]["a1"]["summary"]["middle"][
                    "max_min_mean_norm_ratio"
                ] < 3.0
            )
            for seed in SEEDS
        },
    }
    return {
        "gradients": gradients,
        "paired_ratio_delta_a1_minus_a0": paired_ratio_deltas,
        "paired_task_norm_delta_a1_minus_a0": paired_task_norm_deltas,
        "outcomes": outcomes,
        "train_val_accuracy_gaps": train_val_gaps,
        "confirmation_checks": checks,
    }


def _study_decision(cross_seed):
    """Apply PLAN.md's pre-registered rule; report the CI as evidence only."""
    checks = cross_seed["confirmation_checks"]
    paired = cross_seed["paired_ratio_delta_a1_minus_a0"]
    a0_late_replicated = all(
        checks["a0_late_ratio_at_least_3_by_seed"].values()
    )
    a1_late_removed = all(checks["a1_late_ratio_below_3_by_seed"].values())
    a1_still_dominated = all(
        checks["a1_middle_and_late_ratio_at_least_3_by_seed"].values()
    )
    valid = (
        checks["all_exposure_checks_passed"]
        and checks["all_runs_same_git_commit"]
        and checks["git_commits"] == [CONFIRMATION_COMMIT]
        and checks["all_runs_have_notes"]
    )
    if valid and a0_late_replicated and a1_late_removed:
        decision = "CONFIRMED"
    elif valid and a1_still_dominated:
        decision = "REJECTED"
    else:
        decision = "INCONCLUSIVE"
    return {
        "decision": decision,
        "decision_basis": {
            "protocol_valid": valid,
            "a0_late_ratio_at_least_3_all_seeds": a0_late_replicated,
            "a1_late_ratio_below_3_all_seeds": a1_late_removed,
            "a1_middle_and_late_ratio_at_least_3_all_seeds": a1_still_dominated,
            "descriptive_paired_ratio_delta": {
                phase: paired[phase] for phase in ("middle", "late")
            },
            "scope": (
                "Tests whether the F9 norm-dominance signal is data-regime "
                "sensitive. Ordinary-split ER accuracy remains context only; "
                "this does not establish an ER generalization effect."
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
        (
            f"tags.study_id = '{STUDY_ID}' and tags.stage = 'confirm' "
            f"and tags.git_commit = '{CONFIRMATION_COMMIT}'"
        ),
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
        for method in METHODS.values()
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
            runs_by_arm = {
                arm: by_key[(method, seed)] for arm, method in METHODS.items()
            }
            gradients = {
                arm: _download_gradient(client, run, destination)
                for arm, run in runs_by_arm.items()
            }
            exposure_check = _exposure_check(gradients["a0"], gradients["a1"])
            run_records = {
                arm: _full_run_record(client, run)
                for arm, run in runs_by_arm.items()
            }
            for arm in ("a0", "a1"):
                run_records[arm]["train_val_accuracy_gap"] = {
                    task: (
                        run_records[arm]["final_train_accuracy"][task]
                        - run_records[arm]["final_val_accuracy"][task]
                    )
                    for task in TASKS
                }
            seed_records[str(seed)] = {
                "runs": run_records,
                "exposure_check": exposure_check,
                "gradients": gradients,
            }

    cross_seed = _cross_seed_summary(seed_records)
    decision = _study_decision(cross_seed)
    completed_at = datetime.now(timezone.utc).isoformat()
    final_note = (STUDY_DIR / "NOTE.md").read_text().strip()
    for run in by_key.values():
        # Every confirmation run was launched with --device-index 0: GPU 0 ran
        # seeds 0,2,4 first, then seeds 1,3 after GPU 1 stayed occupied by
        # other users' jobs. Recorded so the ledger states where runs executed.
        for key, value in {
            "status": decision["decision"].lower(),
            "study_decision": decision["decision"],
            "runtime_device_index": "0",
            "confirmation_completed_at": completed_at,
            "mlflow.note.content": final_note,
        }.items():
            client.set_tag(run.info.run_id, key, value)

    result = {
        "study_id": STUDY_ID,
        "title": "ER data-regime gradient-scale control confirmation",
        "type": "diagnostic",
        "stage": "confirm",
        "generated_at": completed_at,
        "confirmation_commit": CONFIRMATION_COMMIT,
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
        "checks": cross_seed["confirmation_checks"],
        "middle_ratio": {
            arm: cross_seed["gradients"][arm]["middle"][
                "max_min_mean_norm_ratio"
            ]
            for arm in ("a0", "a1")
        },
        "late_ratio": {
            arm: cross_seed["gradients"][arm]["late"][
                "max_min_mean_norm_ratio"
            ]
            for arm in ("a0", "a1")
        },
        "paired_ratio_delta": cross_seed["paired_ratio_delta_a1_minus_a0"],
        "epoch_outcomes": cross_seed["outcomes"]["epoch"],
    }, indent=2))


if __name__ == "__main__":
    main()
