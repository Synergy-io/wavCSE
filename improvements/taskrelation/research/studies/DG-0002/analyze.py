"""Analyze the matched DG-0002 baseline/MTRL diagnostic runs."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


STUDY_ID = "DG-0002"
EXPERIMENT_NAME = "taskrelation-diagnostics"
TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
TASKS = ("ks", "si", "er")
PAIRS = ("ks_si", "ks_er", "si_er")
PHASES = ("early", "middle", "late")


def _one_file(pattern):
    matches = sorted(STUDY_DIR.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def _metric(run, key):
    value = run.data.metrics.get(key)
    if value is None:
        raise RuntimeError(f"Run {run.info.run_id} is missing metric {key}")
    return float(value)


def _history_summary(client, run_id, key, maximize):
    history = client.get_metric_history(run_id, key)
    if not history:
        raise RuntimeError(f"Run {run_id} is missing metric history {key}")
    ordered = sorted(history, key=lambda metric: metric.step)
    best = (max if maximize else min)(ordered, key=lambda metric: metric.value)
    return {
        "first": float(ordered[0].value),
        "final": float(ordered[-1].value),
        "best": float(best.value),
        "best_step": int(best.step),
        "points": len(ordered),
    }


def _run_record(client, run):
    metrics = {}
    for checkpoint in ("epoch", "opt", "best"):
        metrics[checkpoint] = {
            "aggregate": _metric(run, f"test_{checkpoint}_acc_all"),
            **{
                task: _metric(run, f"test_{checkpoint}_{task}_acc")
                for task in TASKS
            },
        }

    training_behavior = {}
    for phase in ("train", "val"):
        training_behavior[phase] = {
            "aggregate_accuracy": _history_summary(
                client, run.info.run_id, f"{phase}_acc_all", maximize=True
            ),
            "aggregate_loss": _history_summary(
                client, run.info.run_id, f"{phase}_loss_all", maximize=False
            ),
        }
        for task in TASKS:
            training_behavior[phase][f"{task}_accuracy"] = _history_summary(
                client, run.info.run_id, f"{phase}_{task}_acc", maximize=True
            )
            training_behavior[phase][f"{task}_loss"] = _history_summary(
                client, run.info.run_id, f"{phase}_{task}_loss", maximize=False
            )

    duration_seconds = None
    if run.info.end_time is not None:
        duration_seconds = (run.info.end_time - run.info.start_time) / 1000.0

    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "git_commit": run.data.tags.get("git_commit"),
        "source_git_commit": run.data.tags.get("mlflow.source.git.commit"),
        "seed": int(run.data.tags["seed"]),
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "training_behavior": training_behavior,
        "standard_tags": {
            key: run.data.tags.get(key)
            for key in (
                "study_id", "stage", "family", "method", "hypothesis_slug",
                "task_set", "representation", "pooling", "layers", "seed",
                "parent_study", "baseline_study", "agent_generated",
                "git_commit", "status", "mlflow.note.content",
            )
        },
    }


def _gradient_record(path):
    with path.open() as artifact:
        payload = json.load(artifact)
    valid_example_summary = {}
    for task in payload["task_array"]:
        counts = [
            record["valid_examples"][task] for record in payload["records"]
        ]
        valid_example_summary[task] = {
            "mean": sum(counts) / len(counts),
            "min": min(counts),
            "max": max(counts),
        }
    return {
        "artifact": str(path.relative_to(REPO_ROOT)),
        "task_array": payload["task_array"],
        "sample_interval_steps": payload["sample_interval_steps"],
        "total_training_steps": payload["total_training_steps"],
        "sampled_steps": payload["sampled_steps"],
        "skipped_sample_steps_missing_tasks": payload[
            "skipped_sample_steps_missing_tasks"
        ],
        "shared_parameter_names": payload["shared_parameter_names"],
        "shared_parameter_count": payload["shared_parameter_count"],
        "valid_examples_per_sampled_batch": valid_example_summary,
        "summary": payload["summary"],
        "records": payload["records"],
    }


def _validate_pairing(baseline, mtrl):
    paired_fields = (
        "task_array", "sample_interval_steps", "total_training_steps",
        "shared_parameter_names", "shared_parameter_count",
    )
    mismatches = {
        field: {"baseline": baseline[field], "mtrl": mtrl[field]}
        for field in paired_fields
        if baseline[field] != mtrl[field]
    }

    baseline_records = baseline["records"]
    mtrl_records = mtrl["records"]
    if len(baseline_records) != len(mtrl_records):
        mismatches["sampled_record_count"] = {
            "baseline": len(baseline_records),
            "mtrl": len(mtrl_records),
        }
    else:
        for baseline_record, mtrl_record in zip(baseline_records, mtrl_records):
            step = baseline_record["step"]
            if step != mtrl_record["step"]:
                mismatches.setdefault("steps", []).append({
                    "baseline": step,
                    "mtrl": mtrl_record["step"],
                })
            if baseline_record["valid_examples"] != mtrl_record["valid_examples"]:
                mismatches.setdefault("valid_examples", []).append({
                    "step": step,
                    "baseline": baseline_record["valid_examples"],
                    "mtrl": mtrl_record["valid_examples"],
                })

    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "paired_sample_count": min(len(baseline_records), len(mtrl_records)),
    }


def _diagnostic_comparison(baseline, mtrl):
    phase_deltas = {}
    persistent_conflicts = []
    norm_dominance_phases = []
    dynamic_pairs = []
    mitigated_pairs = []

    for phase in PHASES:
        phase_deltas[phase] = {
            "pairwise_cosine_mean_delta_mtrl_minus_baseline": {},
            "negative_frequency_delta_mtrl_minus_baseline": {},
            "mean_norm_ratio_mtrl_over_baseline": {},
        }
        for pair in PAIRS:
            baseline_pair = baseline["summary"][phase]["pairwise_cosines"][pair]
            mtrl_pair = mtrl["summary"][phase]["pairwise_cosines"][pair]
            phase_deltas[phase][
                "pairwise_cosine_mean_delta_mtrl_minus_baseline"
            ][pair] = mtrl_pair["mean"] - baseline_pair["mean"]
            phase_deltas[phase][
                "negative_frequency_delta_mtrl_minus_baseline"
            ][pair] = (
                mtrl_pair["negative_frequency"]
                - baseline_pair["negative_frequency"]
            )
        for task in TASKS:
            baseline_norm = baseline["summary"][phase]["task_norms"][task]["mean"]
            mtrl_norm = mtrl["summary"][phase]["task_norms"][task]["mean"]
            phase_deltas[phase]["mean_norm_ratio_mtrl_over_baseline"][task] = (
                mtrl_norm / baseline_norm
            )

        if baseline["summary"][phase]["max_min_mean_norm_ratio"] >= 3.0:
            norm_dominance_phases.append(phase)

    for pair in PAIRS:
        conflict_phases = []
        mitigation_phases = []
        phase_means = []
        for phase in PHASES:
            baseline_pair = baseline["summary"][phase]["pairwise_cosines"][pair]
            mtrl_pair = mtrl["summary"][phase]["pairwise_cosines"][pair]
            phase_means.append(baseline_pair["mean"])
            if (
                baseline_pair["mean"] <= -0.05
                and baseline_pair["negative_frequency"] >= 0.60
            ):
                conflict_phases.append(phase)
            if (
                baseline_pair["negative_frequency"]
                - mtrl_pair["negative_frequency"]
                >= 0.10
            ):
                mitigation_phases.append(phase)
        if len(conflict_phases) >= 2:
            persistent_conflicts.append({
                "pair": pair,
                "phases": conflict_phases,
            })
        if len(mitigation_phases) >= 2:
            mitigated_pairs.append({
                "pair": pair,
                "phases": mitigation_phases,
            })
        if min(phase_means) < 0.0 < max(phase_means) and (
            max(phase_means) - min(phase_means) >= 0.10
        ):
            dynamic_pairs.append({
                "pair": pair,
                "phase_means": dict(zip(PHASES, phase_means)),
            })

    benign = True
    for phase in PHASES:
        if baseline["summary"][phase]["max_min_mean_norm_ratio"] >= 3.0:
            benign = False
        for pair in PAIRS:
            pair_summary = baseline["summary"][phase]["pairwise_cosines"][pair]
            if not (
                -0.05 <= pair_summary["mean"] <= 0.05
                and 0.40 <= pair_summary["negative_frequency"] <= 0.60
            ):
                benign = False

    return {
        "phase_deltas": phase_deltas,
        "persistent_conflicts": persistent_conflicts,
        "norm_dominance_phases": norm_dominance_phases,
        "dynamic_pairs": dynamic_pairs,
        "mitigated_pairs": mitigated_pairs,
        "baseline_meets_benign_falsification_condition": benign,
        "screening_signal": bool(
            persistent_conflicts or len(norm_dominance_phases) >= 2 or dynamic_pairs
        ),
    }


def _omega_record(path, mtrl_gradient):
    with path.open() as artifact:
        payload = json.load(artifact)
    final = payload["final_omega"]
    pair_values = {
        "ks_si": final[0][1],
        "ks_er": final[0][2],
        "si_er": final[1][2],
    }
    agreement = {}
    for pair, omega_value in pair_values.items():
        cosine = mtrl_gradient["summary"]["all"]["pairwise_cosines"][pair]["mean"]
        agreement[pair] = {
            "omega": omega_value,
            "gradient_cosine_mean": cosine,
            "same_sign": math.copysign(1.0, omega_value) == math.copysign(1.0, cosine),
        }
    return {
        "artifact": str(path.relative_to(REPO_ROOT)),
        "snapshots": len(payload["history"]),
        "final_omega": final,
        "final_pair_values": pair_values,
        "off_diagonal_range": max(pair_values.values()) - min(pair_values.values()),
        "gradient_sign_agreement": agreement,
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
        f"tags.study_id = '{STUDY_ID}' and tags.stage = 'paired_screen'",
        max_results=10,
    )
    by_method = {run.data.tags.get("method"): run for run in runs}
    expected_methods = {"wavcse-baseline", "mtrl"}
    if set(by_method) != expected_methods:
        raise RuntimeError(f"Expected {expected_methods}, found {set(by_method)}")
    unfinished = [
        method for method, run in by_method.items() if run.info.status != "FINISHED"
    ]
    if unfinished:
        raise RuntimeError(f"Runs are not finished: {unfinished}")

    baseline_gradient = _gradient_record(
        _one_file("outputs/baseline/**/gradient_diagnostics.json")
    )
    mtrl_gradient = _gradient_record(
        _one_file("outputs/mtrl/**/gradient_diagnostics.json")
    )
    pairing = _validate_pairing(baseline_gradient, mtrl_gradient)
    diagnostics = _diagnostic_comparison(baseline_gradient, mtrl_gradient)
    omega = _omega_record(
        _one_file("outputs/mtrl/**/omega_history.json"),
        mtrl_gradient,
    )

    run_records = {
        "baseline": _run_record(client, by_method["wavcse-baseline"]),
        "mtrl": _run_record(client, by_method["mtrl"]),
    }
    outcome_deltas = {}
    for checkpoint in ("epoch", "opt", "best"):
        outcome_deltas[checkpoint] = {
            metric: (
                run_records["mtrl"]["metrics"][checkpoint][metric]
                - run_records["baseline"]["metrics"][checkpoint][metric]
            )
            for metric in ("aggregate", *TASKS)
        }

    preliminary_decision = "INCONCLUSIVE"
    if pairing["passed"] and diagnostics["screening_signal"]:
        preliminary_decision = "PROMISING"
    elif pairing["passed"] and diagnostics[
        "baseline_meets_benign_falsification_condition"
    ]:
        preliminary_decision = "REJECTED"

    final_note = (STUDY_DIR / "NOTE.md").read_text().strip()
    completed_at = datetime.now(timezone.utc).isoformat()
    for method, run in by_method.items():
        for key, value in {
            "status": preliminary_decision.lower(),
            "study_decision": preliminary_decision,
            "screen_completed_at": completed_at,
            "mlflow.note.content": final_note,
        }.items():
            client.set_tag(run.info.run_id, key, value)
        record_key = "baseline" if method == "wavcse-baseline" else "mtrl"
        run_records[record_key]["standard_tags"]["status"] = (
            preliminary_decision.lower()
        )
        run_records[record_key]["standard_tags"]["mlflow.note.content"] = (
            final_note
        )

    result = {
        "study_id": STUDY_ID,
        "title": "Exposure-controlled gradient compatibility baseline",
        "type": "diagnostic",
        "stage": "paired_screen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": "75e31b81e860d54f6125dd4a623d85e93718b7ec",
        "seed": 42,
        "protocol": {
            "task_set": "ks_si_er",
            "pooling": {"type": "smp", "parameter": 0.5},
            "layers": "all",
            "epochs": 30,
            "batch_size": 2048,
            "primary_checkpoint": "epoch",
            "er_protocol": "ordinary speaker-leaky split; outcome screening only",
        },
        "runs": run_records,
        "outcome_delta_mtrl_minus_baseline": outcome_deltas,
        "pairing_check": pairing,
        "gradient_diagnostics": {
            "baseline": baseline_gradient,
            "mtrl": mtrl_gradient,
            "comparison": diagnostics,
        },
        "omega": omega,
        "preliminary_decision": preliminary_decision,
    }
    output_path = STUDY_DIR / "result.json"
    with output_path.open("w") as output:
        json.dump(result, output, indent=2)

    print(json.dumps({
        "result": str(output_path),
        "runs": {
            method: record["run_id"] for method, record in run_records.items()
        },
        "pairing_passed": pairing["passed"],
        "outcome_epoch_delta": outcome_deltas["epoch"],
        "persistent_conflicts": diagnostics["persistent_conflicts"],
        "norm_dominance_phases": diagnostics["norm_dominance_phases"],
        "dynamic_pairs": diagnostics["dynamic_pairs"],
        "mitigated_pairs": diagnostics["mitigated_pairs"],
        "preliminary_decision": preliminary_decision,
    }, indent=2))


if __name__ == "__main__":
    main()
