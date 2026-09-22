"""Analyze DG-0005's matched standard and ER-weighted screening arms."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


STUDY_ID = "DG-0005"
EXPERIMENT_NAME = "taskrelation-diagnostics"
TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
TASKS = ("ks", "si", "er")
PHASES = ("early", "middle", "late")
METHODS = {
    "a0": "wavcse-standard-composition",
    "a1": "wavcse-er-weighted-composition",
}
IMPLEMENTATION_COMMIT = "0162224e63c1f77d3aaa7228bc2a13f3f2d85b14"


def _latest_file(pattern):
    matches = sorted(STUDY_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise RuntimeError(f"Expected at least one {pattern}")
    return matches[-1]


def _metric(run, key):
    value = run.data.metrics.get(key)
    if value is None:
        raise RuntimeError(f"Run {run.info.run_id} is missing metric {key}")
    return float(value)


def _history_final(client, run_id, key):
    history = client.get_metric_history(run_id, key)
    if not history:
        raise RuntimeError(f"Run {run_id} is missing metric history {key}")
    return float(max(history, key=lambda metric: metric.step).value)


def _run_record(client, run):
    duration_seconds = None
    if run.info.end_time is not None:
        duration_seconds = (run.info.end_time - run.info.start_time) / 1000.0
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "git_commit": run.data.tags.get("git_commit"),
        "seed": int(run.data.tags["seed"]),
        "duration_seconds": duration_seconds,
        "fixed_epoch_metrics": {
            "aggregate": _metric(run, "test_epoch_acc_all"),
            **{task: _metric(run, f"test_epoch_{task}_acc") for task in TASKS},
        },
        "final_train_accuracy": {
            task: _history_final(client, run.info.run_id, f"train_{task}_acc")
            for task in TASKS
        },
        "final_val_accuracy": {
            task: _history_final(client, run.info.run_id, f"val_{task}_acc")
            for task in TASKS
        },
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
    valid_examples = {}
    for task in payload["task_array"]:
        counts = [record["valid_examples"][task] for record in payload["records"]]
        valid_examples[task] = {
            "mean": sum(counts) / len(counts),
            "min": min(counts),
            "max": max(counts),
        }
    er_to_smallest = {}
    for phase in PHASES:
        task_norms = payload["summary"][phase]["task_norms"]
        er_to_smallest[phase] = (
            task_norms["er"]["mean"]
            / min(task_norms["ks"]["mean"], task_norms["si"]["mean"])
        )
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
        "valid_examples_per_sampled_batch": valid_examples,
        "er_to_smallest_ratio": er_to_smallest,
        "summary": payload["summary"],
        "record_steps": [record["step"] for record in payload["records"]],
    }


def _configuration_check():
    with (STUDY_DIR / "configs" / "a0_baseline.yml").open() as source:
        a0 = yaml.safe_load(source)
    with (STUDY_DIR / "configs" / "a1_er_weighted.yml").open() as source:
        a1 = yaml.safe_load(source)

    a1_weight = a1["training"].pop("task_sampling_weights", None)
    ignored_paths = [
        ("device", "index"),
        ("paths", "results_root"),
        ("paths", "checkpoints_root"),
        ("research", "method"),
        ("research", "baseline_study"),
    ]
    normalized_a0 = copy.deepcopy(a0)
    normalized_a1 = copy.deepcopy(a1)
    for section, key in ignored_paths:
        normalized_a0[section].pop(key, None)
        normalized_a1[section].pop(key, None)
    return {
        "passed": normalized_a0 == normalized_a1 and a1_weight == {"er": 11.5},
        "a1_task_sampling_weights": a1_weight,
        "unmatched_after_allowed_fields": normalized_a0 != normalized_a1,
    }


def _exposure_check(a0, a1):
    shared_fields = (
        "task_array", "sample_interval_steps", "total_training_steps",
        "sampled_steps", "shared_parameter_names", "shared_parameter_count",
        "record_steps",
    )
    mismatches = {
        field: {"a0": a0[field], "a1": a1[field]}
        for field in shared_fields
        if a0[field] != a1[field]
    }
    a1_counts = a1["valid_examples_per_sampled_batch"]
    er_to_ks_count_ratio = a1_counts["er"]["mean"] / a1_counts["ks"]["mean"]
    mean_batch_total = sum(a1_counts[task]["mean"] for task in TASKS)
    return {
        "passed": (
            not mismatches
            and a0["total_training_steps"] == 2820
            and a1["total_training_steps"] == 2820
            and er_to_ks_count_ratio >= 0.5
            and abs(mean_batch_total - 2048.0) < 1e-9
            and a0["skipped_sample_steps_missing_tasks"] == 0
            and a1["skipped_sample_steps_missing_tasks"] == 0
        ),
        "mismatches": mismatches,
        "a1_er_to_ks_mean_count_ratio": er_to_ks_count_ratio,
        "a1_mean_batch_total": mean_batch_total,
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
            f"tags.study_id = '{STUDY_ID}' and tags.stage = 'screen' "
            f"and tags.git_commit = '{IMPLEMENTATION_COMMIT}'"
        ),
        max_results=10,
    )
    by_method = {run.data.tags.get("method"): run for run in runs}
    if set(by_method) != set(METHODS.values()):
        raise RuntimeError(
            f"Expected methods {set(METHODS.values())}, found {set(by_method)}"
        )
    unfinished = [
        method for method, run in by_method.items() if run.info.status != "FINISHED"
    ]
    if unfinished:
        raise RuntimeError(f"Runs are not finished: {unfinished}")

    gradients = {
        "a0": _gradient_record(_latest_file("outputs/a0/**/gradient_diagnostics.json")),
        "a1": _gradient_record(_latest_file("outputs/a1/**/gradient_diagnostics.json")),
    }
    configuration_check = _configuration_check()
    exposure_check = _exposure_check(gradients["a0"], gradients["a1"])
    phase_ratios = {
        arm: {
            phase: gradients[arm]["summary"][phase]["max_min_mean_norm_ratio"]
            for phase in PHASES
        }
        for arm in ("a0", "a1")
    }
    ratio_deltas = {
        phase: phase_ratios["a1"][phase] - phase_ratios["a0"][phase]
        for phase in PHASES
    }

    late_supports_h1 = (
        exposure_check["passed"]
        and configuration_check["passed"]
        and phase_ratios["a0"]["late"] >= 3.0
        and phase_ratios["a1"]["late"] < 3.0
    )
    rejects_h1 = (
        exposure_check["passed"]
        and configuration_check["passed"]
        and phase_ratios["a1"]["middle"] >= 3.0
        and phase_ratios["a1"]["late"] >= 3.0
    )
    if late_supports_h1:
        decision = "PROMISING"
        next_action = "Run matched seeds 0-4 confirmation for A0 and A1."
    elif rejects_h1:
        decision = "REJECTED"
        next_action = "Synthesize H2 evidence; do not run A2 or confirmation."
    else:
        decision = "INCONCLUSIVE"
        next_action = "Run pre-registered reverse-direction arm A2 before deciding."

    screen_completed_at = datetime.now(timezone.utc).isoformat()
    final_note = (STUDY_DIR / "NOTE.md").read_text().strip()
    for run in by_method.values():
        for key, value in {
            "status": decision.lower(),
            "study_decision": decision,
            "screen_completed_at": screen_completed_at,
            "runtime_device_index": "0",
            "mlflow.note.content": final_note,
        }.items():
            client.set_tag(run.info.run_id, key, value)

    superseded_runs = client.search_runs(
        [experiment.experiment_id],
        (
            f"tags.study_id = '{STUDY_ID}' and tags.stage = 'screen' "
            "and tags.git_commit = "
            "'815969611a37e078abdaf0c32c7843c65ede2153'"
        ),
        max_results=10,
    )
    excluded_execution_runs = []
    for run in superseded_runs:
        status = "superseded" if run.info.status == "FINISHED" else "failed"
        reason = (
            "pre-fix matched control; excluded because paired A1 did not execute"
            if status == "superseded"
            else "sampler did not unwrap IEMOCAP's Subset component; no optimizer step"
        )
        client.set_tag(run.info.run_id, "status", status)
        client.set_tag(run.info.run_id, "study_decision", status.upper())
        client.set_tag(run.info.run_id, "exclusion_reason", reason)
        client.set_tag(run.info.run_id, "mlflow.note.content", final_note)
        excluded_execution_runs.append({
            "run_id": run.info.run_id,
            "method": run.data.tags.get("method"),
            "mlflow_status": run.info.status,
            "study_status": status,
            "reason": reason,
        })

    run_records = {
        arm: _run_record(client, by_method[method])
        for arm, method in METHODS.items()
    }
    for arm in ("a0", "a1"):
        run_records[arm]["train_val_accuracy_gap"] = {
            task: (
                run_records[arm]["final_train_accuracy"][task]
                - run_records[arm]["final_val_accuracy"][task]
            )
            for task in TASKS
        }
    outcome_delta = {
        metric: (
            run_records["a1"]["fixed_epoch_metrics"][metric]
            - run_records["a0"]["fixed_epoch_metrics"][metric]
        )
        for metric in ("aggregate", *TASKS)
    }

    result = {
        "study_id": STUDY_ID,
        "title": "ER data-regime gradient-scale control",
        "type": "diagnostic",
        "stage": "screen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "seed": 42,
        "protocol": {
            "task_set": "ks_si_er",
            "pooling": {"type": "smp", "parameter": 0.5},
            "layers": "all",
            "epochs": 30,
            "batch_size": 2048,
            "er_protocol": "ordinary speaker-leaky split; outcome context only",
        },
        "configuration_check": configuration_check,
        "exposure_check": exposure_check,
        "runs": run_records,
        "excluded_execution_runs": excluded_execution_runs,
        "gradient_diagnostics": gradients,
        "phase_max_min_mean_norm_ratios": phase_ratios,
        "a1_minus_a0_ratio_delta": ratio_deltas,
        "fixed_epoch_outcome_delta_a1_minus_a0": outcome_delta,
        "decision": decision,
        "next_action": next_action,
    }
    output_path = STUDY_DIR / "result.json"
    with output_path.open("w") as output:
        json.dump(result, output, indent=2)

    print(json.dumps({
        "result": str(output_path),
        "run_ids": {arm: record["run_id"] for arm, record in run_records.items()},
        "configuration_check": configuration_check,
        "exposure_check": exposure_check,
        "phase_ratios": phase_ratios,
        "ratio_deltas": ratio_deltas,
        "fixed_epoch_outcome_delta": outcome_delta,
        "decision": decision,
        "next_action": next_action,
    }, indent=2))


if __name__ == "__main__":
    main()
