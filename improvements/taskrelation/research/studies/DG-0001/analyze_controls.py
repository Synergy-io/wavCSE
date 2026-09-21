"""Separate DG-0001 operational transfer from optimizer-exposure effects."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv


STUDY_ID = "DG-0001"
EXPERIMENT = "taskrelation-diagnostics"
T_CRIT_DF9 = 2.262


def _summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    sem = std / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_std": std,
        "ci95_t9": [mean - T_CRIT_DF9 * sem, mean + T_CRIT_DF9 * sem],
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "range": [min(values), max(values)],
        "values": values,
    }


def _load(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing control artifact: {path}")
    return json.loads(path.read_text())


def _fold_acc(summary: dict, checkpoint: str, task: str) -> list[float]:
    return [
        float(fold["task_metrics"][checkpoint][task]["acc"])
        for fold in summary["folds"]
    ]


def _latest_parent(client, experiment_id: str, stage: str, task_set: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.study_id = '{STUDY_ID}' and tags.stage = '{stage}' "
            f"and tags.task_set = '{task_set}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    parents = [
        run for run in runs
        if run.info.status == "FINISHED"
        and not run.data.tags.get("mlflow.parentRunId")
    ]
    if not parents:
        raise RuntimeError(f"Missing finished run: stage={stage}, task_set={task_set}")
    return parents[0]


def main() -> None:
    study_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[5]
    load_dotenv(repo_root / ".env")

    base = study_dir / "outputs"
    original_er = _load(
        base / "loso/results/DG-0001/stage_c_loso/er/kfold_summary.json"
    )
    pairs = {
        "ks": _load(
            base / "loso/results/DG-0001/stage_c_loso/ks_er/kfold_summary.json"
        ),
        "si": _load(
            base / "loso/results/DG-0001/stage_c_loso/si_er/kfold_summary.json"
        ),
    }
    step_controls = {
        "ks": _load(
            base / "er_steps_ks/results/DG-0001/stage_d_er_steps_ks/er/kfold_summary.json"
        ),
        "si": _load(
            base / "er_steps_si/results/DG-0001/stage_d_er_steps_si/er/kfold_summary.json"
        ),
    }

    reference_speakers = [fold["held_out_test_speaker"] for fold in original_er["folds"]]
    for label, summary in {**pairs, **{f"step_{k}": v for k, v in step_controls.items()}}.items():
        speakers = [fold["held_out_test_speaker"] for fold in summary["folds"]]
        if speakers != reference_speakers:
            raise RuntimeError(f"Speaker order differs for {label}")

    mlflow.set_tracking_uri("https://dagshub.com/Ke-vin-S/wavCSE.mlflow")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT}")

    reverse_runs = {
        task: _latest_parent(
            client, experiment.experiment_id, "stage_c_reverse_control", task
        )
        for task in ("ks", "si")
    }
    control_runs = {
        "er_steps_ks": _latest_parent(
            client, experiment.experiment_id, "stage_d_er_steps_ks", "er"
        ),
        "er_steps_si": _latest_parent(
            client, experiment.experiment_id, "stage_d_er_steps_si", "er"
        ),
    }

    checkpoints = {}
    for checkpoint in ("epoch", "opt", "best"):
        original = _fold_acc(original_er, checkpoint, "er")
        checkpoint_result = {}
        for auxiliary in ("ks", "si"):
            pair_er = _fold_acc(pairs[auxiliary], checkpoint, "er")
            step_er = _fold_acc(step_controls[auxiliary], checkpoint, "er")
            raw = [pair - base_er for pair, base_er in zip(pair_er, original)]
            optimizer = [step - base_er for step, base_er in zip(step_er, original)]
            residual = [pair - step for pair, step in zip(pair_er, step_er)]

            pair_reverse = _fold_acc(pairs[auxiliary], checkpoint, auxiliary)
            key = f"test_{checkpoint}_{auxiliary}_acc"
            reverse_reference = reverse_runs[auxiliary].data.metrics.get(key)
            if reverse_reference is None:
                raise RuntimeError(f"Missing {key} on reverse control {auxiliary}")
            reverse = [value - float(reverse_reference) for value in pair_reverse]

            checkpoint_result[auxiliary] = {
                "raw_operational_transfer": _summary(raw),
                "optimizer_exposure_gain": _summary(optimizer),
                "residual_pair_minus_step_matched_er": _summary(residual),
                "reverse_pair_minus_5epoch_single": {
                    **_summary(reverse),
                    "single_reference": float(reverse_reference),
                },
                "decomposition_identity_max_abs_error": max(
                    abs(raw_value - optimizer_value - residual_value)
                    for raw_value, optimizer_value, residual_value
                    in zip(raw, optimizer, residual)
                ),
            }
        checkpoints[checkpoint] = checkpoint_result

    primary = checkpoints["epoch"]
    per_auxiliary = {}
    semantic_candidates = []
    for auxiliary in ("ks", "si"):
        residual = primary[auxiliary]["residual_pair_minus_step_matched_er"]
        reverse = primary[auxiliary]["reverse_pair_minus_5epoch_single"]
        if residual["mean"] >= 0.010 and residual["ci95_t9"][0] > 0:
            status = "residual_positive_transfer_candidate"
            semantic_candidates.append(auxiliary)
        elif residual["ci95_t9"][0] <= 0 <= residual["ci95_t9"][1]:
            status = "no_resolved_residual_transfer"
        elif residual["ci95_t9"][1] < 0:
            status = "pair_underperforms_step_matched_er"
        else:
            status = "small_positive_residual"
        per_auxiliary[auxiliary] = {
            "status": status,
            "residual_mean": residual["mean"],
            "residual_ci95": residual["ci95_t9"],
            "reverse_mean": reverse["mean"],
            "reverse_ci95": reverse["ci95_t9"],
        }

    if semantic_candidates:
        classification = "directional_transfer_residual_after_step_matching"
    else:
        classification = "raw_asymmetry_explained_by_optimizer_exposure"

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "study_id": STUDY_ID,
        "stage": "stage_d_optimization_controls",
        "status": "complete",
        "generated_at": generated_at,
        "seed": 42,
        "held_out_speakers": reference_speakers,
        "primary_checkpoint": "epoch",
        "checkpoints": checkpoints,
        "decision": {
            "classification": classification,
            "per_auxiliary": per_auxiliary,
            "semantic_candidates": semantic_candidates,
            "interpretation_constraint": (
                "Step matching changes batch size and approximates, rather than exactly "
                "equalizes, update count. Residuals are diagnostic, not architecture claims."
            ),
        },
        "runs": {
            "reverse_ks": reverse_runs["ks"].info.run_id,
            "reverse_si": reverse_runs["si"].info.run_id,
            "er_steps_ks": control_runs["er_steps_ks"].info.run_id,
            "er_steps_si": control_runs["er_steps_si"].info.run_id,
        },
        "commits": sorted({
            run.data.tags.get("mlflow.source.git.commit")
            for run in [*reverse_runs.values(), *control_runs.values()]
        }),
    }
    output_path = study_dir.parents[1] / "task_relations" / "optimization_control.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    result_path = study_dir / "result.json"
    result = json.loads(result_path.read_text())
    result.update({
        "status": "complete",
        "stage": "stage_d_optimization_controls",
        "completed_at": generated_at,
        "stage_d_optimization_controls": payload,
        "final_decision": payload["decision"],
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
