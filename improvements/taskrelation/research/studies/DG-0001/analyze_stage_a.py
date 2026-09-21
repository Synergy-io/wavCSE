"""Build DG-0001's directed transfer matrix from finished MLflow runs.

This script reports the pre-registered fixed-epoch test metrics. It does not
select a model or checkpoint from test performance.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv


STUDY_ID = "DG-0001"
STAGE = "stage_a_screen"
EXPERIMENT = "taskrelation-diagnostics"
ARMS = ("ks", "si", "er", "ks_si", "ks_er", "si_er", "ks_si_er")
TASKS = ("ks", "si", "er")
PAIR_CELLS = {
    "ks<-si": ("ks_si", "ks", "ks"),
    "si<-ks": ("ks_si", "si", "si"),
    "ks<-er": ("ks_er", "ks", "ks"),
    "er<-ks": ("ks_er", "er", "er"),
    "si<-er": ("si_er", "si", "si"),
    "er<-si": ("si_er", "er", "er"),
}
REVERSE_PAIRS = (
    ("ks<-si", "si<-ks"),
    ("ks<-er", "er<-ks"),
    ("si<-er", "er<-si"),
)


def _metric(run, checkpoint: str, task: str) -> float:
    key = f"test_{checkpoint}_{task}_acc"
    value = run.data.metrics.get(key)
    if value is None:
        raise RuntimeError(f"Run {run.info.run_id} is missing {key}")
    return float(value)


def _screen_decision(cells: dict[str, float]) -> dict:
    near_zero = sorted(key for key, value in cells.items() if abs(value) < 0.002)
    material = sorted(key for key, value in cells.items() if abs(value) >= 0.010)
    asymmetric = []
    for forward, reverse in REVERSE_PAIRS:
        gap = abs(cells[forward] - cells[reverse])
        if gap >= 0.010 and max(abs(cells[forward]), abs(cells[reverse])) >= 0.010:
            asymmetric.append({"directions": [forward, reverse], "absolute_gap": gap})

    if asymmetric:
        classification = "candidate_asymmetry"
    elif near_zero and material:
        classification = "candidate_selectivity"
    elif len(near_zero) == len(cells):
        classification = "all_near_zero"
    elif not near_zero and not material:
        classification = "ambiguous_sub_1pp"
    else:
        classification = "no_preregistered_structure_decision"

    return {
        "classification": classification,
        "near_zero_cells_lt_0.002": near_zero,
        "material_cells_ge_0.010": material,
        "candidate_asymmetries": asymmetric,
        "evidence_level": "single-seed screening",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="https://dagshub.com/Ke-vin-S/wavCSE.mlflow")
    args = parser.parse_args()

    study_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[5]
    load_dotenv(repo_root / ".env")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT}")

    candidates = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.study_id = '{STUDY_ID}' and tags.stage = '{STAGE}'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    finished = [run for run in candidates if run.info.status == "FINISHED"]
    by_arm = {}
    for run in finished:
        task_set = run.data.tags.get("task_set") or run.data.params.get("task_type")
        if task_set in ARMS and task_set not in by_arm:
            by_arm[task_set] = run

    missing = sorted(set(ARMS) - set(by_arm))
    if missing:
        raise RuntimeError(f"Missing finished Stage-A arms: {', '.join(missing)}")

    commits = {run.data.tags.get("mlflow.source.git.commit") for run in by_arm.values()}
    if None in commits or len(commits) != 1:
        raise RuntimeError(f"Stage-A arms do not share one recorded commit: {sorted(commits)}")
    commit = next(iter(commits))

    run_records = {}
    for arm, run in by_arm.items():
        arm_tasks = arm.split("_")
        run_records[arm] = {
            "status": "finished",
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "seed": int(run.data.params["seed"]),
            "git_commit": run.data.tags.get("mlflow.source.git.commit"),
            "metrics": {
                checkpoint: {
                    task: _metric(run, checkpoint, task) for task in arm_tasks
                }
                for checkpoint in ("epoch", "opt", "best")
            },
        }

    directed = {}
    checkpoint_sensitivity = {checkpoint: {} for checkpoint in ("opt", "best")}
    for cell, (pair_arm, target, single_arm) in PAIR_CELLS.items():
        directed[cell] = (
            run_records[pair_arm]["metrics"]["epoch"][target]
            - run_records[single_arm]["metrics"]["epoch"][target]
        )
        for checkpoint in checkpoint_sensitivity:
            checkpoint_sensitivity[checkpoint][cell] = (
                run_records[pair_arm]["metrics"][checkpoint][target]
                - run_records[single_arm]["metrics"][checkpoint][target]
            )

    matrix = {
        "ks": {"ks": None, "si": directed["ks<-si"], "er": directed["ks<-er"]},
        "si": {"ks": directed["si<-ks"], "si": None, "er": directed["si<-er"]},
        "er": {"ks": directed["er<-ks"], "si": directed["er<-si"], "er": None},
    }
    decision = _screen_decision(directed)
    generated_at = datetime.now(timezone.utc).isoformat()

    transfer_payload = {
        "study_id": STUDY_ID,
        "stage": STAGE,
        "status": "screening_complete",
        "generated_at": generated_at,
        "seed": 42,
        "git_commit": commit,
        "primary_checkpoint": "epoch",
        "definition": "T(target <- auxiliary) = accuracy(target jointly trained with auxiliary) - accuracy(target trained alone)",
        "matrix": matrix,
        "cells": directed,
        "checkpoint_sensitivity": checkpoint_sensitivity,
        "screening_decision": decision,
        "caveats": [
            "Single-seed screening evidence; no effect is confirmed.",
            "ER cells use the speaker-leaky historical split and require LOSO for claims.",
            "Pairwise arms contain more optimizer updates than single-task arms.",
            "Loss scaling and scheduler trajectories depend on task count.",
        ],
        "runs": {arm: record["run_id"] for arm, record in run_records.items()},
    }
    transfer_path = study_dir.parents[1] / "task_relations" / "empirical_transfer.json"
    transfer_path.write_text(json.dumps(transfer_payload, indent=2) + "\n")

    result_path = study_dir / "result.json"
    result = json.loads(result_path.read_text())
    result.update({
        "status": "stage_a_screening_complete",
        "stage": STAGE,
        "completed_at": generated_at,
        "git_commit": commit,
        "arms": run_records,
        "directed_transfer": {
            "primary_checkpoint": "epoch",
            "matrix": matrix,
            "cells": directed,
            "checkpoint_sensitivity": checkpoint_sensitivity,
        },
        "decision": decision,
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(transfer_payload, indent=2))


if __name__ == "__main__":
    main()
