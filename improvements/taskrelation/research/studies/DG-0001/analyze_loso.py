"""Analyze DG-0001's matched LOSO transfer confirmation.

The primary estimands use fixed-epoch ER accuracy paired by held-out speaker.
Test metrics are reported, never used to select a model or checkpoint.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv


STUDY_ID = "DG-0001"
STAGE = "stage_c_loso"
EXPERIMENT = "taskrelation-diagnostics"
ARMS = ("er", "ks_er", "si_er")
T_CRIT_DF9 = 2.262


def _paired_summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values)
    sem = sample_std / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_std": sample_std,
        "ci95_t9": [mean - T_CRIT_DF9 * sem, mean + T_CRIT_DF9 * sem],
        "positive_folds": sum(value > 0 for value in values),
        "negative_folds": sum(value < 0 for value in values),
        "zero_folds": sum(value == 0 for value in values),
        "range": [min(values), max(values)],
        "fold_values": values,
    }


def _fold_metric(summary: dict, checkpoint: str, task: str) -> list[float]:
    return [
        float(fold["task_metrics"][checkpoint][task]["acc"])
        for fold in summary["folds"]
    ]


def main() -> None:
    study_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[5]
    load_dotenv(repo_root / ".env")

    summaries = {}
    for arm in ARMS:
        path = (
            study_dir / "outputs" / "loso" / "results" / STUDY_ID / STAGE
            / arm / "kfold_summary.json"
        )
        if not path.exists():
            raise RuntimeError(f"Missing LOSO summary: {path}")
        summaries[arm] = json.loads(path.read_text())

    reference_speakers = [
        fold["held_out_test_speaker"] for fold in summaries["er"]["folds"]
    ]
    for arm in ARMS[1:]:
        speakers = [
            fold["held_out_test_speaker"] for fold in summaries[arm]["folds"]
        ]
        if speakers != reference_speakers:
            raise RuntimeError(f"LOSO speaker order differs for {arm}")

    result_path = study_dir / "result.json"
    result = json.loads(result_path.read_text())
    stage_a_arms = result["arms"]

    checkpoint_results = {}
    for checkpoint in ("epoch", "opt", "best"):
        er_single = _fold_metric(summaries["er"], checkpoint, "er")
        checkpoint_result = {}
        for auxiliary, pair_arm in (("ks", "ks_er"), ("si", "si_er")):
            er_pair = _fold_metric(summaries[pair_arm], checkpoint, "er")
            forward_values = [pair - single for pair, single in zip(er_pair, er_single)]
            reverse_pair = _fold_metric(summaries[pair_arm], checkpoint, auxiliary)
            reverse_reference = float(
                stage_a_arms[auxiliary]["metrics"][checkpoint][auxiliary]
            )
            reverse_values = [pair - reverse_reference for pair in reverse_pair]
            checkpoint_result[f"er<-{auxiliary}"] = _paired_summary(forward_values)
            checkpoint_result[f"{auxiliary}<-er"] = {
                **_paired_summary(reverse_values),
                "single_task_reference": reverse_reference,
                "reference_caveat": (
                    "Pairwise values vary over ER folds; the single-task reference "
                    "is the one matched Stage-A seed-42 run, not ten refits."
                ),
            }
        checkpoint_results[checkpoint] = checkpoint_result

    primary = checkpoint_results["epoch"]
    confirmed_directions = []
    for auxiliary in ("ks", "si"):
        forward = primary[f"er<-{auxiliary}"]
        reverse = primary[f"{auxiliary}<-er"]
        if forward["mean"] >= 0.010 and forward["ci95_t9"][0] > 0:
            confirmed_directions.append({
                "forward": f"er<-{auxiliary}",
                "forward_mean": forward["mean"],
                "forward_ci95": forward["ci95_t9"],
                "reverse": f"{auxiliary}<-er",
                "reverse_mean": reverse["mean"],
                "reverse_ci95": reverse["ci95_t9"],
            })

    classification = (
        "speaker_independent_operational_transfer_observed_optimizer_confound_unresolved"
        if confirmed_directions else "stage_a_operational_transfer_not_confirmed"
    )
    decision = {
        "classification": classification,
        "candidate_directions": confirmed_directions,
        "evidence_level": (
            "matched 10-fold speaker-independent LOSO; one seed; task semantics "
            "not identified because optimizer exposure and loss scaling change "
            "with task count"
        ),
    }

    mlflow.set_tracking_uri("https://dagshub.com/Ke-vin-S/wavCSE.mlflow")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.study_id = '{STUDY_ID}' and tags.stage = '{STAGE}'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    parents = [
        run for run in runs
        if run.info.status == "FINISHED"
        and not run.data.tags.get("mlflow.parentRunId")
    ]
    by_arm = {}
    for run in parents:
        arm = run.data.tags.get("task_set") or run.data.params.get("task_type")
        if arm in ARMS and arm not in by_arm:
            by_arm[arm] = run
    missing = sorted(set(ARMS) - set(by_arm))
    if missing:
        raise RuntimeError(f"Missing finished LOSO parent runs: {', '.join(missing)}")
    commits = {run.data.tags.get("mlflow.source.git.commit") for run in by_arm.values()}
    if None in commits or len(commits) != 1:
        raise RuntimeError(f"LOSO arms do not share one commit: {sorted(commits)}")
    commit = next(iter(commits))

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "study_id": STUDY_ID,
        "stage": STAGE,
        "status": "complete",
        "generated_at": generated_at,
        "seed": 42,
        "git_commit": commit,
        "held_out_speakers": reference_speakers,
        "primary_checkpoint": "epoch",
        "transfer": checkpoint_results,
        "decision": decision,
        "runs": {arm: run.info.run_id for arm, run in by_arm.items()},
        "caveats": [
            "One global seed; folds vary held-out speaker, not initialization.",
            "Pairwise arms have more optimizer updates than the single-task ER arm.",
            "Loss scaling and scheduler trajectories depend on task count.",
            "Reverse-direction LOSO sensitivities use one Stage-A single-task reference.",
        ],
    }
    loso_path = study_dir.parents[1] / "task_relations" / "loso_transfer.json"
    loso_path.write_text(json.dumps(payload, indent=2) + "\n")

    has_stage_d = "stage_d_optimization_controls" in result
    result.update({
        "status": "stage_d_optimization_controls" if has_stage_d else "complete",
        "stage": "stage_d_optimization_controls" if has_stage_d else STAGE,
        "completed_at": generated_at,
        "stage_c_loso": {
            "status": "complete",
            "config": "configs/loso.yml",
            "seed": 42,
            "folds": 10,
            "epochs_per_fold": 5,
            "git_commit": commit,
            "arms": {
                arm: {"status": "finished", "run_id": run.info.run_id}
                for arm, run in by_arm.items()
            },
            "primary_checkpoint": "epoch",
            "transfer": checkpoint_results,
            "decision": decision,
        },
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
