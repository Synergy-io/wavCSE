"""Classify the TR-0007 screen under its pre-registered criteria.

Reads the three seed-42 MLflow runs (p-MSSL, classical MTRL and matched
wavCSE baseline) at their explicitly supplied commits. Applies TR-0007's
pre-registered PROMISING / REJECTED / INCONCLUSIVE rule:

  * PROMISING   -- p-MSSL's protocol-checkpoint aggregate test accuracy exceeds both
                   controls' AND every per-task accuracy is within 0.20pp of both.
  * REJECTED    -- otherwise, when all three arms produced a valid result.
  * INCONCLUSIVE-- an arm is missing, failed, or did not run under the shared
                   protocol (the protocol's own word for an arm that cannot run).

Nothing here is a confirmation, and nothing here may move the criteria: a screen can
only produce PROMISING or REJECTED (F1). ER numbers are the ordinary speaker-leaky
split, so they are screening context only and never an ER result (F3).

Usage: uv run python improvements/taskrelation/experiments/TR-0007/analyze_screen.py \
  --commits <external-commits.json> --out <external-result.json>

"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

STUDY_ID = "TR-0007"
STAGE = "screen"
SEED = 42
TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
TASKS = ("ks", "si", "er")
METHODS = ("p-mssl", "classical-mtrl", "wavcse-baseline")
CANDIDATE = "p-mssl"
CONTROLS = ("classical-mtrl", "wavcse-baseline")
# The arm's own run name lives in a different experiment from each control's
# (the naming convention is one experiment per category/architecture), so the
# runs are found by tag, not by experiment.
EXPERIMENTS = ("taskrelation-mssl", "taskrelation-mtrl", "wavcse-baseline")
REGRESSION_LIMIT_PP = 0.20
CHECKPOINT = "epoch"
REPO_ROOT = Path(__file__).resolve().parents[4]


def _expected_commits(path):
    """Validate the exact per-arm commit map supplied with the private record."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not set(METHODS) <= set(document):
        raise ValueError("the commit map must name all three screen arms")
    if set(document) - set(METHODS) - {"note"}:
        raise ValueError("the commit map contains an unknown arm")
    if any(not isinstance(document[name], str) or
           not re.fullmatch(r"[0-9a-f]{40}", document[name]) for name in METHODS):
        raise ValueError("each arm requires an exact full Git commit")
    return {name: document[name] for name in METHODS}

def _runs(client, expected_commits):
    found = {}
    for experiment_name in EXPERIMENTS:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue
        query = "tags.study_id = '{}' and tags.stage = '{}' and tags.seed = '{}'".format(
            STUDY_ID, STAGE, SEED
        )
        for run in client.search_runs([experiment.experiment_id], filter_string=query):
            method = run.data.tags.get("method")
            if method not in METHODS:
                continue
            commit = run.data.tags.get("git_commit")
            if commit != expected_commits[method]:
                continue
            # A duplicate execution that completed but whose outputs were never
            # collected is never screen evidence, however good its metrics look.
            if run.data.tags.get("status") == "duplicate_not_collected":
                continue
            found.setdefault(method, []).append(run)
    return found


def _metrics(run, checkpoint=CHECKPOINT):
    """The protocol's primary checkpoint metrics, plus the other tags for context."""

    record = {}
    for tag in ("epoch", "best", "opt"):
        key = f"test_{tag}_acc_all"
        if key in run.data.metrics:
            record[tag] = {"acc_all": float(run.data.metrics[key])}
            for task in TASKS:
                metric = f"test_{tag}_{task}_acc"
                if metric in run.data.metrics:
                    record[tag][task] = float(run.data.metrics[metric])
    if checkpoint not in record:
        raise RuntimeError(
            "run {} has no {} checkpoint metrics (has: {})".format(
                run.info.run_id, checkpoint, ", ".join(sorted(record)) or "none"
            )
        )
    return record


def _run_summary(run, expected_commits):
    duration = None
    if run.info.end_time is not None:
        duration = (run.info.end_time - run.info.start_time) / 1000.0
    expected_commit = expected_commits[run.data.tags["method"]]
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "experiment": run.data.tags.get("mlflow.experimentName")
        or run.info.experiment_id,
        "status": run.info.status,
        "duration_seconds": duration,
        "git_commit": run.data.tags.get("git_commit"),
        "commit_matches_expected": (
            None if expected_commit is None else run.data.tags.get("git_commit") == expected_commit
        ),
        "seed": run.data.tags.get("seed"),
        "method": run.data.tags.get("method"),
        "representation": run.data.tags.get("representation"),
        "layers": run.data.tags.get("layers"),
        "layer_count": run.data.tags.get("layer_count"),
        "lambda_2": run.data.params.get("model.mssl_lambda_2"),
        "lambda_2_selection": run.data.params.get("mssl.lambda_2_selection"),
        "worker_gpu": run.data.params.get("worker_gpu"),
        "metrics": _metrics(run),
    }


def classify(arms):
    """Apply the pre-registered screen rule. No post-hoc thresholds."""

    missing = [method for method in METHODS if not arms.get(method)]
    if missing:
        return "INCONCLUSIVE", "no collected run for: {}".format(", ".join(missing))
    for method, runs in arms.items():
        for run in runs:
            if run["status"] != "FINISHED":
                return "INCONCLUSIVE", "{} run {} is {}".format(
                    method, run["run_id"], run["status"])

    results = {method: arms[method][0] for method in METHODS}
    candidate = results[CANDIDATE]["metrics"][CHECKPOINT]
    deltas = {}
    winning_controls = []
    regressions = []
    for control in CONTROLS:
        control_metrics = results[control]["metrics"][CHECKPOINT]
        per_task = {}
        for task in TASKS:
            delta_pp = (candidate[task] - control_metrics[task]) * 100.0
            per_task[task] = delta_pp
            if delta_pp < -REGRESSION_LIMIT_PP:
                regressions.append("{} {} {:.4f}pp".format(control, task, delta_pp))
        delta_all_pp = (candidate["acc_all"] - control_metrics["acc_all"]) * 100.0
        per_task["all"] = delta_all_pp
        if delta_all_pp > 0:
            winning_controls.append(control)
        deltas[control] = per_task

    beats_both = len(winning_controls) == len(CONTROLS)
    if beats_both and not regressions:
        return "PROMISING", (
            "p-MSSL beats both controls at the {} checkpoint (deltas: {}) with no task "
            "regressing beyond {:.2f}pp".format(
                CHECKPOINT,
                ", ".join("{} {:.4f}pp".format(k, v["all"]) for k, v in deltas.items()),
                REGRESSION_LIMIT_PP,
            )
        )
    reasons = []
    if not beats_both:
        reasons.append(
            "aggregate accuracy does not exceed: {}".format(
                ", ".join(sorted(set(CONTROLS) - set(winning_controls))) or "no control"
            )
        )
    if regressions:
        reasons.append("per-task regression beyond {:.2f}pp: {}".format(
            REGRESSION_LIMIT_PP, "; ".join(regressions)))
    return "REJECTED", "; ".join(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commits", type=Path, required=True,
                        help="Exact per-arm Git commits from the private Study record")
    parser.add_argument("--out", type=Path, required=True,
                        help="External destination for the screen result")
    args = parser.parse_args()
    try:
        expected_commits = _expected_commits(args.commits)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    uri = os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI)
    client = MlflowClient(tracking_uri=uri)
    runs = _runs(client, expected_commits)

    arms = {}
    for method, found in runs.items():
        found.sort(key=lambda run: run.info.start_time or 0)
        arms[method] = [_run_summary(run, expected_commits) for run in found]

    decision, rationale = classify(arms)

    payload = {
        "study_id": STUDY_ID,
        "stage": STAGE,
        "seed": SEED,
        "checkpoint": CHECKPOINT,
        "expected_commits_per_arm": expected_commits,
        "regression_limit_pp": REGRESSION_LIMIT_PP,
        "classification": decision,
        "rationale": rationale,
        "caveats": [
            "Screen evidence only: one seed, one split; cannot establish an improvement (F1).",
            "ER is the ordinary, speaker-leaky split: screening context only, not an ER result (F3).",
            "lambda_2 = 0.01 is a researcher-fixed screening value (DEC-0016), not a paper default, "
            "so a REJECTED outcome is scoped to this fixed lambda_2.",
            "No confirmation was run and none is authorized by this screen.",
            "Where the arms ran at different commits, the diff between those commits is "
            "control-plane only (job-prep failure classification); no scientific "
            "configuration or training code differs.",
        ],
        "arms": arms,
    }
    out_path = args.out
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("TR-0007 screen classification: {}".format(decision))
    print("  {}".format(rationale))
    for method in METHODS:
        for run in arms.get(method, []) or [{"run_id": "—", "metrics": {}}]:
            metrics = run.get("metrics", {}).get(CHECKPOINT, {})
            print("  {:18s} acc_all={:.4f} ks={:.4f} si={:.4f} er={:.4f} ({})".format(
                method,
                metrics.get("acc_all", float("nan")),
                metrics.get("ks", float("nan")),
                metrics.get("si", float("nan")),
                metrics.get("er", float("nan")),
                run.get("run_id", "—"),
            ))
    print("wrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
