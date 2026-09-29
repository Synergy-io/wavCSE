"""Classify the TR-0007 screen under its pre-registered criteria.

Reads the three seed-42 screen runs from DagsHub (p-MSSL, classical MTRL, matched
wavCSE baseline) at the pre-registration commit and decides PROMISING / REJECTED /
INCONCLUSIVE exactly as `PLAN.md` declared *before* the runs:

  * PROMISING   -- p-MSSL's protocol-checkpoint aggregate test accuracy exceeds both
                   controls' AND every per-task accuracy is within 0.20pp of both.
  * REJECTED    -- otherwise, when all three arms produced a valid result.
  * INCONCLUSIVE-- an arm is missing, failed, or did not run under the shared
                   protocol (the protocol's own word for an arm that cannot run).

Nothing here is a confirmation, and nothing here may move the criteria: a screen can
only produce PROMISING or REJECTED (F1). ER numbers are the ordinary speaker-leaky
split, so they are screening context only and never an ER result (F3).

Usage:
    uv run --locked python improvements/taskrelation/research/studies/TR-0007/analyze_screen.py
"""

import json
import os
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
STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
COMMIT_FILE = STUDY_DIR / "screen_commit.txt"
SCREEN_COMMITS_FILE = STUDY_DIR / "screen_commits.json"


def _expected_commit():
    if COMMIT_FILE.exists():
        return COMMIT_FILE.read_text().strip()
    return None


def _screen_commits():
    """Per-arm expected commits.

    The screen normally runs all three arms at one commit. It ran at two when an
    ENOSPC during job preparation had to be fixed in the control plane (no
    scientific code changed): the candidate arm's collected run predates that
    fix, and ARC's per-(study, arm, seed) output keys make a same-arm re-run
    unable to persist its manifest. The file records that, so the analysis binds
    each arm to the commit its evidence came from instead of to one SHA.
    """
    if SCREEN_COMMITS_FILE.exists():
        return json.loads(SCREEN_COMMITS_FILE.read_text())
    return {}


def _forced_commit():
    """A commit filter supplied on the command line overrides the recorded one."""
    for index, token in enumerate(sys.argv):
        if token == "--commit" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].strip()
    return None


def _runs(client):
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
            forced = _forced_commit()
            if forced and commit != forced:
                continue
            expected = _screen_commits().get(method) or _expected_commit()
            if forced is None and expected and commit != expected:
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


def _run_summary(run):
    duration = None
    if run.info.end_time is not None:
        duration = (run.info.end_time - run.info.start_time) / 1000.0
    expected_commit = _forced_commit() or _expected_commit()
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
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    uri = os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI)
    client = MlflowClient(tracking_uri=uri)
    runs = _runs(client)

    arms = {}
    for method, found in runs.items():
        found.sort(key=lambda run: run.info.start_time or 0)
        arms[method] = [_run_summary(run) for run in found]

    decision, rationale = classify(arms)

    payload = {
        "study_id": STUDY_ID,
        "stage": STAGE,
        "seed": SEED,
        "checkpoint": CHECKPOINT,
        "expected_commit": _forced_commit() or _expected_commit(),
        "expected_commits_per_arm": _screen_commits(),
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
    out_path = STUDY_DIR / "screen_result.json"
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
