"""
Checkpoint-tag rescan for the `er` (emotion recognition) task.

Every run logged through improvements/mlflow_utils.py's log_eval_stats()
already carries three per-task accuracy numbers for `er` -- one per
checkpoint-selection tag (opt/best/epoch) -- but only `test_opt_acc_all`
(the overall, all-tasks number at the "opt" tag) is normally reported as a
run's headline result. Because `er` is small (553 test samples, 4 classes)
and overfits hard, the "opt" tag (selected by avg-of-per-task accuracy
across ks/si/er) sometimes leaves `er` accuracy on the table relative to
the "best" or "epoch" (final) tags for that same run.

This is a pure MLflow query -- no GPU, no re-evaluation, no retraining.
Reuses improvements/mlflow_report.py's experiment-discovery pattern.

Usage:
    python er_tag_report.py [--top 20] [--csv report.csv]
"""

import os
import argparse

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # .../wavCSE (git repo root)

TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"

# Same naming convention as mlflow_report.py.
CATEGORY_PREFIXES = (
    "wavcse-",
    "taskrelation-",
    "lowrank-",
    "clustering-",
    "decomposition-",
)

ER_METRICS = [
    "metrics.test_opt_er_acc",
    "metrics.test_best_er_acc",
    "metrics.test_epoch_er_acc",
]

DISPLAY_COLUMNS = [
    "tags.category",
    "tags.model",
    "params.task_type",
    "tags.pooling_frame",
    "tags.pooling_layer",
] + ER_METRICS + [
    "er_gap",
    "er_best_any_tag",
    "experiment_name",
    "run_name",
    "start_time",
]


def _find_experiments(client):
    experiments = client.search_experiments()
    return [e for e in experiments if e.name.startswith(CATEGORY_PREFIXES)]


def build_report():
    client = MlflowClient()
    experiments = _find_experiments(client)
    if not experiments:
        print("No experiments matching the naming convention were found.")
        return None

    exp_id_to_name = {e.experiment_id: e.name for e in experiments}
    df = mlflow.search_runs(
        experiment_ids=list(exp_id_to_name.keys()),
        output_format="pandas",
    )
    if df.empty:
        print("Matching experiments exist but have no runs yet.")
        return None

    df["experiment_name"] = df["experiment_id"].map(exp_id_to_name)
    df["run_name"] = df.get("tags.mlflow.runName", "")

    for col in DISPLAY_COLUMNS:
        if col not in df.columns and col not in ("er_gap", "er_best_any_tag"):
            df[col] = None

    # Only rows that actually have at least one `er` metric logged are
    # relevant here (runs that never trained/evaluated `er` show all-NaN).
    has_er = df[ER_METRICS].notna().any(axis=1)
    df = df[has_er].copy()
    if df.empty:
        print("No runs found with logged er metrics.")
        return None

    df["er_best_any_tag"] = df[ER_METRICS].max(axis=1)
    df["er_gap"] = df["er_best_any_tag"] - df["metrics.test_opt_er_acc"]

    return df[DISPLAY_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description="Rescan already-logged runs for the best er accuracy per checkpoint tag"
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to print per view")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write the full report as CSV")
    args = parser.parse_args()

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    mlflow.set_tracking_uri(TRACKING_URI)

    report = build_report()
    if report is None:
        return

    if args.csv:
        report.to_csv(args.csv, index=False)
        print(f"Wrote {len(report)} rows to {args.csv}")

    print("\n=== Runs where 'opt' leaves the most er accuracy on the table (sorted by er_gap) ===")
    print(report.sort_values("er_gap", ascending=False).head(args.top).to_markdown(index=False))

    print("\n=== Best er accuracy ever recorded, any tag (sorted by er_best_any_tag) ===")
    print(report.sort_values("er_best_any_tag", ascending=False).head(args.top).to_markdown(index=False))


if __name__ == "__main__":
    main()
