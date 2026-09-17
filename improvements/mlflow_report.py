"""
Cross-category experiment comparison / leaderboard for wavCSE improvements.

MLflow's own compare / parallel-coordinates UI (DagsHub's hosted view
included) works within one experiment at a time. Since this repo's tracking
convention puts each (category, architecture) pair -- base, taskrelation-gbc,
taskrelation-tsm, taskrelation-pmr, and eventually lowrank-*/clustering-*/
decomposition-* -- in its own experiment (see improvements/README.md), there
is no single place to compare "everything we've tried" without this script.

Pulls every run across every experiment matching the naming convention via
mlflow.search_runs(), and prints a single table -- category, model, task_type,
pooling, and each checkpoint tag's accuracy -- sorted by the best requested
metric. Relies on the `category`/`model`/`pooling_frame`/`pooling_layer` tags
set by improvements/mlflow_utils.py's set_standard_tags(), and the
test_{tag}_acc_all metrics set by its log_eval_stats() -- both only present on
runs from scripts wired up to mlflow_utils (currently base/ and
run_improvements.py; extend as lowrank/clustering/decomposition come online).

Usage:
    python mlflow_report.py [--metric test_opt_acc_all] [--top 20] [--csv report.csv]
"""

import os
import argparse

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # .../wavCSE (git repo root)

TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"

# Only base/base-derived runs carry the "wavcse-" prefix; every other
# category is its own top-level architecture (see improvements/README.md's
# naming convention table).
CATEGORY_PREFIXES = (
    "wavcse-",
    "taskrelation-",
    "lowrank-",
    "clustering-",
    "decomposition-",
)

DISPLAY_COLUMNS = [
    "tags.category",
    "tags.model",
    "params.task_type",
    "tags.pooling_frame",
    "tags.pooling_layer",
    "metrics.test_opt_acc_all",
    "metrics.test_best_acc_all",
    "metrics.test_epoch_acc_all",
    "experiment_name",
    "run_name",
    "start_time",
]


def _find_experiments(client):
    experiments = client.search_experiments()
    return [e for e in experiments if e.name.startswith(CATEGORY_PREFIXES)]


def build_report(metric="metrics.test_opt_acc_all"):
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
        if col not in df.columns:
            df[col] = None

    if metric not in df.columns:
        print(f"Warning: metric '{metric}' not found in any run; sorting by start_time instead.")
        sort_col = "start_time"
    else:
        sort_col = metric

    df = df.sort_values(sort_col, ascending=False)
    return df[DISPLAY_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description="Cross-category wavCSE experiment leaderboard")
    parser.add_argument("--metric", type=str, default="metrics.test_opt_acc_all",
                         help="Column to sort by, e.g. metrics.test_opt_acc_all")
    parser.add_argument("--top", type=int, default=20, help="Rows to print")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write the full report as CSV")
    args = parser.parse_args()

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    mlflow.set_tracking_uri(TRACKING_URI)

    report = build_report(metric=args.metric)
    if report is None:
        return

    if args.csv:
        report.to_csv(args.csv, index=False)
        print(f"Wrote {len(report)} rows to {args.csv}")

    print(report.head(args.top).to_markdown(index=False))


if __name__ == "__main__":
    main()
