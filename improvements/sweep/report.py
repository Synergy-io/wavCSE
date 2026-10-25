"""Aggregate a sweep from MLflow, ranked on validation.

The grouping the sweep promises lives in two places: one study folder on disk
(every config, every ledger, every output root) and one MLflow experiment whose
runs all carry ``sweep_id``, ``combo`` and ``seed`` tags. This module reads the
second one back, because that is where the metrics are and where a reader who
was not present can check the work.

Ranking is on **validation**, not test. A layer combination that wins on the
held-out test split has already spent that split; the selection rule this
project runs on is validation-first, and a report that sorted by test accuracy
would quietly invite the selection it is meant to document. Test metrics are
printed beside the ranking, clearly marked as context.
"""

import json
import os

# The tags a report reads back off each run. Declared as a constant rather than
# inlined so a future field is added in one place, and so a reader can see the
# full grouping vocabulary the sweep promises.
MLFLOW_TAG_KEYS = ("sweep_id", "combo", "seed", "stage", "study_id", "method",
                   "representation", "layer_count", "git_commit")


class ReportError(Exception):
    """The sweep cannot be summarized from the tracking store."""


def _client(tracking_uri=None):
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ReportError("mlflow is not importable in this environment: {}".format(exc))
    return MlflowClient(tracking_uri=tracking_uri) if tracking_uri else MlflowClient()


def _experiment_id(client, name):
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        raise ReportError("no MLflow experiment named {!r} on this tracking "
                          "server".format(name))
    return experiment.experiment_id


def _final(metrics, key):
    return metrics.get(key)


def _peak_val(client, run_id, key="val_acc_all"):
    """The best validation value a run recorded, from its metric history."""
    try:
        history = client.get_metric_history(run_id, key)
    except Exception:  # pragma: no cover - store-dependent
        return None
    return max((point.value for point in history), default=None)


def collect(spec, tracking_uri=None):
    """Every run of this sweep, with the fields a report needs."""
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") \
        or (spec.get("mlflow") or {}).get("tracking_uri")
    client = _client(tracking_uri)
    experiment_id = _experiment_id(client, spec["experiment_name"])
    filter_string = "tags.sweep_id = '{}'".format(spec["sweep_id"])
    runs = client.search_runs([experiment_id], filter_string=filter_string,
                              max_results=5000)
    rows = []
    for run in runs:
        tags = dict(run.data.tags)
        metrics = dict(run.data.metrics)
        peak_val = _peak_val(client, run.info.run_id)
        rows.append({
            "run_id": run.info.run_id,
            "status": run.info.status,
            "combo": tags.get("combo") or tags.get("sweep_combo"),
            "group": tags.get("sweep_group"),
            "k": tags.get("sweep_k"),
            "seed": tags.get("seed"),
            "stage": tags.get("stage"),
            "layer_count": tags.get("layer_count"),
            "representation": tags.get("representation"),
            "commit": (tags.get("git_commit") or "")[:12],
            "val_acc_all": peak_val,
            "test_opt_acc_all": _final(metrics, "test_opt_acc_all"),
            "test_best_acc_all": _final(metrics, "test_best_acc_all"),
            "test_epoch_acc_all": _final(metrics, "test_epoch_acc_all"),
            "test_opt_ks_acc": _final(metrics, "test_opt_ks_acc"),
            "test_opt_si_acc": _final(metrics, "test_opt_si_acc"),
            "test_opt_er_acc": _final(metrics, "test_opt_er_acc"),
        })
    return rows


def rank(rows, metric="val_acc_all"):
    """Sort rows by the ranking metric, dropping runs that never reported one."""
    usable = [row for row in rows if row.get(metric) is not None]
    return sorted(usable, key=lambda row: row[metric], reverse=True)


def by_combo(rows, metric="val_acc_all"):
    """Peak value per combination, and the seeds that produced it."""
    peaks = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        current = peaks.get(row["combo"])
        if current is None or row[metric] > current[metric]:
            peaks[row["combo"]] = row
    return sorted(peaks.values(), key=lambda row: row[metric], reverse=True)


def render(rows, metric="val_acc_all", top=None):
    """A fixed-width table, because the report is read in a terminal."""
    ranked = rank(rows, metric)
    if top:
        ranked = ranked[:top]
    header = "{:<22}{:<28}{:>3}{:>5}{:>6}{:>10}{:>12}{:>12}{:>9}{:>9}{:>9}".format(
        "combo", "group", "k", "seed", "L", "val_all", "test_opt", "test_best",
        "ks", "si", "er")
    lines = [header, "-" * len(header)]
    for row in ranked:
        lines.append(
            "{:<22}{:<28}{:>3}{:>5}{:>6}{:>10}{:>12}{:>12}{:>9}{:>9}{:>9}".format(
                str(row.get("combo"))[:22], str(row.get("group") or "-")[:28],
                str(row.get("k") if row.get("k") is not None else "-"),
                str(row["seed"]), str(row["layer_count"]),
                _fmt(row["val_acc_all"]), _fmt(row["test_opt_acc_all"]),
                _fmt(row["test_best_acc_all"]), _fmt(row["test_opt_ks_acc"]),
                _fmt(row["test_opt_si_acc"]), _fmt(row["test_opt_er_acc"])))
    lines.append("")
    lines.append("ranked on {} (validation); test_* columns are context, not the "
                 "selection rule".format(metric))
    return "\n".join(lines)


def _fmt(value):
    return "-" if value is None else "{:.5f}".format(float(value))


def as_json(rows, metric="val_acc_all"):
    return json.dumps({"metric": metric, "runs": rank(rows, metric)},
                      sort_keys=True, indent=2)
