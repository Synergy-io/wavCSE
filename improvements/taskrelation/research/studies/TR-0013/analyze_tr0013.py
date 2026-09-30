#!/usr/bin/env python
"""TR-0013 post-hoc analysis: the pre-registered classification and the mechanism.

Read-only over the runs' own recorded artifacts. It performs the classification the
pre-registration declares (`PLAN.md` section 6) from the runs' test metrics, makes the
validation-only selection the study registers (section 3), and measures the mechanism
bundle the study requires from the runs' own in-run artifacts and stored checkpoints --
lambda_2 and its selection rule, support/edge count, raw Omega, partial correlations, Omega
spectrum, summary Gram/row norms/cosines, relation- and task-gradient norms and their
ratio, onset epoch, plateau behaviour, per-task metrics and the aggregate.

Run ids are passed explicitly (`--runs studies/TR-0013/runs.json`), so the analysis is bound
to the runs the study collected and can never silently pick up a different one:

    {
      "mssl-l2-0p01": "<mlflow run id>",
      "mssl-l2-0p1": "<mlflow run id>",
      "classical-mtrl": "<mlflow run id>",
      "wavcse-baseline": "<mlflow run id>"
    }

Credentials come from the environment (`MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD`) and are never printed. Nothing here writes to the tracking
store. `--mechanism-dir` is where the in-run mechanism artifacts are copied (default
`studies/TR-0013/mechanism/`), each with its sha256 recorded; pass `--no-fetch` to analyse
only what is already on disk.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
CANDIDATES = ("mssl-l2-0p01", "mssl-l2-0p1")
CONTROLS = ("classical-mtrl", "wavcse-baseline")
PROTOCOL_CHECKPOINT = "epoch"
REGRESSION_LIMIT_PP = 0.2
SELECTION_TIE_BAND_PP = 0.2
SUMMARY_EPSILON = 1e-4
D = 2001
TASKS = ("ks", "si", "er")


def _repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").exists() and (parent / "improvements").is_dir():
            return parent
    raise RuntimeError("repository root not found above %s" % __file__)


REPO_ROOT = _repo_root()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def _fetch(client, run_id, artifact, cache_dir, name):
    local = Path(cache_dir) / name
    if not local.exists():
        downloaded = client.download_artifacts(run_id, artifact, str(cache_dir))
        local.write_bytes(Path(downloaded).read_bytes())
    return local


def _epoch_metrics(client, run_id, prefixes=("val_acc_all", "val_loss_all",
                                             "train_acc_all")):
    """Per-epoch metrics from the tracking record (the live per-epoch log)."""
    series = {}
    for metric in client.get_metric_history(run_id, "val_acc_all"):
        series.setdefault("val_acc_all", {})[int(metric.step)] = metric.value
    for key in prefixes[1:]:
        for metric in client.get_metric_history(run_id, key):
            series.setdefault(key, {})[int(metric.step)] = metric.value
    for key in ("relation_grad_norm_heads", "task_grad_norm_heads",
                "relation_over_task_grad_ratio", "relation_value"):
        history = client.get_metric_history(run_id, key)
        if history:
            series.setdefault(key, {})[int(metric.step)] = metric.value
    return series


def _summary_matrix(state_dict):
    rows = []
    for index in range(3):
        weight = state_dict["classifiers.%d.weight" % index].float()
        bias = state_dict["classifiers.%d.bias" % index].float()
        rows.append(torch.cat([weight.mean(dim=0), bias.mean().unsqueeze(0)]))
    W = torch.stack(rows, dim=0)
    norms = W.norm(dim=1, keepdim=True)
    return W, norms.flatten()


def _geometry(state_dict):
    W, raw_norms = _summary_matrix(state_dict)
    W_unit = W / (raw_norms.unsqueeze(1) + SUMMARY_EPSILON)
    gram = W_unit.double() @ W_unit.double().transpose(0, 1)
    eigenvalues = torch.linalg.eigvalsh(gram)
    entry = {
        "summary_cosines": [round(float(gram[0, 1]), 6), round(float(gram[0, 2]), 6),
                            round(float(gram[1, 2]), 6)],
        "summary_gram": [[round(float(value), 6) for value in row] for row in gram],
        "summary_gram_eigenvalues": [round(float(value), 6) for value in eigenvalues],
        "summary_raw_row_norms": [round(float(value), 6) for value in raw_norms],
        "summary_gram_condition": (
            round(float(eigenvalues[-1] / eigenvalues[0]), 3)
            if float(eigenvalues[0]) > 0 else None
        ),
    }
    if "omega" in state_dict:
        omega = state_dict["omega"].double()
        entry["stored_omega_trace"] = round(float(torch.trace(omega)), 4)
        entry["stored_omega_eigenvalues"] = [
            round(float(value), 4) for value in torch.linalg.eigvalsh(omega)
        ]
    return entry


def _checkpoint_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def _checkpoint_artifact(client, run_id, tag):
    """The run's own checkpoint for one tag, by the wrapper's own selection rule."""
    listing = client.list_artifacts(run_id, "checkpoints")
    names = [item.path for item in listing]
    matches = sorted(name for name in names if tag in os.path.basename(name).lower())
    if not matches:
        return None
    if tag == "epoch":
        numbered = []
        for name in matches:
            stem = os.path.basename(name)
            digits = "".join(character for character in stem.split("_")[-1]
                             if character.isdigit())
            if digits:
                numbered.append((int(digits), name))
        if numbered:
            numbered.sort()
            return numbered[-1][1]
    return matches[0]


def _plateau(series, window=5, tolerance_pp=0.2):
    """How long the validation curve has been flat at the end of training."""
    steps = sorted(series)
    if len(steps) < window:
        return {"available": False}
    values = [series[step] for step in steps]
    final = values[-1]
    flat_from = steps[-1]
    for step, value in zip(reversed(steps), reversed(values)):
        if abs(value - final) * 100.0 > tolerance_pp:
            flat_from = step + 1
            break
        flat_from = step
    best_step = steps[values.index(max(values))]
    tail = values[-window:]
    return {
        "available": True,
        "argmax_epoch": best_step,
        "final_epoch": steps[-1],
        "final_value": round(final, 6),
        "tail_window": window,
        "tail_spread_pp": round((max(tail) - min(tail)) * 100.0, 4),
        "flat_from_epoch": flat_from,
        "flat_epochs": steps[-1] - flat_from + 1,
    }


def _pp(value):
    return round(float(value) * 100.0, 4)



def _exact_support(omega):
    """Exact edge count from the recorded matrix.

    The in-run `support_edges` field of the TR-0013 runs is defective (`torch.triu`
    returns the whole matrix with the lower part zeroed, so the recorded count also
    counted that zeroed region: for a 3x3 snapshot with two zero edges it recorded
    zero_offdiagonals = 8 and support_edges = -5). The raw matrix is recorded
    correctly, so the count is recomputed here and both values are reported.
    """
    size = len(omega)
    zeros = sum(
        1 for i in range(size) for j in range(i + 1, size)
        if float(omega[i][j]) == 0.0
    )
    pairs = size * (size - 1) // 2
    return {"pairs": pairs, "zero_offdiagonals": zeros, "support_edges": pairs - zeros}


def _summarize_omega_history(payload):
    """Per-epoch relation object and summary geometry, as recorded in the run."""
    history = payload.get("history") or []
    rows = []
    for entry in history:
        omega = entry.get("omega") or []
        support = _exact_support(omega) if omega else {}
        rows.append({
            "epoch": entry.get("epoch"),
            "omega_trace": entry.get("omega_trace"),
            "omega_eigenvalues": entry.get("omega_eigenvalues"),
            "support_edges": support.get("support_edges"),
            "zero_offdiagonals": support.get("zero_offdiagonals"),
            "recorded_support_edges": entry.get("support_edges"),
            "recorded_zero_offdiagonals": entry.get("zero_offdiagonals"),
            "mean_abs_offdiagonal": entry.get("mean_abs_offdiagonal"),
            "partial_correlations": entry.get("partial_correlations"),
            "summary_cosines": entry.get("summary_cosines"),
            "summary_gram_eigenvalues": entry.get("summary_gram_eigenvalues"),
            "summary_raw_row_norms": entry.get("summary_raw_row_norms"),
            "coupling_value": entry.get("coupling_value"),
            "relative_duality_gap": entry.get("relative_duality_gap"),
        })
    collinear = [row for row in rows
                 if row["summary_cosines"] and min(row["summary_cosines"]) > 0.9]
    supports = [row["support_edges"] for row in rows if row["support_edges"] is not None]
    summary = {
        "snapshots": len(rows),
        "epochs": [row["epoch"] for row in rows],
        "support_edges_trajectory": supports,
        "support_edges_first": supports[0] if supports else None,
        "support_edges_last": supports[-1] if supports else None,
        "support_edges_final_snapshot": rows[-1]["support_edges"] if rows else None,
        "support_source": ("recomputed from the recorded Omega matrix; the in-run "
                           "support_edges / zero_offdiagonals fields of these runs are "
                           "defective (they counted the region torch.triu zeroes). The "
                           "recorded values are kept beside them for the record."),
        "summary_collinear_from_epoch": collinear[0]["epoch"] if collinear else None,
        "final": rows[-1] if rows else None,
        "rows": rows,
    }
    return summary


def _summarize_coupling_scale(payload):
    """The coupling's value and gradient scale beside the task gradient, per epoch."""
    history = payload.get("history") or []
    ratios = [(row.get("epoch"), row.get("relation_over_task_grad_ratio_heads"))
              for row in history if row.get("relation_over_task_grad_ratio_heads") is not None]
    values = [row.get("relation_value") for row in history
              if row.get("relation_value") is not None]
    task = [row.get("task_grad_norm_heads") for row in history
            if row.get("task_grad_norm_heads") is not None]
    relation = [row.get("relation_grad_norm_heads") for row in history
                if row.get("relation_grad_norm_heads") is not None]
    return {
        "probe": payload.get("probe"),
        "lambda_2": payload.get("lambda_2"),
        "lambda_2_selection": payload.get("lambda_2_selection"),
        "normalize_w": payload.get("normalize_w"),
        "warmup_epochs": payload.get("warmup_epochs"),
        "coupling_onset_epoch": payload.get("coupling_onset_epoch"),
        "epochs_probed": [row.get("epoch") for row in history],
        "coupling_value_trajectory": values,
        "task_grad_norm_heads_trajectory": task,
        "relation_grad_norm_heads_trajectory": relation,
        "ratio_trajectory": [value for _, value in ratios],
        "ratio_max": max((value for _, value in ratios), default=None),
        "ratio_min": min((value for _, value in ratios), default=None),
        "ratio_final": ratios[-1][1] if ratios else None,
        "ratio_epochs_above_10": [epoch for epoch, value in ratios if value > 10.0],
        "rows": history,
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="JSON mapping arm -> MLflow run id")
    parser.add_argument("--out", default=str(Path(__file__).with_name("result.json")))
    parser.add_argument("--mechanism-dir",
                        default=str(Path(__file__).with_name("mechanism")))
    parser.add_argument("--cache-dir", default="/tmp/tr0013-artifacts")
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args(argv)

    runs = json.loads(Path(args.runs).read_text())
    for arm in CANDIDATES + CONTROLS:
        if arm not in runs:
            raise SystemExit("runs file has no entry for arm %r" % arm)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mechanism_dir = Path(args.mechanism_dir)
    mechanism_dir.mkdir(parents=True, exist_ok=True)

    client = None if args.no_fetch else _client()
    result = {
        "study": "TR-0013",
        "kind": "post-hoc analysis of the screen's own runs",
        "changes_the_study_verdict": False,
        "tracking_uri": TRACKING_URI,
        "arms": {},
    }

    for arm, run_id in runs.items():
        entry = {"run_id": run_id}
        if client is not None:
            run = client.get_run(run_id)
            entry["run_name"] = run.data.tags.get("mlflow.runName")
            entry["experiment_id"] = run.info.experiment_id
            entry["status"] = run.info.status
            entry["git_commit"] = run.data.tags.get("git_commit")
            entry["layer_count"] = run.data.tags.get("layer_count")
            entry["layers"] = run.data.tags.get("layers")
            entry["study_id"] = run.data.tags.get("study_id")
            entry["stage"] = run.data.tags.get("stage")
            entry["method"] = run.data.tags.get("method")
            params = run.data.params
            entry["metrics"] = {
                tag: {
                    "acc_all": next((value.value for value in
                                     client.get_metric_history(run_id, "test_%s_acc_all" % tag)),
                                    None),
                    **{task: next((value.value for value in
                                   client.get_metric_history(
                                       run_id, "test_%s_%s_acc" % (tag, task))), None)
                       for task in TASKS},
                }
                for tag in ("epoch", "best", "opt")
            }
            for key in ("model.mssl_lambda_2", "mssl.lambda_2_selection",
                        "research.lambda_2_selection", "worker_gpu",
                        "mtrl_lambda", "model.mtrl_lambda"):
                if key in params:
                    entry[key] = params[key]
            curves = _epoch_metrics(client, run_id)
            entry["val_curve"] = {key: {str(step): value for step, value in sorted(
                series.items())} for key, series in curves.items()}
            entry["plateau_val_acc_all"] = _plateau(curves.get("val_acc_all", {}))
            for tag in ("best", "epoch", "opt"):
                artifact = _checkpoint_artifact(client, run_id, tag)
                if artifact is None:
                    continue
                local = _fetch(client, run_id, artifact, cache_dir,
                               "%s_%s.pth" % (arm, tag))
                entry.setdefault("checkpoints", {})[tag] = {
                    "artifact": artifact,
                    "sha256": _sha256(local),
                    "geometry": _geometry(_checkpoint_state(local)),
                }
            # The candidate's own mechanism artifacts, copied into the repository.
            for artifact, name in (("omega/omega_history.json", "omega_history.json"),
                                   ("mechanism/coupling_scale.json", "coupling_scale.json")):
                try:
                    local = _fetch(client, run_id, artifact, cache_dir,
                                   "%s_%s" % (arm, name))
                except Exception as exc:  # pragma: no cover - absent for control arms
                    entry.setdefault("mechanism_artifacts", {})[name] = None
                    continue
                target = mechanism_dir / ("%s_%s" % (arm, name))
                target.write_bytes(local.read_bytes())
                entry.setdefault("mechanism_artifacts", {})[name] = {
                    "sha256": _sha256(target),
                    "repository_path": str(target.relative_to(REPO_ROOT)),
                }
                payload = json.loads(target.read_text())
                if name == "omega_history.json":
                    entry["omega_history"] = _summarize_omega_history(payload)
                else:
                    entry["coupling_scale"] = _summarize_coupling_scale(payload)
        result["arms"][arm] = entry

    # ---- the registered validation-only selection -------------------------------
    selection = {"rule": "higher val_acc_all at the protocol checkpoint (epoch 30); "
                         "within the %.2fpp tie band prefer lambda_2 = 0.01" % SELECTION_TIE_BAND_PP}
    for arm in CANDIDATES:
        curve = result["arms"][arm].get("val_curve", {}).get("val_acc_all", {})
        selection[arm] = {
            "val_acc_all_epoch30": (curve.get("30") if curve else None),
            "val_acc_all_final_step": (
                curve[str(max(int(step) for step in curve))] if curve else None
            ),
            "final_step": (max(int(step) for step in curve) if curve else None),
        }
    values = {arm: selection[arm]["val_acc_all_final_step"] for arm in CANDIDATES}
    if all(value is not None for value in values.values()):
        spread_pp = abs(values["mssl-l2-0p01"] - values["mssl-l2-0p1"]) * 100.0
        selection["spread_pp"] = round(spread_pp, 4)
        if spread_pp <= SELECTION_TIE_BAND_PP:
            selection["selected"] = "mssl-l2-0p01"
            selection["discriminating"] = False
            selection["reason"] = ("both values are within the %.2fpp tie band; the smaller "
                                   "lambda_2 is the registered representative" % SELECTION_TIE_BAND_PP)
        else:
            selection["selected"] = max(values, key=values.get)
            selection["discriminating"] = True
            selection["reason"] = "higher validation accuracy at the protocol checkpoint"
    result["validation_selection"] = selection

    # ---- the pre-registered classification --------------------------------------
    classification = {"rule": (
        "PROMISING iff the selected candidate's test_epoch_acc_all exceeds both controls' "
        "and every per-task test_epoch_*_acc is within %.2fpp of both; else REJECTED"
        % REGRESSION_LIMIT_PP)}
    selected = selection.get("selected")
    if selected is not None:
        candidate = result["arms"][selected]["metrics"]["epoch"]
        controls = {arm: result["arms"][arm]["metrics"]["epoch"] for arm in CONTROLS}
        classification["selected_arm"] = selected
        classification["candidate"] = candidate
        classification["controls"] = controls
        deltas = {}
        regressions = []
        for arm, control in controls.items():
            delta = {"all_pp": _pp(candidate["acc_all"] - control["acc_all"])}
            for task in TASKS:
                difference = _pp(candidate[task] - control[task])
                delta[task + "_pp"] = difference
                if difference < -REGRESSION_LIMIT_PP:
                    regressions.append("%s %s %.4fpp" % (arm, task, difference))
            deltas["vs_" + arm] = delta
        classification["deltas"] = deltas
        beats = all(candidate["acc_all"] > control["acc_all"] for control in controls.values())
        classification["beats_both_aggregates"] = beats
        classification["per_task_regressions_beyond_0.20pp"] = regressions
        classification["classification"] = (
            "PROMISING" if beats and not regressions else "REJECTED"
        )
        classification["caveats"] = [
            "Screen evidence only: one seed, one split; cannot establish an improvement (F1).",
            "ER is the ordinary, speaker-leaky split: context only, not an ER result (F3).",
            "lambda_2 is validation-selected over the paper's two smallest grid values "
            "(%s), so the classification is scoped to that axis."
            % ("discriminating" if selection.get("discriminating") else
               "indiscriminate at this scale"),
            "No confirmation, no LOSO and no ablation were run, and none is authorized.",
        ]
    result["classification"] = classification

    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
    print("wrote %s" % args.out)
    print("selection:", json.dumps(result["validation_selection"], indent=1))
    if selected is not None:
        print("classification:", classification["classification"],
              "deltas:", json.dumps(classification["deltas"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
