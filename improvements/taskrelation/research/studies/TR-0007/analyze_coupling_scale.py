#!/usr/bin/env python
"""TR-0007 post-hoc: where the coupling term's scale comes from.

Read-only analysis of the TR-0007 screen's own stored artifacts. It changes no
result, re-opens no protocol and promotes nothing: the screen's classification
stays `REJECTED` at the pre-registered researcher-fixed `lambda_2 = 0.01`
(`result.json`, `analysis.md`). What it adds is the measurement the screen's
mechanism section could make only from `Omega` history: the *summary geometry
of all three arms*, obtained from the checkpoints each arm already published,
plus the behaviour of the published `lambda_0` / `lambda_2` axes at that
geometry.

Why a checkpoint can answer this at all: the task-parameter summary `W` is a
pure function of the classifier heads (`mtrl_model.get_task_parameter_matrix`
and its MSSL twin) — per task, `[mean(classifier.weight, dim=0); mean(bias)]`,
unit-normalised because `normalize_w` is on in both relation arms. The
checkpoints on DagsHub carry those heads, so `R = W W^T` (whose off-diagonals
are the pairwise cosine similarities of the three task summaries) and the
stored `omega` can be read without retraining anything.

Three measurements, all deterministic:

1. per-arm summary geometry at the protocol's `best` checkpoint (three arms of
   the seed-42 screen, same commit line, same 30-epoch budget);
2. the published `lambda_2` axis (`{0.01, 0.1, 1, 10, 100}`, JMLR §4.1) solved
   with this repository's own `graphical_lasso_admm` on each measured Gram;
3. the published `lambda_0` axis (`lambda_0 = 1` in the paper; varied here only
   to show what it controls).

Run (network read of the run artifacts, no GPU, no writes to any store):

    uv run --locked python \
        improvements/taskrelation/research/studies/TR-0007/analyze_coupling_scale.py

Credentials come from the environment (`MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD`, as `improvements/mlflow_utils.py` expects) and are
never printed. `--checkpoints-dir DIR` reuses already-downloaded
`<arm>_best.pth` files instead of fetching them.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


def _repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").exists() and (parent / "improvements").is_dir():
            return parent
    raise RuntimeError("repository root not found above %s" % __file__)


REPO_ROOT = _repo_root()

# The three arms of the TR-0007 screen, seed 42, by MLflow run id. The
# candidate's duplicate execution (f7550613, identical metrics to five
# decimals) is deliberately not used: one run per arm, as the study collected.
ARMS = [
    ("p-mssl", "a82144d2d0004b55a8f986c10a78e3f2",
     "checkpoints/train_ks_si_er_best.pth"),
    ("classical-mtrl", "1e40694559d348db9bc0f72687371fc0",
     "checkpoints/train_ks_si_er_best.pth"),
    ("wavcse-baseline", "641b25abd4dc432e8db546bd8ff720f9",
     "checkpoints/train_ks_si_er_best.pth"),
]

TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
SUMMARY_EPSILON = 1e-4        # mtrl_model / mssl_model row-normalisation floor
D = 2001                      # summary length: 2000 hidden + 1 mean bias
LAMBDA_2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)   # JMLR §4.1 classification grid
LAMBDA_0_VALUES = (1.0, 0.1, 0.01, 1e-3, 1.0 / D, 1e-4)
# The scale the screen recorded for its task gradients (`analysis.md`, epoch-4
# comparison "coupling gradient ~1e5 against task gradients ~1e-2"). Used only
# as a reference for reporting a ratio; it is not a measured quantity of this
# script.
RECORDED_TASK_GRADIENT_SCALE = 1e-2
# The coupling term the screen recorded on the candidate run (analysis.md:
# "the reported training loss is dominated by the coupling term (~2015)").
RECORDED_COUPLING_LOSS = 2015.0


def _load_mssl_solver():
    """Import the arm's own solver by path (0N- folders are not packages)."""
    path = REPO_ROOT / "improvements/taskrelation/04-mssl/mssl_model.py"
    spec = importlib.util.spec_from_file_location("tr0007_mssl_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary_matrix(state_dict):
    """The shared class-mean summary adapter, unit-normalised (`normalize_w`)."""
    rows = []
    for index in range(3):
        weight = state_dict["classifiers.%d.weight" % index].float()
        bias = state_dict["classifiers.%d.bias" % index].float()
        rows.append(torch.cat([weight.mean(dim=0), bias.mean().unsqueeze(0)]))
    W = torch.stack(rows, dim=0)
    norms = W.norm(dim=1, keepdim=True)
    return W / (norms + SUMMARY_EPSILON), norms.flatten()


def _gram_from_cosines(c01, c02, c12):
    R = torch.eye(3, dtype=torch.double)
    R[0, 1] = R[1, 0] = c01
    R[0, 2] = R[2, 0] = c02
    R[1, 2] = R[2, 1] = c12
    return R


def _solve(solver, gram, lambda_2, lambda_0):
    cov = (gram.double() / D)
    omega = solver.graphical_lasso_admm(
        cov, lambda_2, D, lambda_0=lambda_0, max_iterations=2000, tolerance=1e-8
    )
    return omega.double()


def _omega_report(omega, gram):
    off = [omega[0, 1], omega[0, 2], omega[1, 2]]
    partial = [
        -float(omega[i, j]) / math.sqrt(float(omega[i, i] * omega[j, j]))
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    return {
        "exact_zero_offdiagonals": int(sum(1 for value in off if abs(float(value)) < 1e-12)),
        "trace": round(float(torch.trace(omega)), 2),
        "diagonal_share_of_abs_mass": round(
            float(torch.diagonal(omega).sum() / omega.abs().sum()), 4
        ),
        "partial_correlations": [round(value, 4) for value in partial],
        "coupling_value": round(float(torch.trace(gram.double() @ omega)), 2),
        "coupling_value_per_task": round(float(torch.trace(gram.double() @ omega)) / 3.0, 2),
    }


def _gradient_scale(omega, lambda_0):
    """|2 * lambda_0 * Omega @ W|_F for unit-norm rows (W = I is that scale)."""
    identity = torch.eye(3, dtype=torch.double)
    return round(float((2.0 * lambda_0 * omega @ identity).norm()), 1)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(arm, run_id, artifact, cache_dir):
    local = cache_dir / ("%s_best.pth" % arm)
    if local.exists():
        return local
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    downloaded = client.download_artifacts(run_id, artifact, str(cache_dir))
    local.write_bytes(Path(downloaded).read_bytes())
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default="/tmp/tr0007-checkpoints")
    parser.add_argument("--out", default=str(Path(__file__).with_name("coupling_scale_result.json")))
    args = parser.parse_args()

    cache_dir = Path(args.checkpoints_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver = _load_mssl_solver()

    result = {
        "study": "TR-0007",
        "kind": "post-hoc read-only analysis of the screen's stored artifacts",
        "changes_the_study_verdict": False,
        "tracking_uri": TRACKING_URI,
        "summary_length": D,
        "recorded_reference_scales": {
            "task_gradient_scale": RECORDED_TASK_GRADIENT_SCALE,
            "coupling_loss_candidate_run": RECORDED_COUPLING_LOSS,
        },
        "arms": {},
    }

    grams = {}
    for arm, run_id, artifact in ARMS:
        path = _fetch(arm, run_id, artifact, cache_dir)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        W, raw_norms = _summary_matrix(state)
        gram = W.double() @ W.double().transpose(0, 1)
        grams[arm] = gram
        eigenvalues = torch.linalg.eigvalsh(gram)
        entry = {
            "run_id": run_id,
            "artifact": artifact,
            "artifact_sha256": _sha256(path),
            "raw_summary_row_norms": [round(float(value), 4) for value in raw_norms],
            "summary_gram": [[round(float(value), 6) for value in row] for row in gram],
            "summary_cosines": [
                round(float(gram[0, 1]), 6),
                round(float(gram[0, 2]), 6),
                round(float(gram[1, 2]), 6),
            ],
            "summary_gram_eigenvalues": [round(float(value), 6) for value in eigenvalues],
            "summary_gram_condition": (
                round(float(eigenvalues[-1] / eigenvalues[0]), 3)
                if float(eigenvalues[0]) > 0 else None
            ),
        }
        if "omega" in state:
            omega = state["omega"].double()
            entry["stored_omega"] = [[round(float(value), 3) for value in row] for row in omega]
            entry.update(_omega_report(omega, gram))
            entry["stored_omega_eigenvalues"] = [
                round(float(value), 2) for value in torch.linalg.eigvalsh(omega)
            ]
        result["arms"][arm] = entry

    # The published lambda_2 axis at every measured Gram. A Gram is the whole
    # input to the Omega step (S = R/d), so these are exact solves of Eq. (8)
    # at the geometry each arm actually reached, not a training run.
    result["lambda_2_axis"] = {}
    for arm, gram in grams.items():
        rows = []
        for lambda_2 in LAMBDA_2_GRID:
            omega = _solve(solver, gram, lambda_2, 1.0)
            row = {"lambda_2": lambda_2}
            row.update(_omega_report(omega, gram))
            row["coupling_gradient_scale"] = _gradient_scale(omega, 1.0)
            row["coupling_gradient_over_recorded_task_scale"] = round(
                row["coupling_gradient_scale"] / RECORDED_TASK_GRADIENT_SCALE, 1
            )
            rows.append(row)
        result["lambda_2_axis"][arm] = rows

    # The pre-activation geometry the screen recorded for the candidate
    # (`analysis.md`: implied cosines +0.261, +0.035, +0.009 at epoch 3, the
    # last epoch before the coupling engaged). Re-solved here so the axis is
    # read at the geometry the coupling *started* from as well as the one it
    # ended at.
    early = _gram_from_cosines(0.261, 0.035, 0.009)
    early_rows = []
    for lambda_2 in LAMBDA_2_GRID:
        omega = _solve(solver, early, lambda_2, 1.0)
        row = {"lambda_2": lambda_2}
        row.update(_omega_report(omega, early))
        row["coupling_gradient_scale"] = _gradient_scale(omega, 1.0)
        row["coupling_gradient_over_recorded_task_scale"] = round(
            row["coupling_gradient_scale"] / RECORDED_TASK_GRADIENT_SCALE, 1
        )
        early_rows.append(row)
    result["lambda_2_axis"]["recorded_epoch3_geometry"] = early_rows
    result["lambda_2_axis_gram_note"] = (
        "recorded_epoch3_geometry uses the three implied cosines the screen's own "
        "analysis reports for epoch 3; it is that record's geometry, not a re-measurement"
    )

    # What lambda_0 controls, at the uncoupled (baseline) Gram, lambda_2 = 0.01.
    lambda_0_rows = []
    for lambda_0 in LAMBDA_0_VALUES:
        omega = _solve(solver, grams["wavcse-baseline"], 0.01, lambda_0)
        row = {
            "lambda_0": lambda_0,
            "lambda_0_over_published": round(lambda_0 / 1.0, 9),
            "trace": round(float(torch.trace(omega)), 2),
            "coupling_value": round(
                float(lambda_0 * torch.trace(grams["wavcse-baseline"].double() @ omega)), 2
            ),
            "coupling_value_per_task": round(
                float(lambda_0 * torch.trace(grams["wavcse-baseline"].double() @ omega)) / 3.0, 2
            ),
            "coupling_gradient_scale": _gradient_scale(omega, lambda_0),
            "exact_zero_offdiagonals": int(
                sum(
                    1 for value in (omega[0, 1], omega[0, 2], omega[1, 2])
                    if abs(float(value)) < 1e-12
                )
            ),
        }
        lambda_0_rows.append(row)
    result["lambda_0_axis_on_baseline_gram"] = lambda_0_rows

    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
    print("wrote %s" % args.out)
    for arm, entry in result["arms"].items():
        print("%-17s cos=%s  gram_eig=%s" % (
            arm, entry["summary_cosines"], entry["summary_gram_eigenvalues"]))


if __name__ == "__main__":
    main()
