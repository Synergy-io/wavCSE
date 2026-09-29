"""DG-0007 runtime faithfulness gate -- deterministic, CPU-only, artifact-level.

`studies/DG-0007/PLAN.md` pre-registers the gate this script applies:

    "In every run, the logged final `Omega` must be the minimizer of
     `tr(Omega^-1 W^T W)` for the **un-normalized** task-parameter matrix to
     within `1e-4` relative, satisfy `tr(Omega) = 1 +- 1e-6`, and the realized
     penalty must be quadratic in `W`'s scale."

It reads a finished run's own artifacts -- the checkpoint the trainer wrote
(`torch.save(model.state_dict(), path)`) and the run-identity record the
training process emitted (`improvements/run_identity.py`, `ARC_RUN_IDENTITY`)
-- and does float64 linear algebra on tiny matrices. Nothing here trains,
imports the model class, or needs a GPU.

Exit status (documented contract):

    0  evidence is usable and the gate passed
    1  evidence is usable and the gate failed  (a *scientific* result)
    2  the evidence is unusable: corrupt, malformed, inconsistent, or not
       evidence for the requested run

Exit 2 is never produced by a scientific failure, and a malformed artifact never
gets to masquerade as a gate failure: every input is validated before any clause
is evaluated, and no exception escapes as the API result. The exit-2 report is
bounded *by value* (`_bounded_detail`), not by truncating serialized JSON: the
detail is evidence-derived and arbitrary, and truncating its serialization
produces a string that is no longer JSON, which is how a hostile or merely large
record used to escape as a traceback instead of exit 2.

Run identity binding. A checkpoint alone says nothing about which run produced
it -- a two-task checkpoint from any experiment will happily satisfy the
mathematics -- so the checker requires the run's `ARC_RUN_IDENTITY` file and
asserts the record names the requested study, stage, arm (`method`), model,
task type, representation and seed. `model` alone cannot separate the two MTRL
arms (both are `model="mtrl"`), which is why the identity record carries
`method`. Evidence whose identity does not match is not evidence for this run
and returns 2.

Two details of that contract matter:

* *Completeness is presence and emptiness, never truthiness.* `seed` is an
  integer and **0 is a valid seed** -- DG-0007's confirmation stage runs seeds
  0-4 -- so a zero-valued field is accepted while an absent field, `None`, an
  empty or whitespace string, or a non-integer seed are refused. `git_commit` is
  still refused when `resolve_git_commit` returned `None`.
* *The artifacts are bound to the record that describes them.* Each run writes
  into its own timestamped directory and `identity_record` records both
  absolute paths, so the checkpoint must live in the record's
  `checkpoints_dir` and the optional `omega_history.json` in its `results_dir`.
  Without that, run A's identity could be paired with run B's checkpoint
  whenever the two share an arm, task shape and seed. What this cannot detect is
  a foreign checkpoint deliberately copied *into* the run's own directory; that
  requires write access to the evidence directory, and the compute path binds
  its staged copies by sha256 in `MANIFEST.json`.

Readings of the clauses (recorded rather than silently chosen):

* Clause 1 (Omega is the closed form for the un-normalized `W`) is the
  artifact-level discriminator: it is evaluated against the checkpoint's own
  head parameters and genuinely fails for a row-normalized run (~+50 % gap).
* Clause 3's phrase "the realized penalty is quadratic in `W`'s scale" has two
  readings. With `Omega` held fixed it is `c^2` for **any** positive-definite
  `Omega`, so that reading cannot discriminate and is not a test; the checker
  applies the reading in which `Omega` is re-derived from the scaled parameters
  through the run's own `W` construction, which is what the audit and its review
  measured. Under that reading clause 3 is an algebraic identity for a
  `normalize_w=false` artifact, so it confirms the *declared* policy rather
  than testing the artifact; clause 2 is the same kind of confirmation. Only
  clause 1 discriminates on the artifact, and it is checked first.

Residual limitation (recorded, not fixed here). The artifact set carries one
`Omega` stream, so the checker cannot detect a run whose `Omega` step and
penalty step disagreed *in-process* (the penalty used `inv(Omega + eps*I)` from
the same buffer, and that inverse is not logged). The plan's diagnostics section
already requires the missing per-epoch instrumentation; until it exists,
clause 1 is the artifact-level proxy.
"""

import argparse
import json
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", ".."))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "downstream")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from improvements.run_identity import read_identity_file  # noqa: E402
from utils.constant_mapping import (  # noqa: E402
    LabelKeywordMapping,
    TaskDatasetMapping,
)

EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_BAD_INPUT = 2

IDENTITY_FIELDS = (
    "study_id", "stage", "method", "model", "task_type", "representation", "seed",
    "git_commit",
)

# The run-identity record also names where the run wrote its artifacts. Those
# paths are what bind an artifact to the record that describes it.
ARTIFACT_DIRECTORY_FIELDS = ("results_dir", "checkpoints_dir")

# `seed` is an integer; every other identity field is a non-empty string. The
# schema comes from `improvements/run_identity.py`: `identity_record` passes
# `seed` through from the entry points' `--seed`/config integer and adds
# `results_dir`/`checkpoints_dir` as absolute paths, and `research_identity`
# adds the research-block strings.
_INTEGER_IDENTITY_FIELDS = frozenset({"seed"})

_DIAGNOSTIC_LIMIT = 240
# A bad-input detail is evidence-derived, so it is bounded by shape as well: at
# most this many entries per container, this many nested containers, and this
# many characters of rendered leaves in total.
_DIAGNOSTIC_ITEMS = 16
_DIAGNOSTIC_DEPTH = 3
_DIAGNOSTIC_CHARS = 4096


class BadEvidence(Exception):
    """The supplied artifacts are unusable, corrupt or for another run."""

    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail or {}


def _bounded(value):
    """A short, safe rendering: never dump whole tensors or objects."""
    text = str(value).replace("\n", " ")
    if len(text) > _DIAGNOSTIC_LIMIT:
        text = text[:_DIAGNOSTIC_LIMIT] + "..."
    return text


def _bounded_detail(value, depth=0, budget=None):
    """`detail`, with every leaf bounded *before* it is serialized.

    A `BadEvidence` detail is evidence-derived and therefore arbitrary: a
    foreign identity record's field values, an unpickler's message, a shape
    list. The bound must therefore be applied to the *values*. Truncating the
    serialized JSON and parsing it back does not work and is what let a large
    enough detail escape as a `JSONDecodeError` traceback instead of the
    documented exit 2 -- the truncation is not valid JSON by construction.

    The result is always a plain JSON-serializable structure, so the caller's
    `json.dumps` cannot fail on it, and no tensor or whole object is ever
    materialized: an unrecognized object becomes `<TypeName>`, never `str(obj)`.
    Field-level diagnostics survive, because the depth budget below admits the
    deepest shape this checker builds (`{"mismatched": {field: {...}}}`).
    """
    budget = [_DIAGNOSTIC_CHARS] if budget is None else budget
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        # `json.dumps` renders a Python int of unbounded size verbatim, so a
        # hostile record could carry one giant number through every other bound.
        if value.bit_length() > 128:
            return "<int with {} bits>".format(value.bit_length())
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        text = _bounded(value)
        budget[0] -= len(text)
        return text
    if depth >= _DIAGNOSTIC_DEPTH or budget[0] <= 0:
        return "<{}>".format(type(value).__name__)
    if isinstance(value, dict):
        items = list(value.items())[:_DIAGNOSTIC_ITEMS]
        bounded = {}
        for key, item in items:
            name = key if isinstance(key, str) else "<{}>".format(type(key).__name__)
            bounded[_bounded(name)] = _bounded_detail(item, depth + 1, budget)
        if len(value) > len(items):
            bounded["..."] = "{} more keys".format(len(value) - len(items))
        return bounded
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)[:_DIAGNOSTIC_ITEMS]
        bounded = [_bounded_detail(item, depth + 1, budget) for item in items]
        if len(value) > len(items):
            bounded.append("... {} more items".format(len(value) - len(items)))
        return bounded
    return "<{}>".format(type(value).__name__)


def bad_input_payload(error):
    """The exit-2 report for one `BadEvidence`, already bounded field by field."""
    payload = {"error": _bounded(error), "kind": "bad_input"}
    if error.detail:
        payload["detail"] = _bounded_detail(error.detail)
    return payload


def identity_field_problem(field, value):
    """Why `value` cannot be used as this identity field, or None if it can.

    Deliberately not a truthiness test. `seed` is an integer and 0 is a
    legitimate seed -- DG-0007's confirmation stage runs seeds 0-4 -- so
    "absent", "None" and "an empty value where emptiness is forbidden" are
    checked explicitly, and a falsey-but-valid value such as the integer 0 is
    accepted. `git_commit` is left to `resolve_git_commit`, which returns None
    when the tree has no history, and None is still refused here.
    """
    if value is None:
        return "missing"
    if field in _INTEGER_IDENTITY_FIELDS:
        # bool is an int subclass, and True == 1 would silently match a seed
        # comparison, so it is refused as a seed rather than accepted.
        if isinstance(value, bool) or not isinstance(value, int):
            return "not an integer seed"
        return None
    if not isinstance(value, str) or not value.strip():
        return "empty"
    return None


def _load_state_dict(path):
    """Load a trainer checkpoint (a bare state_dict, or one under 'state_dict')."""
    if not os.path.isfile(path):
        raise BadEvidence("checkpoint is not a readable file", {"path": path})
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # truncated mid-write, unpickling error, ...
        raise BadEvidence(
            "checkpoint could not be loaded",
            {"path": path, "error": f"{type(error).__name__}: {_bounded(error)}"},
        )
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise BadEvidence(
            "checkpoint is not a state dict", {"type": type(payload).__name__}
        )
    return payload


def _task_parameter_matrix(state, normalize_w, scale=1.0, epsilon=0.0):
    """The adapter's W, as `mtrl_model.get_task_parameter_matrix` builds it.

    `classifiers.{t}` is read in ascending index order, matching the model's
    ModuleList order (which follows task_type). Each row is
    `[mean(weight, dim=0), mean(bias)]`, optionally row-unit-normalized with the
    same epsilon floor the model uses. `scale` multiplies the *parameters*
    first, then the run's own normalization acts on them -- otherwise clause 3
    could never bite.
    """
    indices = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith("classifiers.")}
    )
    if not indices:
        raise BadEvidence("checkpoint has no 'classifiers.*' parameters")
    rows = []
    for index in indices:
        weight = _checked_tensor(state, f"classifiers.{index}.weight")
        bias = _checked_tensor(state, f"classifiers.{index}.bias")
        if weight.dim() != 2:
            raise BadEvidence(
                "classifier weight is not a matrix",
                {"key": f"classifiers.{index}.weight", "shape": list(weight.shape)},
            )
        if bias.dim() != 1:
            raise BadEvidence(
                "classifier bias is not a vector",
                {"key": f"classifiers.{index}.bias", "shape": list(bias.shape)},
            )
        if weight.shape[0] != bias.shape[0]:
            raise BadEvidence(
                "classifier weight and bias disagree on the class count",
                {"index": index, "weight": list(weight.shape),
                 "bias": list(bias.shape)},
            )
        rows.append(torch.cat([weight.mean(dim=0), bias.mean().reshape(1)]))
    widths = {int(row.numel()) for row in rows}
    if len(widths) != 1:
        raise BadEvidence(
            "classifier summaries have inconsistent widths, so no task parameter "
            "matrix exists", {"widths": sorted(widths)}
        )
    matrix = torch.stack(rows, dim=0).double() * scale
    if normalize_w:
        matrix = matrix / (matrix.norm(dim=1, keepdim=True) + epsilon)
    return matrix


def expected_output_dims(task_type):
    """Per-task class counts for a task type, from the repository's own mapping.

    Mirrors `mtrl_model.DownstreamMultiTaskModelMTRL._output_dims_from_task_type`
    (dedupe datasets, preserving task_type order) so the checker can tell
    whether a checkpoint is structurally evidence for the requested task set
    without importing the model.
    """
    dims = []
    seen = set()
    for token in task_type.split("_"):
        dataset = TaskDatasetMapping.get_dataset_key(token)
        if dataset is None:
            raise BadEvidence("unknown task token in the requested task type",
                              {"task_type": task_type, "token": token})
        if dataset in seen:
            continue
        seen.add(dataset)
        label2index, _ = LabelKeywordMapping.get_label_mapping(dataset)
        dims.append(len(label2index))
    return dims


def _tensor(state, key):
    if key not in state:
        raise BadEvidence("checkpoint is missing a required parameter", {"key": key})
    value = state[key]
    if not torch.is_tensor(value):
        raise BadEvidence(
            "checkpoint parameter is not a tensor",
            {"key": key, "type": type(value).__name__},
        )
    return value


def validate_task_set(state, task_type):
    """The checkpoint must be shaped like evidence for the identity's task set."""
    dims = expected_output_dims(task_type)
    indices = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith("classifiers.")}
    )
    if len(indices) != len(dims):
        raise BadEvidence(
            "the checkpoint's task count does not match the run's task type",
            {"task_type": task_type, "expected_tasks": len(dims),
             "found_tasks": len(indices)},
        )
    for position, index in enumerate(indices):
        weight = _tensor(state, f"classifiers.{index}.weight")
        if weight.dim() != 2:
            raise BadEvidence(
                "classifier weight is not a matrix",
                {"key": f"classifiers.{index}.weight",
                 "shape": list(weight.shape)},
            )
        if int(weight.shape[0]) != dims[position]:
            raise BadEvidence(
                "a classifier's class count does not match the run's task type",
                {"task": position, "task_type": task_type,
                 "expected_classes": dims[position],
                 "found_classes": int(weight.shape[0])},
            )
    return dims


def _checked_tensor(state, key):
    if key not in state:
        raise BadEvidence("checkpoint is missing a required parameter", {"key": key})
    value = state[key]
    if not torch.is_tensor(value):
        raise BadEvidence(
            "checkpoint parameter is not a tensor",
            {"key": key, "type": type(value).__name__},
        )
    if not torch.isfinite(value).all():
        raise BadEvidence("checkpoint parameter has non-finite values", {"key": key})
    return value.double()


def sqrt_psd(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp(min=0.0)
    return (eigenvectors * eigenvalues.sqrt()) @ eigenvectors.T


def closed_form_omega(gram):
    root = sqrt_psd(gram)
    return root / torch.trace(root)


def relation_value(omega, gram):
    """tr(Omega^-1 A) -- the paper's subproblem value, TKDD Eq. (14)."""
    return torch.trace(torch.linalg.solve(omega, gram)).item()


def published_minimum(gram):
    """(tr A^(1/2))^2 -- Eq. (14)'s minimum over {Omega >= 0, tr(Omega) = 1}."""
    return torch.trace(sqrt_psd(gram)).item() ** 2


def require_within(path, declared_directory, label, field):
    """The artifact must live in the directory the identity names for it.

    The trainer writes each run into its own timestamped directory
    (`downstream/trainer/trainer_model.py`: `results_dir = <root>/<run_id>`,
    `ckpt_dir = <root>/<run_id>`), and `identity_record` records those two
    absolute paths. Containment is therefore what says "this file is that run's
    artifact": two runs of the same arm, shape and seed differ by directory, and
    the real path is compared so a symlink cannot smuggle a file across.

    What this cannot detect: someone deliberately copying a foreign checkpoint
    *into* the run's own directory. That needs write access to the evidence
    directory and is outside what the identity record can express; the compute
    path binds the staged copies by sha256 in `MANIFEST.json`.
    """
    if not declared_directory:
        raise BadEvidence(
            "the run-identity record names no {} directory, so this artifact "
            "cannot be bound to the run".format(field.replace("_", " ")),
            {"field": field},
        )
    real = os.path.realpath(os.path.abspath(path))
    root = os.path.realpath(os.path.abspath(os.path.expanduser(declared_directory)))
    try:
        contained = os.path.commonpath([real, root]) == root
    except ValueError:  # different drives / mixed absolute-relatives
        contained = False
    if not contained:
        raise BadEvidence(
            "this {} is not from the run the identity describes".format(label),
            {"artifact": real, "expected_{}".format(field): root},
        )


def load_identity(path, expected, expect_commit=None):
    """Read the run-identity file and bind it to the requested run.

    Uses `improvements/run_identity.py::read_identity_file`, the same reader the
    compute worker uses, so no parallel metadata format is introduced.
    """
    if not os.path.isfile(path):
        raise BadEvidence("run-identity file not found", {"path": path})
    records = read_identity_file(path)
    if not records:
        raise BadEvidence("run-identity file holds no readable record",
                          {"path": path})
    record = records[-1]
    problems = {
        field: problem
        for field in IDENTITY_FIELDS + ARTIFACT_DIRECTORY_FIELDS
        for problem in [identity_field_problem(field, record.get(field))]
        if problem is not None
    }
    if problems:
        raise BadEvidence(
            "run-identity record is incomplete; it cannot identify this run",
            {"fields": problems, "path": path},
        )
    mismatched = {
        field: {"expected": expected[field], "found": record.get(field)}
        for field in expected
        if record.get(field) != expected[field]
    }
    if mismatched:
        raise BadEvidence(
            "these artifacts are not evidence for the requested run",
            {"mismatched": mismatched},
        )
    if expect_commit is not None and record["git_commit"] != expect_commit:
        raise BadEvidence(
            "these artifacts were produced at a different commit",
            {"expected": expect_commit, "found": record["git_commit"]},
        )
    return record


def check(checkpoint, normalize_w, identity, expected_identity, omega_history=None,
          penalty_scale=3.0, tolerance_rel=1e-4, trace_tolerance=1e-6,
          scale_tolerance_rel=1e-3, omega_epsilon=1e-4, expect_commit=None):
    """Apply the pre-registered gate. Returns a JSON-serializable verdict.

    Raises `BadEvidence` when the artifacts are unusable or belong to another
    run; any other unexpected failure is converted to `BadEvidence` by `main`,
    so a malformed artifact can never be reported as a gate failure.
    """
    run_identity = load_identity(identity, expected_identity, expect_commit)

    # Bind the artifacts to the run the identity describes. Checking the
    # identity alone would accept run A's record paired with run B's checkpoint
    # whenever both are the same arm, shape and seed.
    require_within(checkpoint, run_identity["checkpoints_dir"], "checkpoint",
                   "checkpoints_dir")
    if omega_history is not None:
        require_within(omega_history, run_identity["results_dir"],
                       "Omega history", "results_dir")

    state = _load_state_dict(checkpoint)
    validate_task_set(state, run_identity["task_type"])
    if "omega" not in state:
        raise BadEvidence("checkpoint has no 'omega' buffer",
                          {"path": checkpoint})
    omega = _checked_tensor(state, "omega")
    if omega.dim() != 2 or omega.shape[0] != omega.shape[1]:
        raise BadEvidence(
            "the omega buffer is not a square matrix",
            {"shape": list(omega.shape)},
        )

    raw = _task_parameter_matrix(state, normalize_w=False)
    if omega.shape[0] != raw.shape[0]:
        raise BadEvidence(
            "the omega buffer and the task parameter matrix disagree on the "
            "task count",
            {"omega": list(omega.shape), "task_parameter_matrix": list(raw.shape)},
        )
    used = _task_parameter_matrix(
        state, normalize_w=normalize_w, epsilon=omega_epsilon
    )
    gram_raw = raw @ raw.T

    trace = torch.trace(omega).item()
    attained = relation_value(omega, gram_raw)
    minimum = published_minimum(gram_raw)
    gap = (attained - minimum) / minimum
    closed_form_residual = (omega - closed_form_omega(gram_raw)).abs().max().item()

    scaled = _task_parameter_matrix(
        state, normalize_w=normalize_w, scale=penalty_scale, epsilon=omega_epsilon
    )
    scaled_gram = scaled @ scaled.T
    ratio_gram = used @ used.T
    ratio = (
        relation_value(closed_form_omega(scaled_gram), scaled_gram)
        / relation_value(closed_form_omega(ratio_gram), ratio_gram)
    )

    if omega_history is not None:
        with open(omega_history, encoding="utf-8") as handle:
            payload = json.load(handle)
        logged = torch.tensor(payload["final_omega"]).double()
        if logged.shape != omega.shape:
            raise BadEvidence(
                "the logged Omega and the checkpoint's Omega have different shapes",
                {"logged": list(logged.shape), "checkpoint": list(omega.shape)},
            )
        if not torch.allclose(logged, omega, atol=1e-6):
            raise BadEvidence(
                "the final logged Omega is not the checkpoint's Omega buffer; the "
                "artifacts are from different runs",
                {"path": omega_history},
            )

    verdict = {
        "checkpoint": os.path.abspath(checkpoint),
        "omega_history": os.path.abspath(omega_history) if omega_history else None,
        "identity": {
            "path": os.path.abspath(identity),
            "record": run_identity,
        },
        "normalize_w": bool(normalize_w),
        "task_parameter_matrix_shape": list(used.shape),
        "task_row_norms": [round(v, 8) for v in raw.norm(dim=1).tolist()],
        "clauses": {
            "trace_equals_one": {
                "required": f"|tr(Omega) - 1| <= {trace_tolerance:g}",
                "tr_omega": trace,
                "passed": abs(trace - 1.0) <= trace_tolerance,
            },
            "omega_is_argmin_for_the_un_normalized_W": {
                "required": f"|gap| <= {tolerance_rel:g}",
                "attained_tr_omega_inv_WtW": attained,
                "published_minimum": minimum,
                "relative_gap": gap,
                "closed_form_max_abs_residual": closed_form_residual,
                "passed": abs(gap) <= tolerance_rel,
            },
            "penalty_is_quadratic_in_W_scale": {
                "required": f"|ratio - {penalty_scale ** 2:g}| / {penalty_scale ** 2:g} "
                            f"<= {scale_tolerance_rel:g}",
                "scale": penalty_scale,
                "ratio": ratio,
                "passed": abs(ratio - penalty_scale ** 2) / penalty_scale ** 2
                          <= scale_tolerance_rel,
            },
        },
    }
    verdict["passed"] = all(c["passed"] for c in verdict["clauses"].values())
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True,
                        help="the run's saved state_dict (.pth)")
    parser.add_argument("--normalize-w", required=True, choices=["true", "false"],
                        help="the run's model.normalize_w, from its committed config")
    parser.add_argument("--identity", required=True,
                        help="the run's ARC_RUN_IDENTITY file")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--method", required=True,
                        help="the arm's research.method, as in its study config")
    parser.add_argument("--model", required=True,
                        help="the runner's --model (mtrl, original, ...)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-type", default="ks_si_er")
    parser.add_argument("--representation", default="smp25")
    parser.add_argument("--expect-commit", default=None,
                        help="optional: the commit the study record pins")
    parser.add_argument("--omega-history", default=None,
                        help="optional results/omega_history.json for a cross-check")
    parser.add_argument("--penalty-scale", type=float, default=3.0)
    parser.add_argument("--tolerance-rel", type=float, default=1e-4)
    parser.add_argument("--trace-tolerance", type=float, default=1e-6)
    parser.add_argument("--scale-tolerance-rel", type=float, default=1e-3)
    parser.add_argument("--omega-epsilon", type=float, default=1e-4,
                        help="the run's model.omega_epsilon (the W row floor)")
    parser.add_argument("--json", action="store_true", help="machine-readable output only")
    args = parser.parse_args(argv)

    expected_identity = {
        "study_id": args.study_id,
        "stage": args.stage,
        "method": args.method,
        "model": args.model,
        "task_type": args.task_type,
        "representation": args.representation,
        "seed": args.seed,
    }
    try:
        verdict = check(
            checkpoint=args.checkpoint,
            normalize_w=args.normalize_w == "true",
            identity=args.identity,
            expected_identity=expected_identity,
            omega_history=args.omega_history,
            penalty_scale=args.penalty_scale,
            tolerance_rel=args.tolerance_rel,
            trace_tolerance=args.trace_tolerance,
            scale_tolerance_rel=args.scale_tolerance_rel,
            omega_epsilon=args.omega_epsilon,
            expect_commit=args.expect_commit,
        )
    except BadEvidence as error:
        print(json.dumps(bad_input_payload(error), indent=2, default=str))
        return EXIT_BAD_INPUT
    except Exception as error:  # never let an uncaught traceback be the result
        print(json.dumps({
            "error": "evidence could not be evaluated",
            "kind": "bad_input",
            "detail": {"error": f"{type(error).__name__}: {_bounded(error)}"},
        }, indent=2))
        return EXIT_BAD_INPUT

    if args.json:
        print(json.dumps(verdict, indent=2, default=str))
    else:
        for name, clause in verdict["clauses"].items():
            print(f"{'PASS' if clause['passed'] else 'FAIL'}  {name}  "
                  f"(required: {clause['required']})")
        identity = verdict["identity"]["record"]
        print(f"identity: {identity['study_id']} | {identity['stage']} | "
              f"{identity['method']} | s{identity['seed']} | "
              f"{identity['git_commit'][:12]}")
    return EXIT_PASS if verdict["passed"] else EXIT_GATE_FAIL


if __name__ == "__main__":
    sys.exit(main())
