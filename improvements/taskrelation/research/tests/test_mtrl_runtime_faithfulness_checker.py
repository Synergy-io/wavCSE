"""DG-0007 runtime-faithfulness gate: classification, validation and binding.

Three contracts are pinned here.

*Classification.* Valid normalization-corrected evidence passes (exit 0); valid
evidence whose mathematics is the historical row-normalized kind is a
scientific failure (exit 1).

*Refusal.* Anything unusable -- corrupt, truncated, malformed, structurally
inconsistent -- exits 2 with an actionable bounded diagnostic, never a gate
failure and never an uncaught traceback.

*Binding.* Evidence must belong to the run the identity record describes. The
record's own `checkpoints_dir`/`results_dir` bind the artifact to it, so run
A's identity paired with run B's checkpoint is refused even when the two share
an arm, task shape, representation and seed. Identity completeness is presence
and emptiness, never truthiness: `seed = 0` is a valid seed (DG-0007's
confirmation stage runs seeds 0-4).

Artifacts are built independently of the checker (numpy float64 closed form), so
a fault in the checker's own algebra cannot hide behind its own helper.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
STUDY_DIR = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "studies", "DG-0007"
)
MTRL_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "01-mtrl")
for path in (STUDY_DIR, REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from check_runtime_faithfulness import (  # noqa: E402
    BadEvidence,
    _DIAGNOSTIC_LIMIT,
    check,
    main,
)

CLASS_COUNTS = (12, 1251, 4)
HIDDEN = 6
COMMIT = "a" * 40

# The trainer writes each run under `<root>/<run_id>`
# (downstream/trainer/trainer_model.py: results_dir = <results_root>/<run_id>,
# ckpt_dir = <checkpoints_root>/<run_id>), so both directories are per-run.
RUN_ID = "2026_09_30_12_00_00"

EXPECTED_IDENTITY = {
    "study_id": "DG-0007",
    "stage": "screen",
    "method": "mtrl_norm_corrected",
    "model": "mtrl",
    "task_type": "ks_si_er",
    "representation": "smp25",
    "seed": 42,
}

# Template only: `GateBase.identity()` fills the directories in from the temp
# directory that actually holds the artifacts, because the checker binds the
# artifact to the record's own paths.
IDENTITY_RECORD = dict(EXPECTED_IDENTITY, git_commit=COMMIT, run_id=RUN_ID,
                       checkpoint_run_id=RUN_ID, results_dir="", checkpoints_dir="")


def _independent_closed_form_omega(gram):
    """numpy float64: A^(1/2)/tr(A^(1/2)) -- deliberately not the checker's code."""
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    return root / np.trace(root)


def _state_dict_with(omega, seed=0, class_counts=CLASS_COUNTS, widths=None):
    generator = torch.Generator().manual_seed(seed)
    state = {"omega": torch.tensor(omega, dtype=torch.float32)}
    widths = widths or [HIDDEN] * len(class_counts)
    for index, classes in enumerate(class_counts):
        state[f"classifiers.{index}.weight"] = torch.randn(
            classes, widths[index], generator=generator
        )
        state[f"classifiers.{index}.bias"] = torch.randn(classes, generator=generator)
    return state


def _head_summary(state, normalize_w, epsilon=1e-4, count=None):
    rows = []
    for index in range(count if count is not None else len(CLASS_COUNTS)):
        weight = state[f"classifiers.{index}.weight"].double()
        bias = state[f"classifiers.{index}.bias"].double()
        rows.append(torch.cat([weight.mean(dim=0), bias.mean().reshape(1)]))
    matrix = torch.stack(rows)
    if normalize_w:
        matrix = matrix / (matrix.norm(dim=1, keepdim=True) + epsilon)
    return matrix


def _write(path, payload):
    torch.save(payload, path)
    return path


class GateBase(unittest.TestCase):
    """Shared scaffolding: one run's real directories, artifacts and record."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = self._tmp.name
        self.results_dir = os.path.join(self.directory, "results", RUN_ID)
        self.checkpoints_dir = os.path.join(self.directory, "checkpoints", RUN_ID)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def artifact(self, normalize_w, name="train_ks_si_er_epoch30.pth",
                 directory=None, **kwargs):
        """A checkpoint as the trainer would have written it, for that policy."""
        state = _state_dict_with(np.eye(3), **kwargs)
        matrix = _head_summary(state, normalize_w=normalize_w)
        state = dict(state)
        state["omega"] = torch.tensor(
            _independent_closed_form_omega((matrix @ matrix.T).numpy()),
            dtype=torch.float32,
        )
        return _write(os.path.join(directory or self.checkpoints_dir, name), state)

    def identity(self, name="identity.jsonl", **overrides):
        record = dict(IDENTITY_RECORD, results_dir=self.results_dir,
                      checkpoints_dir=self.checkpoints_dir)
        record.update(overrides)
        return _identity_file(self.directory, name, record)

    def run_check(self, checkpoint, identity, normalize_w=False, expected=None,
                  **kwargs):
        return check(
            checkpoint=checkpoint,
            normalize_w=normalize_w,
            identity=identity,
            expected_identity=dict(expected or EXPECTED_IDENTITY),
            **kwargs,
        )

    def cli(self, checkpoint, identity, normalize_w=False, seed=None, extra=()):
        return main([
            "--checkpoint", checkpoint,
            "--identity", identity,
            "--normalize-w", "true" if normalize_w else "false",
            "--study-id", EXPECTED_IDENTITY["study_id"],
            "--stage", EXPECTED_IDENTITY["stage"],
            "--method", EXPECTED_IDENTITY["method"],
            "--model", EXPECTED_IDENTITY["model"],
            "--seed", str(EXPECTED_IDENTITY["seed"] if seed is None else seed),
            "--json", *extra,
        ])


def _identity_file(directory, name="identity.jsonl", record=None):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record if record is not None
                                else dict(IDENTITY_RECORD)) + "\n")
    return path


class ValidEvidenceIsClassified(GateBase):
    def test_corrected_run_artifacts_pass_every_clause(self):
        verdict = self.run_check(self.artifact(False), self.identity())
        self.assertTrue(verdict["passed"], verdict)
        for name, clause in verdict["clauses"].items():
            self.assertTrue(clause["passed"], f"{name}: {clause}")
        self.assertEqual(verdict["identity"]["record"]["method"],
                         "mtrl_norm_corrected")

    def test_historical_run_artifacts_fail_the_published_W_clauses(self):
        verdict = self.run_check(self.artifact(True), self.identity(),
                                 normalize_w=True)
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["clauses"]["trace_equals_one"]["passed"])
        self.assertFalse(
            verdict["clauses"]["omega_is_argmin_for_the_un_normalized_W"]["passed"]
        )
        self.assertFalse(
            verdict["clauses"]["penalty_is_quadratic_in_W_scale"]["passed"]
        )

    def test_historical_verdict_reports_the_measured_gap(self):
        verdict = self.run_check(self.artifact(True), self.identity(),
                                 normalize_w=True)
        clause = verdict["clauses"]["omega_is_argmin_for_the_un_normalized_W"]
        self.assertGreater(clause["relative_gap"], 0.40)
        ratio = verdict["clauses"]["penalty_is_quadratic_in_W_scale"]["ratio"]
        self.assertGreater(ratio, 0.99)
        self.assertLess(ratio, 1.01)

    def test_omega_history_cross_check(self):
        checkpoint = self.artifact(False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        history = os.path.join(self.results_dir, "omega_history.json")
        with open(history, "w", encoding="utf-8") as handle:
            json.dump({"final_omega": state["omega"].tolist()}, handle)
        verdict = self.run_check(checkpoint, self.identity(), omega_history=history)
        self.assertEqual(verdict["omega_history"], os.path.abspath(history))

        with open(history, "w", encoding="utf-8") as handle:
            json.dump({"final_omega": (state["omega"] + 0.2).tolist()}, handle)
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(checkpoint, self.identity(), omega_history=history)
        self.assertIn("different runs", str(caught.exception))


class SeedIsValidatedByPresenceNotTruthiness(GateBase):
    """`seed = 0` is a valid seed; the confirmation stage runs seeds 0-4."""

    def confirmation_evidence(self, seed):
        """A confirmation-stage artifact and its matching identity record."""
        checkpoint = self.artifact(
            False, name=f"train_ks_si_er_epoch30.pth"
        )
        identity = self.identity(
            name=f"identity_seed{seed}.jsonl", stage="confirm", seed=seed
        )
        return checkpoint, identity, dict(EXPECTED_IDENTITY, stage="confirm",
                                          seed=seed)

    def test_zero_one_and_forty_two_seeds_are_accepted(self):
        for seed in (0, 1, 42):
            with self.subTest(seed=seed):
                checkpoint, identity, expected = self.confirmation_evidence(seed)
                verdict = self.run_check(checkpoint, identity,
                                         expected=expected)
                self.assertTrue(verdict["passed"], verdict)
                self.assertEqual(verdict["identity"]["record"]["seed"], seed)

    def test_zero_seed_evidence_exits_zero_end_to_end(self):
        """The reviewer's case: seed 0 must not be refused as bad evidence."""
        checkpoint, identity, expected = self.confirmation_evidence(0)
        self.assertEqual(
            self.cli(checkpoint, identity, seed=0,
                     extra=("--stage", "confirm")),
            0,
        )

    def test_missing_and_none_and_empty_seeds_are_refused(self):
        for label, mutate in (
            ("missing", lambda record: record.pop("seed")),
            ("none", lambda record: record.__setitem__("seed", None)),
            ("empty string", lambda record: record.__setitem__("seed", "")),
            ("blank string", lambda record: record.__setitem__("seed", "   ")),
            ("numeric string", lambda record: record.__setitem__("seed", "0")),
            ("boolean", lambda record: record.__setitem__("seed", True)),
        ):
            with self.subTest(case=label):
                checkpoint = self.artifact(False)
                record = dict(IDENTITY_RECORD, results_dir=self.results_dir,
                              checkpoints_dir=self.checkpoints_dir)
                mutate(record)
                identity = _identity_file(
                    self.directory, f"seed_{label.replace(' ', '_')}.jsonl", record
                )
                with self.assertRaises(BadEvidence) as caught:
                    self.run_check(checkpoint, identity)
                self.assertIn("incomplete", str(caught.exception))
                self.assertEqual(self.cli(checkpoint, identity), 2)

    def test_expected_seed_one_against_identity_seed_zero_is_refused(self):
        """Fixing falsey handling must not weaken identity matching."""
        checkpoint, identity, _ = self.confirmation_evidence(0)
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(
                checkpoint, identity,
                expected=dict(EXPECTED_IDENTITY, stage="confirm", seed=1),
            )
        self.assertIn("not evidence for the requested run", str(caught.exception))
        self.assertEqual(
            self.cli(checkpoint, identity, seed=1, extra=("--stage", "confirm")), 2
        )

    def test_empty_string_text_fields_are_still_refused(self):
        for field in ("study_id", "method", "representation", "git_commit"):
            with self.subTest(field=field):
                checkpoint = self.artifact(False)
                with self.assertRaises(BadEvidence) as caught:
                    self.run_check(checkpoint, self.identity(**{field: ""}))
                self.assertIn("incomplete", str(caught.exception))


class MalformedEvidenceIsRefused(GateBase):
    """Every unusable artifact is bad input (exit 2), not a gate failure."""

    def malformed_cases(self):
        good = self.artifact(False)
        identity = self.identity()
        directory = self.checkpoints_dir

        missing = os.path.join(directory, "absent.pth")

        truncated = os.path.join(directory, "truncated.pth")
        with open(good, "rb") as source:
            blob = source.read()
        with open(truncated, "wb") as handle:
            handle.write(blob[: len(blob) // 3])

        non_dict = _write(os.path.join(directory, "non_dict.pth"), [1, 2, 3])

        no_omega = os.path.join(directory, "no_omega.pth")
        state = _state_dict_with(np.eye(3))
        del state["omega"]
        _write(no_omega, state)

        bad_omega = os.path.join(directory, "bad_omega.pth")
        state = _state_dict_with(np.eye(3))
        state["omega"] = torch.zeros(6, 6)
        _write(bad_omega, state)

        bad_omega_vector = os.path.join(directory, "bad_omega_vector.pth")
        state = _state_dict_with(np.eye(3))
        state["omega"] = torch.zeros(3)
        _write(bad_omega_vector, state)

        ragged = os.path.join(directory, "ragged.pth")
        state = _state_dict_with(np.eye(3), widths=[HIDDEN, HIDDEN + 1, HIDDEN])
        state["omega"] = torch.eye(3) / 3.0
        _write(ragged, state)

        swapped = os.path.join(directory, "swapped.pth")
        state = _state_dict_with(np.eye(3))
        state["classifiers.0.bias"] = torch.zeros(11)
        _write(swapped, state)

        non_finite = os.path.join(directory, "non_finite.pth")
        state = _state_dict_with(np.eye(3))
        state["classifiers.1.weight"][0, 0] = float("nan")
        _write(non_finite, state)

        two_task = os.path.join(directory, "two_task.pth")
        state = _state_dict_with(np.eye(2) / 2.0, class_counts=(12, 4),
                                 widths=[HIDDEN] * 2)
        _write(two_task, state)

        return {
            "missing checkpoint": (missing, identity),
            "truncated checkpoint": (truncated, identity),
            "non-dict checkpoint": (non_dict, identity),
            "missing omega": (no_omega, identity),
            "malformed omega shape": (bad_omega, identity),
            "vector omega": (bad_omega_vector, identity),
            "inconsistent classifier widths": (ragged, identity),
            "classifier bias mismatch": (swapped, identity),
            "non-finite values": (non_finite, identity),
            "two-task checkpoint": (two_task, identity),
        }

    def test_each_malformed_artifact_raises_bad_evidence(self):
        for label, (checkpoint, identity) in self.malformed_cases().items():
            with self.subTest(case=label):
                with self.assertRaises(BadEvidence):
                    self.run_check(checkpoint, identity)

    def test_each_malformed_artifact_exits_two(self):
        for label, (checkpoint, identity) in self.malformed_cases().items():
            with self.subTest(case=label):
                self.assertEqual(self.cli(checkpoint, identity), 2)

    def test_the_diagnostic_is_bounded_and_does_not_dump_tensors(self):
        checkpoint, identity = self.malformed_cases()["malformed omega shape"]
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(checkpoint, identity)
        rendered = json.dumps(caught.exception.detail, default=str)
        self.assertLess(len(rendered), 500)
        self.assertNotIn("tensor(", rendered)

    def test_a_scientific_failure_is_not_reported_as_bad_input(self):
        """Valid evidence with the historical mathematics is exit 1, not 2."""
        self.assertEqual(
            self.cli(self.artifact(True), self.identity(), normalize_w=True), 1
        )

    def test_valid_evidence_exits_zero(self):
        self.assertEqual(self.cli(self.artifact(False), self.identity()), 0)


class EvidenceIsBoundToItsRun(GateBase):
    """A checkpoint that satisfies the mathematics is not thereby this run's."""

    def test_wrong_identity_fields_are_refused(self):
        checkpoint = self.artifact(False)
        for field, wrong in (
            ("study_id", "DG-0005"),
            ("stage", "confirm"),
            ("method", "mtrl"),
            ("seed", 7),
            ("model", "original"),
            ("representation", "weighted25"),
            ("task_type", "ks_si"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(BadEvidence) as caught:
                    self.run_check(checkpoint, self.identity(**{field: wrong}))
                self.assertIn("not evidence for the requested run",
                              str(caught.exception))
                self.assertIn(field, json.dumps(caught.exception.detail))

    def test_mismatched_commit_is_refused_when_the_study_pins_one(self):
        checkpoint = self.artifact(False)
        self.run_check(checkpoint, self.identity(), expect_commit=COMMIT)
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(checkpoint, self.identity(), expect_commit="b" * 40)
        self.assertIn("different commit", str(caught.exception))

    def test_missing_or_empty_or_incomplete_identity_is_refused(self):
        checkpoint = self.artifact(False)

        with self.assertRaises(BadEvidence) as caught:
            self.run_check(checkpoint, os.path.join(self.directory, "absent"))
        self.assertIn("not found", str(caught.exception))

        empty = os.path.join(self.directory, "empty.jsonl")
        open(empty, "w", encoding="utf-8").close()
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(checkpoint, empty)
        self.assertIn("no readable record", str(caught.exception))

        for field in ("study_id", "method", "git_commit", "checkpoints_dir"):
            with self.subTest(missing_field=field):
                record = dict(IDENTITY_RECORD, results_dir=self.results_dir,
                              checkpoints_dir=self.checkpoints_dir)
                record[field] = None
                path = _identity_file(self.directory, f"missing_{field}.jsonl",
                                      record=record)
                with self.assertRaises(BadEvidence) as caught:
                    self.run_check(checkpoint, path)
                self.assertIn("incomplete", str(caught.exception))

    def test_wrong_run_evidence_exits_two(self):
        checkpoint = self.artifact(False)
        self.assertEqual(self.cli(checkpoint, self.identity(method="mtrl")), 2)
        self.assertEqual(
            self.cli(checkpoint, self.identity(study_id="DG-0005")), 2
        )
        self.assertEqual(self.cli(checkpoint, self.identity(seed=1)), 2)

    def test_a_non_dg0007_checkpoint_that_used_to_pass_is_now_refused(self):
        """A 2-task checkpoint with a consistent Omega satisfies every clause.

        It is refused because it is not structurally evidence for `ks_si_er`.
        """
        state = _state_dict_with(np.eye(2) / 2.0, class_counts=(12, 4),
                                 widths=[HIDDEN] * 2)
        matrix = _head_summary(state, normalize_w=False, count=2)
        state["omega"] = torch.tensor(
            _independent_closed_form_omega((matrix @ matrix.T).numpy()),
            dtype=torch.float32,
        )
        path = _write(os.path.join(self.checkpoints_dir, "two_task.pth"), state)
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(path, self.identity())
        self.assertIn("task count", str(caught.exception))
        self.assertEqual(self.cli(path, self.identity()), 2)


class CrossRunSubstitutionIsRefused(GateBase):
    """M-REM-1: run A's identity paired with run B's artifact, same everything."""

    def two_runs(self):
        """Two real runs of the same arm, shape, representation and seed.

        They differ only in their per-run directories -- exactly what the
        trainer gives two runs -- so nothing but the identity's own recorded
        directory can tell their artifacts apart.
        """
        runs = []
        for index in (0, 1):
            run_id = f"{RUN_ID}_{index}"
            checkpoints = os.path.join(self.directory, "checkpoints", run_id)
            results = os.path.join(self.directory, "results", run_id)
            os.makedirs(checkpoints, exist_ok=True)
            os.makedirs(results, exist_ok=True)
            checkpoint = self.artifact(False, directory=checkpoints,
                                       name=f"train_ks_si_er_epoch30.pth")
            identity = _identity_file(
                self.directory, f"identity_run{index}.jsonl",
                dict(IDENTITY_RECORD, results_dir=results,
                     checkpoints_dir=checkpoints, run_id=run_id,
                     checkpoint_run_id=run_id),
            )
            runs.append({"checkpoint": checkpoint, "identity": identity,
                         "results_dir": results, "checkpoints_dir": checkpoints})
        return runs

    def test_identity_from_one_run_with_another_runs_checkpoint_is_refused(self):
        run_a, run_b = self.two_runs()
        # both are individually valid, same arm, shape, representation, seed
        self.assertTrue(self.run_check(run_a["checkpoint"], run_a["identity"])["passed"])
        self.assertTrue(self.run_check(run_b["checkpoint"], run_b["identity"])["passed"])

        # the attack: run A's identity, run B's checkpoint
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(run_b["checkpoint"], run_a["identity"])
        self.assertIn("not from the run the identity describes",
                      str(caught.exception))
        self.assertEqual(self.cli(run_b["checkpoint"], run_a["identity"]), 2)

    def test_identity_from_one_run_with_another_runs_omega_history_is_refused(self):
        run_a, run_b = self.two_runs()
        state = torch.load(run_b["checkpoint"], map_location="cpu", weights_only=True)
        foreign_history = os.path.join(run_b["results_dir"], "omega_history.json")
        with open(foreign_history, "w", encoding="utf-8") as handle:
            json.dump({"final_omega": state["omega"].tolist()}, handle)
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(run_a["checkpoint"], run_a["identity"],
                           omega_history=foreign_history)
        self.assertIn("Omega history", str(caught.exception))

    def test_a_checkpoint_outside_the_recorded_directory_is_refused(self):
        """The simplest form: the artifact is not in the run's directory."""
        checkpoint = self.artifact(False, directory=self.directory,
                                   name="copied_elsewhere.pth")
        with self.assertRaises(BadEvidence) as caught:
            self.run_check(checkpoint, self.identity())
        self.assertIn("not from the run the identity describes",
                      str(caught.exception))
        self.assertEqual(self.cli(checkpoint, self.identity()), 2)


class GateAcceptsARealModelCheckpoint(GateBase):
    """The checker reads a genuine trainer-shaped checkpoint, not a toy dict."""

    def real_artifact(self, config_name):
        import yaml

        from improvements.run_improvements import build_model

        with open(os.path.join(MTRL_DIR, config_name), encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        torch.manual_seed(0)
        model = build_model("mtrl", cfg, "ks_si_er", 25)
        model.update_omega()
        return _write(
            os.path.join(self.checkpoints_dir, "train_ks_si_er_epoch30.pth"),
            model.state_dict(),
        )

    def test_real_model_state_dict_passes_under_the_corrected_policy(self):
        path = self.real_artifact("mtrl_norm_corrected_25L_config.yml")
        verdict = self.run_check(path, self.identity())
        self.assertTrue(verdict["passed"], verdict)
        self.assertEqual(verdict["task_parameter_matrix_shape"], [3, 2001])

    def test_real_model_state_dict_fails_under_the_historical_policy(self):
        path = self.real_artifact("mtrl_poolingwinner_25L_config.yml")
        verdict = self.run_check(
            path, self.identity(method="mtrl"), normalize_w=True,
            expected=dict(EXPECTED_IDENTITY, method="mtrl"),
        )
        self.assertFalse(verdict["passed"], verdict)
        self.assertGreater(
            verdict["clauses"]["omega_is_argmin_for_the_un_normalized_W"]["relative_gap"],
            0.4,
        )


class BadInputDiagnosticsAreBounded(GateBase):
    """A large or foreign detail exits 2; it must not raise out of the checker.

    The exit-2 report used to be assembled as
    `json.loads(_bounded(json.dumps(detail)))` -- bounding the *serialized* JSON
    and parsing the truncated string back. Truncation is not JSON by
    construction, so any detail whose serialization exceeded the 240-character
    limit raised `JSONDecodeError` from inside the `except BadEvidence` handler
    and escaped `main()` as a traceback with exit 1: the opposite of the
    documented contract, and reachable from ordinary malformed evidence.

    These are the classes that reproduced it, each of which is now required to
    report exit 2 with a bounded structured diagnostic and no traceback.
    """

    def _reports(self, checkpoint, identity):
        """Run the CLI in-process, capturing its stdout. Returns (code, payload)."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = self.cli(checkpoint, identity)
        return code, json.loads(stream.getvalue())

    def arbitrary_bytes_checkpoint(self):
        """A `.pth` that is not a checkpoint at all: text, not a pickled dict.

        The run directory is deliberately long-named: this class's detail is
        `{"path": ..., "error": "..."}`, so whether its serialization crosses
        the bound depends on the path length, and the regression is only
        reproduced at a realistic one.
        """
        path = os.path.join(
            self.checkpoints_dir,
            "not-a-checkpoint-" + "x" * 220 + ".pth",
        )
        with open(path, "wb") as handle:
            handle.write(b"this is not a checkpoint, it is arbitrary text\n" * 8)
        return path

    def cases(self):
        """label -> (checkpoint, identity, a size the bound had to cut).

        Each `uncut` is a lower bound on the *unbounded* diagnostic the checker
        would have serialized for that case, so `assertGreater(uncut, 240)`
        states that the case genuinely reproduced the escape -- a detail longer
        than the 240-character limit -- rather than exiting 2 for some other
        reason.
        """
        huge = "m" * 5000
        arbitrary = self.arbitrary_bytes_checkpoint()
        near_empty = _identity_file(self.directory, "near_empty.jsonl", {"seed": 0})
        oversized_seed = self.identity(seed=10 ** 400)
        with open(oversized_seed, encoding="utf-8") as handle:
            oversized_seed_bytes = len(handle.read())
        missing = {
            field: "missing"
            for field in ("study_id", "stage", "method", "model", "task_type",
                          "representation", "git_commit", "results_dir",
                          "checkpoints_dir")
        }
        return {
            "arbitrary bytes checkpoint": (
                arbitrary, self.identity(), len(arbitrary)
            ),
            "foreign identity value larger than the bound": (
                self.artifact(False), self.identity(method=huge), len(huge)
            ),
            "foreign identity integer larger than the bound": (
                self.artifact(False), oversized_seed, oversized_seed_bytes
            ),
            "near-empty identity record": (
                self.artifact(False), near_empty,
                len(json.dumps({"fields": missing, "path": near_empty})),
            ),
        }

    def test_each_class_exits_two_with_a_bounded_structured_diagnostic(self):
        for label, (checkpoint, identity, uncut) in self.cases().items():
            with self.subTest(case=label):
                code, payload = self._reports(checkpoint, identity)
                self.assertEqual(code, 2)
                self.assertEqual(payload["kind"], "bad_input")
                self.assertTrue(payload["error"])
                self.assertIn("detail", payload)
                rendered = json.dumps(payload["detail"])
                # The case is not vacuous: its unbounded diagnostic crossed the
                # 240-character limit the old report truncated at, which is the
                # condition that used to escape as a JSONDecodeError.
                self.assertGreater(uncut, _DIAGNOSTIC_LIMIT)
                # ... and the report is nevertheless bounded and JSON.
                self.assertLess(len(rendered), 8192)
                self.assertNotIn("tensor(", rendered)

    def test_the_near_empty_record_names_every_missing_field(self):
        """Bounding must not cost the field-level diagnostic."""
        _checkpoint, identity, _ = self.cases()["near-empty identity record"]
        _code, payload = self._reports(self.artifact(False), identity)
        fields = payload["detail"]["fields"]
        self.assertEqual(
            sorted(fields),
            sorted(("study_id", "stage", "method", "model", "task_type",
                    "representation", "git_commit", "results_dir",
                    "checkpoints_dir")),
        )
        # `seed: 0` is present and valid, so it is not a problem -- the same
        # presence-not-truthiness rule the checker's seed clause applies.
        self.assertNotIn("seed", fields)

    def test_a_foreign_value_is_truncated_not_dumped(self):
        _checkpoint, identity, uncut = self.cases()[
            "foreign identity value larger than the bound"
        ]
        _code, payload = self._reports(self.artifact(False), identity)
        found = payload["detail"]["mismatched"]["method"]["found"]
        self.assertLessEqual(len(found), 244)
        self.assertTrue(found.endswith("..."))
        self.assertGreater(uncut, len(found))

    def test_the_cli_process_exits_two_without_a_traceback(self):
        """End to end: the process contract, not just the in-process call."""
        checker = os.path.join(STUDY_DIR, "check_runtime_faithfulness.py")
        self.assertTrue(os.path.isfile(checker), checker)
        for label, (checkpoint, identity, _uncut) in self.cases().items():
            with self.subTest(case=label):
                completed = subprocess.run(
                    [sys.executable, checker,
                     "--checkpoint", checkpoint,
                     "--identity", identity,
                     "--normalize-w", "false",
                     "--study-id", "DG-0007",
                     "--stage", "screen",
                     "--method", "mtrl_norm_corrected",
                     "--model", "mtrl",
                     "--seed", "42",
                     "--json"],
                    capture_output=True, text=True, cwd=REPO_ROOT,
                )
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(json.loads(completed.stdout)["kind"], "bad_input")


if __name__ == "__main__":
    unittest.main()
