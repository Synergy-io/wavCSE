"""Job-spec generation: determinism, exact commits, and the refusals that matter."""

import os
import unittest

from improvements.compute import jobspec
from improvements.compute.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    RepositoryConflictError,
)
from improvements.compute.tests.fakes import ComputeTestCase, sample_plan

COMMIT = "a" * 40


class PlanValidationTests(ComputeTestCase):
    def test_sample_plan_is_valid(self):
        jobspec.validate_plan(sample_plan())

    def test_arm_must_consume_its_seed(self):
        plan = sample_plan()
        plan["arms"][0]["argv"] = ["python", "-m", "x", "--config", "{config}"]
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_unknown_placeholder_is_rejected(self):
        plan = sample_plan()
        plan["arms"][0]["argv"].append("{unknown}")
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_ssh_remote_is_rejected(self):
        plan = sample_plan(repository="git@github.com:example/wavCSE.git")
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)

    def test_worker_request_is_required(self):
        plan = sample_plan()
        del plan["worker"]
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(plan)


class SpecTests(ComputeTestCase):
    def setUp(self):
        super(SpecTests, self).setUp()
        self.make_repo()
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))

    def build(self, **overrides):
        kwargs = dict(stage="screen", arm="mssl", seed=42, commit=COMMIT,
                      scope="TR-0007", envelope_digest="digest")
        kwargs.update(overrides)
        return jobspec.build_spec(self.plan, **kwargs)

    def test_spec_is_schema_valid_and_carries_the_exact_commit(self):
        spec = self.build()
        self.assertEqual(spec["source"]["commit"], COMMIT)
        self.assertEqual(spec["source"]["repository"],
                         "https://github.com/example/wavCSE.git")
        self.assertEqual(spec["name"], "TR-0007__screen__mssl__s42")
        self.assertEqual(spec["tracking"]["metadata"]["scope"], "TR-0007")
        self.assertEqual(spec["tracking"]["metadata"]["seed"], "42")

    def test_generation_is_byte_identical(self):
        self.assertEqual(jobspec.render(self.build()), jobspec.render(self.build()))

    def test_a_short_commit_is_refused(self):
        with self.assertRaises(RepositoryConflictError):
            self.build(commit="cac803f")

    def test_branch_like_identity_is_refused(self):
        for bad in ("main", "HEAD", "v1.2"):
            with self.subTest(bad=bad):
                with self.assertRaises(RepositoryConflictError):
                    self.build(commit=bad)

    def test_seed_must_be_pre_registered_for_the_stage(self):
        with self.assertRaises(ConfigurationError):
            self.build(seed=7)

    def test_argv_is_a_vector_and_contains_no_shell_string(self):
        spec = self.build()
        argv = spec["command"]["argv"]
        self.assertIsInstance(argv, list)
        self.assertNotIn(";", " ".join(argv))
        self.assertTrue(all(isinstance(token, str) and token for token in argv))

    def test_outputs_are_deterministic_files(self):
        spec = self.build()
        paths = [entry["path"] for entry in spec["outputs"]]
        self.assertIn("outputs/mssl_s42/MANIFEST.json", paths)
        self.assertIn("outputs/mssl_s42/checkpoint_best.pth", paths)
        self.assertIn("outputs/mssl_s42/eval_metrics_opt.txt", paths)
        self.assertTrue(all("__" not in path for path in paths))

    def test_two_seeds_do_not_share_output_paths(self):
        first = {entry["path"] for entry in self.build(seed=0, stage="confirm")["outputs"]}
        second = {entry["path"] for entry in self.build(seed=1, stage="confirm")["outputs"]}
        self.assertFalse(first & second)

    def test_absolute_path_is_never_emitted(self):
        spec = self.build()
        for entry in spec["outputs"] + spec["inputs"]:
            self.assertFalse(entry["path" if "path" in entry else "destination"].startswith("/"))

    def test_tracking_metadata_carries_no_secrets(self):
        spec = self.build()
        blob = jobspec.render(spec)
        for marker in ("X-Amz-", "Bearer ", "AKIA", "?Signature="):
            self.assertNotIn(marker, blob)

    def test_job_key_is_deterministic_and_seed_specific(self):
        key = jobspec.job_key("TR-0007", "TR-0007", "screen", "mssl", 42, COMMIT)
        self.assertEqual(
            key, jobspec.job_key("TR-0007", "TR-0007", "screen", "mssl", 42, COMMIT)
        )
        self.assertNotEqual(
            key, jobspec.job_key("TR-0007", "TR-0007", "screen", "mssl", 43, COMMIT)
        )

    def test_duplicate_detection_matches_the_same_job_only(self):
        spec = self.build()
        record = {
            "name": spec["name"],
            "spec": {"source": {"commit": COMMIT},
                     "tracking": {"metadata": spec["tracking"]["metadata"]}},
        }
        self.assertTrue(jobspec.looks_like_duplicate(record, spec))
        other = dict(record)
        other["spec"] = {"source": {"commit": "b" * 40},
                         "tracking": {"metadata": spec["tracking"]["metadata"]}}
        self.assertFalse(jobspec.looks_like_duplicate(other, spec))


class DirtyTreeTests(ComputeTestCase):
    def test_dirty_tree_cannot_produce_a_recorded_job(self):
        self.make_repo(clean=False)
        with self.assertRaises(RepositoryConflictError):
            jobspec.require_committed_experiment()

    def test_clean_tree_passes_and_reports_head(self):
        repo = self.make_repo()
        state = jobspec.require_committed_experiment(repo)
        self.assertTrue(state["clean"])
        self.assertEqual(len(state["head"]), 40)

    def test_expected_commit_mismatch_is_refused(self):
        self.make_repo()
        with self.assertRaises(RepositoryConflictError):
            jobspec.require_committed_experiment(expected_commit="c" * 40)


if __name__ == "__main__":
    unittest.main()
