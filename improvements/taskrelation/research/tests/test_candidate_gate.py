"""Behavioral tests for the deterministic candidate-change gate.

These tests are the machine-enforced half of the capability boundary: a writing
specialist's candidate must not be able to reach the canonical checkout by
editing the gate, its policy, the validation command, test discovery, an agent
definition, or an existing test. The gate's own adversarial self-test is the
first case; the rest drive it against a disposable git repository.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GATE_PATH = os.path.join(REPO_ROOT, "scripts", "agents", "candidate_gate.py")


def _load_gate():
    spec = importlib.util.spec_from_file_location("candidate_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


class TempRepo(object):
    """A disposable git checkout the gate can be pointed at."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="candidate-gate-")
        self.git("init", "-q")
        self.git("config", "user.email", "gate@example.invalid")
        self.git("config", "user.name", "Candidate Gate")
        self.write("README.md", "baseline\n")
        self.write(os.path.join(".omp", "agents", "research-executor.md"), "agent\n")
        self.write(os.path.join("improvements", "compute", "jobspec.py"), "code\n")
        self.write(os.path.join("improvements", "compute", "tests", "test_jobspec.py"),
                   "def test_x():\n    assert True\n")
        self.write(os.path.join("improvements", "taskrelation", "01-mtrl", "model.py"),
                   "model\n")
        self.write(os.path.join("improvements", "taskrelation", "research", "tests",
                                "test_proposal_check.py"), "def test_y():\n    pass\n")
        self.write("Makefile", "check:\n\t@true\n")
        self.write(os.path.join("scripts", "agents", "candidate_gate.py"), "gate\n")
        self.write(os.path.join("infra", "src", "wavcse_infra", "cli.py"), "cli\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "baseline")
        self.baseline = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        result = subprocess.run(
            ["git", "-C", self.root] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            universal_newlines=True,
        )
        if result.returncode != 0:
            raise AssertionError("git {} failed: {}".format(args, result.stderr))
        return result.stdout

    def write(self, relative, text):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def patch(self, relative, addition="changed\n"):
        self.write(relative, addition)
        return self.git("diff", "--no-color", self.baseline)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class AdversarialSelfTestTests(unittest.TestCase):
    def test_every_adversarial_case_passes(self):
        failures = [entry for entry in gate.selftest() if not entry[1]]
        self.assertEqual([], failures, "gate self-test failures: {}".format(failures))


class WorkspaceGateTests(unittest.TestCase):
    def setUp(self):
        self.repo = TempRepo()

    def tearDown(self):
        self.repo.cleanup()

    def judge(self, role="research-executor"):
        return gate.check(repo=self.repo.root, baseline=self.repo.baseline,
                          workspace=self.repo.root, role=role)

    def test_legitimate_implementation_change_is_accepted(self):
        self.repo.write(os.path.join("improvements", "taskrelation", "01-mtrl", "model.py"),
                        "model v2\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("ACCEPT", verdict)
        self.assertEqual([], reasons)

    def test_new_feature_test_is_accepted(self):
        self.repo.write(os.path.join("improvements", "taskrelation", "research", "tests",
                                     "test_new_feature.py"), "def test_new():\n    pass\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("ACCEPT", verdict, reasons)

    def test_agent_definition_change_is_rejected(self):
        self.repo.write(os.path.join(".omp", "agents", "research-executor.md"), "widened\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["CONTROL_PLANE_MODIFIED"], [reason for reason, _ in reasons])

    def test_gate_policy_change_is_rejected(self):
        self.repo.write(os.path.join("scripts", "agents", "candidate_gate.py"),
                        "PROTECTED_PREFIXES = ()\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["VALIDATION_AUTHORITY_MODIFIED"], [reason for reason, _ in reasons])

    def test_makefile_change_is_rejected(self):
        self.repo.write("Makefile", "check:\n\t@echo weakened\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["VALIDATION_AUTHORITY_MODIFIED"], [reason for reason, _ in reasons])

    def test_existing_test_change_is_rejected(self):
        self.repo.write(os.path.join("improvements", "compute", "tests", "test_jobspec.py"),
                        "def test_x():\n    pass  # assertion removed\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["EXISTING_TEST_MODIFIED"], [reason for reason, _ in reasons])

    def test_paid_execution_seam_change_is_rejected(self):
        self.repo.write(os.path.join("improvements", "compute", "jobspec.py"), "weakened\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["PROTECTED_PATH_MODIFIED"], [reason for reason, _ in reasons])

    def test_out_of_scope_change_is_rejected(self):
        self.repo.write(os.path.join("infra", "src", "wavcse_infra", "cli.py"), "edited\n")
        verdict, reasons, _, _ = self.judge()
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["WRITE_SCOPE_VIOLATION"], [reason for reason, _ in reasons])

    def test_infra_engineer_may_edit_its_own_scope(self):
        self.repo.write(os.path.join("infra", "src", "wavcse_infra", "cli.py"), "edited\n")
        verdict, _, _, _ = self.judge(role="infrastructure-engineer")
        self.assertEqual("ACCEPT", verdict)


class PatchGateTests(unittest.TestCase):
    def setUp(self):
        self.repo = TempRepo()

    def tearDown(self):
        self.repo.cleanup()

    def judge_patch(self, patch_text, role="research-executor"):
        directory = tempfile.mkdtemp(prefix="candidate-patch-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "candidate.patch")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(patch_text)
        return gate.check(repo=self.repo.root, baseline=self.repo.baseline,
                          patch=path, role=role)

    def test_deleted_existing_test_is_rejected(self):
        verdict, reasons, _, _ = self.judge_patch(gate._patch(
            "deleted", os.path.join("improvements", "compute", "tests", "test_jobspec.py")))
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["EXISTING_TEST_DELETED"], [reason for reason, _ in reasons])

    def test_renamed_existing_test_is_rejected(self):
        verdict, reasons, _, _ = self.judge_patch(gate._patch(
            "renamed",
            os.path.join("improvements", "taskrelation", "research", "tests",
                         "test_proposal_check.py"),
            os.path.join("improvements", "taskrelation", "research", "tests",
                         "test_proposal_check_v2.py")))
        self.assertEqual("REJECT", verdict)
        self.assertEqual(["EXISTING_TEST_RENAMED"], [reason for reason, _ in reasons])

    def test_added_test_is_accepted(self):
        verdict, _, _, _ = self.judge_patch(gate._patch(
            "added", os.path.join("improvements", "taskrelation", "research", "tests",
                                  "test_added.py")))
        self.assertEqual("ACCEPT", verdict)


class CliTests(unittest.TestCase):
    """The gate is invoked as a command by Main OMP, so its exit codes matter."""

    def run_gate(self, *args):
        return subprocess.run(
            [sys.executable, GATE_PATH] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            universal_newlines=True)

    def test_selftest_exits_zero(self):
        result = self.run_gate("selftest")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("adversarial cases passed", result.stdout)

    def test_check_rejects_and_exits_one(self):
        repo = TempRepo()
        self.addCleanup(repo.cleanup)
        repo.write("Makefile", "check:\n\t@echo weakened\n")
        result = self.run_gate("check", "--repo", repo.root, "--baseline", repo.baseline,
                               "--workspace", repo.root, "--role", "research-executor")
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertIn("VALIDATION_AUTHORITY_MODIFIED", result.stdout)

    def test_check_json_reports_the_verdict(self):
        repo = TempRepo()
        self.addCleanup(repo.cleanup)
        repo.write(os.path.join("improvements", "taskrelation", "01-mtrl", "model.py"),
                   "model v2\n")
        result = self.run_gate("check", "--repo", repo.root, "--baseline", repo.baseline,
                               "--workspace", repo.root, "--role", "research-executor",
                               "--json")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn('"verdict": "ACCEPT"', result.stdout)


if __name__ == "__main__":
    unittest.main()
