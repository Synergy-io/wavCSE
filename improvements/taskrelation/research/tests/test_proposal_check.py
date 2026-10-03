"""The proposal object cannot silently become a registration or an authorization.

The validator is deterministic, so these tests exercise it directly and through
its CLI: a malformed proposal fails, a legacy pre-schema proposal is skipped, and
a pre-decision proposal whose study id is already registered is rejected.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import proposal_check


REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"

GOOD = """---
proposal_schema: 1
proposal_id: DP-0000
allocated_study_id: XX-0000
status: DRAFT
created_at: 2000-01-01T00:00:00+00:00
review: ""
---

# draft

## Question
## Evidence basis
## Hypotheses
## Proposed study
## Discriminating measurements
## Success and stop conditions
## Expected compute
## Risks and confounds
## Repository effects
## Human decisions required
## Review
"""


def research_dir(tmp, registry_ids=(), study_dirs=()):
    root = Path(tmp)
    (root / "proposals").mkdir(parents=True)
    if registry_ids:
        lines = [json.dumps({"study_id": study_id}) for study_id in registry_ids]
        (root / "STUDIES.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for study_id in study_dirs:
        (root / "studies" / study_id).mkdir(parents=True)
    return root


class FrontmatterTests(unittest.TestCase):
    def test_minimal_draft_is_valid(self):
        kind, problems = proposal_check.check_text(GOOD, RESEARCH_DIR, set())
        self.assertEqual(kind, "proposal")
        self.assertEqual(problems, [])

    def test_template_is_valid(self):
        kind, problems = proposal_check.check_text(proposal_check.TEMPLATE, RESEARCH_DIR, set())
        self.assertEqual(kind, "proposal")
        self.assertEqual(problems, [])

    def test_absent_frontmatter_is_legacy(self):
        kind, problems = proposal_check.check_text("# old draft\n\n## Question\n", RESEARCH_DIR, set())
        self.assertEqual(kind, "legacy")
        self.assertEqual(problems, [])

    def test_missing_schema_is_legacy(self):
        text = GOOD.replace("proposal_schema: 1\n", "")
        kind, _ = proposal_check.check_text(text, RESEARCH_DIR, set())
        self.assertEqual(kind, "legacy")

    def test_unknown_key_is_rejected(self):
        text = GOOD.replace("review: \"\"\n", "review: \"\"\nauthorization: DG-0008\n")
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, set())
        self.assertTrue(any("unknown frontmatter key 'authorization'" in p for p in problems))

    def test_unknown_status_is_rejected(self):
        text = GOOD.replace("status: DRAFT", "status: EXECUTING")
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, set())
        self.assertTrue(any("unknown status" in p for p in problems))

    def test_missing_section_is_rejected(self):
        text = GOOD.replace("## Discriminating measurements\n", "")
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, set())
        self.assertTrue(any("## Discriminating measurements" in p for p in problems))


class LifecycleTests(unittest.TestCase):
    def test_ready_for_human_requires_pass(self):
        text = GOOD.replace("status: DRAFT", "status: READY_FOR_HUMAN")
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, set())
        self.assertTrue(any("requires review: PASS" in p for p in problems))

    def test_approved_requires_human_fields(self):
        text = GOOD.replace("status: DRAFT", "status: APPROVED").replace(
            "review: \"\"", "review: PASS")
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, set())
        self.assertTrue(any("approved_by" in p for p in problems))
        self.assertTrue(any("approved_at" in p for p in problems))

    def test_pre_decision_proposal_must_not_be_registered(self):
        text = GOOD.replace("allocated_study_id: XX-0000", "allocated_study_id: DG-0008")
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, {"DG-0008"})
        self.assertTrue(any("already registered" in p for p in problems))

    def test_pre_decision_proposal_must_not_have_a_study_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = research_dir(tmp, study_dirs=("DG-0008",))
            text = GOOD.replace("allocated_study_id: XX-0000", "allocated_study_id: DG-0008")
            _, problems = proposal_check.check_text(text, root, set())
            self.assertTrue(any("studies/DG-0008/" in p for p in problems))

    def test_approved_proposal_may_be_registered(self):
        text = (
            GOOD.replace("status: DRAFT", "status: APPROVED")
            .replace("review: \"\"", "review: PASS\napproved_by: human\napproved_at: 2000-01-02T00:00:00+00:00")
            .replace("allocated_study_id: XX-0000", "allocated_study_id: DG-0008")
        )
        _, problems = proposal_check.check_text(text, RESEARCH_DIR, {"DG-0008"})
        self.assertEqual(problems, [])


class CliTests(unittest.TestCase):
    def run_check(self, root):
        return subprocess.run(
            [sys.executable, "-m", "improvements.taskrelation.research.proposal_check", "check",
             "--research-dir", str(root)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )

    def test_cli_rejects_and_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = research_dir(tmp, registry_ids=("DG-0008",))
            (root / "proposals" / "draft.md").write_text(
                GOOD.replace("allocated_study_id: XX-0000", "allocated_study_id: DG-0008"),
                encoding="utf-8")
            bad = self.run_check(root)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("already registered", bad.stdout)

            (root / "proposals" / "draft.md").write_text(GOOD, encoding="utf-8")
            good = self.run_check(root)
            self.assertEqual(good.returncode, 0)
            self.assertIn("OK", good.stdout)


if __name__ == "__main__":
    unittest.main()
