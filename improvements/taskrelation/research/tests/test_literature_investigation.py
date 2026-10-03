"""The deterministic LT-investigation lifecycle (INC-018).

Covers registration, id allocation and idempotency, the single-delegation rule,
verification on completion, and the closed/open state transitions.
"""

import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_investigation as li


class InvestigationFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="lt-lifecycle-")
        self.repo_root = Path(self.tempdir.name)
        self.research_dir = (
            self.repo_root / "improvements" / "taskrelation" / "research"
        )
        self.literature_dir = self.research_dir / "literature"
        self.studies_dir = self.research_dir / "studies"
        self.literature_dir.mkdir(parents=True)
        self.studies_dir.mkdir()
        self.studies_path = self.research_dir / "STUDIES.jsonl"
        self._write_catalog()
        self._write_studies()

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_catalog(self):
        record = {
            "schema_version": 1,
            "paper_id": "alpha-2024-method",
            "card_path": "improvements/taskrelation/research/literature/alpha-2024-method.md",
            "title": "Alpha Method",
            "year": 2024,
            "authors": ["A. Author"],
            "venue": "Venue A",
            "external_ids": {"doi": ["10.1234/alpha"]},
            "source_urls": ["https://example.org/alpha"],
            "aliases": [],
        }
        (self.literature_dir / "catalog.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        (self.literature_dir / "alpha-2024-method.md").write_text(
            "# Alpha Method\n\n## Summary\n\nalpha body\n", encoding="utf-8"
        )
        (self.literature_dir / "assessments.jsonl").write_text("", encoding="utf-8")

    def _write_studies(self):
        studies = [
            {
                "study_id": "DG-0001",
                "type": "diagnostic",
                "title": "Not literature",
                "status": "complete",
                "stage": "analysis",
                "path": "improvements/taskrelation/research/studies/DG-0001",
            },
            {
                "study_id": "LT-0001",
                "type": "literature",
                "title": "First literature question",
                "status": "rejected",
                "stage": "review",
                "path": "improvements/taskrelation/research/studies/LT-0001",
            },
            {
                "study_id": "LT-0002",
                "type": "literature",
                "title": "Second literature question",
                "status": "complete",
                "stage": "verification",
                "path": "improvements/taskrelation/research/studies/LT-0002",
            },
        ]
        self.studies_path.write_text(
            "".join(json.dumps(row) + "\n" for row in studies), encoding="utf-8"
        )
        for study_id in ("DG-0001", "LT-0001", "LT-0002"):
            (self.studies_dir / study_id).mkdir()
        (self.studies_dir / "LT-0002" / "analysis.md").write_text(
            "# LT-0002\n\n## Assessment\n\nbody\n", encoding="utf-8"
        )

    def open(self, **kwargs):
        return li.open_investigation(
            kwargs.pop("question", "A bounded question"),
            kwargs.pop("scope", "A bounded scope"),
            repo_root=self.repo_root,
            studies_path=self.studies_path,
            **kwargs
        )


class RegistrationTests(InvestigationFixture):
    def test_open_allocates_the_next_lt_id_and_marks_it_active(self):
        record = self.open()

        self.assertTrue(record["created"])
        self.assertEqual(record["investigation_id"], "LT-0003")
        self.assertEqual(record["status"], "active")
        self.assertTrue((self.studies_dir / "LT-0003" / "PLAN.md").is_file())
        self.assertTrue((self.studies_dir / "LT-0003" / "analysis.md").is_file())
        row = li.load_studies(self.studies_path)["LT-0003"]
        self.assertEqual(row["type"], "literature")
        self.assertEqual(row["question"], "A bounded question")
        self.assertEqual(row["path"], "improvements/taskrelation/research/studies/LT-0003")

    def test_open_is_idempotent_on_request_id(self):
        first = self.open(request_id="req-1")
        second = self.open(request_id="req-1")
        third = self.open(request_id="req-2")

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(second["investigation_id"], first["investigation_id"])
        self.assertEqual(third["investigation_id"], "LT-0004")

    def test_open_records_an_optional_assessment_vocabulary(self):
        record = self.open(assessment_roles=["family_b"], assessment_verdicts=["pass"])

        row = li.load_studies(self.studies_path)[record["investigation_id"]]
        self.assertEqual(row["assessment_scope"]["roles"], ["family_b"])
        self.assertEqual(row["assessment_scope"]["verdicts"], ["pass"])

    def test_open_rejects_blank_question_and_scope(self):
        with self.assertRaises(li.InvestigationError) as raised:
            self.open(question="   ")
        self.assertEqual(raised.exception.kind, li.INVALID_REQUEST)


class DelegationTests(InvestigationFixture):
    def test_delegate_marks_the_writable_scope(self):
        record = self.open()
        delegated = li.delegate_investigation(
            record["investigation_id"], studies_path=self.studies_path
        )

        self.assertTrue(delegated["delegated"])
        self.assertEqual(
            li.delegated_investigation(studies_path=self.studies_path),
            record["investigation_id"],
        )

    def test_only_one_investigation_may_be_delegated(self):
        first = self.open(question="one")
        second = self.open(question="two")
        li.delegate_investigation(first["investigation_id"], studies_path=self.studies_path)

        with self.assertRaises(li.InvestigationError) as raised:
            li.delegate_investigation(second["investigation_id"], studies_path=self.studies_path)
        self.assertEqual(raised.exception.kind, li.INVESTIGATION_ALREADY_DELEGATED)

    def test_delegating_a_retry_is_idempotent(self):
        record = self.open()
        li.delegate_investigation(record["investigation_id"], studies_path=self.studies_path)
        again = li.delegate_investigation(
            record["investigation_id"], studies_path=self.studies_path
        )
        self.assertTrue(again["delegated"])

    def test_closed_investigation_cannot_be_delegated(self):
        with self.assertRaises(li.InvestigationError) as raised:
            li.delegate_investigation("LT-0002", studies_path=self.studies_path)
        self.assertEqual(raised.exception.kind, li.INVESTIGATION_NOT_ACTIVE)

    def test_non_literature_study_is_not_delegatable(self):
        with self.assertRaises(li.InvestigationError) as raised:
            li.delegate_investigation("DG-0001", studies_path=self.studies_path)
        self.assertEqual(raised.exception.kind, li.INVESTIGATION_NOT_LITERATURE)

    def test_unknown_investigation_is_reported(self):
        with self.assertRaises(li.InvestigationError) as raised:
            li.delegate_investigation("LT-9999", studies_path=self.studies_path)
        self.assertEqual(raised.exception.kind, li.INVESTIGATION_NOT_FOUND)


class CompletionTests(InvestigationFixture):
    def _with_assessment(self, investigation_id):
        (self.literature_dir / "assessments.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "investigation_id": investigation_id,
                    "paper_id": "alpha-2024-method",
                    "role": "family_b",
                    "verdict": "pass",
                    "gates": [],
                    "reason_summary": "bounded reason",
                    "assessment_anchor": (
                        "improvements/taskrelation/research/studies/{}/analysis.md"
                        "#assessment".format(investigation_id)
                    ),
                    "assessed_at": "2026-10-04T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_complete_verifies_durable_outputs_and_writes_result(self):
        record = self.open(assessment_roles=["family_b"], assessment_verdicts=["pass"])
        self._with_assessment(record["investigation_id"])
        li.delegate_investigation(record["investigation_id"], studies_path=self.studies_path)

        completion = li.complete_investigation(
            record["investigation_id"],
            decision="COMPLETE",
            summary="done",
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )

        self.assertEqual(completion["status"], "complete")
        self.assertFalse(completion["already_completed"])
        result = json.loads(
            (self.studies_dir / "LT-0003" / "result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["assessment_count"], 1)
        self.assertEqual(result["papers_assessed"], ["alpha-2024-method"])
        row = li.load_studies(self.studies_path)["LT-0003"]
        self.assertEqual(row["status"], "complete")
        self.assertFalse(row["delegated"])

    def test_repeated_completion_is_safe(self):
        record = self.open(assessment_roles=["family_b"], assessment_verdicts=["pass"])
        self._with_assessment(record["investigation_id"])
        first = li.complete_investigation(
            record["investigation_id"],
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )
        second = li.complete_investigation(
            record["investigation_id"],
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )

        self.assertFalse(first["already_completed"])
        self.assertTrue(second["already_completed"])
        self.assertEqual(first["completed_at"], second["completed_at"])

    def test_completion_creates_no_finding_or_decision(self):
        record = self.open()
        li.complete_investigation(
            record["investigation_id"],
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )

        self.assertFalse((self.research_dir / "FINDINGS.md").exists())
        self.assertFalse((self.research_dir / "DECISIONS.md").exists())

    def test_completed_investigation_cannot_complete_again_as_a_transition(self):
        record = self.open()
        li.complete_investigation(
            record["investigation_id"],
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )
        # A second call is idempotent, not an error.
        again = li.complete_investigation(
            record["investigation_id"],
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )
        self.assertTrue(again["already_completed"])

        with self.assertRaises(li.InvestigationError) as raised:
            li.abandon_investigation(
                record["investigation_id"], reason="late", studies_path=self.studies_path
            )
        self.assertEqual(raised.exception.kind, li.INVESTIGATION_NOT_ACTIVE)

    def test_completion_rejects_an_invalid_assessment_registry(self):
        record = self.open()
        (self.literature_dir / "assessments.jsonl").write_text(
            '{"not": "a record"}\n', encoding="utf-8"
        )
        with self.assertRaises(li.InvestigationError) as raised:
            li.complete_investigation(
                record["investigation_id"],
                studies_path=self.studies_path,
                repo_root=self.repo_root,
                literature_dir=self.literature_dir,
            )
        self.assertEqual(raised.exception.kind, "INVESTIGATION_OUTPUT_INVALID")


class QueryTests(InvestigationFixture):
    def test_get_and_list_expose_literature_investigations(self):
        record = self.open()
        fetched = li.get_investigation(record["investigation_id"], studies_path=self.studies_path)
        listed = li.list_investigations(studies_path=self.studies_path)

        self.assertEqual(fetched["investigation_id"], record["investigation_id"])
        self.assertEqual(
            [item["investigation_id"] for item in listed], ["LT-0001", "LT-0002", "LT-0003"]
        )

    def test_abandon_records_reason(self):
        record = self.open()
        abandoned = li.abandon_investigation(
            record["investigation_id"], reason="superseded", studies_path=self.studies_path
        )
        self.assertEqual(abandoned["status"], "abandoned")
        row = li.load_studies(self.studies_path)[record["investigation_id"]]
        self.assertEqual(row["abandon_reason"], "superseded")


if __name__ == "__main__":
    unittest.main()
