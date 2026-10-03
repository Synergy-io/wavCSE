"""Bounded literature reading exposes only approved card, Study and primary refs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

from improvements.taskrelation.research import literature_primary
from improvements.taskrelation.research import literature_read


class FakeQuery:
    def __init__(self, paper, study=None):
        self.paper = paper
        self.study = study

    def resolve_paper(self, identity):
        if identity != self.paper.paper_id:
            raise ValueError("unknown paper identity {!r}".format(identity))
        return self.paper

    def get_study(self, study_id):
        if self.study is None or study_id != self.study.study_id:
            raise ValueError("unknown Study {!r}".format(study_id))
        return self.study


class FakePrimary:
    def __init__(self, artifact=None, error=None):
        self.artifact = artifact
        self.error = error

    def get(self, paper_id):
        if self.error is not None:
            raise self.error
        return self.artifact


class BoundedReadTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="literature-read-")
        self.repo_root = Path(self.tempdir.name) / "repo"
        self.card_path = self.repo_root / "research" / "literature" / "alpha.md"
        self.study_dir = self.repo_root / "research" / "studies" / "LT-0001"
        self.card_path.parent.mkdir(parents=True)
        self.study_dir.mkdir(parents=True)
        self.card_path.write_text("# Alpha\n\ncard evidence\n", encoding="utf-8")
        for name in ("PLAN.md", "NOTE.md", "analysis.md"):
            (self.study_dir / name).write_text(
                "# {}\n\nstudy evidence\n".format(name), encoding="utf-8"
            )
        (self.study_dir / "result.json").write_text(
            json.dumps({"study_id": "LT-0001", "outcome": "bounded"}),
            encoding="utf-8",
        )
        self.paper = self._paper("research/literature/alpha.md")
        self.study = self._study()
        self.query = FakeQuery(self.paper, self.study)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _paper(card_path):
        from improvements.taskrelation.research import literature_query

        return literature_query.PaperRecord(
            paper_id="alpha",
            title="Alpha",
            year=2024,
            authors=("A. Author",),
            venue="Venue",
            external_ids=MappingProxyType({}),
            source_urls=("https://example.org/a",),
            aliases=(),
            card_path=card_path,
        )

    def _study(self):
        from improvements.taskrelation.research import literature_query

        return literature_query.StudyRecord(
            study_id="LT-0001",
            title="Study",
            status="complete",
            stage="analysis",
            decision="COMPLETE",
            path="research/studies/LT-0001",
            plan_path="research/studies/LT-0001/PLAN.md",
            note_path="research/studies/LT-0001/NOTE.md",
            analysis_path="research/studies/LT-0001/analysis.md",
            result_path="research/studies/LT-0001/result.json",
        )

    def reader(self, query=None, primary=None, extractor=None):
        return literature_read.LiteratureReader(
            repo_root=self.repo_root,
            query=query or self.query,
            primary=primary or FakePrimary(
                error=literature_primary.PrimaryError(
                    "not retained", kind=literature_primary.PRIMARY_NOT_AVAILABLE
                )
            ),
            primary_extractor=extractor,
        )

    def test_card_read_returns_card_derived_evidence(self):
        result = self.reader().read_card("alpha")

        self.assertEqual(result.source_kind, "card")
        self.assertEqual(result.evidence_level, "card-derived")
        self.assertEqual(result.paper_id, "alpha")
        self.assertIn("card evidence", result.text)
        self.assertFalse(result.truncated)

    def test_card_read_is_bounded_and_reports_truncation(self):
        result = self.reader().read_card("alpha", max_chars=8)

        self.assertEqual(result.text, "# Alpha\n")
        self.assertEqual(result.characters, 8)
        self.assertTrue(result.truncated)

    def test_unknown_paper_is_a_structured_read_error(self):
        with self.assertRaisesRegex(
            literature_read.LiteratureReadError, "unknown paper identity 'missing'"
        ) as caught:
            self.reader().read_card("missing")

        self.assertEqual(caught.exception.kind, literature_read.UNKNOWN_REFERENCE)

    def test_study_read_allows_only_registered_artifact_kinds(self):
        reader = self.reader()

        result = reader.read_study("LT-0001", "analysis")

        self.assertEqual(result.source_kind, "study")
        self.assertEqual(result.evidence_level, "Study-derived")
        self.assertEqual(result.study_id, "LT-0001")
        self.assertIn("study evidence", result.text)

    def test_unapproved_study_artifact_is_rejected(self):
        with self.assertRaisesRegex(
            literature_read.LiteratureReadError, "artifact must be one of"
        ) as caught:
            self.reader().read_study("LT-0001", "../../DECISIONS.md")

        self.assertEqual(caught.exception.kind, literature_read.INVALID_REFERENCE)

    def test_unavailable_registered_study_artifact_is_explicit(self):
        study = self._study()
        study = type(study)(**{**study.__dict__, "result_path": None})

        with self.assertRaisesRegex(
            literature_read.LiteratureReadError, "has no 'result' artifact"
        ) as caught:
            self.reader(query=FakeQuery(self.paper, study)).read_study(
                "LT-0001", "result"
            )

        self.assertEqual(caught.exception.kind, literature_read.ARTIFACT_NOT_AVAILABLE)

    def test_card_path_escape_is_rejected_even_if_query_returns_it(self):
        outside = self.repo_root.parent / "outside.md"
        outside.write_text("secret", encoding="utf-8")

        for bad_path in ("../outside.md", str(outside)):
            with self.subTest(bad_path=bad_path):
                malicious = self._paper(bad_path)
                with self.assertRaises(literature_read.LiteratureReadError) as caught:
                    self.reader(query=FakeQuery(malicious, self.study)).read_card("alpha")
                self.assertEqual(caught.exception.kind, literature_read.INVALID_REFERENCE)

    def test_primary_unavailable_kind_is_preserved(self):
        with self.assertRaises(literature_primary.PrimaryError) as caught:
            self.reader().read_primary("alpha")

        self.assertEqual(
            caught.exception.kind, literature_primary.PRIMARY_NOT_AVAILABLE
        )

    def test_integrity_failure_is_preserved_and_never_extracted(self):
        calls = []
        primary = FakePrimary(
            error=literature_primary.PrimaryError(
                "corrupt", kind=literature_primary.INTEGRITY_MISMATCH
            )
        )

        with self.assertRaises(literature_primary.PrimaryError):
            self.reader(
                primary=primary, extractor=lambda path: calls.append(path) or "bad"
            ).read_primary("alpha")

        self.assertEqual(calls, [])

    def test_available_primary_is_extracted_only_after_verified_get(self):
        pdf = self.repo_root.parent / "cache" / "alpha" / "source.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF verified")
        artifact = literature_primary.PrimaryArtifact(
            paper_id="alpha",
            role="source",
            sha256="0" * 64,
            size_bytes=13,
            media_type="application/pdf",
            source_url="https://example.org/a",
            object_key="papers/alpha/source.pdf",
            path=pdf,
            source="cache",
        )
        calls = []

        def extract(path):
            calls.append(path)
            return "page 1\nprimary evidence\n"

        result = self.reader(
            primary=FakePrimary(artifact=artifact), extractor=extract
        ).read_primary("alpha")

        self.assertEqual(calls, [pdf])
        self.assertEqual(result.source_kind, "primary")
        self.assertEqual(result.evidence_level, "primary")
        self.assertIn("primary evidence", result.text)

    def test_max_chars_has_a_strict_upper_bound(self):
        for value in (0, -1, 100001, "100"):
            with self.subTest(value=value):
                with self.assertRaises(literature_read.LiteratureReadError):
                    self.reader().read_card("alpha", max_chars=value)


if __name__ == "__main__":
    unittest.main()
