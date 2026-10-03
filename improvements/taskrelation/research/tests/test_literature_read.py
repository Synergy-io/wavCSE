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
    """Stand-in for ``LiteraturePrimary``; `read` mirrors the real page-bounded API."""

    def __init__(self, artifact=None, error=None, text=None):
        self.artifact = artifact
        self.error = error
        self.text = text

    def _raise_if_failing(self):
        if self.error is not None:
            raise self.error

    def get(self, paper_id, role=None):
        self._raise_if_failing()
        return self.artifact

    def read(self, paper_id, role=None, *, page=None, page_end=None, max_chars=None):
        self._raise_if_failing()
        return self.text


def stub_primary_text(pages=((1, "primary evidence\n"),)):
    return literature_primary.PrimaryText(
        paper_id="alpha",
        role="preprint",
        sha256="a" * 64,
        source_url="https://example.org/a",
        source="cache",
        page=pages[0][0],
        page_end=pages[-1][0],
        page_count=max(page for page, _text in pages),
        locator="primary:page:{}".format(pages[0][0]),
        extractor="stub",
        extractor_version="0",
        warnings=("a stub warning",),
        pages=tuple(pages),
        characters=sum(len(text) for _page, text in pages),
        truncated=False,
    )


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

    def reader(self, query=None, primary=None):
        return literature_read.LiteratureReader(
            repo_root=self.repo_root,
            query=query or self.query,
            primary=primary or FakePrimary(
                error=literature_primary.PrimaryError(
                    "not retained", kind=literature_primary.PRIMARY_NOT_AVAILABLE
                )
            ),
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
        primary = FakePrimary(
            error=literature_primary.PrimaryError(
                "corrupt", kind=literature_primary.INTEGRITY_MISMATCH
            )
        )

        with self.assertRaises(literature_primary.PrimaryError) as caught:
            self.reader(primary=primary).read_primary("alpha")

        self.assertEqual(caught.exception.kind, literature_primary.INTEGRITY_MISMATCH)

    def test_primary_read_carries_the_artifact_version_and_locator(self):
        primary = FakePrimary(text=stub_primary_text())

        result = self.reader(primary=primary).read_primary("alpha", role="preprint", page=1)

        self.assertEqual(result.source_kind, "primary")
        self.assertEqual(result.evidence_level, "primary")
        self.assertEqual(result.artifact_role, "preprint")
        self.assertEqual(result.sha256, "a" * 64)
        self.assertEqual(result.locator, "primary:page:1")
        self.assertEqual(result.page, 1)
        self.assertEqual(result.warnings, ("a stub warning",))
        self.assertIn("primary evidence", result.text)

    def test_primary_read_exposes_no_filesystem_path(self):
        primary = FakePrimary(text=stub_primary_text())

        document = self.reader(primary=primary).read_primary("alpha", role="preprint").as_dict()

        self.assertNotIn(str(self.repo_root), json.dumps(document))
        self.assertEqual(document["path"], "")

    def test_max_chars_has_a_strict_upper_bound(self):
        for value in (0, -1, 100001, "100"):
            with self.subTest(value=value):
                with self.assertRaises(literature_read.LiteratureReadError):
                    self.reader().read_card("alpha", max_chars=value)


if __name__ == "__main__":
    unittest.main()
