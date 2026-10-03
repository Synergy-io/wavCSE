"""The literature catalog fixes paper identity without encoding screening state."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_catalog


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_CATALOG = (
    REPO_ROOT
    / "improvements"
    / "taskrelation"
    / "research"
    / "literature"
    / "catalog.jsonl"
)


class CatalogFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="literature-catalog-")
        self.repo_root = Path(self.tempdir.name)
        self.literature_dir = self.repo_root / "research" / "literature"
        self.literature_dir.mkdir(parents=True)
        self.index_path = self.literature_dir / "INDEX.md"
        self.index_path.write_text(
            "# Literature\n\n[Example](example-2024-paper.md)\n",
            encoding="utf-8",
        )
        self.card_path = self.literature_dir / "example-2024-paper.md"
        self.card_path.write_text(
            "# Example (2024)\n\n"
            "## Citation\n\n"
            "A. Author. “An Example Paper.” Example Venue, 2024. "
            "DOI 10.1234/example.1.\n\n"
            "Primary source: https://example.org/paper\n",
            encoding="utf-8",
        )
        self.catalog_path = self.literature_dir / "catalog.jsonl"
        self.record = {
            "schema_version": 1,
            "paper_id": "example-2024-paper",
            "card_path": "research/literature/example-2024-paper.md",
            "title": "An Example Paper",
            "year": 2024,
            "authors": ["A. Author"],
            "venue": "Example Venue",
            "external_ids": {"doi": ["10.1234/example.1"]},
            "source_urls": ["https://example.org/paper"],
            "aliases": [],
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def write_catalog(self, records):
        self.catalog_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def load(self):
        return literature_catalog.load_catalog(
            self.catalog_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            index_path=self.index_path,
        )


class CatalogValidationTests(CatalogFixture):
    def test_valid_catalog_resolves_paper_id_to_the_canonical_card(self):
        self.write_catalog([self.record])

        catalog = self.load()

        entry = catalog.get("example-2024-paper")
        self.assertEqual(entry.paper_id, "example-2024-paper")
        self.assertEqual(
            entry.card_path,
            "research/literature/example-2024-paper.md",
        )

    def test_duplicate_paper_id_is_rejected(self):
        self.write_catalog([self.record, copy.deepcopy(self.record)])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "duplicate paper_id 'example-2024-paper'",
        ):
            self.load()

    def test_missing_referenced_card_is_rejected(self):
        self.card_path.unlink()
        self.write_catalog([self.record])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "card_path does not exist",
        ):
            self.load()

    def test_duplicate_external_identifier_is_rejected(self):
        other_card = self.literature_dir / "other-2024-paper.md"
        other_card.write_text(
            "# Other (2024)\n\n## Citation\n\n"
            "B. Author. “Another Paper.” Example Venue, 2024. "
            "DOI 10.1234/example.1.\n\n"
            "Primary source: https://example.org/other\n",
            encoding="utf-8",
        )
        self.index_path.write_text(
            "# Literature\n\n"
            "[Example](example-2024-paper.md)\n"
            "[Other](other-2024-paper.md)\n",
            encoding="utf-8",
        )
        other = {
            **self.record,
            "paper_id": "other-2024-paper",
            "card_path": "research/literature/other-2024-paper.md",
            "title": "Another Paper",
            "authors": ["B. Author"],
            "source_urls": ["https://example.org/other"],
        }
        self.write_catalog([self.record, other])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "duplicate doi '10.1234/example.1'",
        ):
            self.load()

    def test_malformed_record_is_rejected(self):
        malformed = {**self.record, "year": "2024"}
        self.write_catalog([malformed])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "year must be an integer",
        ):
            self.load()

    def test_malformed_external_identifier_is_rejected(self):
        malformed = {
            **self.record,
            "external_ids": {"doi": ["not-a-doi"]},
        }
        self.card_path.write_text(
            self.card_path.read_text(encoding="utf-8") + "not-a-doi\n",
            encoding="utf-8",
        )
        self.write_catalog([malformed])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "must be a canonical DOI",
        ):
            self.load()

    def test_screening_status_is_rejected_as_non_identity_state(self):
        malformed = {**self.record, "status": "included"}
        self.write_catalog([malformed])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            r"unknown key\(s\): status",
        ):
            self.load()

    def test_unknown_index_card_reference_is_rejected(self):
        self.index_path.write_text(
            "# Literature\n\n[Unknown](unknown-paper.md)\n",
            encoding="utf-8",
        )
        self.write_catalog([self.record])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "literature index references uncataloged card 'unknown-paper'",
        ):
            self.load()

    def test_unregistered_canonical_card_is_rejected_as_drift(self):
        extra_card = self.literature_dir / "uncataloged-2025-paper.md"
        extra_card.write_text(
            "# Uncataloged\n\n## Citation\n\n"
            "C. Author. “Uncataloged Paper.” Example Venue, 2025.\n",
            encoding="utf-8",
        )
        self.write_catalog([self.record])

        with self.assertRaisesRegex(
            literature_catalog.CatalogError,
            "canonical card missing from catalog: uncataloged-2025-paper",
        ):
            self.load()

    def test_index_is_not_treated_as_a_paper_card(self):
        self.write_catalog([self.record])

        catalog = self.load()

        self.assertEqual([entry.paper_id for entry in catalog.entries], ["example-2024-paper"])

    def test_lookup_accepts_paper_id_title_doi_and_versionless_arxiv_id(self):
        record = {
            **self.record,
            "external_ids": {
                "doi": ["10.1234/example.1"],
                "arxiv": ["2410.15875v1"],
            },
        }
        self.card_path.write_text(
            self.card_path.read_text(encoding="utf-8")
            + "arXiv:2410.15875v1\n",
            encoding="utf-8",
        )
        self.write_catalog([record])
        catalog = self.load()

        for query in (
            "example-2024-paper",
            "An Example Paper",
            "https://doi.org/10.1234/EXAMPLE.1",
            "arXiv:2410.15875",
        ):
            with self.subTest(query=query):
                self.assertEqual(catalog.lookup(query).paper_id, "example-2024-paper")


class RealCatalogTests(unittest.TestCase):
    def test_repository_catalog_matches_every_canonical_literature_card(self):
        catalog = literature_catalog.load_catalog(REAL_CATALOG, repo_root=REPO_ROOT)

        discovered = literature_catalog.discover_card_ids(REAL_CATALOG.parent)
        self.assertEqual(
            [entry.paper_id for entry in catalog.entries],
            sorted(discovered),
        )


if __name__ == "__main__":
    unittest.main()
