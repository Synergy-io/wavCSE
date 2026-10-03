"""Deterministic CandidatePaper -> canonical Paper admission is bounded and fail-closed.

INC-017 closes the transition discovery never owned: a normalized
``CandidatePaper`` (external provider metadata, not canonical state) becomes a
durable canonical ``Paper`` only through deterministic identity/provenance
validation. These tests pin the contract with local fixtures and no live
service:

* a new DOI- or arXiv-backed candidate is admitted, and a repeat is ``KNOWN``
  with no duplicate catalog row;
* a strong identifier that contradicts another, or an existing Paper, fails
  closed; exact-title collision with a disagreeing identifier fails closed and
  fuzzy title similarity never merges;
* identifiers are merged across providers regardless of provider order;
* the generated ``paper_id`` is deterministic and a slug collision is detected;
* source URLs follow the catalog's canonical semantics (identifier-derived or
  recognized scholarly hosts only) and an arbitrary provider URL is not stored;
* the catalog mutation is atomic and idempotent, and the existing rows are never
  rewritten;
* admission writes exactly the card, the catalog row and the canonicalization
  ledger — never another research record, credential or raw provider payload;
* the admitted Paper is immediately usable through the existing
  ``literature_acquire`` -> ``literature_primary`` path.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_acquire as la
from improvements.taskrelation.research import literature_admit as ladm
from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_ingest as li
from improvements.taskrelation.research import literature_primary as lp
from improvements.taskrelation.research import literature_primary_text as lpt
from improvements.taskrelation.research import literature_query
from improvements.taskrelation.research import literature_read
from improvements.taskrelation.research.literature_discovery import http as http_layer
from improvements.taskrelation.research.literature_discovery import model as dm


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_CATALOG = (
    REPO_ROOT / "improvements" / "taskrelation" / "research" / "literature"
    / "catalog.jsonl"
)

PDF_BYTES = b"%PDF-1.4 admitted canonical paper artifact\n"
SAFE_ADDRESS = "93.184.216.34"


def candidate(provider="crossref", record_id="r1", **fields):
    fields.setdefault("authors", ("Ada Lovelace",))
    fields.setdefault("year", 2024)
    fields.setdefault("venue", "NeurIPS 2024")
    return dm.CandidatePaper(provider=provider, provider_record_id=record_id, **fields)


def document(pages=("first page", "second page")):
    return lpt.ExtractedDocument(
        page_count=len(pages), pages=tuple(pages), extractor="stub",
        extractor_version="1", warnings=(),
    )


class FakeTransport(http_layer.Transport):
    def __init__(self, route):
        self.route = route
        self.requests = []

    def request(self, url, *, headers, timeout, max_bytes):
        self.requests.append(url)
        response = self.route(url)
        if isinstance(response, http_layer.HttpResponse):
            return response
        status, body = response
        return http_layer.HttpResponse(status=status, headers={}, body=body, url=url)


def pdf_response(url, body=PDF_BYTES):
    return http_layer.HttpResponse(
        status=200, headers={"content-type": "application/pdf"}, body=body, url=url
    )


class AdmitFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="literature-admit-")
        self.root = Path(self.tempdir.name)
        self.repo_root = self.root / "repo"
        self.research_dir = self.repo_root / "improvements" / "taskrelation" / "research"
        self.literature_dir = self.research_dir / "literature"
        self.literature_dir.mkdir(parents=True)
        (self.literature_dir / "INDEX.md").write_text("# Literature\n", encoding="utf-8")
        self.catalog_path = self.literature_dir / "catalog.jsonl"
        self.ledger_path = self.literature_dir / "canonicalizations.jsonl"
        (self.literature_dir / "primary_manifest.jsonl").write_text("", encoding="utf-8")
        self.studies_path = self.research_dir / "studies.jsonl"
        self.assessments_path = self.literature_dir / "assessments.jsonl"
        self.syntheses_path = self.research_dir / "syntheses.jsonl"
        for path in (self.studies_path, self.assessments_path, self.syntheses_path):
            path.write_text("", encoding="utf-8")
        self.papers = [
            {
                "schema_version": 1, "paper_id": "alpha-2024-method",
                "card_path": self._card_rel("alpha-2024-method"), "title": "Alpha Method",
                "year": 2024, "authors": ["Ada Lovelace"], "venue": "NeurIPS 2024",
                "external_ids": {"doi": ["10.1000/alpha"]},
                "source_urls": ["https://doi.org/10.1000/alpha"], "aliases": [],
            },
            {
                "schema_version": 1, "paper_id": "beta-2020-method",
                "card_path": self._card_rel("beta-2020-method"), "title": "Beta Method",
                "year": 2020, "authors": ["Barbara Beta"], "venue": "ICML 2020",
                "external_ids": {"arxiv": ["2001.00002"]},
                "source_urls": ["https://arxiv.org/abs/2001.00002"], "aliases": [],
            },
        ]
        for record in self.papers:
            self._write_card(record)
        self.write_catalog(self.papers)

    def tearDown(self):
        self.tempdir.cleanup()

    def _card_rel(self, paper_id):
        return (
            "improvements/taskrelation/research/literature/{}.md".format(paper_id)
        )

    def _write_card(self, record):
        ids = ""
        for kind, values in record["external_ids"].items():
            for value in values:
                ids += "{}: {}\n".format(kind, value)
        lines = [
            "# {}".format(record["title"]),
            "",
            "## Citation",
            "",
            'A. Author. “{}.” V, {}.'.format(record["title"], record["year"]),
            "",
        ]
        lines += ["Primary source: {}".format(url) for url in record["source_urls"]]
        lines += ["", ids, ""]
        (self.literature_dir / (record["paper_id"] + ".md")).write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def write_catalog(self, records):
        ordered = sorted(records, key=lambda record: record["paper_id"])
        self.catalog_path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in ordered),
            encoding="utf-8",
        )

    def catalog_lines(self):
        return self.catalog_path.read_text(encoding="utf-8").splitlines()

    def admission(self):
        return ladm.LiteratureAdmission(
            repo_root=self.repo_root,
            catalog_path=self.catalog_path,
            literature_dir=self.literature_dir,
            ledger_path=self.ledger_path,
        )

    def ledger_rows(self):
        if not self.ledger_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def card_text(self, paper_id):
        return (self.literature_dir / (paper_id + ".md")).read_text(encoding="utf-8")

    # -- acquisition integration ------------------------------------------

    def open_primary(self):
        policy = lp.StoragePolicy(
            research_dir=self.research_dir,
            cache_root=self.root / "cache",
            bucket="wavcse-primary",
        )
        return lp.LiteraturePrimary(
            policy=policy,
            catalog_path=self.catalog_path,
            manifest_path=self.literature_dir / "primary_manifest.jsonl",
            text_extractor=lambda _path: document(),
        )

    def acquirer(self, route):
        transport = FakeTransport(route)
        retriever = la.ArtifactRetriever(
            transport=transport,
            policy=la.AcquisitionPolicy(resolver=lambda _host: [SAFE_ADDRESS]),
            throttle=la.HostThrottle(intervals={}, default=0.0, sleep=lambda _s: None),
            sleep=lambda _s: None,
        )
        primary = self.open_primary()
        ingest = li.LiteratureIngest(primary=primary)
        return la.PublicAcquirer(
            ingest=ingest, retriever=retriever,
            extractor=lambda _path: document(),
        )


class NewIdentityTests(AdmitFixture):
    def test_new_doi_candidate_is_admitted(self):
        result = self.admission().admit(
            candidate(doi="10.5555/brand-new", title="Brand New Work",
                      url="https://doi.org/10.5555/brand-new")
        )

        self.assertEqual(result.status, ladm.ADMITTED)
        self.assertTrue(result.admitted)
        self.assertEqual(result.paper_id, "lovelace-2024-brand-new-work")
        catalog = literature_catalog.load_catalog(
            self.catalog_path, repo_root=self.repo_root
        )
        self.assertIn(result.paper_id, [entry.paper_id for entry in catalog.entries])

    def test_new_arxiv_candidate_is_admitted(self):
        result = self.admission().admit(
            candidate(provider="arxiv", record_id="2401.00009",
                      title="Fresh Method", arxiv_id="2401.00009v2",
                      url="https://arxiv.org/abs/2401.00009")
        )

        self.assertEqual(result.status, ladm.ADMITTED)
        self.assertEqual(result.source_urls, ("https://arxiv.org/abs/2401.00009",))
        catalog = literature_catalog.load_catalog(
            self.catalog_path, repo_root=self.repo_root
        )
        entry = catalog.get(result.paper_id)
        self.assertEqual(entry.external_ids["arxiv"], ("2401.00009",))

    def test_repeated_candidate_is_known_and_idempotent(self):
        admission = self.admission()
        paper = candidate(doi="10.5555/brand-new", title="Brand New Work")
        first = admission.admit(paper)
        before = self.catalog_lines()

        second = admission.admit(paper)

        self.assertEqual(first.status, ladm.ADMITTED)
        self.assertEqual(second.status, ladm.KNOWN)
        self.assertEqual(second.paper_id, first.paper_id)
        self.assertEqual(second.matched_field, "doi")
        self.assertEqual(self.catalog_lines(), before)
        # The duplicate attempt is one ledger row, not two.
        self.assertEqual(
            len([row for row in self.ledger_rows() if row["status"] == "ADMITTED"]),
            1,
        )


class MultiProviderTests(AdmitFixture):
    def test_same_doi_from_two_providers_yields_one_paper(self):
        admission = self.admission()
        first = admission.admit([
            candidate(provider="crossref", record_id="c", doi="10.7777/shared",
                      title="Shared Work"),
            candidate(provider="semanticscholar", record_id="s", doi="10.7777/SHARED",
                      title="Shared Work"),
        ])
        second = admission.admit(
            candidate(provider="openreview", record_id="o", doi="10.7777/shared",
                      title="Shared Work")
        )

        self.assertEqual(first.status, ladm.ADMITTED)
        self.assertEqual(second.status, ladm.KNOWN)
        self.assertEqual(second.paper_id, first.paper_id)
        rows = [
            line for line in self.catalog_lines()
            if json.loads(line)["paper_id"] == first.paper_id
        ]
        self.assertEqual(len(rows), 1)

    def test_same_arxiv_from_two_providers_yields_one_paper(self):
        admission = self.admission()
        first = admission.admit([
            candidate(provider="arxiv", record_id="a", arxiv_id="2401.00009",
                      title="Shared Preprint"),
            candidate(provider="semanticscholar", record_id="s",
                      arxiv_id="2401.00009v1", title="Shared Preprint"),
        ])
        second = admission.admit(
            candidate(provider="crossref", record_id="c", arxiv_id="2401.00009",
                      title="Shared Preprint")
        )

        self.assertEqual(first.status, ladm.ADMITTED)
        self.assertEqual(second.status, ladm.KNOWN)
        self.assertEqual(second.paper_id, first.paper_id)

    def test_provider_ordering_does_not_change_identity(self):
        admission = self.admission()
        forward = [
            candidate(provider="crossref", record_id="c", doi="10.7777/order",
                      title="Ordered Work"),
            candidate(provider="arxiv", record_id="a", arxiv_id="2405.00001",
                      title="Ordered Work"),
        ]
        first = admission.admit(forward)
        reversed_result = admission.admit(list(reversed(forward)))

        self.assertEqual(first.status, ladm.ADMITTED)
        self.assertEqual(reversed_result.status, ladm.KNOWN)
        self.assertEqual(reversed_result.paper_id, first.paper_id)


class FailClosedTests(AdmitFixture):
    def test_doi_and_arxiv_pointing_at_different_papers_conflict(self):
        result = self.admission().admit(
            candidate(doi="10.1000/alpha", arxiv_id="2001.00002",
                      title="Contradictory Work")
        )

        self.assertEqual(result.status, ladm.REJECTED)
        self.assertEqual(result.kind, ladm.IDENTITY_CONFLICT)

    def test_exact_title_collision_with_a_conflicting_identifier_fails_closed(self):
        result = self.admission().admit(
            candidate(doi="10.9999/other", title="Alpha Method")
        )

        self.assertEqual(result.status, ladm.REJECTED)
        self.assertIn(result.kind, (ladm.AMBIGUOUS_PAPER, ladm.IDENTITY_CONFLICT))

    def test_insufficient_metadata_is_rejected(self):
        result = self.admission().admit(
            candidate(title=None, authors=(), year=None)
        )

        self.assertEqual(result.status, ladm.REJECTED)
        self.assertEqual(result.kind, ladm.IDENTITY_INSUFFICIENT)

    def test_fuzzy_title_similarity_never_merges(self):
        result = self.admission().admit(
            candidate(doi="10.9999/extended", title="Alpha Method Extended Version")
        )

        self.assertEqual(result.status, ladm.ADMITTED)
        self.assertNotEqual(result.paper_id, "alpha-2024-method")

    def test_paper_id_collision_is_detected(self):
        admission = self.admission()
        admission.admit(candidate(doi="10.5555/coll", title="Coll Work"))
        # A different work whose generated slug collides: a leading stopword is
        # dropped from the slug but is part of the bibliographic title.
        result = admission.admit(
            candidate(doi="10.5555/coll-2", title="The Coll Work")
        )

        self.assertEqual(result.status, ladm.REJECTED)
        self.assertEqual(result.kind, ladm.CATALOG_CONFLICT)


class SourceUrlTests(AdmitFixture):
    def test_canonical_source_urls_follow_the_catalog_semantics(self):
        result = self.admission().admit(
            candidate(doi="10.2222/uw", title="URL Work",
                      url="https://api.crossref.org/works/10.2222/uw")
        )

        self.assertEqual(result.status, ladm.ADMITTED)
        self.assertEqual(result.source_urls, ("https://doi.org/10.2222/uw",))
        self.assertNotIn("api.crossref.org", self.card_text(result.paper_id))

    def test_arbitrary_provider_urls_are_not_persisted(self):
        result = self.admission().admit(
            candidate(title="Unidentified Work", url="https://evil.example.com/paper.pdf")
        )

        self.assertEqual(result.status, ladm.REJECTED)
        self.assertEqual(result.kind, ladm.SOURCE_CORRESPONDENCE_INVALID)

    def test_recognized_scholarly_url_is_persisted(self):
        url = "https://proceedings.neurips.cc/paper/2024/hash/abc-Abstract.html"
        result = self.admission().admit(candidate(title="Venue Work", url=url))

        self.assertEqual(result.status, ladm.ADMITTED)
        self.assertEqual(result.source_urls, (url,))
        self.assertIn(url, self.card_text(result.paper_id))


class MutationBoundaryTests(AdmitFixture):
    def test_catalog_mutation_is_atomic_and_idempotent(self):
        admission = self.admission()
        paper = candidate(doi="10.5555/atomic", title="Atomic Work")
        admission.admit(paper)
        after_first = self.catalog_path.read_bytes()

        admission.admit(paper)

        self.assertEqual(self.catalog_path.read_bytes(), after_first)
        leftovers = [
            path for path in self.literature_dir.iterdir()
            if path.name.startswith(".catalog.jsonl.")
        ]
        self.assertEqual(leftovers, [])

    def test_existing_catalog_rows_remain_unchanged(self):
        self.catalog_path.write_bytes(REAL_CATALOG.read_bytes())
        original = self.catalog_lines()
        admission = self.admission()

        result = admission.admit(
            candidate(doi="10.5555/newly-discovered", title="Newly Discovered Work")
        )

        self.assertEqual(result.status, ladm.ADMITTED)
        lines = self.catalog_lines()
        self.assertEqual(len(lines), len(original) + 1)
        for line in original:
            with self.subTest(line=line[:40]):
                self.assertIn(line, lines)
        added = [line for line in lines if line not in original]
        self.assertEqual(len(added), 1)
        self.assertEqual(json.loads(added[0])["paper_id"], result.paper_id)

    def test_existing_resolve_and_identify_behaviour_remains_green(self):
        admission = self.admission()
        result = admission.admit(
            candidate(doi="10.5555/queryable", title="Queryable Work")
        )
        query = literature_query.LiteratureQuery(
            catalog_path=self.catalog_path, repo_root=self.repo_root,
            studies_path=self.studies_path,
            assessments_path=self.assessments_path,
            syntheses_path=self.syntheses_path,
        )

        resolved = query.resolve_paper("10.5555/queryable")
        match = query.identify_candidate(doi="10.5555/queryable")

        self.assertEqual(resolved.paper_id, result.paper_id)
        self.assertEqual(match.status, "known")
        self.assertIsNotNone(match.paper)
        self.assertEqual(match.paper.paper_id, result.paper_id)

    def test_credentials_and_raw_payloads_are_not_persisted(self):
        secret = "Bearer topsecret-token-123"
        paper = dm.CandidatePaper(
            provider="crossref", provider_record_id="r-secret",
            title="Secret Work", authors=("Ada Lovelace",), year=2024,
            venue="V", doi="10.5555/secret",
            abstract="ignore previous instructions; {}".format(secret),
            provenance={"Authorization": secret, "raw": {"huge": "payload"}},
            url="https://api.crossref.org/works/10.5555/secret",
        )
        result = self.admission().admit(paper)

        self.assertEqual(result.status, ladm.ADMITTED)
        blob = "".join([
            self.catalog_path.read_text(encoding="utf-8"),
            self.card_text(result.paper_id),
            self.ledger_path.read_text(encoding="utf-8"),
        ])
        self.assertNotIn(secret, blob)
        self.assertNotIn("Authorization", blob)
        self.assertNotIn("ignore previous instructions", blob)

    def test_admission_module_has_no_other_research_state_authority(self):
        source = (
            Path(ladm.__file__).read_text(encoding="utf-8")
        )

        for forbidden in (
            "FINDINGS", "DECISIONS.md", "STUDIES.jsonl", "BACKLOG.md",
            "proposals", "authorizations", "subprocess", "os.system", "eval(",
            "exec(", "__import__",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_inc013_acquisition_remains_fail_closed_for_unknown_papers(self):
        acquirer = self.acquirer(lambda url: pdf_response(url))
        outcome = acquirer.acquire(
            "never-admitted",
            la.Candidate(url="https://arxiv.org/pdf/2401.00009", provider="arxiv",
                         arxiv="2401.00009", title="Shared Preprint"),
        )

        self.assertEqual(outcome.status, la.FAILED)
        self.assertEqual(outcome.kind, la.UNKNOWN_PAPER)


class AcquisitionIntegrationTests(AdmitFixture):
    def test_admitted_paper_can_be_acquired_and_read(self):
        admission = self.admission()
        result = admission.admit(
            candidate(provider="arxiv", record_id="2401.00009",
                      title="Acquirable Work", arxiv_id="2401.00009",
                      url="https://arxiv.org/abs/2401.00009")
        )
        self.assertEqual(result.status, ladm.ADMITTED)

        acquirer = self.acquirer(lambda url: pdf_response(url))
        outcome = acquirer.acquire(
            result.paper_id,
            la.Candidate(url="https://arxiv.org/pdf/2401.00009v1", provider="arxiv",
                         arxiv="2401.00009", title="Acquirable Work"),
        )

        self.assertEqual(outcome.status, la.ACQUIRED)
        self.assertEqual(outcome.sha256, hashlib.sha256(PDF_BYTES).hexdigest())
        self.assertIn("candidate:arxiv", outcome.identity_evidence)

        primary = self.open_primary()
        view = literature_read.LiteratureReader(primary=primary).read_primary(
            result.paper_id, page=1
        )
        self.assertEqual(view.evidence_level, "primary")
        self.assertEqual(view.sha256, outcome.sha256)
        self.assertIn("first page", view.text)


class LedgerTests(AdmitFixture):
    def test_admission_ledger_validates_and_describes_provenance(self):
        admission = self.admission()
        admission.admit(
            candidate(provider="crossref", record_id="10.5555/x",
                      doi="10.5555/x", title="Ledger Work")
        )

        rows = self.ledger_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ADMITTED")
        self.assertEqual(rows[0]["provenance"],
                         [{"provider": "crossref", "provider_record_id": "10.5555/x"}])
        self.assertEqual(admission.validate(), 1)

    def test_rejected_attempts_are_recorded_with_a_kind(self):
        admission = self.admission()
        admission.admit(candidate(title=None, authors=(), year=None))

        rows = self.ledger_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "REJECTED")
        self.assertEqual(rows[0]["kind"], ladm.IDENTITY_INSUFFICIENT)


class CliTests(AdmitFixture):
    def test_cli_admits_a_discovery_result(self):
        discovery = {
            "kind": "DISCOVERY", "schema_version": 1, "mode": "identity",
            "candidates": [
                {
                    "provider": "crossref", "provider_record_id": "10.5555/cli",
                    "title": "CLI Work", "authors": ["Ada Lovelace"], "year": 2024,
                    "venue": "NeurIPS 2024", "doi": "10.5555/cli",
                    "arxiv_id": None, "openreview_id": None, "other_ids": [],
                    "abstract": None, "url": "https://doi.org/10.5555/cli",
                    "artifact_locations": [], "provenance": {},
                    "identity": {"status": "new"}, "relation": None,
                }
            ],
        }
        path = self.root / "discovery.json"
        path.write_text(json.dumps(discovery), encoding="utf-8")

        candidates = ladm._candidates_from_args(
            ladm._build_parser().parse_args(
                ["admit", "--discovery", str(path), "--index", "0"]
            )
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].provider, "crossref")
        self.assertEqual(candidates[0].doi, "10.5555/cli")

    def test_cli_refuses_a_non_discovery_document(self):
        path = self.root / "free-form.json"
        path.write_text(json.dumps({"title": "Free Form", "doi": "10.1/x"}),
                        encoding="utf-8")

        with self.assertRaises(ladm.AdmissionError):
            ladm._candidates_from_args(
                ladm._build_parser().parse_args(["admit", "--discovery", str(path)])
            )


class ModelSurfaceTests(unittest.TestCase):
    def test_admission_is_not_a_model_facing_tool(self):
        tools = (
            REPO_ROOT / ".omp" / "tools" / "literature.ts"
        ).read_text(encoding="utf-8")

        self.assertNotIn("literature_admit", tools)
        self.assertNotIn("admit", tools)


if __name__ == "__main__":
    unittest.main()
