"""Recorded literature claims stay source-bound, exact, and read-only.

The registry exists because an agent attributed a paper's ``λ₂`` grid to the
per-paper card, which states no grid: the value was real, the provenance was
invented. These tests pin the mechanical guarantees that make that class of
error detectable — every claim names an artifact class and a real section, and a
quoted claim must contain text that is verbatim inside that section.

Scope guard: the registry records *paper-attributed* assertions. Screening
verdicts, research decisions and authorizations are different kinds of state and
must not be expressible here.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"
LITERATURE_DIR = RESEARCH_DIR / "literature"
CLAIMS_PATH = LITERATURE_DIR / "claims.jsonl"
MODULE = "improvements.taskrelation.research.literature_claims"

from improvements.taskrelation.research import literature_claims as claims


def _code_only(source):
    """Strip module docstring and comments so boundary checks test code, not prose."""

    import re

    stripped = re.sub(r'^""".*?"""', "", source, count=1, flags=re.DOTALL)
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", MODULE, *argv],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def record(**overrides):
    """A minimal valid card-sourced claim, overridable per test."""

    base = {
        "schema_version": 1,
        "paper_id": "goncalves-2016-mssl",
        "claim_id": "fixture-claim",
        "assertion_kind": "paraphrase",
        "assertion": "A fixture assertion about the paper.",
        "claim_type": "method-objective",
        "source_level": "card",
        "verification": "derived_existing_record",
        "locator": {"kind": "card", "anchor": "Optimization method"},
        "status": "active",
    }
    base.update(overrides)
    return base


def retained_row(paper_id, role):
    """The real manifest row for a retained artifact, or None."""

    path = LITERATURE_DIR / "primary_manifest.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["paper_id"] == paper_id and row["role"] == role:
            return row
    return None


def load(records, cache_root=None):
    """Validate `records` through the real loader via a temporary file.

    `cache_root` points the disposable primary cache elsewhere, so a test can
    exercise the state where a retained artifact is declared in Git but its bytes
    are not present locally.
    """

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    with handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    try:
        if cache_root is None:
            return claims.load_claims(
                Path(handle.name), repo_root=REPO_ROOT, literature_dir=LITERATURE_DIR
            )
        with unittest.mock.patch.dict(
            os.environ, {"WAVCSE_PRIMARY_CACHE": str(cache_root)}
        ):
            return claims.load_claims(
                Path(handle.name), repo_root=REPO_ROOT, literature_dir=LITERATURE_DIR
            )
    finally:
        Path(handle.name).unlink()


def rejected(records, cache_root=None):
    """Return the ClaimError raised for `records`, or fail the test."""

    try:
        load(records, cache_root=cache_root)
    except claims.ClaimError as exc:
        return exc
    raise AssertionError("expected the claim registry to reject these records")


class RegistryContentTests(unittest.TestCase):
    def setUp(self):
        self.registry = claims.load_claims()

    def test_seeded_registry_loads_with_deterministic_exact_references(self):
        refs = self.registry.claim_ids()

        self.assertEqual(len(refs), 18)
        self.assertEqual(list(refs), sorted(refs))
        self.assertEqual(len(set(refs)), len(refs))
        self.assertIn(
            "goncalves-2016-mssl#published-lambda2-classification-grid", refs
        )

    def test_every_record_names_an_artifact_class_and_a_location(self):
        for claim in self.registry.records:
            with self.subTest(claim=claim.claim_ref):
                self.assertIn(claim.source_level, claims.SOURCE_LEVELS)
                self.assertIn(claim.verification, claims.VERIFICATION_LEVELS)
                self.assertIn(claim.claim_type, claims.CLAIM_TYPES)
                self.assertEqual(claim.locator["kind"], claim.source_level)
                if claim.locator["kind"] == "primary":
                    # A primary locator is version- and digest-bound, not anchored.
                    self.assertTrue(claim.locator["role"])
                    self.assertRegex(claim.locator["sha256"], "^[0-9a-f]{64}$")
                    self.assertTrue(claim.locator["page"])
                else:
                    self.assertTrue(claim.locator["anchor"])
                self.assertEqual(claim.status, "active")

    def test_get_claim_is_exact_and_never_fuzzy(self):
        found = self.registry.get_claim(
            "goncalves-2016-mssl", "published-lambda2-classification-grid"
        )

        self.assertEqual(found.claim_ref, "goncalves-2016-mssl#published-lambda2-classification-grid")
        self.assertEqual(found.source_level, "survey")
        self.assertEqual(found.verification, "unverified_primary")
        self.assertEqual(found.locator["document"], "MSSL_SPARSITY_ANALYSIS.md")

        with self.assertRaises(claims.ClaimError) as caught:
            self.registry.get_claim("goncalves-2016-mssl", "published-lambda2-grid")
        self.assertEqual(caught.exception.kind, "UNKNOWN_CLAIM")
        self.assertIn("published-lambda2-classification-grid", caught.exception.detail["available"])

    def test_claims_for_paper_filters_and_rejects_unknown_papers(self):
        all_claims = self.registry.claims_for_paper("goncalves-2016-mssl")
        objectives = self.registry.claims_for_paper(
            "goncalves-2016-mssl", "method-objective"
        )

        self.assertEqual(len(all_claims), 8)
        self.assertEqual(len(objectives), 4)
        self.assertTrue(all(c.claim_type == "method-objective" for c in objectives))

        with self.assertRaises(claims.ClaimError) as caught:
            self.registry.claims_for_paper("no-such-paper")
        self.assertEqual(caught.exception.kind, "UNKNOWN_PAPER")

        with self.assertRaises(claims.ClaimError) as caught:
            self.registry.claims_for_paper("goncalves-2016-mssl", "not-a-type")
        self.assertEqual(caught.exception.kind, "INVALID_REFERENCE")

    def test_the_primary_claims_separate_the_two_retained_versions(self):
        """The published and preprint formulations are bound to different artifacts."""

        published_formulations = (
            "barrier-placement-and-1-over-d-absorbable",
            "omega-step-is-graphical-lasso",
        )
        digests = {}
        for claim_id in published_formulations:
            claim = self.registry.get_claim("goncalves-2016-mssl", claim_id)
            with self.subTest(claim=claim_id):
                self.assertEqual(claim.source_level, "primary")
                self.assertEqual(claim.verification, "primary_verified")
                self.assertEqual(claim.locator["role"], "published")
                digests[claim_id] = claim.locator["sha256"]

        preprint = self.registry.get_claim(
            "goncalves-2016-mssl", "preprint-barrier-is-task-scaled"
        )
        self.assertEqual(preprint.source_level, "primary")
        self.assertEqual(preprint.verification, "primary_verified")
        self.assertEqual(preprint.locator["role"], "preprint")
        self.assertEqual(preprint.claim_type, "method-objective")
        # Neither version's evidence may be reused as the other's.
        for claim_id in published_formulations:
            self.assertNotEqual(digests[claim_id], preprint.locator["sha256"])

    def test_the_grid_is_recorded_against_the_survey_not_the_card(self):
        """The TR-0007 attribution failure, asserted as a positive fact."""

        grid = self.registry.get_claim(
            "goncalves-2016-mssl", "published-lambda2-classification-grid"
        )

        self.assertEqual(grid.source_level, "survey")
        self.assertNotEqual(grid.source_level, "card")
        # The corrected qualification now records where the retained published
        # artifact places the grid, and keeps the claim at survey provenance
        # because the paper scopes it to the algorithms' regularization parameters.
        self.assertIn("section 4.2", grid.qualification)
        self.assertIn("page 25", grid.qualification)
        self.assertEqual(grid.verification, "unverified_primary")
        selection = self.registry.get_claim(
            "goncalves-2016-mssl", "lambda-penalties-selected-on-data"
        )
        self.assertEqual(selection.source_level, "card")
        self.assertEqual(selection.assertion_kind, "quote")
        self.assertIn("no numeric lambda_2 grid", selection.qualification)


class AttributionTests(unittest.TestCase):
    """The mechanical guarantee: a quote cannot borrow another artifact's authority."""

    def test_quote_absent_from_the_named_artifact_is_rejected(self):
        """Attributing the survey's grid to the card — the original failure."""

        error = rejected([
            record(
                claim_id="grid-on-the-card",
                assertion_kind="quote",
                quote="The paper's classification experiments use the lambda_2 grid",
                locator={"kind": "card", "anchor": "Evidence"},
            )
        ])

        self.assertEqual(error.kind, "QUOTE_NOT_VERBATIM")
        self.assertEqual(error.detail["locator"]["kind"], "card")

    def test_quote_from_another_section_of_the_same_artifact_is_rejected(self):
        verbatim_but_elsewhere = "term is on the off-diagonal"

        error = rejected([
            record(
                claim_id="wrong-section",
                source_level="card",
                locator={"kind": "card", "anchor": "Relation representation"},
                assertion_kind="quote",
                quote=verbatim_but_elsewhere,
            )
        ])

        self.assertEqual(error.kind, "QUOTE_NOT_VERBATIM")

        accepted = load([
            record(
                claim_id="right-section",
                source_level="card",
                locator={
                    "kind": "card",
                    "anchor": "Transcription correction (2026-09-29)",
                },
                assertion_kind="quote",
                quote=verbatim_but_elsewhere,
            )
        ])
        self.assertEqual(len(accepted.records), 1)

    def test_unknown_anchor_and_unknown_study_locator_are_rejected(self):
        error = rejected([
            record(locator={"kind": "card", "anchor": "A Section That Does Not Exist"})
        ])
        self.assertEqual(error.kind, "LOCATOR_NOT_FOUND")

        error = rejected([
            record(
                source_level="study",
                locator={
                    "kind": "study",
                    "study_id": "LT-9999",
                    "artifact": "analysis",
                    "anchor": "Anything",
                },
            )
        ])
        self.assertEqual(error.kind, "LOCATOR_NOT_READABLE")

        error = rejected([
            record(
                source_level="study",
                locator={
                    "kind": "study",
                    "study_id": "LT-0001",
                    "artifact": "transcript",
                    "anchor": "Anything",
                },
            )
        ])
        self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_survey_document_reference_cannot_escape_the_survey_directory(self):
        anchor = "1. What each method actually couples — and what it does not"
        for document in ("../DECISIONS.md", "/etc/passwd", "sub/dir.md", "notes.txt"):
            with self.subTest(document=document):
                error = rejected([
                    record(
                        source_level="survey",
                        verification="unverified_primary",
                        locator={"kind": "survey", "document": document, "anchor": anchor},
                    )
                ])
                self.assertEqual(error.kind, "CLAIM_INVALID")

        # A well-formed name for a document the survey directory does not hold is a
        # resolution failure, not a malformed reference.
        error = rejected([
            record(
                source_level="survey",
                verification="unverified_primary",
                locator={"kind": "survey", "document": "MSSL.md", "anchor": anchor},
            )
        ])
        self.assertEqual(error.kind, "LOCATOR_NOT_READABLE")

    def test_a_paraphrase_must_not_carry_verbatim_text(self):
        error = rejected([
            record(assertion_kind="paraphrase", quote="some text")
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")
        self.assertIn("quote", str(error))

    def test_a_quote_must_not_be_empty(self):
        error = rejected([record(assertion_kind="quote", quote="   ")])

        self.assertEqual(error.kind, "CLAIM_INVALID")


class VerificationLevelTests(unittest.TestCase):
    """Verification is a claim about evidence, so it is checked, not trusted."""

    # A paper the repository does not retain a primary artifact for.
    UNRETAINED = "zhang-yang-2021-mtl-survey"

    def test_primary_verified_is_impossible_without_a_retained_artifact(self):
        error = rejected([
            record(
                paper_id=self.UNRETAINED,
                source_level="primary",
                verification="primary_verified",
                locator={"kind": "primary", "section": "4.1"},
                assertion_kind="quote",
                quote="anything",
            )
        ])

        self.assertIn(error.kind, ("UNSUPPORTED_VERIFICATION", "UNSUPPORTED_SOURCE"))

    def test_primary_source_level_is_refused_without_a_retained_artifact(self):
        error = rejected([
            record(
                paper_id=self.UNRETAINED,
                source_level="primary",
                verification="derived_existing_record",
                locator={"kind": "primary", "section": "4.1"},
            )
        ])

        self.assertEqual(error.kind, "UNSUPPORTED_SOURCE")

    def test_a_primary_claim_binds_to_version_digest_and_page(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")

        loaded = load([
            record(
                paper_id="goncalves-2016-mssl",
                claim_id="primary-bound",
                source_level="primary",
                verification="primary_verified",
                locator={
                    "kind": "primary",
                    "role": "preprint",
                    "sha256": row["sha256"],
                    "page": "6",
                },
            )
        ])

        claim = loaded.records[0]
        self.assertEqual(claim.verification, "primary_verified")
        self.assertEqual(claim.locator["role"], "preprint")
        self.assertEqual(claim.locator["sha256"], row["sha256"])

    def test_a_primary_locator_must_name_a_version_role(self):
        for role in ("source", "", "camera-ready"):
            with self.subTest(role=role):
                error = rejected([
                    record(
                        source_level="primary",
                        verification="primary_verified",
                        locator={
                            "kind": "primary",
                            "role": role,
                            "sha256": "a" * 64,
                            "page": "1",
                        },
                    )
                ])
                self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_a_primary_locator_cannot_name_a_different_digest(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")

        error = rejected([
            record(
                paper_id="goncalves-2016-mssl",
                source_level="primary",
                verification="primary_verified",
                locator={
                    "kind": "primary",
                    "role": "preprint",
                    "sha256": "b" * 64,
                    "page": "6",
                },
            )
        ])

        self.assertEqual(error.kind, "ARTIFACT_MISMATCH")
        self.assertEqual(error.detail["retained_sha256"], row["sha256"])

    def test_a_primary_locator_cannot_name_an_unretained_version(self):
        # A paper may retain the preprint while the published version is absent:
        # the locator must then be refused, not silently bound to the preprint.
        with tempfile.TemporaryDirectory(prefix="claims-literature-") as tmp:
            literature_dir = Path(tmp) / "literature"
            literature_dir.mkdir()
            (literature_dir / "catalog.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "paper_id": "alpha-2024-method",
                        "card_path": "research/literature/alpha-2024-method.md",
                        "title": "Alpha Method",
                        "year": 2024,
                        "authors": ["A. Author"],
                        "venue": "Venue",
                        "external_ids": {},
                        "source_urls": ["https://example.org/alpha-2024-method.pdf"],
                        "aliases": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (literature_dir / "primary_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "paper_id": "alpha-2024-method",
                        "role": "preprint",
                        "object_key": "papers/alpha-2024-method/preprint.pdf",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                        "media_type": "application/pdf",
                        "source_url": "https://example.org/alpha-2024-method.pdf",
                        "retained_at": "2026-10-01T00:00:00+00:00",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            claims_path = Path(tmp) / "claims.jsonl"
            claims_path.write_text(
                json.dumps(
                    record(
                        paper_id="alpha-2024-method",
                        source_level="primary",
                        verification="primary_verified",
                        locator={
                            "kind": "primary",
                            "role": "published",
                            "sha256": "a" * 64,
                            "page": "1",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(claims.ClaimError) as caught:
                claims.load_claims(
                    claims_path, repo_root=REPO_ROOT, literature_dir=literature_dir
                )

        self.assertEqual(caught.exception.kind, "LOCATOR_NOT_READABLE")
        self.assertEqual(caught.exception.detail["role"], "published")

    def test_a_primary_locator_may_bind_an_inclusive_page_range(self):
        row = retained_row("goncalves-2016-mssl", "published")
        if row is None:
            self.skipTest("goncalves-2016-mssl published version is not retained")

        loaded = load([
            record(
                paper_id="goncalves-2016-mssl",
                claim_id="primary-range",
                source_level="primary",
                verification="primary_verified",
                locator={
                    "kind": "primary",
                    "role": "published",
                    "sha256": row["sha256"],
                    "page": "8",
                    "page_end": "9",
                },
            )
        ])

        self.assertEqual(loaded.records[0].locator["page_end"], "9")

    def test_a_primary_locator_rejects_an_inverted_page_range(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")

        error = rejected([
            record(
                source_level="primary",
                verification="primary_verified",
                locator={
                    "kind": "primary",
                    "role": "preprint",
                    "sha256": row["sha256"],
                    "page": "9",
                    "page_end": "8",
                },
            )
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_a_primary_locator_requires_page_role_and_digest(self):
        complete = {
            "kind": "primary",
            "role": "preprint",
            "sha256": "a" * 64,
            "page": "6",
        }
        for missing in ("role", "sha256", "page"):
            with self.subTest(missing=missing):
                locator = dict(complete)
                del locator[missing]
                error = rejected([
                    record(
                        source_level="primary",
                        verification="primary_verified",
                        locator=locator,
                    )
                ])
                self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_a_quoted_primary_locator_needs_the_local_artifact(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")

        quoted = record(
            paper_id="goncalves-2016-mssl",
            claim_id="primary-quote",
            assertion_kind="quote",
            quote="anything",
            source_level="primary",
            verification="primary_verified",
            locator={
                "kind": "primary",
                "role": "preprint",
                "sha256": row["sha256"],
                "page": "6",
            },
        )
        with tempfile.TemporaryDirectory(prefix="empty-primary-cache-") as cache:
            error = rejected([quoted], cache_root=cache)

        self.assertEqual(error.kind, "PRIMARY_ARTIFACT_NOT_LOCAL")

    def test_strengthening_a_proposition_to_primary_verified_keeps_its_identity(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")

        card_sourced = load([
            record(claim_id="same-proposition", source_level="card",
                   locator={"kind": "card", "anchor": "Optimization method"})
        ]).records[0]
        strengthened = load([
            record(
                claim_id="same-proposition",
                source_level="primary",
                verification="primary_verified",
                locator={
                    "kind": "primary",
                    "role": "preprint",
                    "sha256": row["sha256"],
                    "page": "6",
                },
            )
        ]).records[0]

        self.assertEqual(card_sourced.claim_ref, strengthened.claim_ref)
        self.assertEqual(card_sourced.source_level, "card")
        self.assertEqual(strengthened.source_level, "primary")

    def test_version_disagreements_remain_distinct_claims(self):
        preprint = retained_row("goncalves-2016-mssl", "preprint")
        published = retained_row("goncalves-2016-mssl", "published")
        if preprint is None or published is None:
            self.skipTest("both goncalves-2016-mssl versions must be retained")

        loaded = load([
            record(
                claim_id="aaa-preprint-proposition",
                source_level="primary",
                verification="primary_verified",
                locator={"kind": "primary", "role": "preprint",
                         "sha256": preprint["sha256"], "page": "6"},
            ),
            record(
                claim_id="zzz-published-proposition",
                source_level="primary",
                verification="primary_verified",
                locator={"kind": "primary", "role": "published",
                         "sha256": published["sha256"], "page": "8"},
            ),
        ])

        self.assertEqual(len(loaded.records), 2)
        self.assertEqual(
            {claim.locator["role"] for claim in loaded.records},
            {"preprint", "published"},
        )
        self.assertNotEqual(
            loaded.records[0].locator["sha256"], loaded.records[1].locator["sha256"]
        )

    def test_a_primary_verified_record_requires_a_primary_source_level(self):
        error = rejected([record(verification="primary_verified")])

        self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_locator_kind_must_match_the_source_level(self):
        error = rejected([
            record(
                source_level="card",
                verification="unverified_primary",
                locator={"kind": "survey", "document": "MSSL_SPARSITY_ANALYSIS.md", "anchor": "x"},
            )
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_an_unavailable_locator_states_a_reason_and_carries_no_quote(self):
        loaded = load([
            record(
                source_level="card",
                locator={"kind": "unavailable", "reason": "card does not state this"},
                assertion_kind="paraphrase",
            )
        ])
        self.assertEqual(loaded.records[0].locator["kind"], "unavailable")

        error = rejected([
            record(locator={"kind": "unavailable"}),
        ])
        self.assertEqual(error.kind, "CLAIM_INVALID")

        error = rejected([
            record(
                locator={"kind": "unavailable", "reason": "missing"},
                assertion_kind="quote",
                quote="anything",
            )
        ])
        self.assertEqual(error.kind, "CLAIM_INVALID")


class SchemaTests(unittest.TestCase):
    def test_unknown_keys_are_refused_so_other_state_cannot_be_smuggled_in(self):
        for key in ("screening", "verdict", "decision", "authorization", "study_id"):
            with self.subTest(key=key):
                error = rejected([record(**{key: "excluded"})])
                self.assertEqual(error.kind, "CLAIM_INVALID")
                self.assertIn(key, str(error))

    def test_required_fields_and_versions_are_enforced(self):
        incomplete = record()
        del incomplete["verification"]
        self.assertEqual(rejected([incomplete]).kind, "CLAIM_INVALID")

        self.assertEqual(
            rejected([record(schema_version=2)]).kind, "CLAIM_INVALID"
        )
        self.assertEqual(
            rejected([record(claim_id="Not A Slug")]).kind, "CLAIM_INVALID"
        )
        self.assertEqual(
            rejected([record(paper_id="unknown-paper")]).kind, "CLAIM_INVALID"
        )
        self.assertEqual(
            rejected([record(claim_type="performance-benchmark")]).kind,
            "CLAIM_INVALID",
        )
        self.assertEqual(
            rejected([record(status="excluded")]).kind, "CLAIM_INVALID"
        )
        self.assertEqual(
            rejected([record(assertion="   ")]).kind, "CLAIM_INVALID"
        )

    def test_duplicate_identity_and_blank_lines_are_refused(self):
        error = rejected([record(), record()])
        self.assertEqual(error.kind, "DUPLICATE_CLAIM")

        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(json.dumps(record()) + "\n\n")
        try:
            with self.assertRaises(claims.ClaimError) as caught:
                claims.load_claims(
                    Path(handle.name), repo_root=REPO_ROOT, literature_dir=LITERATURE_DIR
                )
            self.assertEqual(caught.exception.kind, "CLAIM_INVALID")
        finally:
            Path(handle.name).unlink()

    def test_records_must_be_sorted_for_a_stable_file_level_diff(self):
        b = record(claim_id="aaa")
        a = record(claim_id="zzz")
        error = rejected([a, b])

        self.assertEqual(error.kind, "CLAIM_INVALID")
        self.assertIn("sorted", str(error))


class ModuleBoundaryTests(unittest.TestCase):
    def test_the_registry_cannot_express_screening_decisions_or_authorizations(self):
        # Screening verdicts, decisions and authorizations are different kinds of
        # state. The invariant is that the schema cannot express them — not that the
        # words are absent, since the refusal message must name what it refuses.
        vocabulary = sorted(
            set(claims._ALLOWED_KEYS)
            | set(claims.CLAIM_TYPES)
            | set(claims.CLAIM_STATUSES)
            | set(claims.SOURCE_LEVELS)
            | set(claims.VERIFICATION_LEVELS)
            | set(claims.LOCATOR_KINDS)
        )
        for forbidden in (
            "screening", "decision", "verdict", "authorization", "approved",
            "excluded", "exclude", "retained_for", "promoted",
        ):
            with self.subTest(needle=forbidden):
                self.assertFalse(
                    [name for name in vocabulary if forbidden in name], vocabulary
                )

        for forbidden in ("write_text(", "open(", "unlink(", "os.replace(", "mkdir("):
            with self.subTest(needle=forbidden):
                self.assertNotIn(forbidden, _code_only(
                    (RESEARCH_DIR / "literature_claims.py").read_text(encoding="utf-8")
                ))

    def test_the_claim_cli_reads_only(self):
        ok = run_cli("validate")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("OK", ok.stdout)

        listed = run_cli("list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(len(json.loads(listed.stdout)["claim_refs"]), 18)

        paper = run_cli("paper", "goncalves-2016-mssl", "--claim-type", "relation-object")
        self.assertEqual(paper.returncode, 0, paper.stderr)
        payload = json.loads(paper.stdout)
        self.assertEqual(len(payload["claims"]), 1)
        self.assertEqual(payload["claims"][0]["claim_ref"], "goncalves-2016-mssl#relation-object-sparse-task-precision")

        exact = run_cli("get", "goncalves-2016-mssl", "omega-step-is-graphical-lasso")
        self.assertEqual(exact.returncode, 0, exact.stderr)
        self.assertEqual(json.loads(exact.stdout)["claim_type"], "method-objective")

        missing = run_cli("get", "goncalves-2016-mssl", "does-not-exist")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(json.loads(missing.stderr)["kind"], "UNKNOWN_CLAIM")

        unknown = run_cli("paper", "no-such-paper")
        self.assertEqual(unknown.returncode, 1)
        self.assertEqual(json.loads(unknown.stderr)["kind"], "UNKNOWN_PAPER")


if __name__ == "__main__":
    unittest.main()
