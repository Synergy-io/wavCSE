"""Recorded literature claims separate the proposition from its evidence.

A claim owns a paper-bound *proposition*; each ``EvidenceReference`` owns the
location that proposition was observed or supported at. These tests pin the
mechanical guarantees that make the original attribution error detectable — every
claim names an artifact class and a real section, and a quoted claim must contain
text that is verbatim inside that section — and pin the separation itself: a
proposition may accumulate evidence without deleting the evidence it was first
recorded from, and ``source_level`` / ``verification`` are derived, never stored.

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
        "schema_version": 2,
        "paper_id": "goncalves-2016-mssl",
        "claim_id": "fixture-claim",
        "assertion_kind": "paraphrase",
        "assertion": "A fixture assertion about the paper.",
        "claim_type": "method-objective",
        "evidence": [{"kind": "card", "anchor": "Optimization method"}],
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


def primary_ref(paper_id="goncalves-2016-mssl", role="preprint", row=None, **extra):
    """A primary evidence reference bound to the real retained artifact digest."""

    row = row if row is not None else retained_row(paper_id, role)
    if row is None:
        raise AssertionError("no retained {!r} artifact for {!r}".format(role, paper_id))
    reference = {
        "kind": "primary",
        "role": role,
        "sha256": row["sha256"],
        "page": "6",
    }
    reference.update(extra)
    return reference


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

    def test_every_record_separates_proposition_from_evidence(self):
        for claim in self.registry.records:
            with self.subTest(claim=claim.claim_ref):
                self.assertGreaterEqual(len(claim.evidence), 1)
                for reference in claim.evidence:
                    self.assertIn(reference.kind, claims.EVIDENCE_KINDS)
                self.assertIn(claim.assertion_kind, claims.ASSERTION_KINDS)
                self.assertIn(claim.claim_type, claims.CLAIM_TYPES)
                self.assertIn(claim.verification, claims.VERIFICATION_LEVELS)
                self.assertEqual(claim.status, "active")

    def test_evidence_provenance_is_preserved_verbatim(self):
        """The exact V1 locator precision survives inside the evidence reference."""

        range_claim = self.registry.get_claim(
            "goncalves-2016-mssl", "barrier-placement-and-1-over-d-absorbable"
        )
        self.assertEqual(
            range_claim.evidence[0].as_dict(),
            {
                "kind": "primary",
                "role": "published",
                "sha256": "5dcca4cf3cc70a0eecf99757628c0dab165e8f499c69ed96ea77a86cd3d1ce2b",
                "page": "8",
                "page_end": "9",
            },
        )

        grid = self.registry.get_claim(
            "goncalves-2016-mssl", "published-lambda2-classification-grid"
        )
        self.assertEqual(grid.evidence[0].kind, "survey")
        self.assertEqual(grid.evidence[0].fields["document"], "MSSL_SPARSITY_ANALYSIS.md")
        self.assertIn("scale `INFERRED`", grid.evidence[0].fields["anchor"])
        self.assertIn("0.01, 0.1, 1, 10, 100", grid.evidence[0].quote)

        card = self.registry.get_claim(
            "goncalves-2016-mssl", "relation-object-sparse-task-precision"
        )
        self.assertEqual(card.evidence[0].kind, "card")
        self.assertEqual(card.evidence[0].fields["anchor"], "Relation representation")
        self.assertTrue(card.evidence[0].quote)

    def test_quotes_live_on_their_evidence_reference_not_the_proposition(self):
        selection = self.registry.get_claim(
            "goncalves-2016-mssl", "lambda-penalties-selected-on-data"
        )

        self.assertEqual(selection.assertion_kind, "quote")
        self.assertEqual(len(selection.evidence), 1)
        self.assertEqual(selection.evidence[0].kind, "card")
        self.assertIn("chosen by cross-validation", selection.evidence[0].quote)

    def test_get_claim_is_exact_and_never_fuzzy(self):
        found = self.registry.get_claim(
            "goncalves-2016-mssl", "published-lambda2-classification-grid"
        )

        self.assertEqual(
            found.claim_ref,
            "goncalves-2016-mssl#published-lambda2-classification-grid",
        )
        self.assertEqual(found.source_level, "survey")
        self.assertEqual(found.verification, "derived_existing_record")
        self.assertEqual(found.evidence[0].fields["document"], "MSSL_SPARSITY_ANALYSIS.md")

        with self.assertRaises(claims.ClaimError) as caught:
            self.registry.get_claim("goncalves-2016-mssl", "published-lambda2-grid")
        self.assertEqual(caught.exception.kind, "UNKNOWN_CLAIM")
        self.assertIn(
            "published-lambda2-classification-grid", caught.exception.detail["available"]
        )

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
                self.assertEqual(claim.evidence[0].fields["role"], "published")
                digests[claim_id] = claim.evidence[0].fields["sha256"]

        preprint = self.registry.get_claim(
            "goncalves-2016-mssl", "preprint-barrier-is-task-scaled"
        )
        self.assertEqual(preprint.source_level, "primary")
        self.assertEqual(preprint.verification, "primary_verified")
        self.assertEqual(preprint.evidence[0].fields["role"], "preprint")
        self.assertEqual(preprint.claim_type, "method-objective")
        # Neither version's evidence may be reused as the other's.
        for claim_id in published_formulations:
            self.assertNotEqual(digests[claim_id], preprint.evidence[0].fields["sha256"])

    def test_the_grid_is_recorded_against_the_survey_not_the_card(self):
        """The TR-0007 attribution failure, asserted as a positive fact."""

        grid = self.registry.get_claim(
            "goncalves-2016-mssl", "published-lambda2-classification-grid"
        )

        self.assertEqual(grid.source_level, "survey")
        self.assertNotEqual(grid.source_level, "card")
        self.assertEqual(grid.evidence[0].kind, "survey")
        # The corrected qualification now records where the retained published
        # artifact places the grid, and keeps the claim at survey provenance
        # because the paper scopes it to the algorithms' regularization parameters.
        self.assertIn("section 4.2", grid.qualification)
        self.assertIn("page 25", grid.qualification)
        self.assertEqual(grid.verification, "derived_existing_record")
        selection = self.registry.get_claim(
            "goncalves-2016-mssl", "lambda-penalties-selected-on-data"
        )
        self.assertEqual(selection.source_level, "card")
        self.assertEqual(selection.assertion_kind, "quote")
        self.assertIn("no numeric lambda_2 grid", selection.qualification)

    def test_the_persisted_records_store_no_derived_authority(self):
        """``source_level`` and ``verification`` are computed, not persisted."""

        for line in CLAIMS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            with self.subTest(claim=raw["claim_id"]):
                self.assertEqual(raw["schema_version"], 2)
                self.assertIn("evidence", raw)
                for derived in ("source_level", "verification", "locator", "quote"):
                    self.assertNotIn(derived, raw)


class EvidenceModelTests(unittest.TestCase):
    """A proposition accumulates evidence; it never deletes the earlier record."""

    def test_evidence_must_be_a_non_empty_list(self):
        self.assertEqual(rejected([record(evidence=[])]).kind, "CLAIM_INVALID")

        incomplete = record()
        del incomplete["evidence"]
        self.assertEqual(rejected([incomplete]).kind, "CLAIM_INVALID")

        self.assertEqual(
            rejected([record(evidence={"kind": "card", "anchor": "Optimization method"})]).kind,
            "CLAIM_INVALID",
        )

    def test_evidence_kind_must_be_known(self):
        error = rejected([record(evidence=[{"kind": "newspaper", "anchor": "x"}])])

        self.assertEqual(error.kind, "CLAIM_INVALID")
        self.assertIn("kind", str(error))

    def test_a_card_claim_can_gain_primary_evidence_without_losing_the_card(self):
        card_only = load([record(claim_id="same-proposition")]).records[0]
        strengthened = load([
            record(
                claim_id="same-proposition",
                evidence=[
                    {"kind": "card", "anchor": "Optimization method"},
                    primary_ref("goncalves-2016-mssl", "preprint"),
                ],
            )
        ]).records[0]

        # Identity and the original provenance are unchanged.
        self.assertEqual(card_only.claim_ref, strengthened.claim_ref)
        self.assertEqual(len(strengthened.evidence), 2)
        self.assertEqual(strengthened.evidence[0].as_dict(),
                         {"kind": "card", "anchor": "Optimization method"})
        self.assertEqual(strengthened.evidence[1].kind, "primary")
        # Only the derived view rises.
        self.assertEqual(card_only.source_level, "card")
        self.assertEqual(card_only.verification, "derived_existing_record")
        self.assertEqual(strengthened.source_level, "primary")
        self.assertEqual(strengthened.verification, "primary_verified")

    def test_multiple_non_primary_kinds_are_joined_without_a_ranking(self):
        anchor = "1. What each method actually couples — and what it does not"
        loaded = load([
            record(
                evidence=[
                    {"kind": "card", "anchor": "Optimization method"},
                    {
                        "kind": "survey",
                        "document": "MSSL_SPARSITY_ANALYSIS.md",
                        "anchor": anchor,
                    },
                ]
            )
        ]).records[0]

        self.assertEqual(loaded.evidence_kinds, ("card", "survey"))
        self.assertEqual(loaded.source_level, "card+survey")
        self.assertEqual(loaded.verification, "derived_existing_record")

    def test_duplicate_evidence_references_are_rejected_deterministically(self):
        error = rejected([
            record(
                evidence=[
                    {"kind": "card", "anchor": "Optimization method"},
                    {"kind": "card", "anchor": "Optimization method"},
                ]
            )
        ])

        self.assertEqual(error.kind, "DUPLICATE_EVIDENCE")

    def test_distinct_references_of_the_same_kind_coexist(self):
        loaded = load([
            record(
                evidence=[
                    {"kind": "card", "anchor": "Optimization method"},
                    {"kind": "card", "anchor": "Relation representation"},
                ]
            )
        ])

        self.assertEqual(len(loaded.records[0].evidence), 2)

    def test_study_evidence_resolves_and_derives_a_study_level(self):
        loaded = load([
            record(
                evidence=[
                    {
                        "kind": "study",
                        "study_id": "LT-0001",
                        "artifact": "analysis",
                        "anchor": "Study decision",
                    }
                ]
            )
        ]).records[0]

        self.assertEqual(loaded.evidence[0].kind, "study")
        self.assertEqual(loaded.source_level, "study")
        self.assertEqual(loaded.verification, "derived_existing_record")

    def test_study_evidence_is_bounded_to_registered_artifacts(self):
        error = rejected([
            record(
                evidence=[
                    {
                        "kind": "study",
                        "study_id": "LT-0001",
                        "artifact": "transcript",
                        "anchor": "Anything",
                    }
                ]
            )
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")


class AttributionTests(unittest.TestCase):
    """The mechanical guarantee: a quote cannot borrow another artifact's authority."""

    def test_quote_absent_from_the_named_artifact_is_rejected(self):
        """Attributing the survey's grid to the card — the original failure."""

        error = rejected([
            record(
                claim_id="grid-on-the-card",
                assertion_kind="quote",
                evidence=[
                    {
                        "kind": "card",
                        "anchor": "Evidence",
                        "quote": "The paper's classification experiments use the lambda_2 grid",
                    },
                ],
            )
        ])

        self.assertEqual(error.kind, "QUOTE_NOT_VERBATIM")
        self.assertEqual(error.detail["kind"], "card")

    def test_quote_from_another_section_of_the_same_artifact_is_rejected(self):
        verbatim_but_elsewhere = "term is on the off-diagonal"

        error = rejected([
            record(
                claim_id="wrong-section",
                evidence=[
                    {
                        "kind": "card",
                        "anchor": "Relation representation",
                        "quote": verbatim_but_elsewhere,
                    }
                ],
                assertion_kind="quote",
            )
        ])

        self.assertEqual(error.kind, "QUOTE_NOT_VERBATIM")

        accepted = load([
            record(
                claim_id="right-section",
                evidence=[
                    {
                        "kind": "card",
                        "anchor": "Transcription correction (2026-09-29)",
                        "quote": verbatim_but_elsewhere,
                    }
                ],
                assertion_kind="quote",
            )
        ])
        self.assertEqual(len(accepted.records), 1)

    def test_a_quote_claim_requires_a_quote_on_some_evidence_reference(self):
        error = rejected([
            record(assertion_kind="quote", evidence=[{"kind": "card", "anchor": "Optimization method"}])
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")
        self.assertIn("quote", str(error))

    def test_unknown_anchor_and_unknown_study_locator_are_rejected(self):
        error = rejected([
            record(evidence=[{"kind": "card", "anchor": "A Section That Does Not Exist"}])
        ])
        self.assertEqual(error.kind, "LOCATOR_NOT_FOUND")

        error = rejected([
            record(
                evidence=[
                    {
                        "kind": "study",
                        "study_id": "LT-9999",
                        "artifact": "analysis",
                        "anchor": "Anything",
                    }
                ]
            )
        ])
        self.assertEqual(error.kind, "LOCATOR_NOT_READABLE")

    def test_survey_document_reference_cannot_escape_the_survey_directory(self):
        anchor = "1. What each method actually couples — and what it does not"
        for document in ("../DECISIONS.md", "/etc/passwd", "sub/dir.md", "notes.txt"):
            with self.subTest(document=document):
                error = rejected([
                    record(
                        evidence=[
                            {"kind": "survey", "document": document, "anchor": anchor}
                        ]
                    )
                ])
                self.assertEqual(error.kind, "CLAIM_INVALID")

        # A well-formed name for a document the survey directory does not hold is a
        # resolution failure, not a malformed reference.
        error = rejected([
            record(
                evidence=[
                    {"kind": "survey", "document": "MSSL.md", "anchor": anchor}
                ]
            )
        ])
        self.assertEqual(error.kind, "LOCATOR_NOT_READABLE")

    def test_a_paraphrase_must_not_carry_verbatim_text(self):
        error = rejected([
            record(
                assertion_kind="paraphrase",
                evidence=[
                    {
                        "kind": "card",
                        "anchor": "Optimization method",
                        "quote": "some text",
                    }
                ],
            )
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")
        self.assertIn("quote", str(error))

    def test_a_carried_quote_must_not_be_empty(self):
        error = rejected([
            record(
                assertion_kind="quote",
                evidence=[{"kind": "card", "anchor": "Optimization method", "quote": "   "}],
            )
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")


class VerificationLevelTests(unittest.TestCase):
    """Verification and source level are derived from evidence, never trusted."""

    # A paper the repository does not retain a primary artifact for.
    UNRETAINED = "zhang-yang-2021-mtl-survey"

    def test_a_primary_reference_needs_a_retained_artifact(self):
        error = rejected([
            record(
                paper_id=self.UNRETAINED,
                evidence=[
                    {"kind": "primary", "role": "preprint", "sha256": "a" * 64, "page": "1"}
                ],
            )
        ])

        self.assertEqual(error.kind, "LOCATOR_NOT_READABLE")

    def test_a_primary_reference_binds_to_version_digest_and_page(self):
        loaded = load([
            record(claim_id="primary-bound", evidence=[primary_ref()])
        ])

        claim = loaded.records[0]
        self.assertEqual(claim.verification, "primary_verified")
        self.assertEqual(claim.source_level, "primary")
        self.assertEqual(claim.evidence[0].fields["role"], "preprint")
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")
        self.assertEqual(claim.evidence[0].fields["sha256"], row["sha256"])

    def test_a_primary_reference_must_name_a_version_role(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")
        for role in ("source", "", "camera-ready"):
            with self.subTest(role=role):
                error = rejected([
                    record(
                        evidence=[
                            {
                                "kind": "primary",
                                "role": role,
                                "sha256": row["sha256"],
                                "page": "1",
                            }
                        ]
                    )
                ])
                self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_a_primary_reference_cannot_name_a_different_digest(self):
        row = retained_row("goncalves-2016-mssl", "preprint")
        if row is None:
            self.skipTest("goncalves-2016-mssl preprint is not retained")
        error = rejected([
            record(
                evidence=[
                    {
                        "kind": "primary",
                        "role": "preprint",
                        "sha256": "b" * 64,
                        "page": "6",
                    }
                ]
            )
        ])

        self.assertEqual(error.kind, "ARTIFACT_MISMATCH")
        self.assertEqual(error.detail["retained_sha256"], row["sha256"])

    def test_a_primary_reference_cannot_name_an_unretained_version(self):
        # A paper may retain the preprint while the published version is absent:
        # the reference must then be refused, not silently bound to the preprint.
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
                        evidence=[
                            {
                                "kind": "primary",
                                "role": "published",
                                "sha256": "a" * 64,
                                "page": "1",
                            }
                        ],
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

    def test_a_primary_reference_may_bind_an_inclusive_page_range(self):
        loaded = load([
            record(
                claim_id="primary-range",
                evidence=[primary_ref(role="published", page="8", page_end="9")],
            )
        ])

        self.assertEqual(loaded.records[0].evidence[0].fields["page_end"], "9")

    def test_a_primary_reference_rejects_an_inverted_page_range(self):
        error = rejected([
            record(evidence=[primary_ref(page="9", page_end="8")])
        ])

        self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_a_primary_reference_requires_page_role_and_digest(self):
        complete = primary_ref()
        for missing in ("role", "sha256", "page"):
            with self.subTest(missing=missing):
                reference = dict(complete)
                del reference[missing]
                error = rejected([record(evidence=[reference])])
                self.assertEqual(error.kind, "CLAIM_INVALID")

    def test_a_primary_paraphrase_is_validated_without_the_local_artifact(self):
        """A paraphrase binds to the manifest alone, so the registry stays
        deterministically validatable from Git without the disposable cache."""

        with tempfile.TemporaryDirectory(prefix="empty-primary-cache-") as cache:
            loaded = load([record(evidence=[primary_ref()])], cache_root=cache)

        self.assertEqual(loaded.records[0].verification, "primary_verified")

    def test_a_quoted_primary_reference_needs_the_local_artifact(self):
        quoted = record(
            claim_id="primary-quote",
            assertion_kind="quote",
            evidence=[primary_ref(quote="anything")],
        )
        with tempfile.TemporaryDirectory(prefix="empty-primary-cache-") as cache:
            error = rejected([quoted], cache_root=cache)

        self.assertEqual(error.kind, "PRIMARY_ARTIFACT_NOT_LOCAL")

    def test_strengthening_a_proposition_to_primary_verified_keeps_its_identity(self):
        card_sourced = load([
            record(claim_id="same-proposition")
        ]).records[0]
        strengthened = load([
            record(claim_id="same-proposition", evidence=[primary_ref()])
        ]).records[0]

        self.assertEqual(card_sourced.claim_ref, strengthened.claim_ref)
        self.assertEqual(card_sourced.source_level, "card")
        self.assertEqual(strengthened.source_level, "primary")
        self.assertEqual(strengthened.verification, "primary_verified")

    def test_version_disagreements_remain_distinct_claims(self):
        preprint = retained_row("goncalves-2016-mssl", "preprint")
        published = retained_row("goncalves-2016-mssl", "published")
        if preprint is None or published is None:
            self.skipTest("both goncalves-2016-mssl versions must be retained")

        loaded = load([
            record(
                claim_id="aaa-preprint-proposition",
                evidence=[primary_ref(role="preprint", row=preprint)],
            ),
            record(
                claim_id="zzz-published-proposition",
                evidence=[primary_ref(role="published", row=published, page="8")],
            ),
        ])

        self.assertEqual(len(loaded.records), 2)
        self.assertEqual(
            {claim.evidence[0].fields["role"] for claim in loaded.records},
            {"preprint", "published"},
        )
        self.assertNotEqual(
            loaded.records[0].evidence[0].fields["sha256"],
            loaded.records[1].evidence[0].fields["sha256"],
        )


class SchemaTests(unittest.TestCase):
    def test_unknown_keys_are_refused_so_other_state_cannot_be_smuggled_in(self):
        for key in (
            "screening", "verdict", "decision", "authorization", "study_id",
            # The V1 fused provenance fields are no longer claim-level keys.
            "locator", "source_level", "verification", "quote",
        ):
            with self.subTest(key=key):
                error = rejected([record(**{key: "excluded"})])
                self.assertEqual(error.kind, "CLAIM_INVALID")
                self.assertIn(key, str(error))

    def test_required_fields_and_versions_are_enforced(self):
        incomplete = record()
        del incomplete["assertion"]
        self.assertEqual(rejected([incomplete]).kind, "CLAIM_INVALID")

        self.assertEqual(
            rejected([record(schema_version=3)]).kind, "CLAIM_INVALID"
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
            | set(claims.EVIDENCE_KINDS)
            | set(claims.VERIFICATION_LEVELS)
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
        self.assertEqual(
            payload["claims"][0]["claim_ref"],
            "goncalves-2016-mssl#relation-object-sparse-task-precision",
        )
        self.assertEqual(payload["claims"][0]["evidence"][0]["kind"], "card")

        exact = run_cli("get", "goncalves-2016-mssl", "omega-step-is-graphical-lasso")
        self.assertEqual(exact.returncode, 0, exact.stderr)
        document = json.loads(exact.stdout)
        self.assertEqual(document["claim_type"], "method-objective")
        self.assertEqual(document["verification"], "primary_verified")
        self.assertEqual(document["evidence"][0]["role"], "published")

        missing = run_cli("get", "goncalves-2016-mssl", "does-not-exist")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(json.loads(missing.stderr)["kind"], "UNKNOWN_CLAIM")

        unknown = run_cli("paper", "no-such-paper")
        self.assertEqual(unknown.returncode, 1)
        self.assertEqual(json.loads(unknown.stderr)["kind"], "UNKNOWN_PAPER")


if __name__ == "__main__":
    unittest.main()
