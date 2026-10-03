"""Primary-paper retrieval is paper_id-driven, checksum-verified and cache-disposable.

The remote transfer is an injected seam: real S3 publication and credentials
belong to the `wavcse-infra` checkout, so these tests use a fake fetcher and
never require live AWS credentials.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_primary as lp


REPO_ROOT = Path(__file__).resolve().parents[4]
PDF_BYTES = b"%PDF-1.4 retained primary artifact\n"
PDF_SHA256 = hashlib.sha256(PDF_BYTES).hexdigest()
OTHER_SHA256 = hashlib.sha256(b"other bytes").hexdigest()


def paper_record(paper_id, title, url):
    return {
        "schema_version": 1,
        "paper_id": paper_id,
        "card_path": "research/literature/{}.md".format(paper_id),
        "title": title,
        "year": 2024,
        "authors": ["A. Author"],
        "venue": "Venue",
        "external_ids": {},
        "source_urls": [url],
        "aliases": [],
    }


def manifest_row(paper_id, **overrides):
    row = {
        "schema_version": 1,
        "paper_id": paper_id,
        "role": "source",
        "object_key": "papers/{}/source.pdf".format(paper_id),
        "sha256": PDF_SHA256,
        "size_bytes": len(PDF_BYTES),
        "media_type": "application/pdf",
        "source_url": "https://example.org/{}.pdf".format(paper_id),
        "retained_at": "2026-10-01T00:00:00+00:00",
    }
    row.update(overrides)
    return row


class PrimaryFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="literature-primary-")
        self.root = Path(self.tempdir.name)
        self.repo_root = self.root / "repo"
        self.research_dir = self.repo_root / "research"
        self.literature_dir = self.research_dir / "literature"
        self.literature_dir.mkdir(parents=True)
        self.cache_root = self.root / "cache"
        self.catalog_path = self.literature_dir / "catalog.jsonl"
        self.manifest_path = self.literature_dir / "primary_manifest.jsonl"
        self.papers = {
            "alpha-2024-method": paper_record(
                "alpha-2024-method", "Alpha Method", "https://example.org/alpha-2024-method.pdf"
            ),
            "beta-2020-method": paper_record(
                "beta-2020-method", "Beta Method", "https://example.org/beta-2020-method.pdf"
            ),
        }
        self.write_catalog()
        self.manifest_path.write_text("", encoding="utf-8")
        self.fetches = []
        self.fetch_policies = []

    def tearDown(self):
        self.tempdir.cleanup()

    def write_catalog(self, records=None):
        records = list(self.papers.values()) if records is None else records
        self.catalog_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def write_manifest(self, rows):
        self.manifest_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def fake_fetch(self, *, object_key, destination, policy):
        """Write the retained bytes to the destination; record the call."""

        self.fetches.append(object_key)
        self.fetch_policies.append(policy)
        Path(destination).write_bytes(PDF_BYTES)

    def open_primary(self, fetcher=None, bucket="wavcse-primary"):
        policy = lp.StoragePolicy(
            research_dir=self.research_dir,
            cache_root=self.cache_root,
            bucket=bucket,
        )
        return lp.LiteraturePrimary(
            policy=policy,
            catalog_path=self.catalog_path,
            manifest_path=self.manifest_path,
            fetcher=fetcher,
        )

    def cache_file(self, paper_id, role="source"):
        return self.open_primary().cache_path(paper_id, role)


class StoragePolicyTests(PrimaryFixture):
    def test_object_key_is_derived_from_paper_id_and_role(self):
        primary = self.open_primary()

        self.assertEqual(
            primary.object_key("alpha-2024-method"),
            "papers/alpha-2024-method/source.pdf",
        )

    def test_cache_path_is_derived_and_outside_the_repository(self):
        primary = self.open_primary()

        path = primary.cache_path("alpha-2024-method")

        self.assertTrue(str(path).startswith(str(self.cache_root)))
        self.assertNotIn(str(self.repo_root), str(path))
        self.assertEqual(path.name, "source.pdf")

    def test_identity_that_cannot_map_to_a_key_is_rejected(self):
        primary = self.open_primary()

        for identity in ("../escape", "a/b", "..", ".", "Upper", "a b", ""):
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(
                    lp.PrimaryManifestError, "not a valid paper_id"
                ):
                    primary.object_key(identity)

    def test_cache_root_inside_the_repository_is_refused(self):
        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "outside the repository"
        ):
            lp.StoragePolicy(
                research_dir=self.research_dir,
                cache_root=self.repo_root / "cache",
                repo_root=self.repo_root,
            )

    def test_policy_reads_bucket_and_cache_root_from_the_environment(self):
        policy = lp.StoragePolicy.from_environment(
            research_dir=self.research_dir,
            repo_root=self.repo_root,
            env={
                "WAVCSE_PRIMARY_BUCKET": "some-bucket",
                "WAVCSE_PRIMARY_PREFIX": "research/",
                "WAVCSE_PRIMARY_CACHE": str(self.cache_root),
            },
        )

        self.assertEqual(policy.bucket, "some-bucket")
        self.assertEqual(policy.prefix, "research/")
        self.assertEqual(policy.cache_root, self.cache_root)
        self.assertEqual(
            policy.object_key("alpha-2024-method"),
            "research/papers/alpha-2024-method/source.pdf",
        )

    def test_policy_without_a_bucket_is_reported_as_unconfigured(self):
        policy = lp.StoragePolicy.from_environment(
            research_dir=self.research_dir,
            repo_root=self.repo_root,
            env={"WAVCSE_PRIMARY_CACHE": str(self.cache_root)},
        )

        self.assertIsNone(policy.bucket)
        self.assertFalse(policy.configured)


class ManifestValidationTests(PrimaryFixture):
    def test_empty_manifest_is_valid_and_means_nothing_is_retained(self):
        primary = self.open_primary()

        self.assertEqual(primary.manifest_rows(), ())

    def test_valid_manifest_row_resolves_to_its_declared_identity(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        primary = self.open_primary()

        status = primary.status("alpha-2024-method")

        self.assertTrue(status.retained)
        self.assertEqual(status.sha256, PDF_SHA256)
        self.assertEqual(status.media_type, "application/pdf")
        self.assertEqual(status.object_key, "papers/alpha-2024-method/source.pdf")
        self.assertEqual(status.cache_state, "absent")

    def test_manifest_keys_are_deterministically_sorted(self):
        self.write_manifest(
            [manifest_row("beta-2020-method"), manifest_row("alpha-2024-method")]
        )
        primary = self.open_primary()

        self.assertEqual(
            [row.paper_id for row in primary.manifest_rows()],
            ["alpha-2024-method", "beta-2020-method"],
        )

    def test_manifest_row_for_unknown_paper_is_rejected(self):
        self.write_manifest([manifest_row("gamma-2024-method")])

        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "unknown paper_id 'gamma-2024-method'"
        ):
            self.open_primary()

    def test_manifest_object_key_must_match_the_derived_key(self):
        self.write_manifest(
            [manifest_row("alpha-2024-method", object_key="papers/elsewhere.pdf")]
        )

        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "object_key must be exactly"
        ):
            self.open_primary()

    def test_manifest_checksum_must_be_a_lowercase_sha256(self):
        self.write_manifest([manifest_row("alpha-2024-method", sha256="deadbeef")])

        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "sha256 must be 64 lowercase hex characters"
        ):
            self.open_primary()

    def test_manifest_source_url_must_come_from_the_catalog(self):
        self.write_manifest(
            [manifest_row("alpha-2024-method", source_url="https://invented.example/x.pdf")]
        )

        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "source_url is not recorded in the catalog"
        ):
            self.open_primary()

    def test_duplicate_paper_and_role_is_rejected(self):
        self.write_manifest(
            [manifest_row("alpha-2024-method"), manifest_row("alpha-2024-method")]
        )

        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "duplicate manifest entry for 'alpha-2024-method'"
        ):
            self.open_primary()

    def test_unsupported_role_and_unknown_key_are_rejected(self):
        self.write_manifest([manifest_row("alpha-2024-method", role="supplement")])
        with self.assertRaisesRegex(
            lp.PrimaryManifestError, "role must be one of: source"
        ):
            self.open_primary()

        self.write_manifest([manifest_row("alpha-2024-method", title="Alpha")])
        with self.assertRaisesRegex(
            lp.PrimaryManifestError, r"unknown key\(s\): title"
        ):
            self.open_primary()


class PrimaryRetrievalTests(PrimaryFixture):
    def test_unknown_paper_id_fails_before_any_storage_access(self):
        primary = self.open_primary(fetcher=self.fake_fetch)

        with self.assertRaisesRegex(lp.PrimaryError, "unknown paper_id 'nope'") as caught:
            primary.get("nope")

        self.assertEqual(caught.exception.kind, lp.UNKNOWN_PAPER)

    def test_paper_without_a_retained_artifact_reports_unavailable(self):
        primary = self.open_primary(fetcher=self.fake_fetch)

        with self.assertRaises(lp.PrimaryError) as caught:
            primary.get("alpha-2024-method")

        self.assertEqual(caught.exception.kind, lp.PRIMARY_NOT_AVAILABLE)
        self.assertEqual(self.fetches, [])

    def test_retained_paper_without_a_transfer_backend_reports_not_configured(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        primary = self.open_primary(fetcher=None)

        with self.assertRaises(lp.PrimaryError) as caught:
            primary.get("alpha-2024-method")

        self.assertEqual(caught.exception.kind, lp.STORAGE_NOT_CONFIGURED)

    def test_unavailable_credentials_are_reported_as_such(self):
        self.write_manifest([manifest_row("alpha-2024-method")])

        def refuses(**_kwargs):
            raise lp.PrimaryError("no credentials", kind=lp.CREDENTIALS_UNAVAILABLE)

        primary = self.open_primary(fetcher=refuses)

        with self.assertRaises(lp.PrimaryError) as caught:
            primary.get("alpha-2024-method")

        self.assertEqual(caught.exception.kind, lp.CREDENTIALS_UNAVAILABLE)

    def test_remote_transfer_failure_is_wrapped_deterministically(self):
        self.write_manifest([manifest_row("alpha-2024-method")])

        def explodes(**_kwargs):
            raise RuntimeError("socket exploded: /home/someone/.aws/credentials")

        primary = self.open_primary(fetcher=explodes)

        with self.assertRaises(lp.PrimaryError) as caught:
            primary.get("alpha-2024-method")

        self.assertEqual(caught.exception.kind, lp.REMOTE_RETRIEVAL_FAILED)
        self.assertEqual(caught.exception.detail["error_type"], "RuntimeError")

    def test_cache_miss_with_a_working_backend_fetches_verifies_and_caches(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        primary = self.open_primary(fetcher=self.fake_fetch)

        artifact = primary.get("alpha-2024-method")

        self.assertEqual(artifact.source, "remote")
        self.assertEqual(artifact.sha256, PDF_SHA256)
        self.assertEqual(artifact.path.read_bytes(), PDF_BYTES)
        self.assertEqual(self.fetches, ["papers/alpha-2024-method/source.pdf"])
        self.assertEqual(self.fetch_policies[0].bucket, "wavcse-primary")

    def test_repeated_retrieval_reuses_the_validated_cache_without_transfer(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        primary = self.open_primary(fetcher=self.fake_fetch)

        first = primary.get("alpha-2024-method")
        second = primary.get("alpha-2024-method")

        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.path, second.path)
        self.assertEqual(second.source, "cache")
        self.assertEqual(len(self.fetches), 1)

    def test_valid_cache_hit_needs_no_backend_at_all(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        self.cache_file("alpha-2024-method").parent.mkdir(parents=True)
        self.cache_file("alpha-2024-method").write_bytes(PDF_BYTES)
        primary = self.open_primary(fetcher=None)

        artifact = primary.get("alpha-2024-method")

        self.assertEqual(artifact.source, "cache")
        self.assertEqual(artifact.sha256, PDF_SHA256)

    def test_corrupt_cache_is_reported_and_removed_never_returned(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        cache = self.cache_file("alpha-2024-method")
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"%PDF corrupted")
        primary = self.open_primary(fetcher=None)

        with self.assertRaises(lp.PrimaryError) as caught:
            primary.get("alpha-2024-method")

        self.assertEqual(caught.exception.kind, lp.INTEGRITY_MISMATCH)
        self.assertEqual(caught.exception.detail["expected_sha256"], PDF_SHA256)
        self.assertTrue(caught.exception.detail["cache_entry_removed"])
        self.assertFalse(cache.exists())

    def test_status_reports_a_corrupt_cache_without_mutating_it(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        cache = self.cache_file("alpha-2024-method")
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"%PDF corrupted")
        primary = self.open_primary(fetcher=None)

        status = primary.status("alpha-2024-method")

        self.assertEqual(status.cache_state, "corrupt")
        self.assertTrue(cache.exists())

    def test_checksum_mismatch_after_download_never_becomes_valid_cache(self):
        self.write_manifest([manifest_row("alpha-2024-method")])

        def wrong_bytes(**kwargs):
            Path(kwargs["destination"]).write_bytes(b"%PDF wrong artifact")

        primary = self.open_primary(fetcher=wrong_bytes)

        with self.assertRaises(lp.PrimaryError) as caught:
            primary.get("alpha-2024-method")

        self.assertEqual(caught.exception.kind, lp.INTEGRITY_MISMATCH)
        self.assertFalse(self.cache_file("alpha-2024-method").exists())
        self.assertEqual(list(self.cache_file("alpha-2024-method").parent.glob("*")), [])

    def test_interrupted_retrieval_leaves_no_partial_cache_entry(self):
        self.write_manifest([manifest_row("alpha-2024-method")])

        def interrupted(**kwargs):
            Path(kwargs["destination"]).write_bytes(b"%PDF-1.4 truncated")
            raise KeyboardInterrupt

        primary = self.open_primary(fetcher=interrupted)

        with self.assertRaises(KeyboardInterrupt):
            primary.get("alpha-2024-method")

        self.assertFalse(self.cache_file("alpha-2024-method").exists())
        self.assertEqual(
            list(self.cache_file("alpha-2024-method").parent.glob("*.partial")), []
        )

    def test_second_call_after_a_corrupt_cache_recovers_cleanly(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        cache = self.cache_file("alpha-2024-method")
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"%PDF corrupted")
        primary = self.open_primary(fetcher=self.fake_fetch)

        with self.assertRaises(lp.PrimaryError):
            primary.get("alpha-2024-method")
        artifact = primary.get("alpha-2024-method")

        self.assertEqual(artifact.source, "remote")
        self.assertEqual(artifact.sha256, PDF_SHA256)
        self.assertEqual(len(self.fetches), 1)

    def test_reported_results_never_contain_credential_material(self):
        self.write_manifest([manifest_row("alpha-2024-method")])
        primary = self.open_primary(fetcher=self.fake_fetch)
        primary.get("alpha-2024-method")

        serialized = json.dumps(primary.status("alpha-2024-method").as_dict()).lower()

        for needle in ("secret", "access_key", "session_token", "aws_"):
            with self.subTest(needle=needle):
                self.assertNotIn(needle, serialized)


class RealRepositoryPrimaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = lp.LiteraturePrimary(repo_root=REPO_ROOT)

    def test_repository_manifest_is_valid_and_declares_no_fabricated_artifacts(self):
        self.assertEqual(self.primary.manifest_rows(), ())
        self.assertEqual(self.primary.validate(), 0)

    def test_every_known_paper_reports_a_precise_unavailable_reason(self):
        from improvements.taskrelation.research import literature_query

        query = literature_query.LiteratureQuery(repo_root=REPO_ROOT)
        for paper in query.list_papers():
            with self.subTest(paper_id=paper.paper_id):
                with self.assertRaises(lp.PrimaryError) as caught:
                    self.primary.get(paper.paper_id)
                self.assertEqual(caught.exception.kind, lp.PRIMARY_NOT_AVAILABLE)

    def test_default_cache_root_is_outside_the_repository(self):
        self.assertNotIn(str(REPO_ROOT), str(self.primary.policy.cache_root))


if __name__ == "__main__":
    unittest.main()
