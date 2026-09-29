"""Input declarations: digests are identity, and only relative keys are allowed."""

import unittest

from improvements.compute import artifacts
from improvements.compute.errors import ArtifactIntegrityError, ConfigurationError
from improvements.compute.tests.fakes import ComputeTestCase

GOOD = {
    "artifact": "embeddings/wavlm-mean-smp25.tar",
    "destination": "embeddings/wavlm.tar",
    "sha256": "b" * 64,
    "size_bytes": 1024,
    "required": True,
    "membership": "verified by the loader",
}


class RequirementTests(ComputeTestCase):
    def document(self, **overrides):
        entry = dict(GOOD)
        entry.update(overrides)
        return {"schema_version": 1, "requirements": [entry]}

    def test_valid_requirement_renders_a_job_input(self):
        rendered = artifacts.to_job_inputs(self.document())
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0]["sha256"], "b" * 64)
        self.assertEqual(rendered[0]["destination"], "embeddings/wavlm.tar")

    def test_missing_digest_is_refused(self):
        document = self.document()
        del document["requirements"][0]["sha256"]
        with self.assertRaises(ConfigurationError):
            artifacts.validate(document)

    def test_short_digest_is_an_integrity_error(self):
        with self.assertRaises(ArtifactIntegrityError):
            artifacts.validate(self.document(sha256="abc123"))

    def test_absolute_destination_is_refused(self):
        with self.assertRaises(ConfigurationError):
            artifacts.validate(self.document(destination="/tmp/embeddings.tar"))

    def test_traversal_destination_is_refused(self):
        with self.assertRaises(ConfigurationError):
            artifacts.validate(self.document(destination="../embeddings.tar"))

    def test_url_or_credential_in_the_key_is_refused(self):
        for bad in ("X-Amz-Signature", "https://bucket/key"):
            with self.subTest(bad=bad):
                with self.assertRaises((ConfigurationError, ArtifactIntegrityError)):
                    artifacts.validate(self.document(artifact=bad))

    def test_duplicate_destinations_are_refused(self):
        document = self.document()
        document["requirements"].append(dict(document["requirements"][0]))
        with self.assertRaises(ConfigurationError):
            artifacts.validate(document)

    def test_unknown_keys_are_refused(self):
        with self.assertRaises(ConfigurationError):
            artifacts.validate(self.document(extra="x"))

    def test_loading_a_missing_file_is_a_configuration_error(self):
        with self.assertRaises((ConfigurationError, OSError)):
            artifacts.load("/nonexistent/inputs.json")


if __name__ == "__main__":
    unittest.main()
