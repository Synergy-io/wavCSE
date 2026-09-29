"""Semantic evidence validation: valid bytes are not yet a valid result.

Every case here starts from a *well-formed* completed job — a manifest and metrics whose
digests are honest, produced by the same fixture the happy-path tests use — and then
breaks exactly one semantic property. The stored bytes and their recorded digests stay
consistent throughout, which is the point: a digest check cannot tell these apart, and
before this validator ARC would have marked every one of them COLLECTED.
"""

import hashlib
import json
import unittest

from improvements.compute import failures, jobspec, ledger, run_study
from improvements.compute.tests.fakes import (
    ComputeTestCase,
    FakeInfra,
    metrics_text,
    sample_plan,
    staged_evidence,
    worker_record,
)


class EvidenceTestCase(ComputeTestCase):
    def setUp(self):
        super(EvidenceTestCase, self).setUp()
        from improvements.compute import envelope as envelope_module

        self.make_repo()
        self.write_envelope("TR-0007")
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()
        self.allow_remote_commit()
        self.view = envelope_module.load("TR-0007")
        self.infra = FakeInfra(workers=[worker_record()])
        self.commit_sha = jobspec.git_state()["head"]
        self.name = jobspec.job_name("TR-0007", "screen", "mssl", 42)
        self.job_id = "job-" + self.name

    # ------------------------------------------------------------------ helpers

    def job_payload(self, *, state="RUNNING", outputs=None):
        return {
            "job_id": self.job_id,
            "name": self.name,
            "state": state,
            "exit_code": 0 if state == "SUCCEEDED" else None,
            "worker_id": "w-1",
            "outputs": outputs or [],
            "spec": {
                "source": {"commit": self.commit_sha},
                "tracking": {"metadata": {
                    "scope": "TR-0007", "stage": "screen", "arm": "mssl", "seed": "42",
                }},
            },
        }

    def staged(self, **kwargs):
        return staged_evidence(self.plan, stage="screen", arm="mssl", seed=42,
                               commit=self.commit_sha, job_id=self.job_id, **kwargs)

    def rewrite_manifest(self, objects, outputs, manifest, mutate):
        """Serialize a mutated manifest back and keep its recorded identity honest."""

        mutate(manifest)
        raw = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        declaration = next(item for item in outputs
                           if item["path"].endswith("MANIFEST.json"))
        objects[declaration["artifact"]] = raw
        declaration["sha256"] = hashlib.sha256(raw).hexdigest()
        declaration["verified_size_bytes"] = len(raw)
        return outputs

    def complete_and_collect(self, objects, outputs):
        """Run the real submit -> reconcile -> collect path for one staged evidence set."""

        self.infra.objects = dict(objects)
        self.infra.submit_payload = self.job_payload(outputs=outputs)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        self.infra.jobs = [self.job_payload(state="SUCCEEDED", outputs=outputs)]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        entry = list(record["entries"].values())[0]
        entry["state"] = run_study.SUCCEEDED
        run_study.save_record("TR-0007", record)
        result = run_study.collect("TR-0007", self.plan, "screen", record=record,
                                   reader=self.infra)
        return record, result

    def entry(self, record):
        return list(record["entries"].values())[0]

    def assert_rejected(self, record, result, *, expect=None):
        self.assertEqual(result["collected"], [])
        self.assertEqual(len(result["invalid"]), 1)
        self.assertEqual(result["unverified"], [])
        entry = self.entry(record)
        self.assertEqual(entry["state"], run_study.FAILED)
        self.assertEqual(entry["failure_class"], failures.EVIDENCE_INVALID)
        self.assertFalse(entry.get("outputs_verified") or False)
        if expect:
            self.assertIn(expect, entry["note"])
        return entry

class EvidenceValidationTests(EvidenceTestCase):
    """Each case breaks exactly one semantic property of otherwise valid evidence."""

    def test_1_an_empty_manifest_is_rejected(self):
        objects, outputs, _ = self.staged()
        declaration = next(item for item in outputs
                           if item["path"].endswith("MANIFEST.json"))
        objects[declaration["artifact"]] = b""
        declaration["sha256"] = hashlib.sha256(b"").hexdigest()
        declaration["verified_size_bytes"] = 0

        record, result = self.complete_and_collect(objects, outputs)

        # An empty object is not even a stored output, so the byte gate refuses it first;
        # either way it must never be collected.
        self.assertEqual(result["collected"], [])
        self.assertEqual(self.entry(record)["state"], run_study.FAILED)

    def test_2_a_malformed_manifest_is_rejected(self):
        objects, outputs, _ = self.staged()
        declaration = next(item for item in outputs
                           if item["path"].endswith("MANIFEST.json"))
        objects[declaration["artifact"]] = b"{not json at all"
        declaration["sha256"] = hashlib.sha256(b"{not json at all").hexdigest()
        declaration["verified_size_bytes"] = len(b"{not json at all")

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="not valid JSON")

    def test_3_a_manifest_from_another_study_is_rejected(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"study": "TR-9999"}))

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="study")

    def test_4_a_manifest_from_another_seed_is_rejected(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"seed": 7}))

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="seed")

    def test_5_a_manifest_from_another_commit_is_rejected(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"commit": "b" * 40}))

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="commit")

    def test_5b_a_manifest_from_another_job_is_rejected(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"job_id": "job-ffffffffffffffff"}))

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="job_id")

    def test_6_a_missing_required_metric_is_rejected(self):
        tasks = ["ks", "si"]
        objects, outputs, _ = self.staged(
            metrics=metrics_text(tasks) + "\n")  # the ``er`` task line is absent

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="missing per-task metrics")

    def test_7_a_non_finite_metric_is_rejected(self):
        objects, outputs, _ = self.staged(
            metrics="loss_all=1.0 | acc_all=0.5\nks | loss=1.0 | acc=nan | samples=10\n"
                    "si | loss=1.0 | acc=0.5 | samples=10\n"
                    "er | loss=1.0 | acc=0.5 | samples=10\n")

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="must be finite")

    def test_7b_an_impossible_accuracy_is_rejected(self):
        objects, outputs, _ = self.staged(
            metrics="loss_all=1.0 | acc_all=0.5\nks | loss=1.0 | acc=1.42 | samples=10\n"
                    "si | loss=1.0 | acc=0.5 | samples=10\n"
                    "er | loss=1.0 | acc=0.5 | samples=10\n")

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="outside the range")

    def test_8_a_file_staged_from_a_neighbouring_run_is_rejected(self):
        """The wrapper staged one file out of the previous seed's directory.

        Digests are honest and every identity field matches, so only the coherence of the
        staged sources catches this: the manifest says the run wrote its results under one
        directory, and one of its files came from another.
        """

        objects, outputs, manifest = self.staged()

        def borrow_another_run(doc):
            doc["files"][0]["source"] = doc["files"][0]["source"].replace(
                "mssl_s42", "mssl_s99")

        self.rewrite_manifest(objects, outputs, manifest, borrow_another_run)

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="does not come from this run")

    def test_8b_metrics_naming_another_protocols_task_are_rejected(self):
        objects, outputs, _ = self.staged(
            metrics=metrics_text(["ks", "si", "er", "ic"]))

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="does not run")

    def test_8c_a_manifest_describing_other_bytes_is_rejected(self):
        """The manifest is internally consistent but does not describe what was stored."""

        objects, outputs, manifest = self.staged()

        def claim_other_bytes(doc):
            doc["files"][0]["sha256"] = "c" * 64

        self.rewrite_manifest(objects, outputs, manifest, claim_other_bytes)

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="not the digest the control plane")

    def test_9_valid_bytes_with_wrong_semantics_is_rejected(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"training_exit_code": 1}))

        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="training exit code")

    def test_9b_an_unmaterialized_required_input_is_rejected(self):
        objects, outputs, _ = self.staged()
        self.infra.objects = dict(objects)
        self.infra.submit_payload = self.job_payload(outputs=outputs)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        payload = self.job_payload(state="SUCCEEDED", outputs=outputs)
        payload["inputs"] = [{"destination": "embeddings/wavlm.tar", "required": True,
                              "materialized": False, "sha256": "d" * 64}]
        self.infra.jobs = [payload]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        entry = self.entry(record)
        entry["state"] = run_study.SUCCEEDED
        run_study.save_record("TR-0007", record)

        result = run_study.collect("TR-0007", self.plan, "screen", record=record,
                                   reader=self.infra)

        self.assert_rejected(record, result, expect="was not materialized")

    def test_10_a_valid_negative_result_is_collected(self):
        """A run that did badly is evidence, and evidence is collected."""

        weak = ("loss_all=4.2 | acc_all=0.041 | \n"
                "ks | loss=4.2 | acc=0.03 | samples=500\n"
                "si | loss=4.9 | acc=0.002 | samples=800\n"
                "er | loss=3.1 | acc=0.24 | samples=120\n").replace(" | \n", "\n")
        objects, outputs, _ = self.staged(metrics=weak)

        record, result = self.complete_and_collect(objects, outputs)

        self.assertEqual(result["invalid"], [])
        self.assertEqual(len(result["collected"]), 1)
        entry = self.entry(record)
        self.assertEqual(entry["state"], run_study.COLLECTED)
        self.assertTrue(entry["evidence_verified"])
        metrics = entry["evidence"]["metrics"]
        self.assertEqual(len(metrics), 1)
        summary = list(metrics.values())[0]
        self.assertEqual(summary["all"]["accuracy"], 0.041)
        self.assertEqual(summary["tasks"]["er"]["accuracy"], 0.24)


class EvidenceGateTests(EvidenceTestCase):
    def test_an_invalid_result_does_not_advance_state_or_allow_cleanup(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"study": "TR-9999"}))
        record, result = self.complete_and_collect(objects, outputs)
        self.assertEqual(result["collected"], [])

        ledger.redeem_create("TR-0007", worker_id="w-1", purpose="test",
                             envelope_digest=self.view.digest,
                             deadline="2999-01-01T00:00:00+00:00")
        finished = run_study.finish("TR-0007", self.plan, infra=self.infra,
                                    view=self.view, record=record)

        self.assertFalse(finished["finished"])
        self.assertEqual(self.infra.destroy_calls, [])

    def test_collection_without_a_reader_is_refused(self):
        """There is no path that marks a result COLLECTED from bytes alone."""

        objects, outputs, _ = self.staged()
        self.infra.objects = dict(objects)
        self.infra.submit_payload = self.job_payload(outputs=outputs)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        self.infra.jobs = [self.job_payload(state="SUCCEEDED", outputs=outputs)]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        self.entry(record)["state"] = run_study.SUCCEEDED
        run_study.save_record("TR-0007", record)

        with self.assertRaises(TypeError):
            run_study.collect("TR-0007", self.plan, "screen", record=record)

    def test_readback_is_bound_to_the_verified_digest(self):
        """A stored object replaced after verification is detected, not accepted."""

        objects, outputs, _ = self.staged()
        declaration = next(item for item in outputs if item["path"].endswith(".txt"))
        objects[declaration["artifact"]] = b"loss_all=0.1 | acc_all=0.99\n"

        record, result = self.complete_and_collect(objects, outputs)

        self.assertEqual(result["collected"], [])
        self.assertEqual(self.entry(record)["failure_class"], failures.ARTIFACT_INTEGRITY)

    def test_a_transient_read_failure_does_not_invalidate_the_run(self):
        """A control-plane read failure is infrastructure, not a verdict on the result."""

        from improvements.compute.errors import TransientInfraError

        objects, outputs, _ = self.staged()
        self.infra.objects = dict(objects)
        self.infra.submit_payload = self.job_payload(outputs=outputs)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        self.infra.jobs = [self.job_payload(state="SUCCEEDED", outputs=outputs)]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        self.entry(record)["state"] = run_study.SUCCEEDED
        run_study.save_record("TR-0007", record)

        def unavailable(*args, **kwargs):
            raise TransientInfraError("the control plane is unreachable")

        self.infra.storage_read = unavailable
        with self.assertRaises(TransientInfraError):
            run_study.collect("TR-0007", self.plan, "screen", record=record,
                              reader=self.infra)

        self.assertEqual(self.entry(record)["state"], run_study.SUCCEEDED)

    def test_an_invalid_result_is_not_retried_and_the_class_is_kept(self):
        objects, outputs, manifest = self.staged()
        self.rewrite_manifest(objects, outputs, manifest,
                              lambda doc: doc.update({"seed": 3}))
        record, result = self.complete_and_collect(objects, outputs)

        outcomes = run_study.assess(record, stage="screen")

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["class"], failures.EVIDENCE_INVALID)
        self.assertFalse(outcomes[0]["retry"])
        self.assertEqual(self.entry(record)["state"], run_study.FAILED)

    def test_a_valid_result_keeps_its_collected_state_through_assess(self):
        objects, outputs, _ = self.staged()
        record, result = self.complete_and_collect(objects, outputs)

        outcomes = run_study.assess(record, stage="screen")

        self.assertEqual(outcomes, [])
        self.assertEqual(self.entry(record)["state"], run_study.COLLECTED)


class EmbeddingLayoutEvidenceTests(EvidenceTestCase):
    """The run's loader-root record is checked against the plan's declared inputs.

    The wrapper writes which artifacts it turned into the loader's tree. Those bytes are
    the ones every embedding this run reads came from, so the record has to be proven
    against the plan rather than believed: a manifest reporting a different artifact, a
    different digest or an undeclared dataset describes some other run's inputs.
    """

    ARTIFACT = "embeddings/v1/speechcommand/training-000.tar"
    DIGEST = hashlib.sha256(b"verified shard bytes").hexdigest()

    def setUp(self):
        super(EmbeddingLayoutEvidenceTests, self).setUp()
        import os

        repo = os.environ["WAVCSE_REPO_ROOT"]
        self.relative = "studies/TR-0007/compute/inputs.json"
        self.write_inputs([{
            "artifact": self.ARTIFACT,
            "destination": "embeddings/training-000.tar",
            "sha256": self.DIGEST,
            "size_bytes": 20,
            "required": True,
        }], repo=repo, relative=self.relative)
        self.layout = {
            "root": "embedding",
            "upstream_model_type": "wavlm_large",
            "frame_pool_id": "mean",
            "datasets": [{
                "dataset": "speechcommand",
                "artifacts": [{"artifact": self.ARTIFACT, "sha256": self.DIGEST,
                               "files": 20000}],
            }],
        }
        self.plan = jobspec.load_plan(self.write_plan(sample_plan(
            inputs_file=self.relative,
            embedding_layout={
                "root": "embedding",
                "datasets": [{"dataset": "speechcommand", "input": self.ARTIFACT}],
            },
        )))
        # The plan and its inputs file are committed, because a recorded job is only
        # generated from a commit-clean tree and the inputs file must be provable.
        self.commit()
        self.commit_sha = jobspec.git_state()["head"]

    def test_a_layout_built_from_the_declared_inputs_is_recorded(self):
        objects, outputs, _ = self.staged(
            overrides={"embedding_layout": self.layout})
        record, result = self.complete_and_collect(objects, outputs)

        self.assertEqual(len(result["collected"]), 1)
        summary = self.entry(record)["evidence"]
        self.assertEqual(summary["embedding_layout"]["datasets"], {"speechcommand": 1})

    def test_an_undeclared_artifact_is_rejected(self):
        layout = json.loads(json.dumps(self.layout))
        layout["datasets"][0]["artifacts"][0]["artifact"] = "embeddings/v1/other.tar"
        objects, outputs, _ = self.staged(overrides={"embedding_layout": layout})
        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="does not declare")

    def test_another_digest_is_rejected(self):
        layout = json.loads(json.dumps(self.layout))
        layout["datasets"][0]["artifacts"][0]["sha256"] = "b" * 64
        objects, outputs, _ = self.staged(overrides={"embedding_layout": layout})
        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="but the plan declares")

    def test_an_undeclared_dataset_is_rejected(self):
        layout = json.loads(json.dumps(self.layout))
        layout["datasets"].append({"dataset": "iemocap", "artifacts": [
            {"artifact": self.ARTIFACT, "sha256": self.DIGEST, "files": 1}]})
        objects, outputs, _ = self.staged(overrides={"embedding_layout": layout})
        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="never loads")

    def test_a_manifest_without_the_layout_is_rejected(self):
        objects, outputs, _ = self.staged()
        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result,
                             expect="does not report the embedding layout")

    def test_a_layout_the_plan_never_declared_is_rejected(self):
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()
        self.commit_sha = jobspec.git_state()["head"]
        objects, outputs, _ = self.staged(
            overrides={"embedding_layout": self.layout})
        record, result = self.complete_and_collect(objects, outputs)

        self.assert_rejected(record, result, expect="the plan declares none")


if __name__ == "__main__":
    unittest.main()
