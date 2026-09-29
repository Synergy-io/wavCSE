"""Stage orchestration: duplicate safety, resume after restart, and verification."""

import unittest

from improvements.compute import envelope as envelope_module
from improvements.compute import failures, jobspec, ledger, run_study
from improvements.compute.errors import (
    ArtifactIntegrityError,
    AuthorizationError,
    RepositoryConflictError,
    ReconcilableError,
    CostError,
)
from improvements.compute.tests.fakes import ComputeTestCase, FakeInfra, sample_plan, worker_record


class StageTestCase(ComputeTestCase):
    def setUp(self):
        super(StageTestCase, self).setUp()
        self.make_repo()
        self.write_envelope("TR-0007")
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()
        self.view = envelope_module.load("TR-0007")
        self.infra = FakeInfra(workers=[worker_record()])
        self.commit_sha = jobspec.git_state()["head"]

    def job_payload(self, name, *, state="RUNNING", worker_id="w-1", outputs=None):
        return {
            "job_id": "job-{}".format(name),
            "name": name,
            "state": state,
            "exit_code": 0 if state == "SUCCEEDED" else None,
            "worker_id": worker_id,
            "outputs": outputs or [],
            "spec": {
                "source": {"commit": self.commit_sha},
                "tracking": {"metadata": {
                    "scope": "TR-0007", "stage": "screen", "arm": "mssl", "seed": "42",
                }},
            },
        }

    def spec_name(self):
        return jobspec.job_name("TR-0007", "screen", "mssl", 42)


class SubmitTests(StageTestCase):
    def test_multi_seed_stage_submits_only_one_job_per_transition(self):
        payload = self.job_payload(jobspec.job_name("TR-0007", "confirm", "mssl", 0))
        payload["spec"]["tracking"]["metadata"].update({"stage": "confirm", "seed": "0"})
        self.infra.submit_payload = payload
        record = run_study.load_record("TR-0007")
        result = run_study.submit_pending(
            "TR-0007", self.plan, "confirm", infra=self.infra, view=self.view,
            record=record)
        self.assertEqual(len(result["submitted"]), 1)
        self.assertEqual(self.infra.count("job_submit"), 1)
        self.assertEqual(len(record["entries"]), 1)

    def test_submission_records_the_job_and_its_worker(self):
        self.infra.submit_payload = self.job_payload(self.spec_name())
        record = run_study.load_record("TR-0007")
        result = run_study.submit_pending(
            "TR-0007", self.plan, "screen", infra=self.infra, view=self.view,
            record=record,
        )
        self.assertEqual(len(result["submitted"]), 1)
        self.assertEqual(self.infra.count("job_submit"), 1)
        entry = list(record["entries"].values())[0]
        self.assertEqual(entry["state"], run_study.SUBMITTED)
        self.assertEqual(entry["attempts"], 1)

    def test_spec_is_written_to_controller_local_state(self):
        self.infra.submit_payload = self.job_payload(self.spec_name())
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        spec_files = []
        for root, _dirs, files in __import__("os").walk(
            __import__("improvements.compute.state", fromlist=["state"]).state_root()
        ):
            spec_files.extend(name for name in files if name.endswith(".json"))
        self.assertTrue(any("TR-0007__screen__mssl__s42" in name for name in spec_files))

    def test_lost_acknowledgement_is_adopted_not_resubmitted(self):
        # The provider accepted the job; the controller saw no record.
        accepted = self.job_payload(self.spec_name())
        def lost_ack(spec_path, worker_id):
            self.infra.calls.append(("job_submit", spec_path, worker_id))
            self.infra.jobs.append(accepted)
            return type("R", (), {"returncode": 1, "payload": None})()
        self.infra.job_submit = lost_ack
        record = run_study.load_record("TR-0007")
        result = run_study.submit_pending("TR-0007", self.plan, "screen",
                                          infra=self.infra, view=self.view,
                                          record=record)
        self.assertEqual(self.infra.count("job_submit"), 1)
        self.assertEqual(result["submitted"], ["job-TR-0007__screen__mssl__s42"])

    def test_unreconcilable_submission_never_repeats_the_request(self):
        self.infra.submit_payload = None
        self.infra.submit_returncode = 1
        record = run_study.load_record("TR-0007")
        with self.assertRaises(ReconcilableError):
            run_study.submit_pending("TR-0007", self.plan, "screen",
                                     infra=self.infra, view=self.view, record=record)
        self.assertEqual(self.infra.count("job_submit"), 1)
        with self.assertRaises(ReconcilableError):
            run_study.advance("TR-0007", self.plan, "screen", infra=self.infra,
                              record=run_study.load_record("TR-0007"))
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_an_existing_job_is_never_submitted_twice(self):
        self.infra.submit_payload = self.job_payload(self.spec_name())
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_restart_does_not_resubmit(self):
        self.infra.submit_payload = self.job_payload(self.spec_name())
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        # A restart is a fresh read of the run ledger.
        reloaded = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=reloaded)
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_no_authorization_means_no_submission(self):
        import os
        import shutil

        shutil.rmtree(os.path.join(os.environ["WAVCSE_REPO_ROOT"],
                                   "improvements", "taskrelation", "research",
                                   "authorizations"), ignore_errors=True)
        self.commit("drop the envelope")
        with self.assertRaises(AuthorizationError):
            envelope_module.load("TR-0007")
        self.assertEqual(self.infra.count("job_submit"), 0)

    def test_dirty_tree_cannot_submit_a_recorded_job(self):
        with open(__import__("os").path.join(
                __import__("os").environ["WAVCSE_REPO_ROOT"], "tracked.txt"),
                "a", encoding="utf-8") as handle:
            handle.write("uncommitted\n")
        record = run_study.load_record("TR-0007")
        with self.assertRaises(RepositoryConflictError):
            run_study.submit_pending("TR-0007", self.plan, "screen",
                                     infra=self.infra, view=self.view, record=record)
        self.assertEqual(self.infra.count("job_submit"), 0)

    def test_untracked_scientific_file_cannot_submit_a_recorded_job(self):
        import os
        path = os.path.join(os.environ["WAVCSE_REPO_ROOT"], "new_model.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# uncommitted model\n")
        with self.assertRaises(RepositoryConflictError):
            run_study.submit_pending("TR-0007", self.plan, "screen",
                                     infra=self.infra, view=self.view,
                                     record=run_study.load_record("TR-0007"))
        self.assertEqual(self.infra.count("job_submit"), 0)

    def test_dry_run_previews_without_submitting(self):
        record = run_study.load_record("TR-0007")
        result = run_study.submit_pending(
            "TR-0007", self.plan, "screen", infra=self.infra, view=self.view,
            record=record, dry_run=True,
        )
        self.assertEqual(len(result["planned"]), 1)
        self.assertEqual(self.infra.count("job_submit"), 0)


class AdvanceTests(StageTestCase):
    def test_resume_preserves_verified_seed_and_submits_only_missing_seed(self):
        old_key = jobspec.job_key("TR-0007", "TR-0007", "confirm", "mssl", 0,
                                  self.commit_sha)
        record = run_study.load_record("TR-0007")
        record["entries"][old_key] = {
            "job_key": old_key, "arm": "mssl", "seed": 0, "stage": "confirm",
            "state": run_study.COLLECTED, "job_id": "job-seed-zero",
            "outputs_verified": True, "attempts": 1,
        }
        run_study.save_record("TR-0007", record)
        payload = self.job_payload(jobspec.job_name("TR-0007", "confirm", "mssl", 1))
        payload["spec"]["tracking"]["metadata"].update({"stage": "confirm", "seed": "1"})
        self.infra.submit_payload = payload
        result = run_study.advance("TR-0007", self.plan, "confirm", infra=self.infra,
                                   record=run_study.load_record("TR-0007"))
        self.assertEqual(result["step"], "submitted")
        reloaded = run_study.load_record("TR-0007")
        self.assertEqual(reloaded["entries"][old_key]["state"], run_study.COLLECTED)
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_budget_exhaustion_stops_before_resuming_a_preparing_job(self):
        ledger.redeem_create("TR-0007", worker_id="w-1", purpose="test",
                             envelope_digest=self.view.digest,
                             deadline="2999-01-01T00:00:00+00:00")
        self.view.envelope["expires_at"] = "2000-01-01T00:00:00+00:00"
        self.infra.jobs = [self.job_payload(self.spec_name(), state="PREPARING")]
        with self.assertRaises(CostError):
            run_study.advance("TR-0007", self.plan, "screen", infra=self.infra,
                              record=run_study.load_record("TR-0007"), view=self.view)
        self.assertEqual(self.infra.stop_calls, ["w-1"])
        self.assertEqual(self.infra.count("job_status"), 0)

    def test_nothing_is_submitted_while_a_job_is_in_flight(self):
        self.infra.submit_payload = self.job_payload(self.spec_name())
        record = run_study.load_record("TR-0007")
        first = run_study.advance("TR-0007", self.plan, "screen", infra=self.infra,
                                  record=record)
        self.assertEqual(first["step"], "submitted")
        second = run_study.advance("TR-0007", self.plan, "screen", infra=self.infra,
                                   record=record)
        self.assertEqual(second["step"], "monitor")
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_a_failed_job_is_not_retried_when_it_is_an_implementation_bug(self):
        name = self.spec_name()
        self.infra.submit_payload = self.job_payload(name)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        # The provider reports a code-level failure.
        failed = self.job_payload(name, state="FAILED")
        failed["exit_code"] = 1
        self.infra.jobs = [failed]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        outcomes = run_study.assess(record, stage="screen")
        entry = list(record["entries"].values())[0]
        self.assertEqual(outcomes[0]["class"], failures.IMPLEMENTATION_BUG)
        self.assertFalse(outcomes[0]["retry"])
        self.assertEqual(entry["state"], run_study.FAILED)

    def test_job_status_with_neighbor_identity_is_rejected(self):
        name = self.spec_name()
        self.infra.submit_payload = self.job_payload(name)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        wrong = self.job_payload(name)
        wrong["spec"]["tracking"]["metadata"]["scope"] = "TR-0008"
        self.infra.jobs = [wrong]
        with self.assertRaises(ArtifactIntegrityError):
            run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                                record=record)

    def test_disappeared_worker_is_classified_as_infra_failure(self):
        name = self.spec_name()
        self.infra.submit_payload = self.job_payload(name)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        gone = self.job_payload(name, state="FAILED")
        gone["worker_absent"] = True
        gone["remote_status"] = "worker_absent"
        self.infra.jobs = [gone]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        outcomes = run_study.assess(record, stage="screen")
        self.assertEqual(outcomes[0]["class"], failures.TRANSIENT_INFRA)

    def test_claimed_success_with_nonzero_exit_is_rejected(self):
        name = self.spec_name()
        self.infra.submit_payload = self.job_payload(name)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        wrong = self.job_payload(name, state="SUCCEEDED")
        wrong["exit_code"] = 1
        self.infra.jobs = [wrong]
        with self.assertRaises(ArtifactIntegrityError):
            run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                                record=record)

    def test_transient_failure_is_retried_until_the_budget_runs_out(self):
        name = self.spec_name()
        self.infra.submit_payload = self.job_payload(name)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        oom = self.job_payload(name, state="FAILED")
        oom["exit_code"] = 137
        self.infra.jobs = [oom]
        self.infra.logs["job-" + name] = "CUDA out of memory"
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        outcomes = run_study.assess(record, stage="screen")
        self.assertEqual(outcomes[0]["class"], failures.RESOURCE_OOM)
        self.assertTrue(outcomes[0]["retry"])
        entry = list(record["entries"].values())[0]
        self.assertEqual(entry["state"], run_study.PENDING)
        self.assertEqual(entry["attempts"], 1)
        self.assertEqual(entry["previous_job_ids"], ["job-" + name])
        # A restart between retry scheduling and submission must not adopt the
        # old failed attempt merely because its deterministic spec matches.
        restarted = run_study.load_record("TR-0007")
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=restarted)
        self.assertIsNone(list(restarted["entries"].values())[0]["job_id"])


class CollectAndFinishTests(StageTestCase):
    def complete_outputs(self):
        return [{"path": item["path"], "artifact": item["artifact"],
                 "required": item["required"], "persisted": True,
                 "verified_size_bytes": 10, "sha256": "a" * 64}
                for item in jobspec.declared_outputs(
                    self.plan, jobspec.arm_by_name(self.plan, "mssl"), 42)]

    def verified_entry(self, outputs):
        name = self.spec_name()
        self.infra.submit_payload = self.job_payload(name, outputs=outputs)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        self.infra.jobs = [self.job_payload(name, state="SUCCEEDED", outputs=outputs)]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        entry = list(record["entries"].values())[0]
        entry["state"] = run_study.SUCCEEDED
        run_study.save_record("TR-0007", record)
        return record

    def test_verified_outputs_are_collected(self):
        outputs = self.complete_outputs()
        record = self.verified_entry(outputs)
        result = run_study.collect("TR-0007", self.plan, "screen", record=record)
        self.assertEqual(len(result["collected"]), 1)
        self.assertEqual(result["unverified"], [])

    def test_unverified_output_fails_the_entry_and_spares_the_worker(self):
        outputs = [
            {"path": "outputs/mssl_s42/MANIFEST.json", "required": True,
             "persisted": True, "verified_size_bytes": 10},
            {"path": "outputs/mssl_s42/checkpoint_best.pth", "required": True,
             "persisted": False},
        ]
        record = self.verified_entry(outputs)
        result = run_study.collect("TR-0007", self.plan, "screen", record=record)
        self.assertEqual(result["collected"], [])
        self.assertEqual(len(result["unverified"]), 1)
        entry = list(record["entries"].values())[0]
        self.assertEqual(entry["failure_class"], failures.ARTIFACT_INTEGRITY)
        self.assertEqual(self.infra.destroy_calls, [])

    def test_missing_declared_output_cannot_be_collected(self):
        outputs = self.complete_outputs()
        outputs.pop(0)
        record = self.verified_entry(outputs)
        result = run_study.collect("TR-0007", self.plan, "screen", record=record)
        self.assertEqual(result["collected"], [])
        self.assertEqual(len(result["unverified"]), 1)

    def test_empty_required_output_cannot_be_collected(self):
        outputs = self.complete_outputs()
        outputs[0]["verified_size_bytes"] = 0
        record = self.verified_entry(outputs)
        result = run_study.collect("TR-0007", self.plan, "screen", record=record)
        self.assertEqual(result["collected"], [])

    def test_wrong_digest_or_artifact_cannot_be_collected(self):
        for change in ({"sha256": "b"}, {"artifact": "neighbor/checkpoint.pth"}):
            outputs = self.complete_outputs()
            outputs[0].update(change)
            record = self.verified_entry(outputs)
            result = run_study.collect("TR-0007", self.plan, "screen", record=record)
            self.assertEqual(result["collected"], [])

    def test_finish_destroys_only_after_everything_is_collected(self):
        outputs = self.complete_outputs()
        record = self.verified_entry(outputs)
        pending = run_study.finish("TR-0007", self.plan, infra=self.infra,
                                   view=self.view, record=record)
        self.assertFalse(pending["finished"])
        self.assertEqual(self.infra.destroy_calls, [])

        ledger.redeem_create("TR-0007", worker_id="w-1", purpose="test",
                            envelope_digest=self.view.digest,
                            deadline="2999-01-01T00:00:00+00:00")
        run_study.collect("TR-0007", self.plan, "screen", record=record)
        done = run_study.finish("TR-0007", self.plan, infra=self.infra,
                                view=self.view, record=record)
        self.assertTrue(done["finished"])
        self.assertEqual(self.infra.destroy_calls, ["w-1"])


if __name__ == "__main__":
    unittest.main()
