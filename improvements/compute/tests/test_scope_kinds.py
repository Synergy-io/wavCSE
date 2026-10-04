"""Execution-scope kind: a research Study and an infrastructure-validation scope.

The contract these tests cover: an authorized execution scope is a pair of
(identity, kind). A Study scope names a Study and carries research semantics; an
``infrastructure_validation`` scope is first-class, needs no Study registration,
and can never borrow, imply or confer a Study's scientific authority. Kind and
identity are compared together wherever a plan meets its authorization, so a
mismatch fails closed.

The other half of the suite is backward compatibility: a plan or an envelope
written before kinds existed carries none, is deterministically a Study, and
must behave exactly as it did before — including DG-0008's approved 201-job
topology and every committed Study record.
"""

import copy
import hashlib
import json
import os
import unittest
from unittest import mock

from improvements.compute import (
    envelope,
    jobspec,
    ledger,
    providers,
    resolve as resolve_module,
    run_study,
    scopes,
    sweep,
    worker,
)
from improvements.compute import __main__ as compute_main
from improvements.compute.errors import (
    AuthorizationError,
    ConfigurationError,
    CostError,
)
from improvements.compute.tests.fakes import (
    ComputeTestCase,
    FakeInfra,
    sample_plan,
    worker_record,
)
from improvements.compute.tests.test_provider_neutral import (
    ProviderInfra,
    colab_envelope,
    empty_facts,
    legacy_runpod_envelope,
    view_for,
)
from improvements.compute.tests.test_stage_arms import dg0008_plan

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SMOKE_PLAN = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "execution",
    "IN-0001", "smoke_plan.json")


def smoke_plan():
    """The committed IN-0001 plan, without its load-time annotation."""

    plan = jobspec.load_plan(SMOKE_PLAN)
    plan.pop("_path", None)
    return copy.deepcopy(plan)


def in0001_envelope():
    document = colab_envelope()
    document["scope"] = "IN-0001"
    document["scope_kind"] = scopes.INFRASTRUCTURE_VALIDATION
    return document


def studies_rows():
    path = os.path.join(REPO_ROOT, "improvements", "taskrelation", "research",
                        "STUDIES.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ScopeKindParsingTests(unittest.TestCase):
    def test_a_missing_kind_is_deterministically_study(self):
        self.assertEqual(scopes.STUDY, scopes.normalize_kind(None))
        self.assertEqual(scopes.STUDY, scopes.normalize_kind(""))

    def test_both_kinds_are_recognised(self):
        for kind in scopes.KINDS:
            self.assertEqual(kind, scopes.normalize_kind(kind))

    def test_an_unknown_kind_is_refused(self):
        for value in ("research", "STUDY", "infra", 3, ["study"]):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    scopes.normalize_kind(value)

    def test_scope_identity_shape_is_shared(self):
        for value in ("IN-0001", "DG-0008", "TR-0007"):
            self.assertEqual(value, scopes.require_scope_id(value))
        for value in ("in-0001", "IN-1", "IN0001", "", None, "IN-00010"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    scopes.require_scope_id(value)


class LegacyPlanTests(unittest.TestCase):
    """A plan written before kinds existed keeps its exact meaning and shape."""

    def test_legacy_plan_is_a_study_scope(self):
        plan = sample_plan()
        self.assertEqual(scopes.STUDY, jobspec.plan_scope_kind(plan))
        self.assertEqual("TR-0007", jobspec.plan_scope(plan))
        jobspec.validate_plan(plan)

    def test_explicit_study_kind_matches_the_legacy_spelling(self):
        plan = sample_plan(scope_kind=scopes.STUDY)
        self.assertEqual(jobspec.plan_scope(sample_plan()), jobspec.plan_scope(plan))
        self.assertEqual(jobspec.plan_scope_kind(sample_plan()),
                         jobspec.plan_scope_kind(plan))
        jobspec.validate_plan(plan)

    def test_a_study_plan_may_not_also_name_a_scope(self):
        plan = sample_plan(scope="IN-0001")
        with self.assertRaises(ConfigurationError) as caught:
            jobspec.validate_plan(plan)
        self.assertIn("scope", str(caught.exception))

    def test_an_unknown_plan_kind_fails_closed(self):
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(sample_plan(scope_kind="research"))

    def test_study_environment_secrets_are_still_required_in_full(self):
        for secrets in ([], ["MLFLOW_TRACKING_USERNAME"], None):
            plan = sample_plan()
            if secrets is None:
                plan.pop("environment_secrets")
            else:
                plan["environment_secrets"] = secrets
            with self.subTest(secrets=secrets):
                with self.assertRaises(ConfigurationError):
                    jobspec.validate_plan(plan)


class StudySpecShapeTests(ComputeTestCase):
    """The Study job spec — and therefore what a Study job executes — is unchanged."""

    def setUp(self):
        super(StudySpecShapeTests, self).setUp()
        self.make_repo()
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()
        self.commit_sha = jobspec.git_state()["head"]

    def build(self):
        return jobspec.build_spec(
            self.plan, stage="screen", arm="mssl", seed=42,
            commit=self.commit_sha, scope="TR-0007", envelope_digest="d" * 64,
        )

    def test_the_spec_still_names_the_committed_research_wrapper(self):
        spec = self.build()
        self.assertEqual([
            "uv", "run", "--locked", "python", "-m",
            "improvements.compute.worker_stage",
            "--plan", "studies/TR-0007/compute/plan.json",
            "--stage", "screen", "--arm", "mssl", "--seed", "42",
            "--outputs-root", "outputs/mssl_s42",
        ], spec["command"]["argv"])

    def test_the_identity_metadata_is_unchanged(self):
        metadata = self.build()["tracking"]["metadata"]
        self.assertEqual("TR-0007", metadata["study_id"])
        self.assertNotIn("scope_id", metadata)
        self.assertNotIn("scope_kind", metadata)

    def test_the_declared_outputs_still_include_the_evidence_manifest(self):
        declared = jobspec.declared_outputs(
            self.plan, jobspec.arm_by_name(self.plan, "mssl"), 42)
        self.assertEqual(
            "TR-0007/mssl_s42/MANIFEST.json",
            [item["artifact"] for item in declared
             if item["path"].endswith("MANIFEST.json")][0],
        )

    def test_the_study_setup_and_secrets_are_unchanged(self):
        spec = self.build()
        self.assertEqual(["uv", "sync", "--locked"], spec["setup"]["argv"])
        self.assertEqual(["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"],
                         spec["runtime"]["environment_secrets"])


class AuthorizationIsolationTests(ComputeTestCase):
    """A grant is isolated by scope identity and by scope kind."""

    def setUp(self):
        super(AuthorizationIsolationTests, self).setUp()
        self.make_repo()
        self.allow_remote_commit()
        self.plan = jobspec.load_plan(self.write_plan(smoke_plan(), directory="IN-0001"))
        self.write_envelope("IN-0001", envelope=in0001_envelope())
        self.commit()
        self.view = envelope.load("IN-0001", os.environ["WAVCSE_REPO_ROOT"])

    def test_the_authorization_carries_its_kind(self):
        self.assertEqual(scopes.INFRASTRUCTURE_VALIDATION,
                         self.view.as_dict()["scope_kind"])

    def test_an_unknown_envelope_kind_fails_closed(self):
        document = in0001_envelope()
        document["scope_kind"] = "research"
        with self.assertRaises(ConfigurationError):
            envelope.validate(document)

    def test_a_legacy_envelope_is_a_study_grant(self):
        document = legacy_runpod_envelope()
        envelope.validate(document)
        self.assertEqual(scopes.STUDY, envelope.envelope_scope_kind(document))

    def test_the_plan_and_the_grant_must_agree(self):
        run_study._require_scope("IN-0001", self.plan, self.view)

    def test_a_study_plan_cannot_spend_an_infrastructure_grant(self):
        # The same identity, a different kind: a Study plan named IN-0001 is
        # refused by the infrastructure grant of that name.
        study_plan = sample_plan(study="IN-0001")
        jobspec.validate_plan(study_plan)
        with self.assertRaises(AuthorizationError):
            run_study._require_scope("IN-0001", study_plan, self.view)

    def test_an_infrastructure_plan_cannot_spend_a_study_grant(self):
        document = colab_envelope()
        document["scope"] = "IN-0001"
        with self.assertRaises(AuthorizationError):
            run_study._require_scope("IN-0001", self.plan, view_for(document))

    def test_a_plan_for_another_scope_is_refused(self):
        other = smoke_plan()
        other["scope"] = "IN-0002"
        with self.assertRaises(ConfigurationError):
            run_study._require_scope("IN-0001", other, self.view)

    def test_the_envelope_check_refuses_a_kind_the_grant_does_not_hold(self):
        decision = envelope.check(
            view_for(legacy_runpod_envelope()), envelope.ACTION_CREATE_WORKER,
            empty_facts(),
            requested={"provider": providers.RUNPOD, "hourly_usd": "0.5",
                       "projected_hours": "1", "container_disk_gb": 60,
                       "scope_kind": scopes.INFRASTRUCTURE_VALIDATION},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual("AUTHORIZATION", decision.klass)

    def test_a_matching_kind_is_still_checked_against_the_cost_rules(self):
        decision = envelope.check(
            view_for(in0001_envelope()), envelope.ACTION_CREATE_WORKER,
            empty_facts(),
            requested={"provider": providers.COLAB, "projected_hours": "1",
                       "scope_kind": scopes.INFRASTRUCTURE_VALIDATION},
        )
        self.assertTrue(decision.allowed)

    def test_worker_ensure_refuses_a_kind_mismatch_before_any_provider_call(self):
        infra = ProviderInfra()
        # The right scope identity and the right provider, but a legacy (Study)
        # grant: the kind, not the identity or provider, is what differs.
        document = colab_envelope()
        document["scope"] = "IN-0001"
        with self.assertRaises(AuthorizationError):
            worker.ensure_worker(self.plan, view_for(document), infra=infra)
        self.assertEqual([], infra.verbs())

    def test_worker_ensure_refuses_a_scope_identity_mismatch(self):
        document = in0001_envelope()
        document["scope"] = "IN-0002"
        wrong = view_for(document)
        with self.assertRaises(AuthorizationError):
            worker.ensure_worker(self.plan, wrong, infra=ProviderInfra())


class InfrastructureValidationPlanTests(unittest.TestCase):
    """The committed IN-0001 plan is a valid, minimal, non-Study workload."""

    def setUp(self):
        self.plan = jobspec.load_plan(SMOKE_PLAN)

    def test_in0001_is_not_a_registered_study(self):
        self.assertNotIn("IN-0001", {row["study_id"] for row in studies_rows()})

    def test_the_plan_validates_without_any_study_registration(self):
        jobspec.validate_plan(self.plan)
        self.assertEqual(scopes.INFRASTRUCTURE_VALIDATION,
                         jobspec.plan_scope_kind(self.plan))
        self.assertEqual("IN-0001", jobspec.plan_scope(self.plan))

    def test_it_is_exactly_one_job(self):
        self.assertEqual([("smoke", 0)], run_study.plan_jobs(self.plan, "smoke"))

    def test_it_resolves_to_colab(self):
        self.assertEqual(providers.COLAB, jobspec.plan_provider(self.plan))
        self.assertEqual({"gpu_type": "T4", "gpu_count": 1}, self.plan["worker"])

    def test_it_declares_no_research_credential(self):
        self.assertEqual([], self.plan["environment_secrets"])

    def test_it_needs_no_embeddings_or_study_inputs(self):
        self.assertNotIn("inputs_file", self.plan)
        self.assertNotIn("embedding_layout", self.plan)
        self.assertEqual([], run_study.inputs_for(self.plan, REPO_ROOT))

    def test_the_job_is_a_direct_probe_with_no_research_wrapper(self):
        spec = jobspec.build_spec(
            self.plan, stage="smoke", arm="smoke", seed=0,
            commit="a" * 40, scope="IN-0001", envelope_digest="d" * 64)
        argv = spec["command"]["argv"]
        self.assertEqual("python3", argv[0])
        self.assertNotIn("improvements.compute.worker_stage", argv)
        # "No setup" is the absent key: the control plane's own model refuses an
        # empty setup argv.
        self.assertNotIn("setup", spec)
        self.assertNotIn("environment_secrets", spec["runtime"])

    def test_the_job_declares_only_its_own_artifact(self):
        declared = jobspec.declared_outputs(
            self.plan, jobspec.arm_by_name(self.plan, "smoke"), 0)
        self.assertEqual(["outputs/smoke_s00/smoke_probe.txt"],
                         [item["path"] for item in declared])
        self.assertEqual("IN-0001/smoke_s00/smoke_probe.txt",
                         declared[0]["artifact"])

    def test_the_identity_metadata_never_claims_a_study(self):
        spec = jobspec.build_spec(
            self.plan, stage="smoke", arm="smoke", seed=0,
            commit="a" * 40, scope="IN-0001", envelope_digest="d" * 64)
        metadata = spec["tracking"]["metadata"]
        self.assertNotIn("study_id", metadata)
        self.assertEqual("IN-0001", metadata["scope_id"])
        self.assertEqual(scopes.INFRASTRUCTURE_VALIDATION, metadata["scope_kind"])


class InfrastructureValidationRunTests(ComputeTestCase):
    """A non-Study scope still owns, isolates and cleans up its own compute."""

    def setUp(self):
        super(InfrastructureValidationRunTests, self).setUp()
        self.make_repo()
        self.allow_remote_commit()
        self.plan = jobspec.load_plan(self.write_plan(smoke_plan(), directory="IN-0001"))
        self.write_envelope("IN-0001", envelope=in0001_envelope())
        self.commit()
        self.commit_sha = jobspec.git_state()["head"]
        self.view = envelope.load("IN-0001", os.environ["WAVCSE_REPO_ROOT"])
        self.infra = ProviderInfra()
        self.name = jobspec.job_name("IN-0001", "smoke", "smoke", 0)

    def submitted(self):
        created, _ = worker.ensure_worker(self.plan, self.view, infra=self.infra)
        record = run_study.load_record("IN-0001")
        self.infra.submit_payload = self.job_payload(
            state="RUNNING", worker_id=created["id"])
        run_study.submit_pending("IN-0001", self.plan, "smoke", infra=self.infra,
                                 view=self.view, record=record)
        return created, record

    def job_payload(self, *, state, outputs=None, worker_id="wavcse-colab-fixture"):
        return {
            "job_id": "job-" + self.name,
            "name": self.name,
            "state": state,
            "exit_code": 0 if state == "SUCCEEDED" else None,
            "worker_id": worker_id,
            "outputs": outputs or [],
            "spec": {
                "source": {"commit": self.commit_sha},
                "tracking": {"metadata": {
                    "scope": "IN-0001", "stage": "smoke", "arm": "smoke", "seed": "0",
                }},
            },
        }

    def persisted_probe(self):
        raw = json.dumps({
            "schema_version": 1, "protocol": "in0001-smoke-v1", "seed": 0,
            "job_id": "job-" + self.name, "commit": self.commit_sha,
        }, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        artifact = "IN-0001/smoke_s00/smoke_probe.txt"
        self.infra.objects[artifact] = raw
        return [{
            "artifact": artifact,
            "path": "outputs/smoke_s00/smoke_probe.txt",
            "persisted": True,
            "required": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "verified_size_bytes": len(raw),
        }]

    def test_the_job_is_submitted_on_the_colab_provider(self):
        _, record = self.submitted()
        entry = list(record["entries"].values())[0]
        self.assertEqual(1, entry["attempts"])
        self.assertEqual("IN-0001", entry["scope"])
        self.assertEqual(scopes.INFRASTRUCTURE_VALIDATION, entry["scope_kind"])
        self.assertNotIn("study", entry)
        self.assertIn("IN-0001", entry["job_key"])
        self.assertEqual(providers.COLAB, self.infra.last_job_submit_provider)

    def test_the_record_is_isolated_by_execution_scope(self):
        self.submitted()
        self.assertEqual({}, run_study.load_record("DG-0008")["entries"])
        self.assertEqual({}, run_study.load_record("TR-0007")["entries"])

    def test_the_lease_and_worker_are_isolated_by_execution_scope(self):
        created, _ = self.submitted()
        self.assertEqual([], ledger.leases_for("DG-0008"))
        self.assertEqual([], ledger.leases_for("TR-0007"))
        self.assertEqual(1, len(ledger.leases_for("IN-0001")))
        self.assertIsNone(run_study.available_worker_id(
            self.infra, "DG-0008", provider=providers.COLAB))
        self.assertEqual(created["id"], run_study.available_worker_id(
            self.infra, "IN-0001", provider=providers.COLAB))

    def test_a_persisted_artifact_is_collected_after_a_read_back(self):
        _, record = self.submitted()
        outputs = self.persisted_probe()
        self.infra.jobs = [self.job_payload(state="SUCCEEDED", outputs=outputs)]
        run_study.reconcile("IN-0001", self.plan, "smoke", infra=self.infra,
                            record=record)
        run_study.collect("IN-0001", self.plan, "smoke", record=record,
                          reader=self.infra)
        entry = list(record["entries"].values())[0]
        self.assertEqual(run_study.COLLECTED, entry["state"])
        self.assertEqual(scopes.INFRASTRUCTURE_VALIDATION,
                         entry["evidence"]["scope_kind"])
        self.assertNotIn("metrics", entry["evidence"])

    def test_an_unreadable_artifact_is_not_collected(self):
        _, record = self.submitted()
        outputs = self.persisted_probe()
        self.infra.objects.pop("IN-0001/smoke_s00/smoke_probe.txt")
        self.infra.jobs = [self.job_payload(state="SUCCEEDED", outputs=outputs)]
        run_study.reconcile("IN-0001", self.plan, "smoke", infra=self.infra,
                            record=record)
        result = run_study.collect("IN-0001", self.plan, "smoke", record=record,
                                   reader=self.infra)
        self.assertEqual([], result["collected"])
        self.assertEqual(1, len(result["unverified"]))
        self.assertEqual(run_study.FAILED,
                         list(record["entries"].values())[0]["state"])

    def test_cleanup_destroys_only_the_leased_colab_identity(self):
        created, _ = self.submitted()
        stranger = dict(created, id="wavcse-unowned", name="wavcse-unowned")
        leases = ledger.leases_for("IN-0001")
        self.assertTrue(ledger.worker_matches_lease(created, leases[0]))
        plan = sweep.plan_actions("IN-0001", workers=[created, stranger],
                                  leases=leases, jobs_by_worker={}, now=None)
        self.assertEqual(1, len(plan["unowned_workers"]))
        result = worker.stop_worker(self.infra, created["id"], reason="test",
                                    worker=created)
        self.assertEqual(0, result.returncode)
        self.assertEqual([created["id"]], self.infra.destroy_calls)

    def test_another_scope_never_owns_this_leases_worker(self):
        created, _ = self.submitted()
        # Exactly how the sweep is called in production: it reads this scope's own
        # leases, so a worker leased by IN-0001 is not attributable to DG-0008.
        self.assertEqual([], ledger.leases_for("DG-0008"))
        plan = sweep.plan_actions("DG-0008", workers=[created],
                                  leases=ledger.leases_for("DG-0008"),
                                  jobs_by_worker={}, now=None)
        self.assertEqual([], plan["employees"])
        self.assertEqual([created["id"]],
                         [entry["worker_id"] for entry in plan["unowned_workers"]])


class ProviderCostGuardTests(unittest.TestCase):
    """Provider and native cost-unit enforcement is untouched by kinds."""

    def test_a_colab_plan_cannot_spend_a_runpod_grant(self):
        view = view_for(legacy_runpod_envelope())
        with self.assertRaises(CostError):
            worker.ensure_worker(smoke_plan(), view, infra=FakeInfra())

    def test_a_usd_request_on_a_colab_grant_is_refused(self):
        decision = envelope.check(
            view_for(in0001_envelope()), envelope.ACTION_CREATE_WORKER, empty_facts(),
            requested={"provider": providers.COLAB, "hourly_usd": "0.5"},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual("COST", decision.klass)

    def test_a_colab_grant_cannot_carry_runpod_budget_fields(self):
        document = in0001_envelope()
        document["budget"] = legacy_runpod_envelope()["budget"]
        with self.assertRaises(ConfigurationError):
            envelope.validate(document)

    def test_a_runpod_grant_cannot_carry_compute_unit_fields(self):
        document = legacy_runpod_envelope()
        document["budget"] = in0001_envelope()["budget"]
        with self.assertRaises(ConfigurationError):
            envelope.validate(document)

    def test_colab_resources_must_stay_empty(self):
        document = in0001_envelope()
        document["resources"] = {"new_persistent_resources": False}
        with self.assertRaises(ConfigurationError):
            envelope.validate(document)


class ControlPlaneBindingTests(ComputeTestCase):
    """A recorded mutation fails closed unless the control plane is explicit."""

    def setUp(self):
        super(ControlPlaneBindingTests, self).setUp()
        self.make_repo()
        self.checkout, self.cli, self.log_path = self.fake_control_plane()
        self.plan_path = self.write_plan(smoke_plan(), directory="IN-0001")
        self.write_envelope("IN-0001", envelope=in0001_envelope())
        self.commit()

    def location(self, source):
        return type("Location", (), {
            "checkout": self.checkout, "cli": self.cli, "source": source,
        })()

    def calls(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def test_only_an_explicit_binding_counts(self):
        for source in ("environment", "environment-cli"):
            resolve_module.require_explicit(self.location(source),
                                            operation="worker allocation")
        for source in ("monorepo", "path", "sibling"):
            with self.subTest(source=source):
                with self.assertRaises(ConfigurationError):
                    resolve_module.require_explicit(self.location(source),
                                                    operation="worker allocation")

    def test_worker_ensure_fails_closed_on_a_discovered_control_plane(self):
        with mock.patch.object(resolve_module, "resolve",
                               return_value=self.location("monorepo")):
            code = compute_main.main(["worker-ensure", "--scope", "IN-0001",
                                      "--plan", self.plan_path, "--json"])
        self.assertEqual(compute_main.EXIT_ERROR, code)
        self.assertEqual([], self.calls())

    def test_advance_fails_closed_on_a_discovered_control_plane(self):
        with mock.patch.object(resolve_module, "resolve",
                               return_value=self.location("path")):
            code = compute_main.main(["advance", "--scope", "IN-0001",
                                      "--plan", self.plan_path, "--stage", "smoke",
                                      "--json"])
        self.assertEqual(compute_main.EXIT_ERROR, code)
        self.assertEqual([], self.calls())

    def test_cleanup_is_never_blocked_by_the_binding_policy(self):
        with mock.patch.object(resolve_module, "resolve",
                               return_value=self.location("monorepo")):
            code = compute_main.main(["stop", "--scope", "IN-0001", "--json"])
        self.assertEqual(compute_main.EXIT_OK, code)


class Dg0008UnchangedTests(unittest.TestCase):
    """DG-0008's approved topology and committed records are untouched."""

    def test_the_approved_topology_is_still_exactly_201_jobs(self):
        plan = dg0008_plan()
        jobspec.validate_plan(plan)
        rows = [(stage, arm, seed)
                for stage in plan["stages"]
                for arm, seed in run_study.plan_jobs(plan, stage)]
        self.assertEqual(201, len(rows))
        self.assertEqual(201, len(set(rows)))
        self.assertEqual(200, sum(stage == "stage1_screen" for stage, _, _ in rows))
        self.assertEqual(1, sum(stage == "determinism_repeat" for stage, _, _ in rows))

    def test_dg0008_is_still_unregistered_for_compute(self):
        rows = {row["study_id"]: row for row in studies_rows()}
        self.assertIn("DG-0008", rows)
        record = rows["DG-0008"]
        self.assertEqual("blocked", record["status"])
        self.assertIn("no authorizations/DG-0008.yaml exists", record["authorization"])

    def test_dg0008_still_has_no_compute_plan(self):
        directory = os.path.join(REPO_ROOT, "improvements", "taskrelation",
                                 "research", "studies", "DG-0008", "compute")
        self.assertEqual(["inputs.json"], sorted(os.listdir(directory)))

    def test_dg0008_inputs_are_still_exactly_the_fifteen_pinned_objects(self):
        path = os.path.join(REPO_ROOT, "improvements", "taskrelation", "research",
                            "studies", "DG-0008", "compute", "inputs.json")
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(15, len(document["requirements"]))
        for requirement in document["requirements"]:
            self.assertRegex(requirement["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(requirement["size_bytes"], 0)

    def test_no_dg0008_compute_authorization_exists(self):
        directory = os.path.join(REPO_ROOT, "improvements", "taskrelation",
                                 "research", "authorizations")
        self.assertNotIn("DG-0008.yaml", sorted(os.listdir(directory)))


if __name__ == "__main__":
    unittest.main()
