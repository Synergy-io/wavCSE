"""Provider-neutral wavCSE ↔ wavcse-infra integration contract.

Canonical infrastructure is `wavcse-infra` commit
`2d7640c7c6454b662ab92c6744beff946bc111fa`. These tests pin only its stable
provider, cost-unit, execution-transport and CLI contract; provider implementation
stays in that repository.
"""

import copy
import os
import unittest

from improvements.compute import envelope, infra_cli, jobspec, ledger, providers
from improvements.compute import run_study, sweep, worker
from improvements.compute.errors import ConfigurationError, ReconcilableError
from improvements.compute.tests.fakes import (
    ComputeTestCase,
    FakeInfra,
    InfraResult,
    sample_plan,
    worker_record,
)
from improvements.compute.tests.test_stage_arms import dg0008_plan


def colab_plan(**overrides):
    plan = sample_plan()
    plan["provider"] = providers.COLAB
    plan["worker"] = {"gpu_type": "T4", "gpu_count": 1}
    plan.update(overrides)
    return plan


def legacy_runpod_envelope():
    return {
        "schema_version": 1,
        "scope": "TR-0007",
        "granted_by": "test",
        "granted_at": "2026-10-04T00:00:00+00:00",
        "expires_at": "2999-01-01T00:00:00+00:00",
        "budget": {
            "max_gpu_hourly_usd": 0.5,
            "max_total_gpu_usd": 8,
            "max_wall_clock_hours": 12,
        },
        "concurrency": {
            "max_simultaneous_workers": 1,
            "replacement_workers_allowed": True,
        },
        "resources": {
            "existing_network_volume_allowed": True,
            "new_persistent_resources": False,
            "container_disk_gb_max": 100,
        },
        "stop_policy": {
            "destroy_on_completion": True,
            "retain_for_reuse_hours": 0,
        },
    }


def explicit_runpod_envelope():
    document = legacy_runpod_envelope()
    document["provider"] = providers.RUNPOD
    document["budget"]["cost_unit"] = providers.USD_PER_HOUR
    return document


def colab_envelope(allow_free_tier=True):
    document = legacy_runpod_envelope()
    document["provider"] = providers.COLAB
    document["budget"] = {
        "cost_unit": providers.COMPUTE_UNITS,
        "max_wall_clock_hours": 12,
        "allow_free_tier": allow_free_tier,
        "max_incremental_rate_cu_per_hour": 3,
        "max_job_cu": 10,
    }
    document["resources"] = {}
    return document


def view_for(document):
    return type(
        "View",
        (),
        {
            "envelope": document,
            "scope": document["scope"],
            "digest": "digest",
            "modified_in_tree": False,
            "uncommitted": False,
        },
    )()


def empty_facts():
    return {
        "live_workers": [],
        "estimated_spend_usd": "0",
        "estimated_wall_clock_hours": "0",
        "estimated_hourly_exposure_usd": "0",
        "accounting_bounded": True,
        "unknowns": [],
    }


class ProviderInfra(FakeInfra):
    """Provider-aware fake added by this module; the baseline fake is untouched."""

    def worker_list(self, provider=None):
        self.calls.append("worker_list")
        self.last_worker_list_provider = provider
        rows = list(self.workers)
        if provider is not None:
            rows = [row for row in rows
                    if str(row.get("provider") or providers.RUNPOD) == provider]
        return rows

    def worker_create(self, *, provider=providers.RUNPOD, name=None, gpu=None,
                      cloud=None, **kwargs):
        self.calls.append(("worker_create", name, gpu, cloud))
        self.last_create_kwargs = dict(
            kwargs, provider=provider, name=name, gpu=gpu, cloud=cloud)
        if self.create_returncode == 0:
            if provider == providers.COLAB:
                self.workers.append({
                    "provider": providers.COLAB,
                    "execution_transport": providers.COLAB_EXEC,
                    "id": "wavcse-colab-fixture",
                    "name": "wavcse-colab-fixture",
                    "state": "RUNNING",
                    "gpu_type": gpu or "T4",
                    "gpu_count": 1,
                    "hourly_cost": None,
                    "created_at": None,
                })
            elif self.create_worker_record is not None:
                record = dict(self.create_worker_record)
                record.setdefault("provider", providers.RUNPOD)
                record.setdefault("execution_transport", providers.SSH)
                record["name"] = "wavcse-{}-fixture".format(
                    str(name or "worker").lower())
                self.workers.append(record)
        return InfraResult(("worker", "create"), self.create_returncode,
                           stderr=self.create_stderr)

    def worker_health(self, worker_id, timeout=None, json_output=True):
        self.calls.append(("worker_health", worker_id))
        self.last_health_json_output = json_output
        return InfraResult(returncode=self.health_returncode)

    def job_submit(self, spec_path, worker_id=None, provider=None):
        self.calls.append(("job_submit", str(spec_path), worker_id))
        self.last_job_submit_provider = provider
        return InfraResult(("job", "submit"), self.submit_returncode,
                           payload=self.submit_payload)


class PlanProviderTests(unittest.TestCase):
    def test_legacy_runpod_plan_remains_valid(self):
        plan = sample_plan()
        self.assertEqual(providers.RUNPOD, jobspec.plan_provider(plan))
        jobspec.validate_plan(plan)

    def test_missing_provider_is_deterministically_runpod(self):
        legacy = sample_plan()
        explicit = sample_plan(provider=providers.RUNPOD)
        self.assertEqual(jobspec.plan_provider(legacy), jobspec.plan_provider(explicit))
        self.assertEqual(run_study.plan_jobs(legacy, "screen"),
                         run_study.plan_jobs(explicit, "screen"))

    def test_explicit_runpod_plan_works(self):
        jobspec.validate_plan(sample_plan(provider=providers.RUNPOD))

    def test_explicit_colab_plan_works(self):
        jobspec.validate_plan(colab_plan())

    def test_unknown_provider_fails_closed(self):
        with self.assertRaises(ConfigurationError):
            jobspec.validate_plan(sample_plan(provider="unknown"))

    def test_colab_requires_no_runpod_fields(self):
        plan = colab_plan()
        self.assertNotIn("cloud", plan["worker"])
        self.assertNotIn("image", plan["worker"])
        self.assertNotIn("network_volume", plan["worker"])
        jobspec.validate_plan(plan)

    def test_runpod_keeps_its_required_fields(self):
        for field in ("gpu_type", "cloud", "image"):
            plan = sample_plan(provider=providers.RUNPOD)
            del plan["worker"][field]
            with self.subTest(field=field):
                with self.assertRaises(ConfigurationError):
                    jobspec.validate_plan(plan)

    def test_dg0008_topology_remains_exactly_201(self):
        runpod = dg0008_plan()
        colab = copy.deepcopy(runpod)
        colab["provider"] = providers.COLAB
        colab["worker"] = {"gpu_type": "T4", "gpu_count": 1}
        jobspec.validate_plan(colab)
        rows = [
            (stage, arm, seed)
            for stage in colab["stages"]
            for arm, seed in run_study.plan_jobs(colab, stage)
        ]
        self.assertEqual(201, len(rows))
        self.assertEqual(201, len(set(rows)))
        self.assertEqual(200, sum(stage == "stage1_screen" for stage, _, _ in rows))
        self.assertEqual(1, sum(stage == "determinism_repeat" for stage, _, _ in rows))

    def test_provider_does_not_change_job_identity_or_scientific_argv(self):
        runpod = dg0008_plan()
        colab = copy.deepcopy(runpod)
        colab["provider"] = providers.COLAB
        colab["worker"] = {"gpu_type": "T4", "gpu_count": 1}
        arm_name = "ks_er_pair_f0"
        arm_runpod = jobspec.arm_by_name(runpod, arm_name)
        arm_colab = jobspec.arm_by_name(colab, arm_name)
        self.assertEqual(jobspec.expand_argv(runpod, arm_runpod, 0),
                         jobspec.expand_argv(colab, arm_colab, 0))
        self.assertEqual(
            jobspec.job_key("DG-0008", "DG-0008", "stage1_screen", arm_name, 0,
                            "a" * 40),
            jobspec.job_key("DG-0008", "DG-0008", "stage1_screen", arm_name, 0,
                            "a" * 40),
        )


class AdapterContractTests(unittest.TestCase):
    def client(self):
        location = type("Location", (), {"checkout": None, "cli": "/bin/true"})()
        return infra_cli.InfraCli(location)

    def test_colab_create_sends_no_runpod_options(self):
        client = self.client()
        captured = {}
        client.run = lambda *args, **kwargs: (
            captured.update({"args": args, "kwargs": kwargs})
            or infra_cli.InfraResult(args, 0, "", "")
        )
        client.worker_create(provider=providers.COLAB, gpu="T4", name=None, cloud=None,
                             start_ssh=False, require_direct_ssh=False, max_price=None)
        self.assertEqual(
            ("worker", "create", "--provider", "colab", "--gpu", "T4", "--yes"),
            captured["args"],
        )
        for forbidden in ("--name", "--cloud", "--image", "--network-volume-id",
                          "--max-price", "--start-ssh", "--require-direct-ssh"):
            self.assertNotIn(forbidden, captured["args"])

    def test_runpod_create_keeps_the_legacy_contract(self):
        client = self.client()
        captured = {}
        client.run = lambda *args, **kwargs: (
            captured.update({"args": args})
            or infra_cli.InfraResult(args, 0, "", "")
        )
        client.worker_create(
            name="wavcse-tr-0007-probe", gpu="A5000", cloud="COMMUNITY",
            image="runpod/pytorch:example", container_disk_gb=100,
        )
        self.assertNotIn("--provider", captured["args"])
        self.assertIn("--start-ssh", captured["args"])
        self.assertIn("--require-direct-ssh", captured["args"])

    def test_colab_job_submission_selects_the_provider(self):
        client = self.client()
        captured = {}
        client.run = lambda *args, **kwargs: (
            captured.update({"args": args})
            or infra_cli.InfraResult(args, 0, "{}", "", payload={})
        )
        client.job_submit("spec.json", "wavcse-colab-fixture", provider=providers.COLAB)
        self.assertEqual(
            ("job", "submit", "spec.json", "--worker", "wavcse-colab-fixture",
             "--provider", "colab"),
            captured["args"],
        )

    def test_runpod_job_submission_keeps_the_legacy_argv(self):
        client = self.client()
        captured = {}
        client.run = lambda *args, **kwargs: (
            captured.update({"args": args})
            or infra_cli.InfraResult(args, 0, "{}", "", payload={})
        )
        client.job_submit("spec.json", "worker-1", provider=providers.RUNPOD)
        self.assertEqual(("job", "submit", "spec.json", "--worker", "worker-1"),
                         captured["args"])


class AuthorizationUnitTests(unittest.TestCase):
    def test_free_tier_is_representable_without_fake_usd(self):
        document = colab_envelope(allow_free_tier=True)
        envelope.validate(document)
        decision = envelope.check(
            view_for(document), envelope.ACTION_CREATE_WORKER, empty_facts(),
            requested={"provider": providers.COLAB, "projected_hours": "1"},
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(providers.COMPUTE_UNITS, decision.details["cost_unit"])
        self.assertTrue(decision.details["allow_free_tier"])
        self.assertNotIn("remaining_usd", decision.details)

    def test_paid_cu_is_representable_in_compute_units(self):
        document = colab_envelope(allow_free_tier=False)
        envelope.validate(document)
        decision = envelope.check(
            view_for(document), envelope.ACTION_SUBMIT_JOB, empty_facts(),
            requested={"provider": providers.COLAB},
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.details["allow_free_tier"])
        self.assertEqual("3", decision.details["max_incremental_rate_cu_per_hour"])
        self.assertEqual("10", decision.details["max_job_cu"])

    def test_legacy_and_explicit_runpod_usd_authorizations_remain_valid(self):
        for document in (legacy_runpod_envelope(), explicit_runpod_envelope()):
            envelope.validate(document)
            decision = envelope.check(
                view_for(document), envelope.ACTION_CREATE_WORKER, empty_facts(),
                requested={"provider": providers.RUNPOD, "hourly_usd": "0.5",
                           "projected_hours": "1", "container_disk_gb": 60},
            )
            self.assertTrue(decision.allowed)
            self.assertEqual(providers.USD_PER_HOUR, decision.details["cost_unit"])
            self.assertIn("remaining_usd", decision.details)

    def test_provider_specific_cost_units_cannot_be_confused(self):
        wrong_colab = colab_envelope()
        wrong_colab["budget"] = explicit_runpod_envelope()["budget"]
        wrong_runpod = explicit_runpod_envelope()
        wrong_runpod["budget"] = colab_envelope()["budget"]
        for document in (wrong_colab, wrong_runpod):
            with self.subTest(provider=document["provider"]):
                with self.assertRaises(ConfigurationError):
                    envelope.validate(document)


class AcquisitionTests(ComputeTestCase):
    def setUp(self):
        super(AcquisitionTests, self).setUp()
        self.make_repo()
        self.allow_remote_commit()

    def colab_view(self):
        document = colab_envelope()
        self.write_envelope("TR-0007", envelope=document)
        return envelope.load("TR-0007", os.environ["WAVCSE_REPO_ROOT"])

    def test_colab_acquisition_dispatches_without_ssh_or_start(self):
        infra = ProviderInfra()
        created, actions = worker.ensure_worker(
            colab_plan(), self.colab_view(), infra=infra)
        self.assertEqual(providers.COLAB, created["provider"])
        self.assertEqual(providers.COLAB, infra.last_create_kwargs["provider"])
        self.assertIsNone(infra.last_create_kwargs["name"])
        self.assertNotIn("worker_wait_ssh", infra.verbs())
        self.assertNotIn("worker_start", infra.verbs())
        self.assertEqual(False, infra.last_health_json_output)
        self.assertIn("Colab readiness satisfied without SSH", actions)

    def test_runpod_retains_ssh_preparation(self):
        infra = FakeInfra()
        setattr(infra, "create_worker_record", worker_record())
        self.write_envelope("TR-0007")
        view = envelope.load("TR-0007", os.environ["WAVCSE_REPO_ROOT"])
        worker.ensure_worker(sample_plan(), view, infra=infra)
        self.assertIn("worker_wait_ssh", infra.verbs())
        self.assertIn("worker_bootstrap", infra.verbs())
        self.assertIn("worker_health", infra.verbs())

    def test_colab_transport_is_not_inferred_from_generic_readiness(self):
        infra = ProviderInfra(workers=[{
            "provider": providers.COLAB,
            "execution_transport": providers.SSH,
            "id": "bad-colab",
            "name": "bad-colab",
            "state": "RUNNING",
        }])
        with self.assertRaises(ReconcilableError):
            worker._prepare_colab(infra, "bad-colab", infra.workers[0])
        self.assertNotIn("worker_wait_ssh", infra.verbs())
        self.assertNotIn("worker_start", infra.verbs())

    def test_provider_identity_is_attributed_by_exact_lease_not_scope_name(self):
        infra = ProviderInfra()
        created, _ = worker.ensure_worker(colab_plan(), self.colab_view(), infra=infra)
        self.assertNotIn("TR-0007", created["name"])
        lease = ledger.leases_for("TR-0007")[0]
        self.assertEqual(providers.COLAB, ledger.lease_provider(lease))
        self.assertTrue(ledger.worker_matches_lease(created, lease))
        self.assertEqual(
            created["id"],
            run_study.available_worker_id(
                infra, "TR-0007", provider=providers.COLAB),
        )

    def test_cleanup_releases_only_the_leased_colab_identity(self):
        infra = ProviderInfra()
        created, _ = worker.ensure_worker(colab_plan(), self.colab_view(), infra=infra)
        stranger = dict(created, id="wavcse-unowned", name="wavcse-unowned")
        leases = ledger.leases_for("TR-0007")
        plan = sweep.plan_actions(
            "TR-0007", workers=[created, stranger], leases=leases,
            jobs_by_worker={}, now=None,
        )
        self.assertEqual(1, len(plan["unowned_workers"]))
        result = worker.stop_worker(
            infra, created["id"], reason="test", worker=created)
        self.assertEqual(0, result.returncode)
        self.assertEqual([created["id"]], infra.destroy_calls)
        self.assertEqual([], infra.stop_calls)


if __name__ == "__main__":
    unittest.main()
