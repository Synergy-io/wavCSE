"""CLI surface: verbs, JSON output, exit codes, and read-only behaviour."""

import json
import os
import unittest

from improvements.compute import __main__ as cli
from improvements.compute import state as state_module
from improvements.compute.tests.fakes import ComputeTestCase


def capture(argv):
    """Run the CLI with stdout captured, returning (exit code, parsed JSON)."""

    import io
    import contextlib

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = cli.main(argv)
    text = stream.getvalue().strip()
    payload = json.loads(text) if text else None
    return code, payload


class CliTests(ComputeTestCase):
    def test_mlflow_credentials_are_loaded_only_into_runtime_environment(self):
        from improvements.compute import infra_cli

        repo = self.make_repo()
        with open(os.path.join(repo, ".env"), "w", encoding="utf-8") as handle:
            handle.write("MLFLOW_TRACKING_USERNAME=fixture-user\n")
            handle.write("MLFLOW_TRACKING_PASSWORD=fixture-value\n")
            handle.write("RUNPOD_API_KEY=must-not-be-forwarded\n")
        saved = {name: os.environ.pop(name, None) for name in
                 ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")}
        try:
            location = type("Location", (), {"checkout": repo, "cli": "/bin/true"})()
            client = infra_cli.InfraCli(location)
            self.assertEqual(client.environ["MLFLOW_TRACKING_PASSWORD"], "fixture-value")
            self.assertNotEqual(client.environ.get("RUNPOD_API_KEY"), "must-not-be-forwarded")
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value

    def test_resolve_fails_loudly_without_infrastructure(self):
        saved = os.environ.get("WAVCSE_INFRA_CHECKOUT")
        os.environ["WAVCSE_INFRA_CHECKOUT"] = self.home
        try:
            code, payload = capture(["resolve", "--json"])
        finally:
            if saved is None:
                os.environ.pop("WAVCSE_INFRA_CHECKOUT", None)
            else:
                os.environ["WAVCSE_INFRA_CHECKOUT"] = saved
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("wavcse-infra", payload["reason"])

    def test_status_without_authorization_is_read_only_and_informative(self):
        self.make_repo()
        code, payload = capture(["status", "--scope", "TR-0007", "--json"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertFalse(payload["envelope"]["present"])
        self.assertEqual(payload["blocked"]["class"], "AUTHORIZATION")
        self.assertEqual(os.path.exists(state_module.state_root()), False)

    def test_envelope_check_refuses_when_absent(self):
        self.make_repo()
        code, payload = capture([
            "envelope-check", "--scope", "TR-0007", "--action", "submit-job", "--json",
        ])
        self.assertEqual(code, cli.EXIT_REFUSED)
        self.assertEqual(payload["error"], "AuthorizationError")

    def test_envelope_check_allows_inside_the_grant(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        self.fake_control_plane()
        code, payload = capture([
            "envelope-check", "--scope", "TR-0007", "--action", "create-worker",
            "--json",
        ])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue(payload["allowed"])

    def test_envelope_check_fails_closed_without_provider_facts(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        saved = os.environ.get("WAVCSE_INFRA_CHECKOUT")
        os.environ["WAVCSE_INFRA_CHECKOUT"] = self.home
        try:
            code, payload = capture(["envelope-check", "--scope", "TR-0007",
                                     "--action", "create-worker", "--json"])
        finally:
            if saved is None:
                os.environ.pop("WAVCSE_INFRA_CHECKOUT", None)
            else:
                os.environ["WAVCSE_INFRA_CHECKOUT"] = saved
        self.assertEqual(code, cli.EXIT_REFUSED)
        self.assertFalse(payload["allowed"])

    def test_sweep_reports_the_block_without_touching_state_when_infra_is_absent(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        for name in ("WAVCSE_INFRA_CHECKOUT", "WAVCSE_INFRA_CLI"):
            os.environ.pop(name, None)
        saved = os.environ.get("PATH")
        os.environ["PATH"] = self.home
        try:
            code, payload = capture(["sweep", "--scope", "TR-0007", "--json"])
        finally:
            if saved is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("error", payload)
        self.assertFalse(os.path.exists(state_module.state_root()))

    def test_plan_verb_previews_and_does_not_submit(self):
        from improvements.compute.tests.fakes import sample_plan

        self.make_repo()
        self.write_envelope("TR-0007")
        plan_path = self.write_plan(sample_plan())
        self.commit()
        _root, _cli, log_path = self.fake_control_plane()
        code, payload = capture([
            "plan", "--scope", "TR-0007", "--plan", plan_path, "--stage", "screen",
            "--json",
        ])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(len(payload["planned"]), 1)
        with open(log_path, encoding="utf-8") as handle:
            calls = handle.read()
        self.assertIn("worker list", calls)
        self.assertNotIn("job submit", calls)
        self.assertNotIn("worker create", calls)

    def test_advance_dry_run_never_writes_or_resumes_a_job(self):
        from improvements.compute.tests.fakes import sample_plan

        self.make_repo()
        self.write_envelope("TR-0007")
        plan_path = self.write_plan(sample_plan())
        self.commit()
        _root, _cli, log_path = self.fake_control_plane()
        before_exists = os.path.exists(state_module.state_root())
        before = sorted(os.path.relpath(os.path.join(root, name), state_module.state_root())
                        for root, _dirs, files in os.walk(state_module.state_root())
                        for name in files)
        code, payload = capture(["advance", "--scope", "TR-0007", "--plan",
                                 plan_path, "--stage", "screen", "--dry-run", "--json"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(payload["step"], "dry-run")
        after = sorted(os.path.relpath(os.path.join(root, name), state_module.state_root())
                       for root, _dirs, files in os.walk(state_module.state_root())
                       for name in files)
        self.assertEqual(after, before)
        self.assertEqual(os.path.exists(state_module.state_root()), before_exists)
        with open(log_path, encoding="utf-8") as handle:
            calls = handle.read()
        self.assertNotIn("job status", calls)
        self.assertNotIn("job submit", calls)

    def test_plan_verb_is_blocked_without_an_authorization(self):
        from improvements.compute.tests.fakes import sample_plan

        self.make_repo()
        plan_path = self.write_plan(sample_plan())
        self.commit()
        self.fake_control_plane()
        code, payload = capture([
            "plan", "--scope", "TR-0007", "--plan", plan_path, "--stage", "screen",
            "--json",
        ])
        self.assertEqual(code, cli.EXIT_REFUSED)
        self.assertEqual(payload["error"], "AuthorizationError")

    def test_unknown_verb_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            cli.main(["nonsense"])

    def test_json_output_is_a_single_object(self):
        self.make_repo()
        code, payload = capture(["status", "--scope", "TR-0007", "--json"])
        self.assertIsInstance(payload, dict)
        self.assertIn("scope", payload)


if __name__ == "__main__":
    unittest.main()

    def test_worker_health_returns_a_result_the_readiness_ladder_can_walk(self):
        """Regression: worker_health used to return its payload.

        `worker.ensure_worker` walks the readiness ladder by exit code, and it
        does so *after* a paid worker exists: returning a dict made every
        provisioning attempt die with AttributeError on a live worker. The
        contract is the same shape `worker_wait_ssh` and `worker_bootstrap`
        return, with the JSON payload available on `.payload`.
        """

        from improvements.compute import infra_cli

        captured = {}

        def fake_run(*args, **kwargs):
            captured["args"] = args
            captured["json_output"] = kwargs.get("json_output")
            return infra_cli.InfraResult(
                returncode=0,
                stdout='{"ready": true, "readiness_state": "READY"}',
                stderr="",
                payload={"ready": True, "readiness_state": "READY"},
            )

        location = type("Location", (), {"checkout": "/nonexistent", "cli": "/bin/true"})()
        client = infra_cli.InfraCli(location)
        client.run = fake_run
        result = client.worker_health("worker-1")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.payload["readiness_state"], "READY")
        self.assertEqual(captured["args"], ("worker", "health", "worker-1"))
        self.assertTrue(captured["json_output"])
