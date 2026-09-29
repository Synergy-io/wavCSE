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
    def test_resolve_fails_loudly_without_infrastructure(self):
        for name in ("WAVCSE_INFRA_CHECKOUT", "WAVCSE_INFRA_CLI"):
            os.environ.pop(name, None)
        saved = os.environ.get("PATH")
        os.environ["PATH"] = self.home
        try:
            code, payload = capture(["resolve", "--json"])
        finally:
            if saved is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved
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
        code, payload = capture([
            "envelope-check", "--scope", "TR-0007", "--action", "create-worker",
            "--json",
        ])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue(payload["allowed"])

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
