"""Two orchestrator instances must never work the same scope at once."""

import os
import unittest
import multiprocessing

from improvements.compute import state as state_module
from improvements.compute.errors import BusyError
from improvements.compute.tests.fakes import ComputeTestCase


def _hold_lock_until_killed(connection):
    with state_module.scope_lock("TR-0007"):
        connection.send("locked")
        connection.recv()


class ScopeLockTests(ComputeTestCase):
    def test_a_second_process_is_refused_and_crash_releases_lock(self):
        parent, child = multiprocessing.Pipe()
        process = multiprocessing.Process(target=_hold_lock_until_killed, args=(child,))
        process.start()
        try:
            self.assertEqual(parent.recv(), "locked")
            with self.assertRaises(BusyError):
                with state_module.scope_lock("TR-0007"):
                    self.fail("second controller must not enter")
        finally:
            process.terminate()
            process.join(timeout=5)
            parent.close()
            child.close()
        self.assertFalse(process.is_alive())
        with state_module.scope_lock("TR-0007"):
            pass

    def test_a_second_holder_is_refused_rather_than_waiting(self):
        with state_module.scope_lock("TR-0007"):
            with self.assertRaises(BusyError) as caught:
                with state_module.scope_lock("TR-0007"):
                    self.fail("the second holder must never acquire the lock")
        self.assertIn("another process holds the lock", str(caught.exception))

    def test_different_scopes_do_not_contend(self):
        with state_module.scope_lock("TR-0007"):
            with state_module.scope_lock("TR-0008"):
                pass

    def test_the_lock_is_released_for_the_next_holder(self):
        with state_module.scope_lock("TR-0007"):
            pass
        with state_module.scope_lock("TR-0007"):
            pass

    def test_the_lock_is_released_after_an_error(self):
        with self.assertRaises(RuntimeError):
            with state_module.scope_lock("TR-0007"):
                raise RuntimeError("boom")
        with state_module.scope_lock("TR-0007"):
            pass

    def test_mutating_verbs_refuse_while_the_scope_is_locked(self):
        from improvements.compute import __main__ as cli
        from improvements.compute.errors import BusyError as CliBusyError

        self.make_repo()
        self.write_envelope("TR-0007")
        self.commit()
        self.fake_control_plane()
        with state_module.scope_lock("TR-0007"):
            with self.assertRaises(CliBusyError):
                with cli._mutation("TR-0007"):
                    self.fail("must not acquire")

    def test_a_second_state_root_cannot_mutate_the_same_host(self):
        """M01: one controller per host, checked outside any single state root.

        The two processes here use *different* state roots — nothing inside either root
        could see the other — and the mutation is still refused, naming the holder's
        root. That is the single-controller deployment invariant, mechanically enforced
        instead of assumed.
        """

        from improvements.compute import __main__ as cli
        from improvements.compute.errors import BusyError as CliBusyError

        self.make_repo()
        self.write_envelope("TR-0007")
        self.commit()
        self.fake_control_plane()
        with state_module.controller_guard():
            original = os.environ["WAVCSE_RESEARCH_STATE"]
            os.environ["WAVCSE_RESEARCH_STATE"] = os.path.join(self.home, "other-root")
            try:
                self.assertNotEqual(state_module.state_root(), original)
                with self.assertRaises(CliBusyError) as caught:
                    with cli._mutation("TR-0007"):
                        self.fail("must not acquire")
                self.assertIn("state root", str(caught.exception))
                self.assertIn(original, str(caught.exception))
            finally:
                os.environ["WAVCSE_RESEARCH_STATE"] = original

    def test_the_host_guard_is_released_when_its_holder_exits(self):
        """A crash must not leave a permanent block: the kernel drops the lock."""

        with state_module.controller_guard():
            pass
        with state_module.controller_guard():
            pass

    def test_read_only_verbs_still_run_while_the_scope_is_locked(self):
        """A status check must never be blocked by a running cycle."""

        import io
        import json
        from contextlib import redirect_stdout

        from improvements.compute import __main__ as cli
        from improvements.compute import status as status_module

        self.make_repo()
        self.write_envelope("TR-0007")
        self.commit()
        self.fake_control_plane()
        with state_module.scope_lock("TR-0007"):
            picture = status_module.build("TR-0007", infra=None, jobs=[])
            self.assertEqual(picture["scope"], "TR-0007")
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = cli.main(["envelope-check", "--scope", "TR-0007",
                                 "--action", "submit-job", "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            self.assertTrue(json.loads(stream.getvalue())["allowed"])


if __name__ == "__main__":
    unittest.main()
