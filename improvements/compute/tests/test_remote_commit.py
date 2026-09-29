"""Exact-commit availability: a paid Pod must never be created for an impossible job.

The probe is exercised against **real** local bare repositories over ``file://`` — the
smart protocol, real ``git-upload-pack``, real object transfer — so what is tested is the
mechanism, not a mock of it. Nothing here needs a network or a provider.
"""

import os
import subprocess
import unittest

from improvements.compute import ledger, remote_commit
from improvements.compute import run_study, worker as worker_module
from improvements.compute.errors import RepositoryConflictError, TransientInfraError
from improvements.compute.tests.fakes import (
    ComputeTestCase,
    FakeInfra,
    sample_plan,
    worker_record,
)


class RemoteCommitTestCase(ComputeTestCase):
    def setUp(self):
        super(RemoteCommitTestCase, self).setUp()
        self.origin = os.path.join(self.home, "origin.git")
        self.work = os.path.join(self.home, "work")
        subprocess.run(["git", "init", "--bare", "--quiet", "-b", "main", self.origin],
                       check=True)
        subprocess.run(["git", "init", "--quiet", "-b", "main", self.work], check=True)
        self.environment = dict(os.environ)
        self.environment.update({
            "GIT_AUTHOR_NAME": "arc-test",
            "GIT_AUTHOR_EMAIL": "arc-test@example.invalid",
            "GIT_COMMITTER_NAME": "arc-test",
            "GIT_COMMITTER_EMAIL": "arc-test@example.invalid",
        })
        self.git("remote", "add", "origin", self.url())
        self.write("tracked.txt", "one\n")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "first")
        self.first = self.git("rev-parse", "HEAD").stdout.strip()
        # The remote carries `main` from the start, so every "unavailable" case below is
        # a commit the remote genuinely does not have rather than an empty remote.
        self.git("push", "--quiet", "origin", "main")

    def url(self):
        return "file://" + self.origin

    def git(self, *args, cwd=None):
        return subprocess.run(
            ["git", "-C", cwd or self.work] + list(args), env=self.environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            universal_newlines=True,
        )

    def write(self, relative, content):
        path = os.path.join(self.work, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def push(self):
        self.git("push", "--quiet", "origin", "main")
        return self.git("rev-parse", "HEAD").stdout.strip()

    def allow_reachable_wants(self):
        """Model a server that answers a bare object want (GitHub does)."""

        subprocess.run(
            ["git", "-C", self.origin, "config",
             "uploadpack.allowReachableSHA1InWant", "true"],
            check=True,
        )


class LocalCommitRefusalTests(RemoteCommitTestCase):
    def test_a_local_only_commit_is_refused(self):
        """The exact unsafe timeline: committed here, never published."""

        self.write("unpushed.txt", "never published\n")
        local = self.commit("local only")
        self.assertNotEqual(local, self.first)

        availability = remote_commit.verify_available(self.url(), local)

        self.assertFalse(availability.available)
        self.assertIn(availability.mechanism, ("shallow-fetch", "reachable-fetch"))

    def test_a_local_only_commit_refusal_names_publication_and_no_substitute(self):
        self.write("unpushed.txt", "never published\n")
        local = self.commit("local only")

        with self.assertRaises(RepositoryConflictError) as caught:
            remote_commit.require_available(self.url(), local)

        message = str(caught.exception)
        self.assertIn(local, message)
        self.assertIn("git push", message)
        self.assertIn("no substitute commit", message)

    def test_a_branch_existing_is_not_accepted_as_proof(self):
        """The remote has `main`; `main` is not the commit we asked about."""

        self.write("unpushed.txt", "never published\n")
        local = self.commit("unpushed")
        refs = self.git("ls-remote", self.url()).stdout
        self.assertIn("refs/heads/main", refs)
        self.assertNotIn(local, refs)

        self.assertFalse(remote_commit.verify_available(self.url(), local).available)

    def test_a_short_sha_is_refused_without_any_remote_read(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            raise AssertionError("a short sha must not reach git")

        with self.assertRaises(RepositoryConflictError):
            remote_commit.verify_available(self.url(), "abc1234", runner=runner)

        self.assertEqual(calls, [])

    def test_a_remote_lookup_failure_fails_closed(self):
        missing = "file://" + os.path.join(self.home, "does-not-exist.git")

        with self.assertRaises(TransientInfraError):
            remote_commit.verify_available(missing, self.first)

    def test_no_git_command_writes_to_the_remote_or_the_checkout(self):
        """The probe is read-only: the remote refs and the local tree are untouched."""

        before = self.git("ls-remote", self.url()).stdout
        self.write("unpushed.txt", "never published\n")
        local = self.commit("local only")
        self.git("stash", "--include-untracked", "--quiet")

        remote_commit.verify_available(self.url(), local)

        self.assertEqual(before, self.git("ls-remote", self.url()).stdout)
        self.assertEqual(self.git("status", "--porcelain").stdout.strip(), "")


class PublishedCommitTests(RemoteCommitTestCase):
    def test_a_published_commit_is_accepted_by_the_cheap_ref_proof(self):
        published = self.push()

        availability = remote_commit.verify_available(self.url(), published)

        self.assertTrue(availability.available)
        self.assertEqual(availability.mechanism, "ls-remote-ref")

    def test_a_commit_behind_a_moved_branch_tip_stays_available(self):
        """`main` moving on must not invalidate a commit the remote still holds."""

        published = self.push()
        self.write("tracked.txt", "two\n")
        moved = self.commit("second")
        self.push()
        tip = self.git("rev-parse", "origin/main").stdout.strip()
        self.assertEqual(tip, moved)
        self.assertNotEqual(tip, published)

        availability = remote_commit.verify_available(self.url(), published)

        self.assertTrue(availability.available)
        self.assertNotEqual(availability.mechanism, "ls-remote-ref")

    def test_a_server_that_answers_a_bare_want_is_used_directly(self):
        published = self.push()
        self.write("tracked.txt", "two\n")
        self.commit("second")
        self.push()
        self.allow_reachable_wants()

        availability = remote_commit.verify_available(self.url(), published)

        self.assertTrue(availability.available)
        self.assertEqual(availability.mechanism, "shallow-fetch")


class PaidCreationGateTests(ComputeTestCase):
    """The gate's wiring: it must run before anything billable, and before any intent."""

    def setUp(self):
        super(PaidCreationGateTests, self).setUp()
        from improvements.compute import envelope as envelope_module
        from improvements.compute import jobspec

        self.make_repo()
        self.write_envelope("TR-0007")
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()
        self.view = envelope_module.load("TR-0007")
        self.infra = FakeInfra()

    def test_an_unavailable_commit_blocks_worker_creation_completely(self):
        self.refuse_remote_commit()
        self.infra.create_worker_record = worker_record(worker_id="w-9")

        with self.assertRaises(RepositoryConflictError):
            worker_module.ensure_worker(self.plan, self.view, infra=self.infra)

        self.assertEqual(self.infra.count("worker_create"), 0)
        # No create intent either: an intent is the record of a request that may have
        # reached the provider, and this request must never be made.
        self.assertEqual(ledger.pending_creates("TR-0007"), [])

    def test_an_unavailable_commit_blocks_a_restart_of_an_existing_worker(self):
        self.refuse_remote_commit()
        ledger.redeem_create("TR-0007", worker_id="w-1", purpose="test",
                             envelope_digest=self.view.digest,
                             deadline="2999-01-01T00:00:00+00:00")

        with self.assertRaises(RepositoryConflictError):
            worker_module.ensure_worker(
                self.plan, self.view, infra=self.infra,
            )

        self.assertNotIn("worker_start", self.infra.verbs())

    def test_an_unavailable_commit_blocks_a_job_submission(self):
        self.allow_remote_commit()
        self.infra.create_worker_record = worker_record(worker_id="w-9")
        worker_module.ensure_worker(self.plan, self.view, infra=self.infra)
        self.refuse_remote_commit()
        self.infra.submit_payload = {"job_id": "job-1"}

        with self.assertRaises(RepositoryConflictError):
            run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                     view=self.view, record=run_study.load_record("TR-0007"))

        self.assertEqual(self.infra.count("job_submit"), 0)

    def test_an_available_commit_lets_the_worker_be_created(self):
        self.allow_remote_commit()
        self.infra.create_worker_record = worker_record(worker_id="w-9")
        self.infra.stop_returncode = 0

        created, actions = worker_module.ensure_worker(self.plan, self.view,
                                                       infra=self.infra)

        self.assertEqual(created["id"], "w-9")
        self.assertEqual(self.infra.count("worker_create"), 1)
        self.assertTrue(any("available on the worker's remote" in action
                            for action in actions))


if __name__ == "__main__":
    unittest.main()
