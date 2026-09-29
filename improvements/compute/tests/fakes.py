"""Offline fixtures for the compute-backend tests.

No test in this package may reach a provider, the network, or the real
configuration: the control plane is replaced by a recording fake, and runtime
state lives in a temporary directory. The fake models the control plane's own
contract closely enough to exercise the failure paths that matter — ambiguous
creates, lost submissions, unverified outputs — without any of them being real.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from improvements.compute import infra_cli  # noqa: E402

SKILL_PATH = os.path.join(".agents", "skills", "wavcse-infra-operator", "SKILL.md")


class InfraResult(object):
    """The subset of ``InfraResult`` the fake needs."""

    def __init__(self, argv=(), returncode=0, stdout="", stderr="", payload=None):
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.payload = payload


class FakeInfra(object):
    """A scriptable stand-in for the control-plane CLI wrapper.

    Every call is recorded, so a test can assert what was *not* done — which is
    how duplicate creates, duplicate submissions and destructive sweeps are
    caught.
    """

    def __init__(self, *, workers=None, volumes=None, jobs=None):
        self.workers = list(workers or [])
        self.volumes = list(volumes or [])
        self.jobs = list(jobs or [])
        self.calls = []
        self.logs = {}
        self.create_returncode = 0
        self.create_stderr = ""
        # Scripted outcomes.
        self.create_worker_record = None      # appended to worker_list on success
        self.submit_payload = None            # payload returned by job_submit
        self.submit_returncode = 0
        self.bootstrap_returncode = 0
        self.wait_ssh_returncode = 0
        self.health_returncode = 0
        self.stop_calls = []
        self.destroy_calls = []

    # read-only
    def worker_list(self):
        self.calls.append("worker_list")
        return list(self.workers)

    def volume_list(self):
        self.calls.append("volume_list")
        return list(self.volumes)

    def job_list(self, state=None, worker_id=None):
        self.calls.append("job_list")
        result = list(self.jobs)
        if state:
            result = [job for job in result if str(job.get("state")) == state]
        if worker_id:
            result = [job for job in result if job.get("worker_id") == worker_id]
        return result

    def job_status(self, job_id):
        self.calls.append(("job_status", job_id))
        for job in self.jobs:
            if job.get("job_id") == job_id:
                return dict(job)
        return {"job_id": job_id, "state": "PREPARING"}

    def job_logs(self, job_id, tail_bytes=200000):
        self.calls.append(("job_logs", job_id))
        return self.logs.get(job_id, "")

    # mutating
    def worker_create(self, *, name, gpu, cloud, **kwargs):
        self.calls.append(("worker_create", name, gpu, cloud))
        self.last_create_kwargs = dict(kwargs, name=name, gpu=gpu, cloud=cloud)
        if self.create_returncode == 0 and self.create_worker_record is not None:
            record = dict(self.create_worker_record)
            record["name"] = "wavcse-{}-fixture".format(name.lower())
            self.workers.append(record)
        return InfraResult(("worker", "create"), self.create_returncode,
                           stderr=self.create_stderr)

    def worker_wait_ssh(self, worker_id, timeout=None):
        self.calls.append(("worker_wait_ssh", worker_id))
        return InfraResult(returncode=self.wait_ssh_returncode)

    def worker_bootstrap(self, worker_id, timeout=None):
        self.calls.append(("worker_bootstrap", worker_id))
        return InfraResult(returncode=self.bootstrap_returncode)

    def worker_health(self, worker_id, timeout=None):
        self.calls.append(("worker_health", worker_id))
        return InfraResult(returncode=self.health_returncode)

    def worker_start(self, worker_id, timeout=None):
        self.calls.append(("worker_start", worker_id))
        return InfraResult()

    def worker_stop(self, worker_id, timeout=None):
        self.calls.append(("worker_stop", worker_id))
        self.stop_calls.append(worker_id)
        for worker in self.workers:
            if worker.get("id") == worker_id:
                worker["state"] = "STOPPED"
        return InfraResult()

    def worker_destroy(self, worker_id, timeout=None):
        self.calls.append(("worker_destroy", worker_id))
        self.destroy_calls.append(worker_id)
        for worker in self.workers:
            if worker.get("id") == worker_id:
                worker["state"] = "DESTROYED"
        return InfraResult()

    def job_submit(self, spec_path, worker_id):
        self.calls.append(("job_submit", str(spec_path), worker_id))
        if self.submit_returncode == 0 and isinstance(self.submit_payload, dict):
            if not any(job.get("job_id") == self.submit_payload.get("job_id")
                       for job in self.jobs):
                self.jobs.append(dict(self.submit_payload))
        return InfraResult(("job", "submit"), self.submit_returncode,
                           payload=self.submit_payload)

    def job_cancel(self, job_id):
        self.calls.append(("job_cancel", job_id))
        return InfraResult(payload={"job_id": job_id})

    def storage_verify(self, artifact, **kwargs):
        self.calls.append(("storage_verify", artifact))
        return InfraResult()

    def storage_download(self, artifact, destination, worker_id, **kwargs):
        self.calls.append(("storage_download", artifact, destination, worker_id))
        return InfraResult()

    def verbs(self):
        return [call[0] if isinstance(call, tuple) else call for call in self.calls]

    def count(self, verb):
        return self.verbs().count(verb)


class ComputeTestCase(unittest.TestCase):
    """Base class: isolated state root, isolated repo root, no real control plane."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="arc-test-")
        self._env = {}
        for name, value in (
            ("WAVCSE_RESEARCH_STATE", os.path.join(self.home, "state")),
            ("WAVCSE_REPO_ROOT", self.home),
        ):
            self._env[name] = os.environ.get(name)
            os.environ[name] = value
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self.home, ignore_errors=True)

    # helpers -------------------------------------------------------------

    def make_repo(self, *, clean=True):
        """A throwaway git repository usable as the scientific checkout."""

        repo = os.path.join(self.home, "repo")
        os.makedirs(repo, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "GIT_AUTHOR_NAME": "arc-test",
            "GIT_AUTHOR_EMAIL": "arc-test@example.invalid",
            "GIT_COMMITTER_NAME": "arc-test",
            "GIT_COMMITTER_EMAIL": "arc-test@example.invalid",
        })

        def git(*args, **kwargs):
            return subprocess.run(["git", "-C", repo] + list(args),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=kwargs.pop("check", False), env=env,
                                  universal_newlines=True)

        git("init", "--quiet", "-b", "main")
        with open(os.path.join(repo, "tracked.txt"), "w", encoding="utf-8") as handle:
            handle.write("tracked\n")
        git("add", "tracked.txt")
        git("commit", "--quiet", "-m", "initial")
        if not clean:
            with open(os.path.join(repo, "tracked.txt"), "a", encoding="utf-8") as handle:
                handle.write("dirty\n")
        os.environ["WAVCSE_REPO_ROOT"] = repo
        return repo

    def fake_control_plane(self, *, responses=None):
        """An executable stand-in for the infra CLI, reachable through the env.

        Returns ``(checkout, cli, log_path)``; every invocation is appended to the
        log so a test can prove what was *not* called. Nothing here touches a
        provider: the script prints fixture JSON and exits.
        """

        root = os.path.join(self.home, "fake-infra")
        skill = os.path.join(root, SKILL_PATH)
        os.makedirs(os.path.dirname(skill), exist_ok=True)
        with open(skill, "w", encoding="utf-8") as handle:
            handle.write("# fake control plane\n")
        cli = os.path.join(root, ".venv", "bin", "infra")
        os.makedirs(os.path.dirname(cli), exist_ok=True)
        log_path = os.path.join(self.home, "fake-infra-calls.log")
        script = (
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "argv = sys.argv[1:]\n"
            "with open({log!r}, 'a', encoding='utf-8') as handle:\n"
            "    handle.write(' '.join(argv) + '\\n')\n"
            "responses = {responses!r}\n"
            "key = ' '.join(t for t in argv if not t.startswith('--'))\n"
            "payload = responses.get(key)\n"
            "if payload is None:\n"
            "    payload = [] if argv[-1] == '--json' else {{}}\n"
            "print(json.dumps(payload))\n"
            "sys.exit(0)\n"
        ).format(log=str(log_path),
                 responses={key: value for key, value in (responses or {}).items()})
        with open(cli, "w", encoding="utf-8") as handle:
            handle.write(script)
        os.chmod(cli, 0o755)
        os.environ["WAVCSE_INFRA_CHECKOUT"] = root
        os.environ.pop("WAVCSE_INFRA_CLI", None)
        return root, cli, log_path

    def commit(self, message="fixture", repo=None):
        """Commit everything in the fixture checkout, so it is commit-clean."""

        repo = repo or os.environ["WAVCSE_REPO_ROOT"]
        subprocess.run(["git", "-C", repo, "add", "-A"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        subprocess.run(
            ["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
             "commit", "--quiet", "--allow-empty", "-m", message],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        return repo

    def write_envelope(self, scope, *, envelope=None, repo=None, commit=True,
                       overrides=None):
        """Write a valid authorization envelope into the fixture checkout."""

        import yaml

        repo = repo or os.environ["WAVCSE_REPO_ROOT"]
        document = {
            "schema_version": 1,
            "scope": scope,
            "granted_by": "test",
            "granted_at": "2026-09-29T00:00:00+00:00",
            "expires_at": "2999-01-01T00:00:00+00:00",
            "budget": {
                "max_gpu_hourly_usd": 0.5,
                "max_total_gpu_usd": 8.0,
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
        if envelope:
            document.update(envelope)
        for key, value in (overrides or {}).items():
            document[key] = value
        directory = os.path.join(repo, "improvements", "taskrelation", "research",
                                 "authorizations")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "{}.yaml".format(scope))
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=False)
        if commit:
            subprocess.run(["git", "-C", repo, "add", os.path.relpath(path, repo)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            subprocess.run(
                ["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                 "commit", "--quiet", "-m", "grant {}".format(scope)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
        return path

    def write_plan(self, plan, *, repo=None, name="plan.json", directory="compute"):
        import json

        repo = repo or os.environ["WAVCSE_REPO_ROOT"]
        target = os.path.join(repo, "studies", "TR-0007", directory, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        return target

    def write_inputs(self, requirements, *, repo=None, relative="studies/TR-0007/compute/inputs.json"):
        import json

        repo = repo or os.environ["WAVCSE_REPO_ROOT"]
        target = os.path.join(repo, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "requirements": requirements}, handle)
        return target


def sample_plan(**overrides):
    """A minimal valid study compute plan."""

    plan = {
        "schema_version": 1,
        "study": "TR-0007",
        "repository": "https://github.com/example/wavCSE.git",
        "task_type": "ks_si_er",
        "environment_secrets": ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"],
        "timeout_seconds": 3600,
        "device_index": 0,
        "arms": [
            {
                "arm": "mssl",
                "method": "mssl",
                "config": "improvements/taskrelation/04-mssl/configs/mssl.yml",
                "argv": [
                    "python", "-m", "improvements.run_improvements",
                    "--model", "mssl", "--task_type", "{task_type}",
                    "--config", "{config}", "--device_index", "{device_index}",
                    "--seed", "{seed}",
                ],
            }
        ],
        "stages": {"screen": {"seeds": [42]}, "confirm": {"seeds": [0, 1]}},
        "outputs": [
            {"name": "checkpoint_best", "kind": "checkpoint", "tag": "best", "required": True},
            {"name": "eval_metrics_opt", "kind": "results_file", "required": True},
        ],
        "worker": {
            "gpu_type": "NVIDIA GeForce RTX 4090",
            "cloud": "SECURE",
            "gpu_count": 1,
            "image": "runpod/pytorch:example",
            "container_disk_gb": 60,
        },
    }
    plan.update(overrides)
    return plan


def worker_record(worker_id="w-1", name="wavcse-tr-0007-abc123def456",
                  state="RUNNING", hourly="0.44", created="__recent__"):
    """A provider worker record.

    ``created`` defaults to a few minutes ago so that a test exercising the
    envelope's wall-clock budget is not accidentally past it; pass an explicit
    timestamp when the elapsed time is the point of the test.
    """

    if created == "__recent__":
        import datetime

        created = (datetime.datetime.now(datetime.timezone.utc)
                   - datetime.timedelta(minutes=5)).isoformat()
    return {
        "id": worker_id,
        "name": name,
        "state": state,
        "hourly_cost": hourly,
        "created_at": created,
        "gpu_type": "NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
    }
