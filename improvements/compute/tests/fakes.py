"""Offline fixtures for the compute-backend tests.

No test in this package may reach a provider, the network, or the real
configuration: the control plane is replaced by a recording fake, and runtime
state lives in a temporary directory. The fake models the control plane's own
contract closely enough to exercise the failure paths that matter — ambiguous
creates, lost submissions, unverified outputs — without any of them being real.
"""

import hashlib
import json
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

from improvements.compute import infra_cli, jobspec  # noqa: E402
from improvements.compute import state as state_module  # noqa: E402
from improvements.compute.errors import ArtifactIntegrityError  # noqa: E402

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
        self.objects = {}
        self.records = {}
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
        """The persisted job record, as the control plane reports it.

        ``jobs`` is the scripted provider-visible state a test controls; when a test
        removes a job from it (the worker that held it has gone), the record this job
        store still holds is what the control plane answers with, exactly as production
        does when the Pod disappears but the job record survives.
        """

        self.calls.append(("job_status", job_id))
        for job in self.jobs:
            if job.get("job_id") == job_id:
                return dict(job)
        if job_id in self.records:
            return dict(self.records[job_id])
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
            record = dict(self.submit_payload)
            if not any(job.get("job_id") == record.get("job_id") for job in self.jobs):
                self.jobs.append(record)
            if record.get("job_id"):
                self.records.setdefault(record["job_id"], record)
        return InfraResult(("job", "submit"), self.submit_returncode,
                           payload=self.submit_payload)

    def job_cancel(self, job_id):
        self.calls.append(("job_cancel", job_id))
        return InfraResult(payload={"job_id": job_id})

    def storage_verify(self, artifact, **kwargs):
        self.calls.append(("storage_verify", artifact))
        return InfraResult()

    def storage_read(self, artifact, *, expected_sha256=None, max_bytes=None):
        """Serve one registered evidence object, with the real digest contract.

        Unlike the other fakes this one is not a stub: it hashes the bytes it holds and
        fails exactly as the control plane does when they do not match the expected
        digest, so a test cannot accidentally pass by trusting the caller's digest.
        """

        self.calls.append(("storage_read", artifact))
        payload = self.objects.get(artifact)
        if payload is None:
            exc = ArtifactIntegrityError(
                "no stored object at {!r} (the fixture registers only what was staged)"
                .format(artifact)
            )
            raise exc
        if isinstance(payload, bytes):
            raw = payload
        else:
            raw = str(payload).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and digest != str(expected_sha256).lower():
            raise ArtifactIntegrityError(
                "the stored object at {!r} does not contain the expected bytes"
                .format(artifact)
            )
        if max_bytes is not None and len(raw) > int(max_bytes):
            raise ArtifactIntegrityError(
                "the stored object at {!r} exceeds the requested read limit".format(artifact)
            )
        return {"artifact": artifact, "size_bytes": len(raw), "sha256": digest,
                "text": raw.decode("utf-8")}

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
            # The host-level controller guard lives at a fixed path outside every state
            # root, so a test must redirect it or it would contend with the real
            # controller and write into the developer's home.
            (state_module.CONTROLLER_LOCK_ENV,
             os.path.join(self.home, "controller.lock")),
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

    def allow_remote_commit(self, *, mechanism="test-stub"):
        """Stub the paid-creation commit preflight with a proven-available answer.

        The gate itself is exercised for real against local bare repositories in
        ``tests/test_remote_commit.py``; a test about worker lifecycle needs the seam
        closed, not a pushable remote, so it substitutes the proof explicitly rather
        than relying on a default that could hide a missing gate.
        """

        from unittest import mock

        from improvements.compute import remote_commit

        def fake(plan, commit, **kwargs):
            return remote_commit.CommitAvailability(
                str(commit), plan.get("repository"), available=True,
                mechanism=mechanism, detail="stubbed for a lifecycle test",
            )

        patcher = mock.patch.object(remote_commit, "require_plan_commit", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def refuse_remote_commit(self, reason=None):
        """Stub the preflight with the refusal a local-only commit produces."""

        from unittest import mock

        from improvements.compute import remote_commit
        from improvements.compute.errors import RepositoryConflictError

        def fake(plan, commit, **kwargs):
            raise RepositoryConflictError(
                reason or "commit {} is not available on {}".format(
                    commit, plan.get("repository"))
            )

        patcher = mock.patch.object(remote_commit, "require_plan_commit", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

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


def metrics_text(tasks, *, base=0.5, seed=0):
    """The evaluator's metrics vocabulary, one line per task plus the all-task line."""

    lines = ["loss_all={:.6f} | acc_all={:.6f}".format(1.0 - base, base)]
    for index, task in enumerate(tasks):
        lines.append("{} | loss={:.6f} | acc={:.6f} | samples={}".format(
            task, 1.0 - base, base + 0.01 * (index + 1) + 0.001 * seed, 100 + index))
    return "\n".join(lines) + "\n"


def gradient_diagnostics_text(tasks, *, sampled_steps=3):
    records = []
    for step in range(1, sampled_steps + 1):
        records.append({
            "step": step,
            "progress": step / float(sampled_steps),
            "phase": "early",
            "valid_examples": {task: 100 for task in tasks},
            "gradient_norms": {task: 0.5 + 0.1 * step for task in tasks},
            "pairwise_cosines": {
                "{}_{}".format(tasks[a], tasks[b]): -0.25
                for a in range(len(tasks)) for b in range(a + 1, len(tasks))
            },
        })
    return json.dumps({
        "task_array": list(tasks),
        "sample_interval_steps": 1,
        "total_training_steps": sampled_steps,
        "sampled_steps": len(records),
        "skipped_sample_steps_missing_tasks": 0,
        "shared_parameter_names": ["backbone.weight"],
        "shared_parameter_count": 1024,
        "records": records,
        "summary": {},
    }, indent=2, sort_keys=True)


def staged_evidence(plan, *, stage, arm, seed, commit, job_id,
                    task_type=None, metrics=None, diagnostics=None,
                    run_id=None, checkpoint_run_id=None, job_directory=None,
                    overrides=None):
    """Build one well-formed completed job's staged bytes and its output records.

    Returns ``(objects, outputs, manifest)`` where ``objects`` maps an artifact identity
    to the exact bytes staged at it — the same shape the control plane serves back — and
    ``outputs`` is the ``outputs[]`` list a job record would carry for them. Tests mutate
    a copy to reproduce a specific defect; nothing here is a stub of the validator.
    """

    arm_spec = jobspec.arm_by_name(plan, arm)
    declared = jobspec.declared_outputs(plan, arm_spec, seed)
    tasks = [token for token in (task_type or plan["task_type"]).split("_") if token]
    run_id = run_id or "{}_s{:02d}".format(arm, seed)
    checkpoint_run_id = checkpoint_run_id or "{}_s{:02d}".format(arm, seed)
    job_directory = job_directory or "/workspace/wavcse-jobs/job-0000000000000000"

    kinds = {}
    for output in plan["outputs"]:
        name = output["name"]
        if output["kind"] == "checkpoint":
            filename = "checkpoint_{}.pth".format(output["tag"])
        elif output["kind"] == "gradient_diagnostics":
            filename = "gradient_diagnostics.json"
        else:
            filename = name + ".txt"
        kinds[filename] = output["kind"]

    metrics = metrics if metrics is not None else metrics_text(tasks)
    diagnostics = diagnostics if diagnostics is not None else \
        gradient_diagnostics_text(tasks)

    outputs = []
    staged = []
    objects = {}
    for declaration in declared:
        filename = declaration["path"].rsplit("/", 1)[-1]
        if filename == "MANIFEST.json":
            continue
        kind = kinds[filename]
        if kind == "checkpoint":
            text = "checkpoint-bytes {} seed {}".format(arm, seed)
            source = "{}/checkpoints/{}/{}".format(job_directory, checkpoint_run_id,
                                                   filename)
        elif kind == "gradient_diagnostics":
            text = diagnostics
            source = "{}/results/{}/{}".format(job_directory, run_id, filename)
        else:
            text = metrics
            source = "{}/results/{}/{}".format(job_directory, run_id, filename)
        raw = text.encode("utf-8")
        objects[declaration["artifact"]] = raw
        outputs.append({
            "path": declaration["path"],
            "artifact": declaration["artifact"],
            "required": declaration["required"],
            "persisted": True,
            "verified_size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        staged.append({
            "name": filename.rsplit(".", 1)[0],
            "kind": kind,
            "required": declaration["required"],
            "target": filename,
            "source": source,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        })

    manifest = {
        "schema_version": 1,
        "study": plan["study"],
        "stage": stage,
        "arm": arm,
        "method": arm_spec.get("method", arm),
        "seed": seed,
        "task_type": plan["task_type"],
        "job_id": job_id,
        "commit": commit,
        "run_id": run_id,
        "checkpoint_run_id": checkpoint_run_id,
        "training_exit_code": 0,
        "staged_at": "2026-09-29T00:00:00+00:00",
        "files": sorted(staged, key=lambda entry: entry["name"]),
    }
    for key, value in (overrides or {}).items():
        manifest[key] = value
    manifest_declaration = [item for item in declared
                            if item["path"].rsplit("/", 1)[-1] == "MANIFEST.json"][0]
    manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    objects[manifest_declaration["artifact"]] = manifest_raw
    outputs.append({
        "path": manifest_declaration["path"],
        "artifact": manifest_declaration["artifact"],
        "required": True,
        "persisted": True,
        "verified_size_bytes": len(manifest_raw),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    })
    return objects, outputs, manifest


def payload_digest(payload):
    """The digest a job record would carry for one staged payload."""

    raw = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resign(outputs, objects):
    """Recompute the recorded digest/size of every output from the objects staged.

    Used after a fixture mutates a payload, so a test always produces bytes whose
    recorded identity is honest and the *content* is the only thing under test.
    """

    for output in outputs:
        payload = objects.get(output["artifact"])
        if payload is None:
            continue
        raw = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        output["sha256"] = payload_digest(raw)
        output["verified_size_bytes"] = len(raw)
    return outputs
