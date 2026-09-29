"""Thin, typed wrapper around the wavcse-infra CLI.

The backend's only infrastructure dependency is this module: everything else in
``improvements/compute`` speaks in terms of these methods, so no provider, SSH,
storage, or job-state detail leaks into the research side, and no infra Python
module is ever imported.

Two deliberate properties:

* **Nothing is retried here.** Retry bounds and reconciliation live in
  ``failures.py`` / ``worker.py`` / ``run_study.py``, which can distinguish a
  safe read retry from a billable mutation that must never be repeated.
* **Failures are classified, not paraphrased.** :func:`classify_failure` maps the
  CLI's exit status and named error text onto the failure taxonomy, defaulting
  to non-retryable when a condition is not explicitly recognised — the same
  philosophy the control plane applies to transient S3 conditions.
"""

import json
import os
import subprocess

from dotenv import dotenv_values

from improvements.compute.errors import (
    ArtifactIntegrityError,
    CapacityError,
    ComputeError,
    ConfigurationError,
    CostError,
    ReconcilableError,
    SecurityError,
    TransientInfraError,
    UsageError,
)
from improvements.compute import resolve as resolve_module

DEFAULT_TIMEOUT_SECONDS = 600.0
_RESEARCH_SECRET_NAMES = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")

# Substrings that name a genuinely transient condition. Everything else is
# treated as non-retryable: an unrecognised failure is not evidence of a
# retryable one.
_TRANSIENT_MARKERS = (
    "SshReadinessTimeoutError",
    "SshConnectionError",
    "SshEndpointUnavailableError",
    "ProviderUnavailableError",
    "StorageUnavailableError",
    "RemoteOperationInterruptedError",
    "ArtifactTransferTransientError",
    "LifecycleTimeoutError",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "connection refused",
    "connection reset",
    "throttl",
    "HTTP 429",
    "HTTP 500",
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "retryable",
)

_ZERO_RETRY_MARKERS = (
    ("CostGuardError", CostError),
    ("cannot enforce --max-price", CostError),
    ("ResourceUnavailableError", CapacityError),
    ("no confirmed capacity", CapacityError),
    ("ProviderAuthenticationError", SecurityError),
    ("CredentialError", SecurityError),
    ("SshAuthenticationError", SecurityError),
    ("SshHostKeyError", SecurityError),
    ("ProviderPermissionError", SecurityError),
    ("ProviderValidationError", ConfigurationError),
    ("JobSpecError", ConfigurationError),
    ("StorageKeyError", ConfigurationError),
    ("StorageVerificationError", ArtifactIntegrityError),
    ("ArtifactChecksumMismatchError", ArtifactIntegrityError),
    ("ArtifactSizeMismatchError", ArtifactIntegrityError),
    ("AmbiguousCreateError", ReconcilableError),
    ("UnresolvedCreateError", ReconcilableError),
    ("ArtifactTransferInProgressError", ReconcilableError),
    ("reconciliation_required", ReconcilableError),
)


class InfraResult(object):
    """One finished control-plane invocation."""

    def __init__(self, argv, returncode, stdout, stderr, payload=None):
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.payload = payload

    def __repr__(self):  # pragma: no cover - debugging aid
        return "InfraResult(returncode={!r}, argv={!r})".format(
            self.returncode, self.argv
        )


def classify_failure(action, returncode, output):
    """Map one failed invocation onto the failure taxonomy."""

    text = output or ""
    for marker, klass in _ZERO_RETRY_MARKERS:
        if marker in text:
            return klass(
                "infra {} failed (exit {}): {}".format(action, returncode, text.strip())
            )
    for marker in _TRANSIENT_MARKERS:
        if marker in text:
            return TransientInfraError(
                "infra {} failed (exit {}) with a recognisably transient "
                "condition: {}".format(action, returncode, text.strip())
            )
    return ComputeError(
        "infra {} failed (exit {}): {}".format(action, returncode, text.strip())
    )


class InfraCli(object):
    """Invoke the control plane over its CLI contract."""

    def __init__(self, location=None, *, timeout=DEFAULT_TIMEOUT_SECONDS,
                 global_args=(), environ=None):
        self.location = location or resolve_module.resolve()
        self.timeout = float(timeout)
        self.global_args = tuple(global_args)
        self.environ = dict(os.environ if environ is None else environ)
        # The controller's gitignored .env is never copied to a worker. Only
        # explicitly supported MLflow credentials enter the infra process and
        # only a job spec naming them forwards them to the worker at runtime.
        if environ is None:
            local_values = dotenv_values(os.path.join(resolve_module.repo_root(), ".env"))
            for name in _RESEARCH_SECRET_NAMES:
                if not self.environ.get(name) and local_values.get(name):
                    self.environ[name] = local_values[name]

    # ------------------------------------------------------------------ plumbing

    def _invoke(self, args, timeout):
        argv = [self.location.cli]
        argv.extend(self.global_args)
        argv.extend(args)
        try:
            completed = subprocess.run(
                argv,
                cwd=self.location.checkout or None,
                env=self.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout if timeout else None,
                check=False,
                universal_newlines=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientInfraError(
                "infra {} did not finish within {}s".format(
                    " ".join(args), exc.timeout
                )
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                "cannot execute the control-plane CLI at {}: {}".format(
                    self.location.cli, exc
                )
            ) from exc
        return completed.returncode, completed.stdout or "", completed.stderr or ""

    def run(self, *args, json_output=False, check=True, timeout=None):
        """Run one control-plane verb and optionally require success."""

        if not args:
            raise UsageError("no control-plane command supplied")
        tail = list(args)
        if json_output:
            tail.append("--json")
        returncode, stdout, stderr = self._invoke(tail, timeout or self.timeout)
        payload = None
        if json_output and stdout.strip():
            try:
                payload = json.loads(stdout)
            except ValueError:
                payload = None
        result = InfraResult(tail, returncode, stdout, stderr, payload)
        if check and returncode != 0:
            raise classify_failure(
                " ".join(str(part) for part in tail),
                returncode,
                (stderr or stdout).strip(),
            )
        if json_output and check and payload is None:
            raise ComputeError(
                "infra {} returned exit 0 without parseable JSON".format(
                    " ".join(str(part) for part in tail)
                )
            )
        return result

    # ------------------------------------------------------------ read-only verbs

    def doctor(self):
        return self.run("doctor")

    def worker_list(self):
        payload = self.run("worker", "list", "--read-only", json_output=True).payload
        if not isinstance(payload, list):
            raise ConfigurationError("infra worker list returned an incompatible JSON shape")
        return payload

    def worker_show(self, worker_id):
        return self.run("worker", "show", worker_id, "--read-only", json_output=True).payload

    def worker_health(self, worker_id):
        return self.run("worker", "health", worker_id, json_output=True).payload

    def volume_list(self):
        result = self.run("volume", "list", "--read-only", json_output=True).payload or {}
        return result.get("volumes", []) if isinstance(result, dict) else result

    def volume_show(self, volume_id):
        return self.run("volume", "show", volume_id, "--read-only", json_output=True).payload

    def job_list(self, state=None, worker_id=None):
        args = ["job", "list"]
        if state:
            args.extend(["--state", state])
        if worker_id:
            args.extend(["--worker", worker_id])
        payload = self.run(*args, json_output=True).payload
        if not isinstance(payload, list):
            raise ConfigurationError("infra job list returned an incompatible JSON shape")
        return payload

    def job_status(self, job_id):
        """Reconcile one job. A FAILED job still returns its record."""

        result = self.run("job", "status", job_id, json_output=True, check=False)
        if result.returncode == 0:
            return result.payload
        # Exit 1 with a parseable record means the job itself failed, which is
        # evidence rather than a control-plane failure.
        if isinstance(result.payload, dict):
            return result.payload
        raise classify_failure(
            "job status {}".format(job_id), result.returncode, result.stderr.strip()
        )

    def job_logs(self, job_id, tail_bytes=200000):
        return self.run(
            "job", "logs", job_id, "--tail-bytes", str(int(tail_bytes)), check=False
        ).stdout

    def storage_verify(self, artifact, expected_size=None, manifest=None,
                       manifest_file=None):
        """Existence/size/manifest consistency. Makes no content claim."""

        args = ["storage", "verify", artifact]
        if expected_size is not None:
            args.extend(["--expected-size", str(int(expected_size))])
        if manifest:
            args.extend(["--manifest", manifest])
        if manifest_file:
            args.extend(["--manifest-file", str(manifest_file)])
        return self.run(*args, json_output=True, check=False)

    def storage_download(self, artifact, destination, worker_id, *,
                         expected_size=None, expected_sha256=None,
                         overwrite=False, timeout=None):
        """Materialize one artifact on a worker with verified identity."""

        args = ["storage", "download", artifact, str(destination),
                "--worker", worker_id]
        if expected_size is not None:
            args.extend(["--expected-size", str(int(expected_size))])
        if expected_sha256:
            args.extend(["--expected-sha256", expected_sha256])
        if overwrite:
            args.append("--overwrite")
        return self.run(*args, json_output=True, check=False, timeout=timeout)

    def storage_list(self, prefix=None):
        args = ["storage", "list"]
        if prefix:
            args.extend(["--prefix", prefix])
        return self.run(*args, json_output=True).payload or []

    def cache_stats(self, worker_id):
        return self.run(
            "volume", "cache", "stats", "--worker", worker_id,
            json_output=True, check=False,
        )

    # --------------------------------------------------------------- mutating verbs

    def job_submit(self, spec_path, worker_id):
        return self.run(
            "job", "submit", str(spec_path), "--worker", worker_id,
            json_output=True, check=False,
        )

    def job_cancel(self, job_id):
        return self.run("job", "cancel", job_id, json_output=True, check=False)

    def worker_create(self, *, name, gpu, cloud, image=None, template=None,
                      gpu_count=1, container_disk_gb=20, volume_gb=0,
                      volume_mount_path=None, network_volume_id=None,
                      data_centers=(), max_price=None, start_ssh=True,
                      require_direct_ssh=True, timeout=None):
        """Request one Pod. Never retried by this layer.

        ``--yes`` is supplied because the caller is non-interactive; it bypasses
        only the confirmation prompt, and the price guard still applies.
        """

        args = [
            "worker", "create",
            "--name", name,
            "--gpu", gpu,
            "--cloud", cloud,
            "--gpu-count", str(int(gpu_count)),
            "--container-disk", str(int(container_disk_gb)),
            "--yes",
        ]
        if image:
            args.extend(["--image", image])
        if template:
            args.extend(["--template", template])
        if volume_gb:
            args.extend(["--volume", str(int(volume_gb))])
        if volume_mount_path:
            args.extend(["--volume-mount-path", volume_mount_path])
        if network_volume_id:
            args.extend(["--network-volume-id", network_volume_id])
        for data_center in data_centers or ():
            args.extend(["--data-center", data_center])
        if max_price is not None:
            args.extend(["--max-price", str(max_price)])
        if start_ssh:
            args.append("--start-ssh")
        if require_direct_ssh:
            args.append("--require-direct-ssh")
        return self.run(*args, check=False, timeout=timeout)

    def worker_wait_ssh(self, worker_id, timeout=None):
        return self.run("worker", "wait-ssh", worker_id, check=False, timeout=timeout)

    def worker_bootstrap(self, worker_id, timeout=None):
        return self.run("worker", "bootstrap", worker_id, check=False, timeout=timeout)

    def worker_start(self, worker_id, timeout=None):
        return self.run("worker", "start", worker_id, check=False, timeout=timeout)

    def worker_stop(self, worker_id, timeout=None):
        return self.run("worker", "stop", worker_id, check=False, timeout=timeout)

    def worker_destroy(self, worker_id, timeout=None):
        return self.run(
            "worker", "destroy", worker_id, "--yes", check=False, timeout=timeout
        )
