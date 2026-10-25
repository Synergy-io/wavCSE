"""Transport adapter for declared static hosts through standalone ``infra``.

No raw SSH host, SSH key, rsync command or provider transport lives in the
public model repository. Paid RunPod sweep launch is deliberately unsupported:
its spend and job identity require the private authorized execution backend.
"""

import json
import os
import subprocess
from pathlib import Path


class RemoteError(Exception):
    """The explicit infra binding or a named-worker transport operation failed."""


def _infra():
    configured = os.environ.get("WAVCSE_INFRA_CHECKOUT", "").strip()
    if not configured:
        raise RemoteError(
            "set WAVCSE_INFRA_CHECKOUT to the standalone wavcse-infra checkout; "
            "implicit host, sibling and PATH discovery are forbidden"
        )
    checkout = Path(configured).expanduser().resolve()
    if not (checkout / "pyproject.toml").is_file() or not (
        checkout / "src/wavcse_infra/__init__.py"
    ).is_file():
        raise RemoteError("WAVCSE_INFRA_CHECKOUT is not a standalone infra package")
    executable = os.environ.get("WAVCSE_INFRA_CLI", "").strip() or str(
        checkout / ".venv/bin/infra"
    )
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise RemoteError("run `uv sync --locked` in the standalone infra checkout")
    if os.path.realpath(executable) != os.path.realpath(checkout / ".venv/bin/infra"):
        raise RemoteError("WAVCSE_INFRA_CLI must belong to WAVCSE_INFRA_CHECKOUT")
    return str(checkout), executable


def _call(argv, *, check=True, input_bytes=None, timeout=None):
    checkout, executable = _infra()
    try:
        completed = subprocess.run(
            [executable, *argv], cwd=checkout, input=input_bytes,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RemoteError("named-worker infra transport failed: {}".format(exc)) from exc
    if check and completed.returncode:
        reason = completed.stderr.decode("utf-8", "replace").strip()[:400]
        raise RemoteError(
            "infra {} failed (exit {}): {}".format(" ".join(argv[:3]), completed.returncode, reason)
        )
    return completed


def _static_worker(worker_id):
    """A remote sweep must never become an unaccounted paid-RunPod job."""
    completed = _call(["worker", "show", worker_id, "--read-only", "--json"])
    try:
        worker = json.loads(completed.stdout)
    except ValueError as exc:
        raise RemoteError("infra worker show returned invalid JSON") from exc
    if not isinstance(worker, dict) or worker.get("provider") != "static_ssh":
        raise RemoteError(
            "remote layer sweeps require a declared static_ssh worker ID; "
            "paid RunPod training must use an authorized exact-commit job"
        )


def exec(worker_id, argv, *, timeout=None, check=True):
    """Run an argv on the named static worker, with infra owning SSH."""
    _static_worker(worker_id)
    command = ["worker", "exec", worker_id]
    if timeout is not None:
        command.extend(["--command-timeout", str(timeout)])
    return _call([*command, "--", *argv], check=check)


def rsync(worker_id, source, destination, *, delete=False, extra=(), source_is_remote=False):
    """Sync directories through infra; never expose deletion or custom rsync flags."""
    if delete or extra:
        raise RemoteError("remote sweep sync does not permit deletion or custom rsync flags")
    _static_worker(worker_id)
    return _call([
        "worker", "sync", worker_id,
        "--direction", "pull" if source_is_remote else "push",
        "--source", source, "--destination", destination, "--json",
    ])


def read_json(worker_id, path):
    result = exec(worker_id, ["cat", path])
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise RemoteError("{}:{} is not valid JSON".format(worker_id, path)) from exc


def start_detached(worker_id, checkout, argv, log_path, env_file=None):
    """Start only on the static worker; infra validates root and builds the shell command."""
    _static_worker(worker_id)
    command = ["worker", "start-detached", worker_id,
               "--checkout", checkout, "--log-path", log_path]
    if env_file:
        command.extend(["--env-file", env_file])
    return _call([*command, "--", *argv])


def write_remote_file(worker_id, path, content):
    """Send an exact small marker over stdin; never interpolate it into a shell."""
    _static_worker(worker_id)
    return _call(["worker", "write-file", worker_id, "--path", path],
                 input_bytes=content.encode("utf-8"))


def exists(worker_id, path):
    result = exec(worker_id, ["test", "-e", path], check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 1 and not result.stderr.strip():
        return False
    raise RemoteError("infra could not verify whether {}:{} exists".format(worker_id, path))
