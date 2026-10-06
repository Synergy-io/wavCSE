"""The controller-side seam to an already-hired pod: ssh, rsync, and file reads.

The split of responsibilities here is deliberate. This module knows how to move
bytes and run one command on another host; it does not know how to *hire* a
host, hold an API token, or manage a worker lifecycle. Provisioning already
exists as its own subsystem (``infra``) and is not reimplemented here -- the
sweep assumes the device is up, which is exactly the situation it was written
for: a VM orchestrating a large RunPod machine that somebody already rented.

Everything is a thin wrapper over ``ssh``/``rsync`` with the exit status and the
captured output preserved, because the CLI's job is to report what happened, not
to smooth it over.
"""

import json
import os
import shlex
import subprocess


class RemoteError(Exception):
    """A remote command or transfer failed, with its own output preserved."""


def _run(argv, *, timeout=None):
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False)


def ssh(host, command, *, timeout=None, check=True):
    """Run one shell command on ``host`` and return its completed process."""
    completed = _run(["ssh", host, command], timeout=timeout)
    if check and completed.returncode != 0:
        raise RemoteError("ssh {} failed (exit {}): {}".format(
            host, completed.returncode,
            completed.stderr.decode("utf-8", "replace").strip() or command))
    return completed


def ssh_argv(host, argv, *, timeout=None, check=True):
    """Run an argument vector on ``host`` without shell quoting surprises."""
    return ssh(host, " ".join(shlex.quote(str(token)) for token in argv),
               timeout=timeout, check=check)


def rsync(host, source, destination, *, delete=False, extra=(),
          source_is_remote=False):
    """Copy between this host and ``host``.

    The direction is explicit rather than inferred from a ``host:`` prefix: a
    pull that silently became a push would overwrite the pod's outputs with an
    empty local directory, which is the kind of mistake that looks like a
    successful command.
    """
    argv = ["rsync", "-az", "--info=stats1"]
    if delete:
        argv.append("--delete")
    argv.extend(extra)
    if source_is_remote:
        argv.extend(["{}:{}".format(host, source), destination])
    else:
        argv.extend([source, "{}:{}".format(host, destination)])
    completed = _run(argv)
    if completed.returncode != 0:
        raise RemoteError("rsync {} failed (exit {}): {}".format(
            "from " + host if source_is_remote else "to " + host,
            completed.returncode,
            completed.stderr.decode("utf-8", "replace").strip()))


def read_json(host, path):
    """Read and parse a JSON file from the remote host."""
    completed = ssh(host, "cat {}".format(shlex.quote(path)))
    try:
        return json.loads(completed.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        raise RemoteError("{}:{} is not valid JSON: {}".format(host, path, exc))


def detach_command(checkout, argv, log_path, env_file=None):
    """The remote shell command that starts a supervisor and survives the ssh.

    ``setsid`` plus a redirection means the sweep outlives the connection that
    started it (the repo's documented ``nohup ... & disown`` intent).

    ``env_file`` is sourced before the command when given, because a
    non-interactive ``ssh host 'cmd'`` runs a shell that sources neither
    ``~/.bashrc`` nor ``~/.profile``: credentials exported in an interactive
    session are simply absent, and a run would start with no tracking
    configured and log nowhere. It is a path, not a value, so no secret is
    written into the command line or into any log.
    """
    parts = ["cd {}".format(shlex.quote(checkout))]
    if env_file:
        parts.append("set -a && . {} && set +a".format(shlex.quote(env_file)))
    parts.append("setsid nohup {} > {} 2>&1 < /dev/null & echo started $!".format(
        " ".join(shlex.quote(str(token)) for token in argv), shlex.quote(log_path)))
    return " && ".join(parts)


def start_detached(host, checkout, argv, log_path, env_file=None):
    """Launch the supervisor on the remote host, detached from this session."""
    return ssh(host, detach_command(checkout, argv, log_path, env_file))


def write_remote_file(host, path, content):
    """Create a small file remotely (used for the DRAIN flag)."""
    command = "mkdir -p {dir} && cat > {path} <<'ARC_EOF'\n{content}\nARC_EOF".format(
        dir=shlex.quote(os.path.dirname(path)), path=shlex.quote(path),
        content=content)
    return ssh(host, command)


def exists(host, path):
    completed = ssh(host, "test -e {}".format(shlex.quote(path)), check=False)
    return completed.returncode == 0
