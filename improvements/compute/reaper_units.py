"""The controller-local mechanism that runs the reaper without a session.

The reaper (:mod:`improvements.compute.reaper`) is only useful if something executes it
when nothing else is running. On this controller that something is a systemd timer: a
persistent host mechanism that survives the end of any agent session, survives a reboot,
and is not part of the process it is protecting against.

Two properties are deliberate:

* **Rendering is pure and installing is idempotent.** The unit documents are a
  deterministic function of facts this machine can prove — the checkout, the state root,
  the resolved control-plane checkout, the interval — so installing twice writes the same
  bytes and reports ``changed: false``. Nothing here reads a developer-local path: the
  controller bootstrap and this installer both discover the same checkout the backend
  already resolves.
* **Installing is not enabling.** Writing a unit file changes no running system; enabling
  the timer is a separate, explicit flag because the timer's whole purpose is to stop
  paid compute unattended, and that must never be switched on by accident or by a code
  path that merely meant to describe itself.
"""

import os
import shutil
import subprocess

from improvements.compute.errors import ConfigurationError

SERVICE_NAME = "wavcse-arc-reaper.service"
TIMER_NAME = "wavcse-arc-reaper.timer"
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_BOOT_DELAY_SECONDS = 120
SYSTEM_UNIT_DIR = "/etc/systemd/system"
USER_UNIT_DIR = os.path.join("~", ".config", "systemd", "user")

# The reaper needs the research state (the authoritative leases) and the control plane it
# drives. Both are passed explicitly so the unit cannot silently resolve a different one.
STATE_ENV = "WAVCSE_RESEARCH_STATE"
CHECKOUT_ENV = "WAVCSE_INFRA_CHECKOUT"


def _environment_lines(values):
    return ["Environment={}={}".format(name, value) for name, value in values if value]


def build_units(*, repo_root, command_argv, state_root, infra_checkout,
                interval_seconds=DEFAULT_INTERVAL_SECONDS,
                boot_delay_seconds=DEFAULT_BOOT_DELAY_SECONDS,
                run_as=None, extra_environment=()):
    """Render the service and timer documents for one controller."""

    command_argv = [str(part) for part in command_argv]
    if not command_argv:
        raise ConfigurationError("the reaper unit needs an ExecStart command")
    environment = [
        (STATE_ENV, state_root),
        (CHECKOUT_ENV, infra_checkout),
    ]
    environment.extend(extra_environment)
    service_lines = [
        "[Unit]",
        "Description=ARC cost reaper: end paid compute whose lease deadline passed",
        "Documentation=file:{}/improvements/compute/README.md".format(repo_root),
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=oneshot",
        "WorkingDirectory={}".format(repo_root),
    ]
    service_lines.extend(_environment_lines(environment))
    if run_as:
        service_lines.append("User={}".format(run_as))
    service_lines.extend([
        "ExecStart={}".format(" ".join(command_argv)),
        # A failing tick must leave the record of the failure, not disappear.
        "StandardOutput=journal",
        "StandardError=journal",
        "",
    ])
    timer_lines = [
        "[Unit]",
        "Description=Run the ARC cost reaper on a fixed interval",
        "Documentation=file:{}/improvements/compute/README.md".format(repo_root),
        "",
        "[Timer]",
        "OnBootSec={}".format(int(boot_delay_seconds)),
        "OnUnitActiveSec={}".format(int(interval_seconds)),
        # A controller that was off for an hour must still reconcile once on the way
        # back up: that is exactly the crash this mechanism exists for.
        "Persistent=true",
        "AccuracySec=30s",
        "Unit=" + SERVICE_NAME,
        "",
        "[Install]",
        "WantedBy=timers.target",
        "",
    ]
    return {
        SERVICE_NAME: "\n".join(service_lines),
        TIMER_NAME: "\n".join(timer_lines),
    }


def install(units, target_dir, *, dry_run=False):
    """Write unit documents into ``target_dir``; idempotent, never enables anything."""

    target_dir = os.path.abspath(os.path.expanduser(target_dir))
    written = []
    unchanged = []
    for name, content in sorted(units.items()):
        path = os.path.join(target_dir, name)
        existing = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                existing = handle.read()
        if existing == content:
            unchanged.append(path)
            continue
        written.append(path)
        if dry_run:
            continue
        staging = path + ".staging"
        try:
            os.makedirs(target_dir, exist_ok=True)
            with open(staging, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, path)
        except OSError as exc:
            # The unit directory is a privilege boundary, not a crash: `--user` exists
            # precisely for a controller that cannot write the system directory. A
            # refused install leaves no staging file and no half-written unit behind.
            try:
                os.unlink(staging)
            except OSError:
                pass
            raise ConfigurationError(
                "the unit {} could not be written into {}: {}. Install with "
                "sufficient privileges for that directory, or install user units with "
                "`--user` (`loginctl enable-linger` is then required for the timer to "
                "run with no session)".format(name, target_dir, exc)
            ) from exc
    return {"target_dir": target_dir, "written": written, "unchanged": unchanged,
            "changed": bool(written), "dry_run": bool(dry_run)}


def _run(argv):
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, universal_newlines=True)


def enable(*, user=False, runner=None):
    """Enable and start the timer. Only ever called by an explicit operator flag."""

    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise ConfigurationError(
            "systemctl is not available on this host, so the reaper timer cannot be "
            "enabled here; the unit files are installed and can be enabled manually"
        )
    argv = [systemctl]
    if user:
        argv.append("--user")
    argv.extend(["enable", "--now", TIMER_NAME])
    completed = (runner or _run)(argv)
    if completed.returncode != 0:
        raise ConfigurationError(
            "systemctl {} failed: {}".format(
                " ".join(argv[1:]), (completed.stderr or completed.stdout).strip()
            )
        )
    return {"enabled": True, "argv": argv}


def default_unit_dir(*, user=False, environ=None):
    environ = os.environ if environ is None else environ
    if user:
        configured = environ.get("XDG_CONFIG_HOME", "").strip()
        base = configured if configured else os.path.expanduser("~/.config")
        return os.path.join(base, "systemd", "user")
    return SYSTEM_UNIT_DIR
