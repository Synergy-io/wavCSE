"""Live resource probes for admission control, and the parsers behind them.

The supervisor decides how many trainings to run at once from what the machine
actually has free right now, so this module is the only place that talks to the
host: ``nvidia-smi`` for VRAM and utilisation, ``/proc/meminfo`` for available
RAM, ``shutil.disk_usage`` for the checkpoint/results filesystem, and
``os.getloadavg``/``os.cpu_count`` for CPU pressure.

Every probe degrades instead of raising. A box with no NVIDIA driver (a
controller VM, a CPU-only smoke host) must still be able to *report* what it
is, because the honest answer -- "no GPU here" -- is what stops a training
sweep from being admitted onto a machine that cannot run it. The parsers are
pure functions over text so the interesting cases (a missing binary, a driver
mismatch, a 0-byte read) are testable without any GPU at all.
"""

import os
import shutil
import subprocess

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
]

MIB = 1024 * 1024
KIB = 1024


def parse_nvidia_smi(text):
    """Parse ``--format=csv,noheader,nounits`` output into per-GPU records.

    A line that does not have the six expected fields is skipped rather than
    failing the probe: a half-answering driver is still useful evidence, and a
    sweep that refuses to describe the machine cannot decide anything.
    """
    devices = []
    for line in (text or "").splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, total, used, free, util = fields
        try:
            devices.append({
                "index": int(index),
                "name": name,
                "memory_total_bytes": int(total) * MIB,
                "memory_used_bytes": int(used) * MIB,
                "memory_free_bytes": int(free) * MIB,
                "utilization_pct": int(util),
            })
        except ValueError:
            continue
    return devices


def parse_meminfo(text):
    """Parse ``/proc/meminfo`` into bytes for the two fields admission needs."""
    wanted = {"MemTotal": "total_bytes", "MemAvailable": "available_bytes"}
    values = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if key not in wanted:
            continue
        number = rest.strip().split(" ")[0]
        try:
            values[wanted[key]] = int(number) * KIB
        except ValueError:
            continue
    return values


def parse_loadavg(text):
    """Parse ``/proc/loadavg`` into (load_1m, load_5m, load_15m)."""
    fields = (text or "").split()
    if len(fields) < 3:
        return None
    try:
        return (float(fields[0]), float(fields[1]), float(fields[2]))
    except ValueError:
        return None


def _run(argv, timeout=15):
    try:
        completed = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "{}: {}".format(type(exc).__name__, exc)
    if completed.returncode != 0:
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
        return None, detail.splitlines()[0] if detail else "exit {}".format(
            completed.returncode)
    return completed.stdout.decode("utf-8", "replace"), None


def probe_gpu():
    """Report the GPU picture without ever raising."""
    text, error = _run(NVIDIA_SMI_QUERY)
    if text is None:
        return {"available": False, "devices": [], "error": error}
    devices = parse_nvidia_smi(text)
    if not devices:
        return {"available": False, "devices": [],
                "error": "nvidia-smi answered but reported no devices"}
    return {"available": True, "devices": devices, "error": None}


def probe_ram():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            values = parse_meminfo(handle.read())
    except OSError as exc:
        return {"total_bytes": None, "available_bytes": None,
                "error": "{}: {}".format(type(exc).__name__, exc)}
    if "total_bytes" not in values:
        values["error"] = "MemTotal missing from /proc/meminfo"
    return values


def probe_disk(path):
    """Free bytes on the filesystem holding ``path`` (its nearest existing parent)."""
    candidate = os.path.abspath(os.path.expanduser(path or "."))
    while candidate and not os.path.isdir(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    try:
        usage = shutil.disk_usage(candidate)
    except OSError as exc:
        return {"free_bytes": None, "path": candidate,
                "error": "{}: {}".format(type(exc).__name__, exc)}
    return {"free_bytes": usage.free, "total_bytes": usage.total,
            "path": candidate, "error": None}


def probe_load():
    try:
        load = os.getloadavg()
    except OSError:  # pragma: no cover - unavailable on some platforms
        load = None
    return {"count": os.cpu_count() or 1, "load_1m": load[0] if load else None}


def probe(disk_path="."):
    """One consistent snapshot for a single admission decision."""
    return {
        "gpu": probe_gpu(),
        "ram": probe_ram(),
        "disk": probe_disk(disk_path),
        "cpu": probe_load(),
    }


def describe(snapshot):
    """A short human line for logs and ``status`` output."""
    gpu = snapshot["gpu"]
    if gpu["available"]:
        device = gpu["devices"][0]
        gpu_text = "{} {} MiB free {}% util".format(
            device["name"], device["memory_free_bytes"] // MIB,
            device["utilization_pct"],
        )
    else:
        gpu_text = "none ({})".format(gpu.get("error") or "unavailable")
    ram = snapshot["ram"]
    ram_text = "{} GiB".format(
        round((ram.get("available_bytes") or 0) / (1024 ** 3), 1))
    disk = snapshot["disk"]
    disk_text = "{} GiB".format(
        round((disk.get("free_bytes") or 0) / (1024 ** 3), 1))
    cpu = snapshot["cpu"]
    return "gpu={} | ram_avail={} | disk_free={} | load1={}/{}".format(
        gpu_text, ram_text, disk_text, cpu.get("load_1m"), cpu["count"])
