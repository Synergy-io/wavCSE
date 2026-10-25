"""The admission decision: how many trainings may run at once, and why.

This is deliberately a *pure* function of (snapshot, running, policy). Every
interesting question about the scheduler -- does it stop when VRAM is tight,
does it refuse a GPU-less host, does it stop admitting into a saturated GPU --
is therefore answerable in a unit test with a hand-written snapshot, on a
machine with no GPU at all. Nothing here launches, sleeps, or reads the host;
that belongs to ``supervisor``, which only ever calls this and obeys it.

The policy answers "as many as fit" without a fixed concurrency number: the
cap is recomputed on every pass from free VRAM, free RAM, cores and disk, and
the binding resource is named in the reason so an operator can see *why* the
sweep is not going faster. ``max_concurrent`` may still be pinned to an integer
for a deliberate, reproducible cap; ``ceiling`` bounds an automatic decision
without turning it into a fixed number.
"""

from collections import namedtuple

GIB = 1024 ** 3

Decision = namedtuple("Decision", "admit reason capacity")

DEFAULT_POLICY = {
    # "auto" = derive the cap from live headroom; an int pins it.
    "max_concurrent": "auto",
    # Upper bound on an automatic cap (0 = no ceiling).
    "ceiling": 0,
    "vram_per_run_gb": 2.5,
    "ram_per_run_gb": 4.0,
    "ram_reserve_gb": 8.0,
    "cpu_cores_per_run": 4,
    "min_free_disk_gb": 20.0,
    "max_gpu_util_pct": 95,
    "max_load_fraction": 0.9,
    "workers_per_run": 2,
    "probe_interval_s": 15,
    "max_wall_seconds": 21600,
    "device_index": 0,
    # A CPU-only host is refused unless this is set: an accidental CPU run is
    # exactly what improvements/device_utils.py exists to catch.
    "allow_cpu": False,
    "concurrency_basis": "",
}


def _gb(value):
    return float(value or 0) / GIB


def vram_headroom_gb(snapshot):
    """Free VRAM on the selected device, or ``None`` when there is no GPU."""
    gpu = snapshot.get("gpu") or {}
    if not gpu.get("available"):
        return None
    devices = gpu.get("devices") or []
    return max(_gb(device["memory_free_bytes"]) for device in devices) if devices else None


def capacity(snapshot, policy):
    """The number of concurrent trainings this host can currently hold.

    Returns ``(capacity, binding)`` where ``binding`` names the resource that
    set the number -- the resource an operator must free to go faster.
    """
    limits = []

    headroom = vram_headroom_gb(snapshot)
    if headroom is None:
        if not policy.get("allow_cpu"):
            return 0, "no usable GPU on this host"
    else:
        limits.append((int(headroom // float(policy["vram_per_run_gb"])),
                       "VRAM ({:.1f} GiB free / {:.1f} GiB per run)".format(
                           headroom, float(policy["vram_per_run_gb"]))))

    ram = snapshot.get("ram") or {}
    available = ram.get("available_bytes")
    if available is None:
        return 0, "available RAM is unknown"
    usable = _gb(available) - float(policy["ram_reserve_gb"])
    limits.append((int(usable // float(policy["ram_per_run_gb"])),
                   "RAM ({:.1f} GiB usable / {:.1f} GiB per run)".format(
                       max(usable, 0.0), float(policy["ram_per_run_gb"]))))

    cpu = snapshot.get("cpu") or {}
    cores = int(cpu.get("count") or 1)
    limits.append((max(1, cores // int(policy["cpu_cores_per_run"])),
                   "CPU ({} cores / {} per run)".format(
                       cores, policy["cpu_cores_per_run"])))

    disk = snapshot.get("disk") or {}
    free = _gb(disk.get("free_bytes"))
    if disk.get("free_bytes") is None:
        return 0, "free disk space is unknown"
    limits.append((10 ** 6 if free >= float(policy["min_free_disk_gb"]) else 0,
                   "disk ({:.1f} GiB free / {:.1f} GiB required)".format(
                       free, float(policy["min_free_disk_gb"]))))

    count, binding = min(limits, key=lambda item: item[0])
    ceiling = int(policy.get("ceiling") or 0)
    if ceiling and count > ceiling:
        return ceiling, "policy ceiling ({})".format(ceiling)
    return max(count, 0), binding


def admit(snapshot, running, policy):
    """Decide whether one more training may start right now."""
    gpu = snapshot.get("gpu") or {}
    if not gpu.get("available") and not policy.get("allow_cpu"):
        return Decision(False, "no GPU available ({}); set policy.allow_cpu to "
                               "run on CPU anyway".format(
                                   gpu.get("error") or "unavailable"), 0)

    if running:
        utilisation = max((device.get("utilization_pct") or 0)
                          for device in (gpu.get("devices") or [{"utilization_pct": 0}]))
        if utilisation > int(policy["max_gpu_util_pct"]):
            return Decision(False, "GPU already {:.0f}% utilised (limit {}%), not "
                                   "admitting into a saturated device".format(
                                       utilisation, policy["max_gpu_util_pct"]),
                            capacity(snapshot, policy)[0])
        load = (snapshot.get("cpu") or {}).get("load_1m")
        cores = int((snapshot.get("cpu") or {}).get("count") or 1)
        if load is not None and load / cores > float(policy["max_load_fraction"]):
            return Decision(False, "load {:.1f} on {} cores exceeds the {:.0%} "
                                   "fraction".format(load, cores,
                                                     float(policy["max_load_fraction"])),
                            capacity(snapshot, policy)[0])

    count, binding = capacity(snapshot, policy)
    pinned = policy.get("max_concurrent")
    if isinstance(pinned, int) and not isinstance(pinned, bool):
        if running >= pinned:
            return Decision(False, "pinned concurrency cap reached ({}/{})".format(
                running, pinned), count)
        return Decision(True, "ok (pinned cap {}, capacity {})".format(pinned, count),
                        count)

    if count <= 0:
        return Decision(False, "no headroom: {}".format(binding), 0)
    if running >= count:
        return Decision(False, "at capacity {}/{}: {}".format(running, count, binding),
                        count)
    return Decision(True, "ok ({}/{} running, capacity set by {})".format(
        running, count, binding), count)
