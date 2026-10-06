"""Admission control: the decision must be explainable from a fake snapshot.

Every case here runs on a machine with no GPU, which is the point of keeping
``scheduler`` free of any host access.
"""

import os
import sys
import unittest

from improvements.sweep import scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GIB = 1024 ** 3


def snapshot(*, gpu=True, vram_free_gb=48.0, util=10, ram_free_gb=128.0,
             cores=32, load1=1.0, disk_free_gb=200.0):
    devices = ([{"index": 0, "name": "Fake", "memory_free_bytes": int(vram_free_gb * GIB),
                 "memory_total_bytes": int(80 * GIB), "memory_used_bytes": 0,
                 "utilization_pct": util}] if gpu else [])
    return {
        "gpu": {"available": bool(devices), "devices": devices,
                "error": None if gpu else "nvidia-smi: not found"},
        "ram": {"total_bytes": int(200 * GIB), "available_bytes": int(ram_free_gb * GIB)},
        "cpu": {"count": cores, "load_1m": load1},
        "disk": {"free_bytes": int(disk_free_gb * GIB), "path": "/workspace"},
    }


def policy(**overrides):
    merged = dict(scheduler.DEFAULT_POLICY)
    merged.update(overrides)
    return merged


class CapacityTest(unittest.TestCase):
    def test_capacity_is_the_smallest_headroom_and_names_it(self):
        count, binding = scheduler.capacity(
            snapshot(vram_free_gb=10.0, ram_free_gb=20.0, cores=8),
            policy(vram_per_run_gb=2.5, ram_per_run_gb=4.0, ram_reserve_gb=8.0,
                   cpu_cores_per_run=4))
        # VRAM: 4, RAM: (20-8)/4 = 3, CPU: 8/4 = 2 -> CPU binds.
        self.assertEqual(count, 2)
        self.assertIn("CPU", binding)

    def test_ceiling_caps_an_automatic_cap(self):
        count, binding = scheduler.capacity(snapshot(), policy(ceiling=3))
        self.assertEqual(count, 3)
        self.assertIn("ceiling", binding)

    def test_disk_below_the_floor_blocks_everything(self):
        count, binding = scheduler.capacity(snapshot(disk_free_gb=5.0),
                                            policy(min_free_disk_gb=20.0))
        self.assertEqual(count, 0)
        self.assertIn("disk", binding)

    def test_vram_headroom_from_a_single_device(self):
        self.assertEqual(scheduler.vram_headroom_gb(snapshot(vram_free_gb=7.5)), 7.5)
        self.assertIsNone(scheduler.vram_headroom_gb(snapshot(gpu=False)))


class AdmitTest(unittest.TestCase):
    def test_gpu_less_host_is_refused_by_default(self):
        decision = scheduler.admit(snapshot(gpu=False), 0, policy())
        self.assertFalse(decision.admit)
        self.assertIn("no GPU", decision.reason)

    def test_gpu_less_host_proceeds_only_with_allow_cpu(self):
        decision = scheduler.admit(snapshot(gpu=False), 0, policy(allow_cpu=True))
        self.assertTrue(decision.admit)

    def test_admits_then_holds_at_capacity(self):
        snap = snapshot(vram_free_gb=10.0, ram_free_gb=40.0, cores=16)
        pol = policy(vram_per_run_gb=2.5, ram_per_run_gb=4.0, ram_reserve_gb=8.0,
                     cpu_cores_per_run=4)
        # (40-8)/4 = 8 RAM, 10/2.5 = 4 VRAM, 16/4 = 4 CPU -> 4.
        running = 0
        while scheduler.admit(snap, running, pol).admit and running < 20:
            running += 1
        self.assertEqual(running, 4)
        held = scheduler.admit(snap, running, pol)
        self.assertFalse(held.admit)
        self.assertIn("at capacity", held.reason)

    def test_pinned_cap_is_obeyed_even_with_headroom(self):
        pol = policy(max_concurrent=2)
        self.assertTrue(scheduler.admit(snapshot(), 1, pol).admit)
        held = scheduler.admit(snapshot(), 2, pol)
        self.assertFalse(held.admit)
        self.assertIn("pinned", held.reason)

    def test_saturated_gpu_is_not_admitted_into(self):
        decision = scheduler.admit(snapshot(util=99), 1, policy(max_gpu_util_pct=95))
        self.assertFalse(decision.admit)
        self.assertIn("utilised", decision.reason)

    def test_loaded_cpu_stops_admission(self):
        decision = scheduler.admit(snapshot(cores=8, load1=16.0), 2,
                                   policy(max_load_fraction=0.9))
        self.assertFalse(decision.admit)
        self.assertIn("load", decision.reason)

    def test_unknown_ram_is_refused_rather_than_assumed(self):
        snap = snapshot()
        snap["ram"] = {"total_bytes": None, "available_bytes": None}
        decision = scheduler.admit(snap, 0, policy())
        self.assertFalse(decision.admit)
        self.assertIn("RAM", decision.reason)

    def test_ram_reserve_is_respected(self):
        snap = snapshot(vram_free_gb=100.0, ram_free_gb=12.0, cores=64)
        pol = policy(ram_per_run_gb=4.0, ram_reserve_gb=8.0)
        self.assertTrue(scheduler.admit(snap, 0, pol).admit)
        self.assertFalse(scheduler.admit(snap, 1, pol).admit)


if __name__ == "__main__":
    unittest.main()
