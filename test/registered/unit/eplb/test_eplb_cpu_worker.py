"""Persistent planner IPC, staging reuse, and failure propagation."""

import dataclasses
import os
import signal
import unittest

import torch

from sglang.srt.eplb.eplb_async_migrator import plan_layer_pull
from sglang.srt.eplb.eplb_cpu_worker import (
    METADATA_UPLOAD_FIELDS,
    CPUPlanner,
    PlannerConfig,
    allocate_upload_buffers,
    compute_plan,
    decode_layer_plan,
)


class TestCPUPlanner(unittest.TestCase):
    def test_process_reuse_staging_and_errors(self):
        torch.set_num_threads(1)
        config = PlannerConfig(16, 4, 1, 1, 0, "auto", True, False)
        old_map = torch.arange(16).remainder(12).repeat(3, 1)
        counts = torch.arange(36).reshape(3, 12).float() + 1
        reference = compute_plan(config, counts, old_map, [0, 1, 2])
        buffers = allocate_upload_buffers(reference.metadata)
        pointers = {name: tensor.data_ptr() for name, tensor in buffers.items()}
        worker = CPUPlanner(buffers)
        snapshots = []
        try:
            for step in range(4):
                loads = counts.roll(step, -1)
                expected = compute_plan(config, loads, old_map, [0, 1, 2])
                actual = worker.submit(config, loads, old_map, [0, 1, 2]).result(
                    timeout=30
                )
                self.assertNotEqual(actual.worker_pid, os.getpid())
                self.assertEqual(actual.worker_pid, worker.pid)
                for name in METADATA_UPLOAD_FIELDS:
                    torch.testing.assert_close(
                        getattr(actual.metadata, name),
                        getattr(expected.metadata, name),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        getattr(actual.metadata, name).data_ptr(), pointers[name]
                    )
                for layer in range(3):
                    self.assertEqual(
                        decode_layer_plan(actual.pull_plans[layer]),
                        plan_layer_pull(
                            old_map[layer].tolist(),
                            expected.placement.physical_to_logical_map[layer].tolist(),
                            4,
                            4,
                            0,
                        ),
                    )
                snapshots.append(
                    (
                        actual.metadata.physical_to_logical_map_cpu,
                        expected.metadata.physical_to_logical_map_cpu.clone(),
                    )
                )
                for snapshot, expected_snapshot in snapshots:
                    torch.testing.assert_close(
                        snapshot, expected_snapshot, rtol=0, atol=0
                    )
            bad = dataclasses.replace(config, num_physical_experts=17)
            with self.assertRaisesRegex(RuntimeError, "evenly distributed"):
                worker.submit(bad, counts, old_map, [0, 1, 2]).result(timeout=30)
            self.assertEqual(
                worker.submit(config, counts, old_map, [0, 1, 2])
                .result(timeout=30)
                .worker_pid,
                worker.pid,
            )
        finally:
            worker.shutdown()
        self.assertIsNotNone(worker._process.poll())

    @unittest.skipUnless(os.name == "posix", "POSIX worker transport")
    def test_dead_worker_propagates(self):
        worker = CPUPlanner()
        try:
            os.kill(worker.pid, signal.SIGKILL)
            config = PlannerConfig(4, 2, 1, 1, 0, "auto", False, False)
            with self.assertRaises(RuntimeError):
                worker.submit(
                    config, torch.ones(1, 3), torch.tensor([[0, 1, 2, 0]]), [0]
                ).result(timeout=30)
        finally:
            worker.shutdown()


if __name__ == "__main__":
    unittest.main()
