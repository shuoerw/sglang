"""Automatic async selection must preserve optional dependencies and rank agreement."""

import argparse
import tempfile
import unittest
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators import mooncake_transfer_engine as te
from sglang.srt.eplb.eplb_manager import EPLBManager


def _runtime(*, enabled=True, elastic=None, ep=2, tp=2, pp=1, device="cuda"):
    stack = ExitStack()
    stack.enter_context(
        patch.object(
            te,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(
                    enable_eplb=enabled,
                    elastic_ep_backend=elastic,
                    enable_elastic_expert_backup=False,
                    mooncake_ib_device=None,
                )
            ),
        )
    )
    stack.enter_context(
        patch.object(
            te,
            "get_parallel",
            return_value=SimpleNamespace(ep_size=ep, tp_size=tp, pp_size=pp),
        )
    )
    stack.enter_context(
        patch.object(te, "get_device", return_value=SimpleNamespace(device=device))
    )
    stack.enter_context(
        patch.object(
            te,
            "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_mode="null",
                disaggregation_transfer_backend="mooncake",
                encoder_only=False,
                language_only=False,
                disaggregation_ib_device=None,
            ),
        )
    )
    stack.enter_context(
        patch.object(
            te,
            "get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=False),
        )
    )
    return stack


def _select_on_rank(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with (
            _runtime(),
            patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=SimpleNamespace(cpu_group=dist.group.WORLD),
            ),
            patch(
                "sglang.srt.eplb.eplb_async_migrator.AsyncExpertMigrator"
            ) as migrator,
            patch("sglang.srt.eplb.eplb_async_migrator.MooncakeExpertTransport"),
        ):
            for missing_rank in (None, 0, 1):
                engine = (
                    None
                    if rank == missing_rank
                    else SimpleNamespace(
                        engine=SimpleNamespace(batch_transfer_sync_read=Mock())
                    )
                )
                manager = EPLBManager.__new__(EPLBManager)
                manager._async_migrator = None
                manager._rebalance_layers_per_chunk = 1
                migrator.reset_mock()
                with patch.object(
                    te, "get_mooncake_transfer_engine", return_value=engine
                ):
                    manager._init_async_migrator()
                assert (manager._async_migrator is not None) == (missing_rank is None)
                assert migrator.call_count == int(missing_rank is None)
    finally:
        dist.destroy_process_group()


class TestAutomaticAsyncEPLB(unittest.TestCase):
    def test_cli_needs_no_async_flag(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "unused",
                "--enable-eplb",
                "--eplb-rebalance-num-iterations",
                "1",
                "--expert-distribution-recorder-buffer-size",
                "1",
            ]
        )
        self.assertTrue(args.enable_eplb)
        self.assertEqual(args.eplb_rebalance_num_iterations, 1)
        self.assertFalse(hasattr(args, "eplb_async_rebalance"))
        self.assertNotIn("--eplb-async-rebalance", parser._option_string_actions)

    def test_supported_config(self):
        with _runtime():
            self.assertIsNone(te.eplb_async_unavailable_reason())

    def test_unsupported_config_preserves_existing_path(self):
        for kwargs in (
            {"enabled": False},
            {"elastic": "mooncake"},
            {"elastic": "nccl"},
            {"ep": 2, "tp": 4},
            {"pp": 2},
            {"device": "cpu"},
            {"device": "xpu"},
        ):
            with self.subTest(**kwargs), _runtime(**kwargs):
                self.assertIsNotNone(te.eplb_async_unavailable_reason())

    def test_eplb_initializes_optional_engine(self):
        with (
            _runtime(),
            patch.object(te, "init_mooncake_transfer_engine") as initialize,
            patch.object(te, "get_local_ip_auto", return_value="127.0.0.1"),
        ):
            te.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)
            initialize.assert_called_once_with(
                hostname="127.0.0.1", gpu_id=0, ib_device=None
            )

    def test_optional_engine_failure_falls_back(self):
        for error in (
            ImportError("missing package"),
            OSError("no device"),
            RuntimeError("init failed"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                _runtime(),
                patch.object(te, "init_mooncake_transfer_engine", side_effect=error),
                patch.object(te, "get_local_ip_auto", return_value="127.0.0.1"),
                self.assertLogs(te.logger, level="WARNING") as logs,
            ):
                te.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)
                self.assertIn("synchronous rebalance", logs.output[0])

    def test_required_engine_failure_is_not_swallowed(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                _runtime(enabled=enabled),
                patch.object(
                    te,
                    "get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_mode="decode",
                        disaggregation_transfer_backend="mooncake",
                        disaggregation_ib_device=None,
                    ),
                ),
                patch.object(
                    te,
                    "init_mooncake_transfer_engine",
                    side_effect=RuntimeError("required"),
                ),
                patch.object(te, "get_local_ip_auto", return_value="127.0.0.1"),
                self.assertRaisesRegex(RuntimeError, "required"),
            ):
                te.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)

    def test_eplb_disabled_does_not_initialize_engine(self):
        with (
            _runtime(enabled=False),
            patch.object(te, "init_mooncake_transfer_engine") as initialize,
        ):
            te.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)
            initialize.assert_not_called()

    def test_unsupported_manager_does_not_enter_new_collectives(self):
        manager = EPLBManager.__new__(EPLBManager)
        manager._async_migrator = None
        with _runtime(elastic="nccl"), patch.object(dist, "all_reduce") as collective:
            manager._init_async_migrator()
            self.assertIsNone(manager._async_migrator)
            collective.assert_not_called()

    def test_manager_rejects_engine_without_expert_read_api(self):
        manager = EPLBManager.__new__(EPLBManager)
        manager._async_migrator = None
        with (
            _runtime(),
            patch.object(
                te,
                "get_mooncake_transfer_engine",
                return_value=SimpleNamespace(engine=object()),
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=SimpleNamespace(cpu_group=None),
            ),
            patch.object(dist, "all_reduce"),
        ):
            manager._init_async_migrator()
            self.assertIsNone(manager._async_migrator)

    def test_two_rank_agreement_when_one_engine_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_select_on_rank, args=(f"{directory}/gloo",), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
