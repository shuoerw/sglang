"""CPU-only regression tests for scheduler hierarchical Gloo exchange."""

import ast
import importlib.util
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt"


def load_helper():
    # Load without importing the GPU runtime through distributed/__init__.py.
    spec = importlib.util.spec_from_file_location(
        "hierarchical_gloo", SOURCE / "distributed/hierarchical_gloo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exchange(helper, enabled):
    source = SOURCE / "managers/scheduler_components/dp_attn.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "exchange_sync_rows"
    )
    namespace = {
        "torch": torch,
        "all_gather_single": dist.all_gather_into_tensor,
        "try_hierarchical_all_gather": helper,
        "_INTERNODE_HIER_ALLGATHER": enabled,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["exchange_sync_rows"]


def check_rows(helper, group, layout, expected_hierarchy):
    world = dist.get_world_size(group)
    global_rank = dist.get_rank()
    parent_ranks = dist.get_process_group_ranks(group)
    helper._hierarchy.cache_clear()

    def same_node(group, source_rank):
        node = next(members for members in layout if source_rank in members)
        return [i in node for i in range(world)]

    with patch.object(helper, "_same_node_members", side_effect=same_node) as discovery:
        for iteration, width in enumerate((1, 9, 13, 9)):
            # Include a non-contiguous input; metadata widths can change.
            row = torch.arange(width * 2, dtype=torch.int64)[::2]
            row += global_rank * 10000 + iteration * 100
            output = torch.full((world * width,), -1, dtype=torch.int64)
            used = helper.try_hierarchical_all_gather(output, row, group)
            assert used == expected_hierarchy, (layout, used)
            if not used:
                assert (output == -1).all(), "fallback mutated the output"
                dist.all_gather_into_tensor(output, row.contiguous(), group=group)
            expected = torch.cat(
                [
                    torch.arange(width, dtype=torch.int64) * 2
                    + r * 10000
                    + iteration * 100
                    for r in parent_ranks
                ]
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            calls = discovery.call_count
            if iteration == 0:
                initial_calls = calls
            else:
                assert calls == initial_calls, (
                    "topology was rediscovered on the hot path"
                )


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=6,
        timeout=timedelta(seconds=60),
    )
    helper = load_helper()
    try:
        cases = (
            ("uniform three-node layout", [[0, 1], [2, 3], [4, 5]], True),
            ("single-node fallback", [list(range(6))], False),
            ("one rank per node fallback", [[i] for i in range(6)], False),
            ("interleaved topology fallback", [[0, 2], [1, 3], [4, 5]], False),
            ("uneven topology fallback", [[0, 1], [2, 3, 4], [5]], False),
        )
        for name, layout, expected in cases:
            check_rows(helper, dist.group.WORLD, layout, expected)
            if expected:
                row = torch.tensor([rank])
                output = torch.empty(6, dtype=torch.int64)
                load_exchange(helper.try_hierarchical_all_gather, True)(
                    output, row, dist.group.WORLD
                )
                torch.testing.assert_close(output, torch.arange(6))
            if rank == 0:
                print(f"PASS: {name}", flush=True)

        subgroup = dist.new_group(ranks=[0, 2, 3, 5], backend="gloo")
        if rank in (0, 2, 3, 5):
            check_rows(helper, subgroup, [[0, 1], [2, 3]], False)
            forbidden = Mock(side_effect=AssertionError("OFF consulted hierarchy"))
            exchange = load_exchange(forbidden, False)
            row = torch.tensor([rank, rank + 100])
            output = torch.empty(8, dtype=torch.int64)
            exchange(output, row, subgroup)
            torch.testing.assert_close(
                output, torch.tensor([0, 100, 2, 102, 3, 103, 5, 105])
            )
            forbidden.assert_not_called()
            # Exercise the enabled scheduler integration with real collectives.
            with patch.object(
                helper,
                "_same_node_members",
                side_effect=AssertionError("subgroup topology discovery"),
            ):
                load_exchange(helper.try_hierarchical_all_gather, True)(
                    output, row, subgroup
                )
            torch.testing.assert_close(
                output, torch.tensor([0, 100, 2, 102, 3, 103, 5, 105])
            )
        dist.barrier()
        if rank == 0:
            print(
                "PASS: subgroup fallback, global-rank mapping, OFF isolation, ON fallback",
                flush=True,
            )

        pair = dist.new_group(ranks=[1, 4], backend="gloo")
        if rank in (1, 4):
            with patch.object(
                helper,
                "_same_node_members",
                side_effect=AssertionError("two-rank discovery"),
            ):
                output = torch.empty(2, dtype=torch.int64)
                load_exchange(helper.try_hierarchical_all_gather, True)(
                    output, torch.tensor([rank]), pair
                )
                torch.testing.assert_close(output, torch.tensor([1, 4]))
        dist.barrier()
        if rank == 0:
            print("PASS: two-rank default path", flush=True)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class TestHierarchicalGloo(unittest.TestCase):
    def test_collectives_and_scheduler_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                worker, args=(str(Path(directory) / "store"),), nprocs=6, join=True
            )

    def test_non_cpu_does_not_discover_topology(self):
        helper = load_helper()
        with patch.object(
            helper, "_hierarchy", side_effect=AssertionError("topology lookup")
        ):
            self.assertFalse(
                helper.try_hierarchical_all_gather(
                    torch.empty(4, device="meta"), torch.empty(1, device="meta"), None
                )
            )

    def test_non_gloo_does_not_discover_topology(self):
        helper = load_helper()
        with (
            patch.object(helper.dist, "get_backend", return_value="nccl"),
            patch.object(
                helper, "_hierarchy", side_effect=AssertionError("topology lookup")
            ),
        ):
            self.assertFalse(
                helper.try_hierarchical_all_gather(torch.empty(4), torch.empty(1), None)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
