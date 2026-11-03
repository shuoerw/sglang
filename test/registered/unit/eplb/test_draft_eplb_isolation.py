"""A draft's expert IDs and checkpoint placement must not depend on target EPLB."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)


class TestDraftEplbIsolation(unittest.TestCase):
    def test_mega_routing_keeps_draft_ids_in_its_own_expert_range(self):
        from sglang.srt.layers.moe import mega_moe as mega

        # A valid target placement can point beyond every draft physical slot.
        target = ExpertLocationDispatchInfo(
            "static",
            torch.full((384,), 400),
            torch.empty(384, 0, dtype=torch.long),
            torch.ones(384, dtype=torch.long),
            448,
        )
        for is_nextn in (False, True):
            with self.subTest(is_nextn=is_nextn):
                logical_ids = torch.tensor([[0, 1, 127]], dtype=torch.int32)
                experts = SimpleNamespace(
                    num_experts=192 if is_nextn else 448,
                    moe_runner_config=SimpleNamespace(swiglu_limit=None),
                    should_fuse_routed_scaling_factor_in_topk=True,
                )

                def topk(hidden, logits, expert_location_dispatch_info=None, **kwargs):
                    return SimpleNamespace(
                        topk_ids=topk_ids_logical_to_physical(
                            logical_ids, expert_location_dispatch_info
                        ),
                        topk_weights=torch.ones(1, 3),
                    )

                def run(experts, hidden, ids, scores, **kwargs):
                    self.assertLess(
                        int(ids.max()),
                        experts.num_experts,
                        "target placement escaped into draft routing",
                    )
                    return ids

                moe = SimpleNamespace(
                    config=SimpleNamespace(
                        hidden_size=16, moe_intermediate_size=8, num_experts_per_tok=3
                    ),
                    layer_id=0,
                    is_nextn=is_nextn,
                    is_hash=False,
                    gate=lambda h, **kw: torch.zeros(1, 128 if is_nextn else 384),
                    topk=topk,
                    experts=experts,
                    num_fused_shared_experts=0,
                    routed_scaling_factor=1.0,
                    mega_shared_l1_weights=None,
                    mega_shared_l2_weights=None,
                )
                with (
                    patch.object(
                        mega.ExpertLocationDispatchInfo, "init_new", return_value=target
                    ),
                    patch.object(mega, "run_mega_routed_experts", side_effect=run),
                ):
                    actual = mega._run_mega_routed(
                        moe, torch.ones(1, 16), None, None, 1
                    )
                torch.testing.assert_close(
                    actual,
                    logical_ids if is_nextn else torch.full_like(logical_ids, 400),
                )

    def test_draft_checkpoint_loading_uses_its_own_storage_rank(self):
        from sglang.srt.layers.moe.fused_moe_triton import layer as module

        param = torch.nn.Parameter(torch.zeros(2, 2))
        loader = Mock()
        expert = SimpleNamespace(
            quant_config=None,
            layer_id=0,
            uses_expert_location=False,
            _has_fused_shared=False,
            _map_global_expert_id_to_local_expert_id=Mock(return_value=1),
            _weight_loader_impl=loader,
        )
        target_metadata = Mock()
        target_metadata.logical_to_all_physical.side_effect = AssertionError(
            "draft loading consulted target placement"
        )
        with patch.object(
            module, "get_global_expert_location_metadata", return_value=target_metadata
        ):
            module.FusedMoE.weight_loader(
                expert, param, torch.ones(2, 2), "weight", "w1", 17
            )
        expert._map_global_expert_id_to_local_expert_id.assert_called_once_with(17)
        self.assertEqual(loader.call_args.kwargs["expert_id"], 1)
        target_metadata.logical_to_all_physical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
