"""Replica loading and cached execution storage must agree with placement."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata


class TestReplicaLoading(unittest.TestCase):
    def test_loading_uses_all_resident_copies_not_dispatch_subset(self):
        physical = torch.tensor([[0, 1, 0, 2, 3, 2]])
        # The nearest-replica dispatch map intentionally omits two resident slots.
        logical = torch.full((1, 4, 6), -1, dtype=torch.int64)
        logical[0, :, 0] = torch.tensor([0, 1, 3, 4])
        metadata = ExpertLocationMetadata(
            physical_to_logical_map=physical,
            physical_to_logical_map_cpu=physical,
            logical_to_all_physical_map=logical,
            logical_to_all_physical_map_cpu=logical,
            logical_to_all_physical_map_num_valid=(logical != -1).sum(-1),
            ep_size=2,
            logical_to_rank_dispatch_physical_map=None,
        )
        loaded = torch.full((6,), -1)
        for expert in range(4):
            loaded[metadata.logical_to_all_physical(0, expert)] = expert
        torch.testing.assert_close(loaded, physical[0])
        self.assertEqual(metadata.logical_to_all_physical(0, 0), [0, 2])
        self.assertEqual(metadata.logical_to_all_physical(1, 2), [2])
        self.assertEqual(metadata.logical_to_all_physical(0, 0, True), [0, 4])
        with self.assertRaises(IndexError):
            metadata.logical_to_all_physical(0, 4)


class TestMegaMoEStorage(unittest.TestCase):
    def test_relocation_covers_cached_scales_and_preserves_shared_slots(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        experts = torch.nn.Module()
        experts.num_local_experts = 5
        for name in ("w13_weight", "w2_weight"):
            experts.register_parameter(
                name, torch.nn.Parameter(torch.arange(20).reshape(5, 4).float())
            )
            experts.register_parameter(
                name + "_scale_inv",
                torch.nn.Parameter(torch.arange(20).reshape(5, 2, 2).float()),
            )
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.detach().clone(),
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.detach().clone(),
        )
        layer = SimpleNamespace(experts=experts, num_fused_shared_experts=1)
        tensors = DeepseekV2MoE.get_moe_weights(layer)
        self.assertEqual(len(tensors), 6)
        self.assertTrue(all(t.shape[0] == 4 for t in tensors))
        originals = [t.clone() for t in tensors]
        pointers = [t.data_ptr() for t in tensors]
        shared = [
            pair[1][-1].clone()
            for pair in (experts.mega_l1_weights, experts.mega_l2_weights)
        ]
        perm = torch.tensor([3, 2, 1, 0])
        for tensor, original in zip(tensors, originals):
            tensor.copy_(original[perm])
        for pair, original_shared in zip(
            (experts.mega_l1_weights, experts.mega_l2_weights), shared
        ):
            torch.testing.assert_close(
                pair[1][:4], torch.arange(20).reshape(5, 2, 2).float()[:4][perm]
            )
            torch.testing.assert_close(pair[1][-1], original_shared)
        self.assertEqual([t.data_ptr() for t in tensors], pointers)
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        self.assertEqual(len(DeepseekV2MoE.get_moe_weights(layer)), 5)
        del experts.mega_l1_weights, experts.mega_l2_weights
        self.assertEqual(len(DeepseekV2MoE.get_moe_weights(layer)), 4)


if __name__ == "__main__":
    unittest.main()
