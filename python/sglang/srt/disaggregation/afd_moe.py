"""AFD-specific MoE classes used at MoE positions in the model.

- AFDATTNMoE runs on prefill / decode (attn role). It owns no expert weights;
  its forward computes routing locally (gate + topk happen above this in
  Qwen2MoeSparseMoeBlock.forward_afd), then ships the result over the wire to
  the expert pool and waits for the post-MoE hidden states back.

- AFDFFNMoE runs on the expert role. It subclasses FusedMoE so the standard
  weight loader and quant_method machinery work normally. forward is overridden
  to:
    1. block on the AFD dispatcher ffn_recv (per-layer, per-source-arena)
    2. run grouped GEMM via run_moe_core (same path as DeepEPMoE)
    3. ship the post-MoE hidden states back via ffn_send

Both classes hold a per-layer AFD dispatcher instance, set lazily at first use
(the dispatcher needs the global Mooncake-EP buffer + bootstrap to be ready,
which happens after model construction).
"""

from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _get_layer_dispatcher(layer_id):
    from sglang.srt.disaggregation.afd_dispatcher import get_or_create_layer_dispatcher
    return get_or_create_layer_dispatcher(layer_id)


class AFDATTNMoE(nn.Module):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        layer_id,
        num_fused_shared_experts=0,
        params_dtype=None,
        quant_config=None,
        prefix="",
        activation="silu",
        routed_scaling_factor=None,
        **kwargs,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self._dispatcher = None

    def _ensure_dispatcher(self):
        if self._dispatcher is None:
            self._dispatcher = _get_layer_dispatcher(self.layer_id)
        return self._dispatcher

    def forward(self, hidden_states, topk_output, forward_batch=None):
        d = self._ensure_dispatcher()
        d.attn_send(hidden_states, topk_output)
        return d.attn_recv()


class AFDFFNMoE(FusedMoE):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        layer_id,
        num_fused_shared_experts=0,
        params_dtype=None,
        quant_config=None,
        prefix="",
        activation="silu",
        routed_scaling_factor=None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        self._dispatcher = None

    def _ensure_dispatcher(self):
        if self._dispatcher is None:
            self._dispatcher = _get_layer_dispatcher(self.layer_id)
        return self._dispatcher

    def forward(self, hidden_states, topk_output):
        d = self._ensure_dispatcher()
        dispatch_output = d.ffn_recv()
        combine_input = self.run_moe_core(dispatch_output)
        d.ffn_send(combine_input)
        return hidden_states
