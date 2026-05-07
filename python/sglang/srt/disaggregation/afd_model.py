"""Pass-through stubs used on the AFD expert (FFN) role.

When `disaggregation_mode == "expert"`, the model is constructed with these
stubs in place of the real attention/Mamba modules. Expert-side forward only
needs to drive the MoE blocks; everything else is a no-op pass-through.
"""

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AFDProxyAttention(nn.Module):
    """Stand-in for self-attention layers on the expert role."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return hidden_states


class AFDProxyLinearAttention(nn.Module):
    """Stand-in for Mamba / GatedDeltaNet layers on the expert role."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return hidden_states
