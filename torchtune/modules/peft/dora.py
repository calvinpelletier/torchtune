# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import List

import torch.nn.functional as F

from torch import nn, Tensor

from torchtune.modules.peft.peft_utils import AdapterModule


class LoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.adapter_scaling = alpha / rank
        assert not use_bias, "use_bias not yet implemented for DoRA"
        assert not quantize_base, "quantize_base not yet implemented for DoRA"
        weight, bias = self._create_weight_and_bias()
        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter("bias", None)
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.dora_magnitude = nn.Parameter(torch.empty(1, out_dim))
        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        nn.init.ones_(self.dora_magnitude)

    def initialize_dora(self):
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(self.weight, lora_weight, scaling)
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        weight = linear.weight
        bias = None
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight", "dora_magnitude"]
        return adapter_params

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        main_out = F.linear(x, self.weight, self.bias)
        dora_out = self._dora_forward(x)
        return main_out + dora_out

    def _lora_forward(self, x):
        x = self.dropout(x)

        lora_out = self.lora_b(self.lora_a(x))

        # x_eye = torch.eye(self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype)
        # lora_weight = self.lora_b(self.lora_a(x_eye)).T
        lora_weight = self.lora_b.weight @ self.lora_a.weight

        weight = self.weight.to(x.dtype)
        weight_norm = self.get_weight_norm(weight, lora_weight.detach())
        weight_norm = weight_norm.detach()
        mag_norm_scale = (self.dora_magnitude / weight_norm).view(1, -1)
        dora_out = (mag_norm_scale - 1) * (
            F.linear(x, weight)
        ) + mag_norm_scale * lora_out * self.adapter_scaling

        return dora_out

    def _get_weight_norm(self, weight, lora_weight, dtype):
        weight = weight + self.adapter_scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
