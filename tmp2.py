import math
from copy import deepcopy

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4

from torchtune.modules.peft.lora import LoRALinear
from torchtune.modules.peft.peft_utils import get_merged_lora_ckpt


IN_DIM = 256
OUT_DIM = 256
RANK = 2
ALPHA = 1.


def main():
    for dropout in (0., 0.1):
        for use_bias in (False, True):
            for quantize_base in (False, True):
                for decompose in (False, True):
                    compare_lora(dropout, use_bias, quantize_base, decompose)


def compare_lora(dropout, use_bias, quantize_base, decompose):
    if use_bias and quantize_base:
        # not supported yet
        return
    print(dropout, use_bias, quantize_base, decompose)

    m1 = LoraLinearReference(
        IN_DIM, OUT_DIM, RANK, ALPHA, 
        dropout=dropout, use_bias=use_bias, quantize_base=quantize_base, decompose=decompose,
    )
    m2 = LoRALinear(
        IN_DIM, OUT_DIM, RANK, ALPHA, 
        dropout=dropout, use_bias=use_bias, quantize_base=quantize_base, decompose=decompose,
    )

    sd = m1.state_dict()
    if quantize_base:
        sd['weight'] = sd['weight'].to(torch.float32)
    m2.load_state_dict(sd)
    
    m1.weight.requires_grad_(False)
    m2.weight.requires_grad_(False)
    if use_bias:
        m1.bias.requires_grad_(False)
        m2.bias.requires_grad_(False)
    
    if decompose:
        m1.initialize_dora()
        m2.initialize_dora()    
    
    compare_models(m1, m2, use_bias, quantize_base, decompose)
    
    opt1 = torch.optim.Adam(m1.parameters())
    opt2 = torch.optim.Adam(m2.parameters())
    opt1.zero_grad()
    opt2.zero_grad()

    x = torch.randn(8, IN_DIM)
    y = torch.randn(8, OUT_DIM)
    torch.manual_seed(0)
    y1 = m1(x.detach())
    torch.manual_seed(0)
    y2 = m2(x.detach())
    F.mse_loss(y1, y.detach()).backward()
    F.mse_loss(y2, y.detach()).backward()
    assert torch.equal(y1, y2)
    if decompose:
        assert torch.equal(m1.lora_magnitude.grad, m2.lora_magnitude.grad)
    assert torch.equal(m1.lora_a.weight.grad, m2.lora_a.weight.grad)
    assert torch.equal(m1.lora_b.weight.grad, m2.lora_b.weight.grad)

    opt1.step()
    opt2.step()
    compare_models(m1, m2, use_bias, quantize_base, decompose)

    sd = get_merged_lora_ckpt(Wrapper(m2).state_dict(), RANK, ALPHA, decompose)
    m3 = Wrapper(nn.Linear(IN_DIM, OUT_DIM, bias=use_bias))
    m3.load_state_dict(sd)
    m3 = m3.module

    m2.eval()
    m3.eval()
    with torch.no_grad():
        x = torch.randn(8, IN_DIM)
        y2 = m2(x)
        y3 = m3(x)
        assert torch.allclose(y2, y3, rtol=1e-4, atol=1e-6)


def compare_models(m1, m2, use_bias, quantize_base, decompose):
    if quantize_base:
        assert torch.equal(m1.weight.to(torch.float32), m2.weight.to(torch.float32))
    else:
        assert torch.equal(m1.weight, m2.weight)
    if use_bias:
        assert torch.equal(m1.bias, m2.bias)
    assert torch.equal(m1.lora_a.weight, m2.lora_a.weight)
    assert torch.equal(m1.lora_b.weight, m2.lora_b.weight)
    if decompose:
        assert torch.equal(m1.lora_magnitude, m2.lora_magnitude)


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)
    

class LoraLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
        decompose: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self.use_bias = use_bias
        self.quantize_base = quantize_base
        self.decompose = decompose

        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False

        weight, bias = self._create_weight_and_bias()
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )

        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        if decompose:
            self.lora_magnitude = nn.Parameter(torch.empty(1, out_dim))

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
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        if self.decompose:
            # NOTE: This initialization is just a fallback. The magnitude is initialized
            # after loading the base model weights in `self.initialize_dora()`.
            nn.init.ones_(self.lora_magnitude)

    def initialize_dora(self) -> None:
        base_weight = self.weight.to(torch.float32)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(base_weight, lora_weight)
        self.lora_magnitude.data = weight_norm

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self.quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            if self.quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def _get_weight_norm(self, base_weight, lora_weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = base_weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        base_out = self._base_forward(x)
        if self.disabled:
            return base_out

        x = self.dropout(x)
        lora_out = self.scaling * self.lora_b(self.lora_a(x))
        if self.decompose:
            lora_out = self._dora_forward(x, lora_out)
        return base_out + lora_out

    def _base_forward(self, x):
        if self.quantize_base:
            return linear_nf4(input=x, weight=self.weight)
        return F.linear(x, self.weight, self.bias)
    
    def _dora_forward(self, x, lora_out):
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        base_weight = self.weight.to(x.dtype)
        weight_norm = self._get_weight_norm(base_weight, lora_weight.detach()).detach()

        mag_norm_scale = (self.lora_magnitude / weight_norm).view(1, -1)
        base_out = F.linear(x, base_weight)
        dora_out = (mag_norm_scale - 1) * base_out + mag_norm_scale * lora_out
        return dora_out


class LoraLinearReference(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
        decompose: bool = False,
    ):
        super().__init__()
        self.use_bias = use_bias
        self.quantize_base = quantize_base
        self.decompose = decompose
        
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not quantize_base else to_nf4(linear.weight)
        bias = None
        if use_bias:
            if quantize_base:
                raise NotImplementedError()
            bias = linear.bias
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )

        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank
        if decompose:
            self.lora_magnitude = nn.Parameter(torch.empty(1, out_dim))
        self.dropout = nn.Dropout(p=dropout)

    def initialize_dora(self) -> None:
        lora_a = self.lora_a.weight
        lora_b = self.lora_b.weight

        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_a.dtype == torch.float16
        if dtype_is_fp16:
            lora_a = lora_a.float()
            lora_b = lora_b.float()

        # weight = dequantize_module_weight(base_layer)
        # weight = self.base_layer.weight
        weight = self.weight.to(self.lora_a.weight.dtype)
        
        lora_weight = lora_b @ lora_a

        if dtype_is_fp16:
            lora_weight = lora_weight.half()
        weight_norm = self._get_weight_norm(weight, lora_weight)

        self.lora_magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self._base_forward(x)
        torch_result_dtype = result.dtype
        x = x.to(self.lora_a.weight.dtype)
        x = self.dropout(x)
        if not self.decompose:
            result = result + self.lora_b(self.lora_a(x)) * self.scaling
        else:
            result = result + self._dora_forward(x)
        result = result.to(torch_result_dtype)
        return result

    def _base_forward(self, x):
        if self.quantize_base:
            return linear_nf4(input=x, weight=self.weight)
        return F.linear(x, self.weight, self.bias)
    
    def _dora_forward(self, x):
        lora_result = self.lora_b(self.lora_a(x))

        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        x_eye = torch.eye(self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype)
        lora_weight = self.lora_b(self.lora_a(x_eye)).T

        magnitude = self.lora_magnitude
        
        # weight = dequantize_module_weight(base_layer)
        # weight = self.base_layer.weight
        weight = self.weight.to(x.dtype)
        
        weight = weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight.detach())
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, weight)
        ) + mag_norm_scale * lora_result * self.scaling

        # Note: Computation could potentially be accelerated by using the code below instead of calculating X@W again.
        # This is only correct if dropout=0, otherwise results will differ:
        # https://github.com/huggingface/peft/pull/1474#issuecomment-1964682771
        # bias = self.get_base_layer().bias
        # if bias is not None:
        #     result = result - bias
        # result = mag_norm_scale * result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        # if bias is not None:
        #     result = result + bias

        return result_dora

    def _get_weight_norm(self, weight, lora_weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
        
        
if __name__ == '__main__':
    main()
