from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


IN_DIM = 256
OUT_DIM = 256


def main():
    for dropout in (0., 0.1):
        for use_bias in (False, True):
            for quantize_base in (False, True):
                compare_dora(dropout, use_bias, quantize_base)


def compare_dora(dropout, use_bias, quantize_base):
    if use_bias and quantize_base:
        # not supported yet
        return
    print(dropout, use_bias, quantize_base)

    m1 = DoraLinear(
        IN_DIM, OUT_DIM, 2, 1., 
        dropout=dropout, use_bias=use_bias, quantize_base=quantize_base, decompose=True,
    )
    m2 = DoraLinear2(
        IN_DIM, OUT_DIM, 2, 1., 
        dropout=dropout, use_bias=use_bias, quantize_base=quantize_base, decompose=True,
    )
    m2.load_state_dict(m1.state_dict())
    m1.weight.requires_grad_(False)
    m2.weight.requires_grad_(False)
    if use_bias:
        m1.bias.requires_grad_(False)
        m2.bias.requires_grad_(False)
    if quantize_base:
        assert torch.equal(m1.weight.to(torch.float32), m2.weight.to(torch.float32))
    else:
        assert torch.equal(m1.weight, m2.weight)
    if use_bias:
        assert torch.equal(m1.bias, m2.bias)
    assert torch.equal(m1.lora_a.weight, m2.lora_a.weight)
    assert torch.equal(m1.lora_b.weight, m2.lora_b.weight)
    
    m1.dora_init()
    m2.dora_init()
    assert torch.equal(m1.magnitude, m2.magnitude)

    x = torch.randn(8, IN_DIM)
    y = torch.randn(8, OUT_DIM)
    torch.manual_seed(0)
    y1 = m1(x.detach())
    torch.manual_seed(0)
    y2 = m2(x.detach())
    F.mse_loss(y1, y.detach()).backward()
    F.mse_loss(y2, y.detach()).backward()
    assert torch.equal(y1, y2)
    assert torch.equal(m1.magnitude.grad, m2.magnitude.grad)
    assert torch.equal(m1.lora_a.weight.grad, m2.lora_a.weight.grad)
    assert torch.equal(m1.lora_b.weight.grad, m2.lora_b.weight.grad)


class DoraLinear2(nn.Module):
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
        self.magnitude = nn.Parameter(torch.empty(1, out_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.decompose = decompose

    def dora_init(self) -> None:
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

        self.magnitude = nn.Parameter(weight_norm, requires_grad=True)

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

        magnitude = self.magnitude
        
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


class DoraLinear(nn.Module):
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
        self.magnitude = nn.Parameter(torch.empty(1, out_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.decompose = decompose

    def dora_init(self) -> None:
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

        self.magnitude = nn.Parameter(weight_norm, requires_grad=True)

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

        magnitude = self.magnitude
        
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
