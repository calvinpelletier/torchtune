from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLI:
    def dora(self):
        m = DoraLinear(8, 4, 2, 1., dropout=0., decompose=True)
        m.dora_init()
        x = torch.randn(16, 8)
        y = m(x)
        print(y.shape)
        y_ = torch.randn(16, 4)
        loss = F.mse_loss(y, y_)
        loss.backward()


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
        self.base_layer = nn.Linear(in_dim, out_dim, bias=False)
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
        weight = self.base_layer.weight
        
        lora_weight = lora_b @ lora_a

        if dtype_is_fp16:
            lora_weight = lora_weight.half()
        weight_norm = self._get_weight_norm(weight, lora_weight)

        self.magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        torch_result_dtype = result.dtype
        x = x.to(self.lora_a.weight.dtype)
        x = self.dropout(x)
        if not self.decompose:
            result = result + self.lora_b(self.lora_a(x)) * self.scaling
        else:
            result = result + self._dora_forward(x)
        result = result.to(torch_result_dtype)
        return result
    
    def _dora_forward(self, x):
        lora_result = self.lora_b(self.lora_a(x))

        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        x_eye = torch.eye(self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype)
        lora_weight = self.lora_b(self.lora_a(x_eye)).T

        magnitude = self.magnitude
        
        # weight = dequantize_module_weight(base_layer)
        weight = self.base_layer.weight
        
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
        self.base_layer = nn.Linear(in_dim, out_dim, bias=False)
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
        weight = self.base_layer.weight
        
        lora_weight = lora_b @ lora_a

        if dtype_is_fp16:
            lora_weight = lora_weight.half()
        weight_norm = self._get_weight_norm(weight, lora_weight)

        self.magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        torch_result_dtype = result.dtype
        x = x.to(self.lora_a.weight.dtype)
        x = self.dropout(x)
        if not self.decompose:
            result = result + self.lora_b(self.lora_a(x)) * self.scaling
        else:
            result = result + self._dora_forward(x)
        result = result.to(torch_result_dtype)
        return result
    
    def _dora_forward(self, x):
        lora_result = self.lora_b(self.lora_a(x))

        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        x_eye = torch.eye(self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype)
        lora_weight = self.lora_b(self.lora_a(x_eye)).T

        magnitude = self.magnitude
        
        # weight = dequantize_module_weight(base_layer)
        weight = self.base_layer.weight
        
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
    Fire(CLI)
