import torch
from torch._dynamo import register_backend
import typing
from typing import List

from triton_kernels import bmm as bmm
from triton_kernels import mm as mm

@register_backend
def welder(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx
    new_fx = compile_fx(gm, example_inputs, **kwargs)
    # do welder optimize here 
    return new_fx

# Supported patterns more than torch.inductor
# [ ] Gemm + Pointwise + Gemm
# [ ] Conv + Pointwise
# [ ] Gemm + Pointwise
# [ ] Pointwise + Gemm/Conv

# Workflow
# 1. Setup model + input tensor
# 2. Generate Schedule + Read Welder Schedule
# 3. Generate Python/Triton Code
# 4. Compile, Run and Get Pefromance
