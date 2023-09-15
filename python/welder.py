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
    print("Using welder backend")
    return compile_fx(gm, example_inputs, **kwargs)

#def fn(x, y):
#    return torch.add(x, y)

#new_fn = torch.compile(fn, backend='welder')
#input_tensor = torch.randn(2, 3).cuda()
#print(new_fn(input_tensor, input_tensor))
