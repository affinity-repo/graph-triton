import torch
import sys
import logging
import typing
from typing import List, Dict, Any, Optional
import functools
import functorch
from torch._inductor import config, pattern_matcher, ir
import torch._dynamo.config as dynamo_config
from torch._inductor.virtualized import V
from torch._inductor.utils import has_incompatible_cudagraph_ops, developer_warning
from torch._inductor.graph import GraphLowering
from torch._inductor.debug import DebugContext
from torch._inductor.ir import (
    IRNode,
    Loops,
    LoopBody,
    LoopBodyBlock,
    Pointwise,
    Scatter,
    Reduction,
    TileHint,
    Reduction,
    WelfordReduction,
    BaseView,
    ExpandView,
    PermuteView,
    SqueezeView,
    GenericView,
    View,
    ReinterpretView,
    SliceView,
    BaseConstant,
    Constant,
    IndexingConstant,
    Layout,
    FixedLayout,
    FlexibleLayout,
    AliasedLayout,
    MutationLayout,
    Buffer,
    InputBuffer,
    ConstantBuffer,
    NoneAsConstantBuffer,
    ShapeAsConstantBuffer,
    ComputedBuffer,
    TemplateBuffer,
    InputsKernel,
    NopKernel,
    ConcatKernel,
    ExternKernel,
    ExternKernelOut,
    RandomSeeds,
    ExternKernelAlloc,
    InplaceBernoulliFallback,
    ScatterFallback,
    IndexPutFallback,
    DeviceCopy,
    DynamicScalar,
    FallbackKernel,
    MultiOutputLayout,
    MultiOutput,
    ConvolutionUnary,
    ConvolutionBinary,
    ConvolutionBinaryInplace,
    LinearUnary,
    LinearBinary,
    MutableBox,
    TensorBox,
    StorageBox,
    Wait,
    CollectiveKernel,
    InPlaceCollectiveKernel,
    OutOfPlaceCollectiveKernel,
    InPlaceHint,
    OutputBuffer,
    AllReduce,
    AllGatherIntoTensor,
    ReduceScatterTensor,
    InterpreterShim,
    InputsKernel,
    MultiOutput,
    TemplateBuffer,
    ComputedBuffer,
)

from torch.utils._sympy.functions import (
    FloorDiv,
    ModularIndexing,
    Mod,
    CleanDiv,
    CeilDiv,
    LShift,
    RShift,
)
import hashlib
import sympy
from torch._inductor.codegen.triton import TritonKernel
from affinity.inductor.lowering import make_pointwise
from torch._inductor.graph import GraphLowering
from torch.fx import GraphModule, symbolic_trace


def test_graph_module():
    def SelfAttention(Q: torch.tensor, K: torch.tensor, V: torch.tensor):
        P = torch.matmul(Q, K.t())
        S = torch.nn.functional.softmax(P)
        attn = torch.matmul(S, V)
        return attn

    gm = symbolic_trace(
        SelfAttention,
        concrete_args={},  # for control flow variable
    )

    temp_tensor = torch.rand(128, 128).cuda()
    out = gm(temp_tensor, temp_tensor, temp_tensor)


test_graph_module()
