import logging
import operator
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set

import sympy

import torch
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import dynamo_timed
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._mode_utils import no_dispatch

from torch._dynamo import config as dynamo_config

from torch._inductor import config, ir
from torch._inductor.codegen.wrapper import CppWrapperCodeGen, WrapperCodeGen
from torch._inductor.exc import (
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)
from torch._inductor.ir import (
    Constant,
    FixedLayout,
    InputBuffer,
    Pointwise,
    Reduction,
    TensorBox,
)
from torch._inductor.lowering import (
    FALLBACK_ALLOW_LIST,
    layout_constraints,
    lowerings,
    make_fallback,
    needs_realized_inputs,
)
from torch._inductor.sizevars import CppSizeVarAllocator, SizeVarAllocator
from torch._inductor.utils import (
    convert_shape_to_inductor,
    gather_origins,
    get_dtype_size,
    sympy_product,
)
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from .schedule import WelderScheduler

log = logging.getLogger(__name__)

output_triton_code_file = "triton_code.py"


class WelderGraphLowering(GraphLowering):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        shape_env=None,
        num_static_inputs=None,
        graph_id=None,
    ):
        super().__init__(gm, shape_env, num_static_inputs, graph_id)

    def codegen(self):
        self.init_wrapper_code()

        self.scheduler = WelderScheduler(self.buffers)
        assert self.scheduler is not None  # mypy can't figure this out
        self.scheduler.codegen()
        assert self.wrapper_code is not None
        return self.wrapper_code.generate()

    @dynamo_timed
    def compile_to_module(self):
        from torch._inductor.codecache import PyCodeCache

        code = self.codegen()
        mod = PyCodeCache.load(code)
        for name, value in self.constants.items():
            setattr(mod, name, value)

        # save code
        codefile = open(output_triton_code_file, "w")
        codefile.write(code)
        codefile.close()

        return mod

    def compile_to_fn(self):
        return self.compile_to_module().call

    # V.debug.output_code(mod.__file__)
