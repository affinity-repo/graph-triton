import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, List, Optional, Tuple, Union

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._prims_common import (
    canonicalize_dim,
    canonicalize_dims,
    check,
    dtype_to_type,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    Number,
    type_to_dtype,
)
from torch.fx.experimental.symbolic_shapes import magic_methods, method_to_operator
from torch.utils._pytree import tree_flatten
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch._dynamo.utils import import_submodule

from torch._inductor import config, inductor_prims, ir, test_operators  # NOQA: F401
from torch._inductor.decomposition import decompositions, get_decompositions
from torch._inductor.ir import (
    ExpandView,
    IndexingConstant,
    is_triton,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    SqueezeView,
    TensorBox,
    validate_ir,
    View,
)
from torch._inductor.utils import ceildiv, decode_device, pad_listlike, sympy_product
from torch._inductor.virtualized import ops, V

from torch._inductor.lowering import (
    mul,
    clone,
    promote_constants,
    log,
    lowerings,
    layout_constraints,
    fallbacks,
    aten,
    tr_c10d,
    prims,
    needs_realized_inputs,
    foreach_ops,
)


def make_pointwise(
    fn,
    override_return_dtype=None,
    override_device=None,
    override_fn_when_input_bool=None,
    override_fn_when_cuda_float64=None,
    allow_alpha=False,
):
    def inner(*inputs: List[TensorBox], alpha=None):
        inputs = promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            assert alpha is None
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_cuda = decode_device(inputs[0].get_device()).type == "cuda"

        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        def inner_fn(index):
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif override_fn_when_cuda_float64 and is_cuda and dtype == torch.float64:
                return override_fn_when_cuda_float64(*[load(index) for load in loaders])
            else:
                return fn(*[load(index) for load in loaders])

        if not override_device:
            device = None
            for i in inputs:
                if i.get_device().type == "cuda":
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        device = override_device or device

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner


def make_foreach_pointwise(pw_fn, allow_alpha=False):
    def inner(*inputs: List[List[TensorBox]], alpha=1):
        def is_dynamic(*args):
            return any(
                isinstance(t, TensorBox)
                and any(x.free_symbols for x in t.data.get_size())  # type: ignore[attr-defined]
                for t in args
            )

        # group by device, whether any of the inputs are dynamic, and whether their types match
        # (proxy for type promotion)
        def group_args(arg_pairs):
            out = defaultdict(list)
            for i, args in enumerate(arg_pairs):
                use_foreach = not is_dynamic(*args)
                device = None
                for t in args:
                    if isinstance(t, TensorBox):
                        device = t.data.get_device()  # type: ignore[attr-defined]
                        break
                assert (
                    device is not None
                ), "foreach op should have at least one tensor arg"
                out[(device, use_foreach)].append((i, args))
            return out

        realize_outputs = False
        for node in V.graph.current_node.users:
            for user in node.users:
                if not (user.op == "call_function" and user.target in foreach_ops):
                    realize_outputs = True

        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        assert (
            a_list_input is not None
        ), "at least one input must be a list to a foreach op"

        # broadcast scalar inputs to match length of list inputs
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                broadcast_inputs.append(input)

        groups = group_args(zip(*broadcast_inputs))

        outputs = [None] * len(a_list_input)
        for (device, use_foreach), group in groups.items():
            buffer_list = []
            for (
                output_ind,
                args,
            ) in group:
                if allow_alpha:
                    output = pw_fn(*args, alpha=alpha)
                else:
                    output = pw_fn(*args)

                outputs[output_ind] = output

                if device.type == "cuda" and use_foreach and realize_outputs:
                    buffer_list.append(output.realize())

            if buffer_list:
                V.graph.register_list(buffer_list)

        assert all(x is not None for x in outputs)
        return outputs

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype, copy=False):
    if x.get_dtype() == dtype:
        return clone(x) if copy else x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype)

    return make_pointwise(_to_dtype, override_return_dtype=dtype)(x)
