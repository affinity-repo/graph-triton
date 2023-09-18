import torch
import sys
import logging
from torch._dynamo import register_backend
import typing
from typing import List, Dict, Any, Optional
import functools
import functorch
from functorch.compile import min_cut_rematerialization_partition
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    BoxedBool,
    _graph_counter,
    count_bytes_inner,
    count_tangents,
    is_tf32_warning_applicable,
    _warn_tf32_disabled,
    _step_logger,
    _shape_env_from_inputs,
    complex_memory_overlap,
    cudagraphify,
    align_inputs
)
from torch._inductor import config, overrides, pattern_matcher
import torch._dynamo.config as dynamo_config
from torch._dynamo import logging as dynamo_logging, utils as dynamo_utils
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.mkldnn import convert_outplace_to_inplace
from torch._inductor.decomposition import select_decomp_table
from torch._functorch.aot_autograd import make_boxed_func
from torch._inductor.virtualized import V
from torch._dynamo.utils import fake_mode_from_tensors
from torch._inductor.utils import has_incompatible_cudagraph_ops, developer_warning
from torch._inductor.graph import GraphLowering
from torch._inductor.debug import DebugContext

from .graph import WelderGraphLowering

@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
def welder_compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs=None,
    num_fixed=0,
    is_backward=False,
    graph_id=None,
):
    if is_tf32_warning_applicable(gm):
        _warn_tf32_disabled()

    if dynamo_utils.count_calls(gm.graph) == 0:
        return make_boxed_func(gm.forward)

    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    _step_logger()(
        logging.INFO,
        "torchinductor compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )
    V.debug.fx_graph(gm, example_inputs)

    if cudagraphs is None:
        cudagraphs = config.triton.cudagraphs

    shape_env = _shape_env_from_inputs(example_inputs)
    fake_mode = fake_mode_from_tensors(
        example_inputs
    ) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)

    with V.set_fake_mode(fake_mode):
        pattern_matcher.fx_passes(gm)
        V.debug.fx_graph_transformed(gm, example_inputs)

        graph = WelderGraphLowering(
            gm,
            shape_env=shape_env,
            num_static_inputs=num_fixed,
            graph_id=graph_id,
        )
        with V.set_graph_handler(graph):
            graph.run(*example_inputs)
            compiled_fn = graph.compile_to_fn()

    if cudagraphs:
        complex_memory_overlap_inputs = any(
            complex_memory_overlap(t) for t in example_inputs
        )

        if (
            set(graph.device_types) == {"cuda"}
            and not graph.mutated_inputs
            and not has_incompatible_cudagraph_ops(gm)
            and not complex_memory_overlap_inputs
        ):
            compiled_fn = cudagraphify(
                compiled_fn, example_inputs, static_input_idxs=range(num_fixed)
            )
        else:
            BoxedBool.disable(cudagraphs)

            if len(set(graph.device_types)) > 1:
                developer_warning("skipping cudagraphs due to multiple devices")
            elif set(graph.device_types) == {"cuda"}:
                if graph.mutated_inputs:
                    developer_warning("skipping cudagraphs due to input mutation")
                elif complex_memory_overlap_inputs:
                    developer_warning(
                        "skipping cudagraphs due to complex input striding"
                    )

    result = align_inputs(compiled_fn, example_inputs, range(num_fixed))
    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # aot autograd needs to know to pass in inputs as a list
    result._boxed_call = True
    return result


def welder_compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    inner_compile=welder_compile_fx_inner,
    config_patches: Optional[Dict[str, Any]] = None,
):
    """Main entrypoint to a compile given FX graph"""
    if config_patches:
        with config.patch(config_patches):
            return compile_fx(
                model_,
                example_inputs_,
                # need extra layer of patching as backwards is compiled out of scope
                inner_compile=config.patch(config_patches)(inner_compile),
            )

    assert not config._raise_error_for_testing

    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = True

    with overrides.patch_functions():
        model_ = overrides.replace_fx(model_)
        model_ = overrides.fuse_fx(model_, example_inputs_)
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(
        config.triton.cudagraphs and not dynamo_config.dynamic_shapes
    )

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        # Why convert outplace op to inplace? Inductor can support inplace operations well and for custom
        # inplace ops which are lowered as ExternKernel, it is beneficial to performance when the inplace
        # implementation is used if available.
        model = convert_outplace_to_inplace(model)
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
        )

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            is_backward=True,
            graph_id=graph_id,
        )

    with overrides.patch_functions():
        # TODO: can add logging before/after the call to create_aot_dispatcher_function
        # in torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
        # once torchdynamo is merged into pytorch
        return aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
            keep_inference_input_mutations=True,
        )(model_, example_inputs_)


@register_backend
def welder(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    new_fx = welder_compile_fx(gm, example_inputs, **kwargs)
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
