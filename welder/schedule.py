import collections
import dataclasses
import functools
import itertools
import os
import pprint
import textwrap
from typing import Dict, List, Optional, Set
from overrides import override
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, dependencies, ir, metrics
from torch._inductor.dependencies import StarDep, WeakDep
from torch._inductor.sizevars import SimplifyIndexing
from torch._inductor.utils import cache_on_self, cmp, has_triton
from torch._inductor.virtualized import V
from torch._inductor.scheduler import (
    Scheduler,
    FusedSchedulerNode,
    SchedulerNode,
    NopKernelSchedulerNode,
    ExternKernelSchedulerNode,
    BaseSchedulerNode,
    pformat,
)

from loguru import logger


class WelderSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: "Scheduler", node: ir.ComputedBuffer, group_fn):
        super().__init__(scheduler, node)
        (
            self._sizes,
            self._body,
        ) = node.simplify_and_reorder()

        self.group = (node.get_device(), group_fn(self._sizes))

        if self.is_template():
            self.set_read_writes(node.normalized_read_writes())
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=True
                )
            )

        if self.is_reduction():
            # reduction has last (reduced) dim in its sizes, and some
            # downstream dependencies get confused by it
            self.read_writes.writes = self.read_writes.writes | {
                w.strip_last_size() for w in self.read_writes.writes
            }
            # reduction not on the last dim swaps the sizes, and downstream
            # dependencies expect unswapped
            # TODO swapping sizes doesn't work, leads to
            # File "/scratch/ngimel/work/repos/torchdynamo/torchinductor/sizevars.py", line 130, in guard_equals
            # if len(right.free_symbols) < len(left.free_symbols):
            # AttributeError: 'int' object has no attribute 'free_symbols'
            # even though memory dep looks correct
            # self.read_writes.writes = self.read_writes.writes | {
            #     w.maybe_swap_sizes() for w in self.read_writes.writes
            # }

    def debug_str_extra(self):
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        if self.get_aliases():
            lines.append(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            lines.append(f"{name}.mutations = {pformat(self.get_mutations())}")
        if isinstance(self._body, ir.LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))
        return "\n".join(lines)

    def get_ranges(self):
        return self._sizes

    def is_reduction(self):
        return bool(self.node.get_reduction_type())

    def is_template(self):
        return isinstance(self.node, ir.TemplateBuffer)

    def run(self, *index_vars):
        self.mark_run()
        self.codegen(index_vars)

    def mark_run(self):
        self.allocate()

    def ranges_from_index_vars(self, index_vars):
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars):
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with V.set_ops_handler(
                SimplifyIndexing(V.get_ops_handler(), var_ranges)
            ), V.kernel.set_current_node(self):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_read_writes(self):
        """
        Get the memory dependencies in the non-reduction axis.
        """
        sizes, reduction_sizes = self._sizes

        def fn(index):
            return self._body(index, [sympy.Integer(0) for _ in reduction_sizes])

        return dependencies.extract_read_writes(fn, sizes)

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False
        if len(self.read_writes.writes) == 1 and hasattr(read_dep, "index"):
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False


class WelderScheduler(Scheduler):
    # @override
    def fuse_nodes(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.
        """
        # for _ in range(10):
        #   old_len = len(self.nodes)
        #    self.fuse_nodes_once()
        #    if len(self.nodes) == old_len:
        #        break
        # TODO(wenxh): Replace this by using fuse_nodes_using_welder_result();

    def fuse_nodes_using_welder_result(self):
        pass

    def fuse_nodes_once(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuses(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        fused_nodes = set(self.nodes)
        for node1, node2 in self.get_possible_fusions():
            node1 = self.name_to_fused_node[node1.get_first_name()]
            node2 = self.name_to_fused_node[node2.get_first_name()]
            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                node3 = FusedSchedulerNode.fuse(node1, node2)
                fused_nodes.remove(node1)
                fused_nodes.remove(node2)
                fused_nodes.add(node3)
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )
        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        self.topological_sort_schedule()
        self.prune_redundant_deps()

    @dynamo_timed
    @override
    def codegen(self):
        for node in self.nodes:
            self.buffer_names_no_longer_needed.update(node.last_usage)

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if device.type == "cuda":
                        if self.current_device and self.current_device.type == "cuda":
                            V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                        assert device.index is not None, "device should have an index"
                        V.graph.wrapper_code.codegen_cuda_device_guard_enter(
                            device.index
                        )
                    elif self.current_device and self.current_device.type == "cuda":
                        V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                    self.current_device = device

            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                node, *epilogue = node.get_nodes()
                self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                self.codegen_extern_call(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_nodes(node.get_nodes())
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.allocate()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_names())

        self.flush()
