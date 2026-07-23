# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Copyright (c) 2013 the respective contributors
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
    MemoryPlanningLine,
    MemoryPlanningState,
    EnterSubgraphLine,
    ExitSubgraphLine,
)
from torch._inductor.utils import cache_on_self
from torch._inductor.virtualized import V
from torch._inductor import config
import torch
import torch._inductor.config


class NPUWrapperCodeGen(PythonWrapperCodegen):
    @staticmethod
    def create(
        is_subgraph,
        subgraph_name,
        parent_wrapper,
        partition_signatures=None,
    ):
        # Upstream create() hardcodes ``return PythonWrapperCodegen()`` for the
        # non-subgraph case, so despite registration it produced a base-class
        # instance and every override here was ignored. Return NPUWrapperCodeGen
        # for the main graph; subgraphs keep the upstream wrapper (not customized).
        if is_subgraph:
            if subgraph_name is None:
                raise ValueError("[triton_experimental] subgraph_name must not be None for subgraph wrapper")
            if parent_wrapper is None:
                raise ValueError("[triton_experimental] parent_wrapper must not be None for subgraph wrapper")
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return NPUWrapperCodeGen()

    def memory_plan_reuse(self):
        # Upstream, but the trailing-pop loop uses getattr(node, "name", None)
        # instead of node.name: the r-axis rsplit path appends a WorkspaceArg
        # (no ``.name`` field) which would raise AttributeError. A WorkspaceArg is
        # never a graph output, so None ∉ out_names pops it (upstream's intent);
        # for a normal Buffer getattr is identical to .name.
        outputs = self.get_graph_outputs()
        out_names = V.graph._get_output_names(outputs)

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            and getattr(getattr(self.lines[-1], "node", None), "name", None)
            not in out_names
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        if len(planning_states) != 0:
            raise RuntimeError("[triton_experimental] planning_states must be empty after memory planning")

    @cache_on_self
    def write_triton_header_once(self) -> None:
        import_str = """
            import triton
            import triton.language as tl
            from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
            from torch_npu._inductor.triton_experimental import npu_triton_heuristics
            """
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)
            self.imports.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    def write_header(self) -> None:
        # After the upstream header, bind empty_strided_npu to the low-overhead
        # torch_npu C++ entry (make_allocation emits it for NPU buffers), bypassing
        # the dispatcher like upstream's _empty_strided_<device>.
        super().write_header()
        if not V.graph.cpp_wrapper:
            self.header.writeline("import torch_npu")
            self.header.writeline(
                "empty_strided_npu = torch_npu._C._empty_strided_npu"
            )

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        # Upstream make_allocation plus ONE branch: NPU buffers allocate via
        # empty_strided_npu() (the low-overhead C++ path; at::empty_strided is
        # ~2us slower per call and backward graphs allocate many buffers/step).
        # Non-NPU branches are byte-identical to upstream.
        if allocation_shape is None:
            allocation_shape = shape

        codegen_shape_tuple = self.codegen_python_shape_tuple(shape)
        codegen_allocation_shape_tuple = self.codegen_python_shape_tuple(
            allocation_shape
        )
        codegen_stride_tuple = self.codegen_python_shape_tuple(stride)
        if torch._inductor.config.test_configs.track_memory_lifecycle:
            out = (
                f"{name} = tracked_empty_strided("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"dtype={dtype}, "
                f"device='{device.type}', "
                f"name='{name}')"
            )
        elif device.type == "cpu" and is_pinned:
            out = (
                f"{name} = empty_strided_cpu_pinned("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype})"
            )
        elif device.type == "npu":
            # optimized NPU path: bypass the dispatcher (torch_npu C++ binding)
            out = (
                f"{name} = empty_strided_npu("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype})"
            )
        elif device.type in ("cpu", "cuda", "xpu", "mtia"):
            # optimized path for faster allocations, saving ~2us versus the stuff below
            out = (
                f"{name} = empty_strided_{device.type}("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype})"
            )
        # all other devices:
        else:
            out = (
                f"{name} = empty_strided("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"device='{device.type}', dtype={dtype})"
            )
        if codegen_shape_tuple != codegen_allocation_shape_tuple:
            # need an extra as_strided call
            out = out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"
        return out
