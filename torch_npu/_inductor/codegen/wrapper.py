import copy
from typing import Optional
from torch._inductor import config
from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SymbolicCallArg,
    pexpr,
)
from torch._inductor.utils import (
    cache_on_self,
)
from torch._inductor.virtualized import V
from torch._inductor.ir import GraphPartitionSignature

import torch_npu.npu.aclnn


class NPUWrapperCodeGen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            return super().create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures)
        return NPUWrapperCodeGen()

    @cache_on_self
    def write_triton_header_once(self) -> None:
        super().write_triton_header_once()
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(
                "import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics"
            )
        if not V.graph.cpp_wrapper:
            self.imports.splice(
                "import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics"
            )

    # generate numel expr for range_tree_node
    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"
        self.writeline(f"{expr} = {pexpr(numel_expr)}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, numel_expr)

    # don't assert
    def codegen_input_size_asserts(self) -> None:
        pass

    def get_next_kernel_suffix(self) -> str:
        iter_val = copy.copy(self._names_iter)
        return f"{next(iter_val)}"

    def write_prefix(self) -> None:
        super().write_prefix()
        if torch_npu.npu.aclnn._use_static_aclnn_kernel:
            with self.prefix.indent():
                self.prefix.writeline('global has_initialized')
                self.prefix.writeline('if not has_initialized:')
            self.prefix.do_indent()
            with self.prefix.indent():
                self.prefix.writeline('from torch_npu._inductor.npu_static_kernel import StaticKernelCompiler')
                self.prefix.writeline('static_kernel_complier = StaticKernelCompiler()')
                self.prefix.writeline('static_kernel_complier.__enter__()')
                self.prefix.writeline('has_initialized = True')
            self.prefix.do_unindent()

    def generate_return(self, output_refs: list[str]) -> None:
        if torch_npu.npu.aclnn._use_static_aclnn_kernel:
            self.wrapper_call.do_unindent()
            with self.wrapper_call.indent():
                self.wrapper_call.writeline('if not has_initialized:')
            self.wrapper_call.do_indent()
            with self.wrapper_call.indent():
                self.wrapper_call.writeline('exc_info=(None, None, None)')
                self.wrapper_call.writeline('static_kernel_complier.__exit__(*exc_info)')
        super().generate_return(output_refs)