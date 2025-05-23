import copy
from torch._inductor import config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg, SubgraphPythonWrapperCodegen
from torch._inductor.runtime import triton_heuristics
from torch._inductor.utils import (
    cache_on_self,
)
from torch._inductor.virtualized import V


class NPUWrapperCodeGen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create(
            is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
    ):
        if is_subgraph:
            return SubgraphPythonWrapperCodegen(subgraph_name, parent_wrapper)
        return NPUWrapperCodeGen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            f"""
                import torch_npu
            """,
            strip=True,
        )

    @cache_on_self
    def write_triton_header_once(self) -> None:
        import_str = f"""
            import triton
            import triton.language as tl
            from {triton_heuristics.__name__} import (                
                split_scan_grid,
                grid_combo_kernels,
                start_graph,
                end_graph,
                cooperative_reduction_grid,
            )
            from torch_npu._inductor.npu_triton_heuristics import grid
            import torch_npu
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

    # generate numel expr for range_tree_node
    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"
        if (expr, V.graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((expr, V.graph))
            self.writeline(
                f"{self.declare}{expr} = {self.expr_printer(numel_expr)}{self.ending}"
            )
        else:
            self.writeline(f"{expr} = {self.expr_printer(numel_expr)}{self.ending}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, numel_expr)

    # don't free anything
    def make_buffer_free(self, buffer):
        return ""

    # don't assert
    def codegen_input_size_asserts(self) -> None:
        pass

    def get_next_kernel_suffix(self) -> str:
        iter_val = copy.copy(self._names_iter)
        return f"{next(iter_val)}"
