import os
import copy
import hashlib
import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg, SubgraphPythonWrapperCodegen
from torch._inductor.runtime import triton_heuristics
from torch._inductor.utils import (
    cache_on_self,
)
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._sympy.singleton_int import SingletonInt

from torch_npu._inductor import config as npu_config


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

    def add_benchmark_harness(self, output):
        """
        Override, add aot-inductor debug kernel support.
        """
        if not config.benchmark_harness:
            return None
        
        if npu_config.aot_inductor.debug_kernel:
            return self.add_npu_repro(output)

        return super().add_benchmark_harness(output)

    def add_npu_repro(self, output):
        self.add_repro_func(output)
        self.add_benchmark_func(output)

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            # List how to use. Read details in torch_npu/_inductor/config.py.
            output.writelines(
                [
                    "# torch_npu._inductor.config.force_fallback_kernel_id = 'all'",
                    "# or",
                    "# torch_npu._inductor.config.force_fallback_kernel_id = [1, 2, 10]",
                    "torch_npu._inductor.config.aot_inductor.debug_kernel_in_run = True",
                    "result = benchmark_compiled_module()",
                    "print(result)",
                ]
            )
    
    def add_repro_func(self, output):
        seen_constants = set()

        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def get_hash(name):
            byte = name.encode('utf-8')
            sha1 = hashlib.sha1()
            sha1.update(byte)
            return sha1.hexdigest()
        
        def save_tensor(tensor, path):
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            torch.save(tensor, path)

        def add_real_tensor(name, tensor):
            tensor_dir = npu_config.aot_inductor.repro_tensor_path
            if isinstance(tensor, FakeTensor):
                raise RuntimeError(f"Could not generate repro func because detected {name} is FakeTensor "
                                   f"when trying to dump it. Set repro and debug_kernel false to avoid it.")
            hash_name = get_hash(name)
            tensor_path = os.path.join(os.getcwd(), tensor_dir, f"{hash_name}.pt")
            if name not in seen_constants:
                save_tensor(tensor, tensor_path)
                seen_constants.add(name)
            output.writeline(
                f"{name} = torch.load('{tensor_path}')"
            )

        def add_torchbind_input(name, value):
            import pickle

            output.writeline(f"{name} = pickle.loads({pickle.dumps(value)!r})")
        output.writelines(
            ["", "", f"def repro_run({', '.join(V.graph.graph_inputs.keys())}):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )
            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_real_tensor(name, value)

            if len(V.graph.torchbind_constants) > 0:
                output.writeline("import pickle")
                for name, torchbind_obj in V.graph.torchbind_constants.items():
                    # all the constants are global variables, that's why we need
                    # these 'global var_name' lines
                    output.writeline(f"global {name}")
                    add_torchbind_input(name, torchbind_obj)
            
            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(f"fn = lambda: {call_str}")
            output.writeline("return fn()")
    
    def add_benchmark_func(self, output):
        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        output.writelines(
            ["", "", "def benchmark_compiled_module(times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )
            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.var_to_val.get(value, None), SingletonInt
                ):
                    continue
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    add_expr_input(name, V.graph.sizevars.size_hint(value, fallback=42))
                else:
                    shape = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_size()
                    ]
                    stride = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_stride()
                    ]
                    add_fake_input(
                        name,
                        shape,
                        stride,
                        value.get_device(),
                        value.get_dtype(),
                    )
            
            call_str = f"repro_run({', '.join(V.graph.graph_inputs.keys())})"
            output.writeline(f"fn = lambda: {call_str}")
            output.writeline("return fn()")
