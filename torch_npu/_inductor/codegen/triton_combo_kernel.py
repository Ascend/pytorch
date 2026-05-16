from typing import Any, cast, Optional

import numpy as np
import sympy
from sympy import Integer
from torch._inductor import config
from torch._inductor.codegen.common import ArgName, ConstexprArg
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton_combo_kernel import ComboKernel
from torch._inductor.codegen.triton_utils import signature_to_meta, config_of
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.runtime.triton_heuristics import SequentialComboKernelGrid
from torch._inductor.utils import IndentedBuffer, Placeholder
from torch._inductor.virtualized import V

from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel


class NPUComboKernel(ComboKernel):
    @staticmethod
    def create_triton_kernel(
        tiling: dict[str, sympy.Expr],
        features: SIMDKernelFeatures,
        optimize_mask: bool,
    ) -> TritonKernel:
        kernel = NPUIndexTritonKernel(
            tiling,
            features=features,
            pid_cache={"tl.program_id(0)": "pid_offset"},
            optimize_mask=optimize_mask,
            # foreach kernels don't work with cooperative reductions
            override_cooperative_reduction=False,
        )
        if len(kernel.split_axis) > 1:
            kernel.split_axis = [kernel.split_axis[0]]
            for axis in kernel.sorted_axis:
                if axis not in kernel.split_axis:
                    axis.is_split_axis = False

        return kernel

    def select_heuristics(self, sub_kernel: TritonKernel) -> tuple[str, dict[str, int]]:
        assert sub_kernel is not NPUIndexTritonKernel
        heuristics = sub_kernel._get_heuristic()
        size_hints = sub_kernel.get_size_hints()
        return heuristics, size_hints

    def select_dispatch_strategy(self) -> None:
        if self.dispatch_class is None:
            self.dispatch_class = NPUComboKernel.SIMDDispatch

    def select_combo_heuristics(
        self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]
    ) -> tuple[str, dict[str, int], TritonKernel]:
        if not self.enable_autotune:
            return "foreach", size_hints_list[0], self.sub_kernels[0]
        return heuristics_list[0], size_hints_list[0], self.sub_kernels[0]

    def codegen_static_numels_sub_kernel(
        self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int
    ) -> list[str]:
        uniquify_block_sizes = []
        for tree in sub_kernel.sorted_axis:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.length)
            if isinstance(simplified_tree_numel, (Integer, int)):
                code.writeline(f"{tree.name}_numel = {int(simplified_tree_numel)}")
            else:
                assert f"{tree.name}_numel_{num}" in self.dynamic_shape_args
                uniquify_block_sizes.append(f"{tree.prefix}numel")

            if tree.is_no_loop_axis:
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                if sub_kernel.is_unified_simt_kernel():
                    code.writeline(f"{tree.name.upper()}BLOCK_SUB: tl.constexpr = {next_power_of_2(val)}")
                else:
                    code.writeline(f"{tree.name.upper()}BLOCK_SUB: tl.constexpr = {val}")


    def jit_line(
        self,
        heuristics: str,
        size_hints: dict[str, int],
        selected_kernel: TritonKernel,
        signature: list[Any],
        argdefs: list[ArgName],
        pointwise_with_reduce: bool = False,
    ) -> str:
        assert selected_kernel is not NPUIndexTritonKernel
        size_dtype = "tl.int64"
        for i, sub in enumerate(self.sub_kernels):
            self.min_x_blocks_sub_kernel(sub, i)
        self.select_dispatch_strategy()
        triton_meta = {
            "signature": signature_to_meta(
                signature, size_dtype=size_dtype, argdefs=argdefs
            ),
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
            "mix_mode": "aiv"
        }

        triton_meta["configs"] = [config_of(signature)]
        mutated_args = self.get_mutated_args_sub_kernels()
        inductor_meta = self.sub_kernels[0].create_inductor_meta()
        dispatch = self.dispatch_class
        assert dispatch is not None
        inductor_meta["grid_type"] = dispatch.grid_expr.__name__
        inductor_meta["combo_grid_meta"] = self.combo_grid_meta()

        sub_kernel = selected_kernel
        if heuristics == "foreach":
            heuristics_line = f"""
                @triton_heuristics.foreach(
                    size_hints={size_hints!r},
                    num_warps={self.num_warps},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                )
                @triton.jit
            """
        elif sub_kernel.inside_reduction:
            reduction_hint = sub_kernel.features.get_reduction_hint()
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                tile_hint = "tile_hint=TileHint.SQUARE,"
            else:
                tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """

        return heuristics_line

    class SIMDDispatch:

        grid_expr = SequentialComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            if num == 0:
                cls._calculate_xblocks(kernel, code)
                code.splice(f"if pid < num_xblocks_{num}:")
                with code.indent():
                    code.splice("pid_offset = pid")
            else:
                code.splice(f"elif pid < num_xblocks_{num}:")
                with code.indent():
                    code.splice(f"pid_offset = pid - num_xblocks_{num - 1}")

        @classmethod
        def _calculate_xblocks(
                cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            x_numels_list = kernel.x_numels_list
            for i in range(len(x_numels_list)):
                xnumels, no_x_dim = (
                    (x_numels_list[i], False)
                    if isinstance(x_numels_list[i], str)
                       and cast(str, x_numels_list[i])[0] != "-"
                       or (
                               isinstance(x_numels_list[i], int)
                               and cast(int, x_numels_list[i]) > 0
                       )
                    else (kernel.min_x_blocks_list[i], True)
                )
                xblock_str = (
                    f"tl.cdiv({xnumels}, {kernel.block_args[0]})" if not no_x_dim else f"{xnumels}"
                )
                if i == 0:
                    code.splice(f"num_xblocks_{i} = {xblock_str}")
                else:
                    code.splice(f"num_xblocks_{i} = num_xblocks_{i - 1} + {xblock_str}")

    # BLOCK and SUB_BLOCK definitions
    def add_autotune_args(self, kernel, argdefs):
        if self.enable_autotune:
            for axis in kernel.split_axis:
                argdefs.append(ArgName(f"{axis.name.upper()}BLOCK", is_constexpr=True))

        for axis in kernel.tiling_axis:
            if axis.name[0] == 'r' and kernel.persistent_reduction:
                continue
            if axis.is_no_loop_axis:
                continue
            argdefs.append(ArgName(f"{axis.name.upper()}BLOCK_SUB", is_constexpr=True))

    def codegen_blocks(self, code: IndentedBuffer) -> None:
        for block in self.block_args:
            code.splice(f"{block}: tl.constexpr = 4096")

    def get_block_args(self) -> list[ConstexprArg]:
        block_names = {}
        for sub_kernel in self.sub_kernels:
            # TODO: we assume all sub_kernels have the same block size
            for tree in sub_kernel.split_axis:
                if tree.is_reduction and (
                    not sub_kernel.inside_reduction or sub_kernel.persistent_reduction
                ):
                    continue
                if tree.prefix == "x" and sub_kernel.no_x_dim:
                    continue
                block_names[f"{tree.name.upper()}BLOCK"] = tree.name
        self.block_args = list(block_names.keys())

        return [ConstexprArg(x) for x in block_names.keys()]

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        # TODO: is it correct to use the first sub kernel's heuristics?
        heuristics_list, size_hints_list = [], []
        for subkernel in self.sub_kernels:
            h, s = self.select_heuristics(subkernel)
            heuristics_list.append(h)
            size_hints_list.append(s)
        heuristics, size_hints, selected_kernel = self.select_combo_heuristics(
            heuristics_list, size_hints_list
        )

        pointwise_with_reduction, heuristics = (
            (True, "pointwise")
            if heuristics == "pointwise_with_reduction"
            else (False, heuristics)
        )
        code = IndentedBuffer()

        from torch._inductor.codegen.triton import gen_common_triton_imports
        code.splice(gen_common_triton_imports())
        if config.benchmark_combo_kernel:
            code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()
        argdefs = self.add_numel_to_args(argdefs, signature)
        block_args = self.get_block_args()
        # remove enable_autotune condition and add autotune args
        self.add_autotune_args(self.sub_kernels[0], argdefs)

        code.splice(
            self.jit_line(
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduce=False,
                signature=signature,
                argdefs=argdefs,
            )
        )
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(x.full_name() for x in argdefs)}):"
        )

        with code.indent():
            code.splice("pid = tl.program_id(0)")
            if not self.enable_autotune:
                self.codegen_blocks(code)

            for num, sub_kernel in enumerate(self.sub_kernels):
                assert self.dispatch_class is not None
                self.dispatch_class.codegen_pid_range(self, num, code)
                with code.indent():
                    uniquify = self.codegen_static_numels_sub_kernel(
                        code, sub_kernel, num
                    )
                    sub_kernel.codegen_body()
                    uniquified_body = self.uniquify_block_sizes(
                        sub_kernel.body, num, uniquify
                    )
                    code.splice(uniquified_body)

            code.splice("else:")
            with code.indent():
                code.splice("pass")

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=0))

        return code.getvalue()
