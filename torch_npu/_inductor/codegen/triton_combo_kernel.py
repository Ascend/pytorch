from typing import Any, cast, Optional

import numpy as np
import sympy
from sympy import Integer
from torch._inductor import config
from torch._inductor.codegen.common import ArgName, ConstexprArg, SizeArg
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton_combo_kernel import ComboKernel
from torch._inductor.codegen.triton_utils import signature_to_meta, config_of, equal_1_arg_indices
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import IndentedBuffer, Placeholder, triton_version_uses_attrs_dict
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ..codegen.npu_kernel_features import NumelList
from ..codegen.triton import NPUIndexTritonKernel
from ..runtime.triton_heuristics import SequentialComboKernelGrid


class NPUComboKernel(ComboKernel):

    def __init__(
        self, enable_autotune: bool = False, mixed_sizes: bool = False
    ) -> None:
        super().__init__(enable_autotune, mixed_sizes)
        self.block_size_1d = 4096

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
            # Grouped plans use a single kernel's axis arguments and grid.
            allow_grouped_autotune=False,
        )
        kernel.split_axis = [
            axis for axis in kernel.split_axis
            if not (axis.is_reduction and kernel.persistent_reduction)
            and not (axis.prefix == "x" and kernel.no_x_dim)
        ][:1]
        for axis in kernel.sorted_axis:
            if axis not in kernel.split_axis:
                axis.is_split_axis = False

        return kernel

    def select_heuristics(self, sub_kernel: TritonKernel) -> tuple[str, dict[str, int]]:
        assert sub_kernel is not NPUIndexTritonKernel
        heuristics = sub_kernel._get_heuristic()
        size_hints = sub_kernel.get_size_hints()
        if not isinstance(size_hints, dict):
            size_hints = dict(zip((axis.name for axis in sub_kernel.sorted_axis), size_hints))
        return heuristics, size_hints

    def select_dispatch_strategy(self) -> None:
        if self.dispatch_class is None:
            self.dispatch_class = NPUComboKernel.SIMDDispatch

    @staticmethod
    def horizontal_partition(
        nodes, triton_scheduling, kernel_map, node_info_map, custom_algorithm=False
    ):
        original_groups = []
        try:
            if custom_algorithm:
                # Community partitioning expects scalar sizes; NPU codegen needs
                # the original per-axis groups after the partition is selected.
                for node in nodes:
                    device, sizes = node.group
                    if any(isinstance(size, NumelList) for size in sizes):
                        original_groups.append((node, node.group))
                        node.group = (
                            device,
                            tuple(
                                sympy.sympify(size.numels())
                                if isinstance(size, NumelList)
                                else size
                                for size in sizes
                            ),
                        )
            partitions = ComboKernel.horizontal_partition(
                nodes, triton_scheduling, kernel_map, node_info_map, custom_algorithm
            )
        finally:
            for node, group in original_groups:
                node.group = group
        compatible_partitions = []
        for partition in partitions:
            groups = {}
            for node in partition:
                kernel = kernel_map[node]
                # One launcher/autotuner supplies block arguments for the whole
                # combo. Only combine kernels with the same block ABI and mode.
                key = (
                    kernel.npu_kernel_type,
                    kernel.inside_reduction,
                    kernel.persistent_reduction,
                    tuple(
                        (
                            axis.name, axis.is_split_axis, axis.is_tiling_axis,
                            axis.is_no_loop_axis, axis.is_reduction,
                        )
                        for axis in kernel.sorted_axis
                    ),
                )
                groups.setdefault(key, []).append(node)
            compatible_partitions.extend(groups.values())
        return compatible_partitions

    def select_combo_heuristics(
        self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]
    ) -> tuple[str, dict[str, int], TritonKernel]:
        # This shape only describes the shared search envelope. Preserve every
        # subkernel's actual shape separately for grid and resource calculations.
        size_hints = {}
        split_names = {axis.name for axis in self.sub_kernels[0].split_axis}
        for subkernel_hints in size_hints_list:
            for axis_name, extent in subkernel_hints.items():
                if axis_name in split_names:
                    size_hints[axis_name] = size_hints.get(axis_name, 0) + extent
                else:
                    size_hints[axis_name] = max(size_hints.get(axis_name, 1), extent)
        if not self.enable_autotune:
            return "foreach", size_hints, self.sub_kernels[0]
        return heuristics_list[0], size_hints, self.sub_kernels[0]

    def autotune_subkernel_meta(self) -> list[dict[str, Any]]:
        """Carry actual per-subkernel constraints into the shared tile search."""
        metadata = []
        fields = (
            "axis_names", "tiling_axis", "no_loop_axis", "split_axis",
            "low_dims", "split_axis_dtype", "dual_reduction", "npu_kernel_type",
        )
        for subkernel in self.sub_kernels:
            subkernel_meta = subkernel.create_inductor_meta()
            item = {field: subkernel_meta[field] for field in fields}
            _, item["size_hints"] = self.select_heuristics(subkernel)
            item["persistent_reduction"] = subkernel.persistent_reduction
            metadata.append(item)
        return metadata

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
                code.writeline(f"{tree.name}_numel = {tree.name}_numel_{num}")

            if tree.is_no_loop_axis or (tree.is_reduction and sub_kernel.persistent_reduction):
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                if sub_kernel.is_unified_simt_kernel():
                    code.writeline(f"{tree.name.upper()}BLOCK_SUB: tl.constexpr = {next_power_of_2(val)}")
                else:
                    code.writeline(f"{tree.name.upper()}BLOCK_SUB: tl.constexpr = {val}")

        return uniquify_block_sizes


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
        can_use_32bit = all(k.index_dtype == "tl.int32" for k in self.sub_kernels)
        size_dtype = "tl.int32" if can_use_32bit else "tl.int64"
        for i, sub in enumerate(self.sub_kernels):
            self.min_x_blocks_sub_kernel(sub, i)
        self.select_dispatch_strategy()
        triton_meta = {
            "signature": signature_to_meta(
                signature, size_dtype=size_dtype, argdefs=argdefs
            ),
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {
                signature[index].name: 1 for index in equal_1_arg_indices(signature)
            },
            "mix_mode": "aiv"
        }

        triton_meta["configs"] = [config_of(signature)]
        mutated_args = self.get_mutated_args_sub_kernels()
        inductor_meta = self.sub_kernels[0].create_inductor_meta()
        self.sub_kernels[0]._disable_grouped_autotune(
            inductor_meta, "Combo kernels do not support grouped autotune"
        )
        # Keep runtime BLOCK arguments, but use the actual Combo signature
        # rather than a member's cached grouped plan.
        inductor_meta["runtime_block_arg_names"] = tuple(self.block_args)
        inductor_meta["mutated_arg_names"] = mutated_args
        dispatch = self.dispatch_class
        assert dispatch is not None
        inductor_meta["grid_type"] = dispatch.grid_expr.__name__
        inductor_meta["combo_grid_meta"] = self.combo_grid_meta()
        inductor_meta["combo_subkernel_meta"] = self.autotune_subkernel_meta()

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
                cls._calculate_split_blocks(kernel, code)
                code.splice(f"if pid < num_blocks_{num}:")
                with code.indent():
                    code.splice("pid_offset = pid")
            else:
                code.splice(f"elif pid < num_blocks_{num}:")
                with code.indent():
                    code.splice(f"pid_offset = pid - num_blocks_{num - 1}")

        @classmethod
        def _calculate_split_blocks(
                cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            split_numels_list = []
            for i, sub_kernel in enumerate(kernel.sub_kernels):
                if len(sub_kernel.split_axis) == 0:
                    split_numels_list.append(None)
                    continue
                tree = sub_kernel.split_axis[0]
                simplified_tree_numel = V.graph.sizevars.simplify(tree.length)
                if isinstance(simplified_tree_numel, (Integer, int)):
                    split_numel = int(simplified_tree_numel)
                else:
                    split_numel = f"{tree.name}_numel_{i}"
                split_numels_list.append(split_numel)

            for i in range(len(split_numels_list)):
                if split_numels_list[i] is None:
                    block_str = "1"
                else:
                    block_str = (
                        f"tl.cdiv({split_numels_list[i]}, {kernel.block_args[0]})"
                    )
                if i == 0:
                    code.splice(f"num_blocks_{i} = {block_str}")
                else:
                    code.splice(f"num_blocks_{i} = num_blocks_{i - 1} + {block_str}")

    # BLOCK and SUB_BLOCK definitions
    def add_numel_to_args(self, argdefs, signature):
        self._dynamic_axis_numels = []
        for num, kernel in enumerate(self.sub_kernels):
            for axis in kernel.sorted_axis:
                length = V.graph.sizevars.simplify(axis.length)
                if not isinstance(length, (Integer, int)):
                    name = f"{axis.name}_numel_{num}"
                    signature.append(SizeArg(name, length))
                    argdefs.append(ArgName(name))
                    self.dynamic_shape_args.append(name)
                    self._dynamic_axis_numels.append((name, length))
        return argdefs

    def add_numel_to_call_args(self, name, call_args, arg_types):
        for _, length in self._dynamic_axis_numels:
            call_args.append(length)
            arg_types.append(type(length))

    def kernel_benchmark_extra_args(self) -> list[str]:
        return [
            str(
                V.graph.sizevars.size_hint(
                    length, fallback=config.unbacked_symint_fallback
                )
            )
            for _, length in self._dynamic_axis_numels
        ]

    def add_autotune_args(self, kernel, argdefs, signature):
        def add_constexpr_arg(arg_name):
            if triton_version_uses_attrs_dict():
                signature.append(ConstexprArg(arg_name))
            argdefs.append(ArgName(arg_name, is_constexpr=True))

        for block_arg in self.get_block_args():
            signature.append(block_arg)
            argdefs.append(ArgName(block_arg.name))

        for axis in kernel.tiling_axis:
            if axis.name[0] == 'r' and kernel.persistent_reduction:
                continue
            if axis.is_no_loop_axis:
                continue
            add_constexpr_arg(f"{axis.name.upper()}BLOCK_SUB")

    def codegen_blocks(self, code: IndentedBuffer) -> None:
        for block in self.block_args:
            if block.startswith("X"):
                code.splice(f"{block}: tl.constexpr = {self.block_size_1d}")
            if block.startswith("Y"):
                code.splice(f"{block}: tl.constexpr = {self.block_size_1d}")

    def get_block_args(self) -> list[SizeArg]:
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
                block_names[f"{tree.name.upper()}BLOCK"] = tree.length
        self.block_args = list(block_names.keys())

        return [SizeArg(name, length) for name, length in block_names.items()]

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        heuristics_list, size_hints_list = [], []
        for subkernel in self.sub_kernels:
            self.removed_buffers |= subkernel.removed_buffers
            self.inplaced_to_remove |= subkernel.inplaced_to_remove
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

        # Render Triton helper functions (e.g. the combine helper generated for
        # tl.associative_scan / cumsum, ``_triton_helper_fn_addN``). The NPU
        # combo codegen previously dropped this block that upstream ComboKernel
        # has, so a fused reduction kernel referencing the helper called it but
        # never defined it -> NameError at ast_to_ttir and NoTritonConfigsError.
        seen_helpers: OrderedSet[str] = OrderedSet()
        for sub_kernel in self.sub_kernels:
            for helper in sub_kernel.helper_functions:
                if helper not in seen_helpers:
                    code.writeline("")
                    code.splice(helper)
                    seen_helpers.add(helper)
        argdefs, _, signature, _ = self.args.python_argdefs()
        argdefs = self.add_numel_to_args(argdefs, signature)
        # remove enable_autotune condition and add autotune args
        self.add_autotune_args(self.sub_kernels[0], argdefs, signature)

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

            for num, sub_kernel in enumerate(self.sub_kernels):
                assert self.dispatch_class is not None
                self.dispatch_class.codegen_pid_range(self, num, code)
                with code.indent(), V.set_kernel_handler(sub_kernel):
                    uniquify = self.codegen_static_numels_sub_kernel(
                        code, sub_kernel, num
                    )
                    if len(sub_kernel.range_tree_nodes) == 0:
                        sub_kernel.write_scalar()
                    else:
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

        # Deferred stores must see the subkernels' removed buffers.
        # Keep this state local until every partition passes validation.
        with V.set_kernel_handler(self):
            return code.getvalue()

    def combo_grid_meta(self) -> dict[str, Any]:
        dynamic_shape = bool(self.dynamic_shape_args)
        num_kernels = len(self.sub_kernels)
        min_blocks = (
            max(self.min_x_blocks_list) * num_kernels if not dynamic_shape else None
        )

        # NPU dispatch flattens the selected split axis onto program_id(0),
        # even when that axis is named Y0 or X1. Match that exact 1-D grid.
        default_config = {"XBLOCK": self.block_args[0]} if self.block_args else None

        meta = {
            "num_kernels": num_kernels,
            "min_blocks": min_blocks,
            "default_config": default_config,
            "split_numels": [],
            "no_split": [],
            "split_block": self.block_args[0] if self.block_args else None,
        }

        for num, sub_kernel in enumerate(self.sub_kernels):
            meta[f"no_x_dim_{num}"] = not sub_kernel.split_axis
            if sub_kernel.split_axis:
                extent = V.graph.sizevars.simplify(sub_kernel.split_axis[0].length)
                static = isinstance(extent, (Integer, int))
                meta[f"xnumel_{num}"] = int(extent) if static else None
                meta["split_numels"].append(
                    int(extent) if static
                    else f"{sub_kernel.split_axis[0].name}_numel_{num}"
                )
            else:
                meta[f"xnumel_{num}"] = 1
                meta["split_numels"].append(1)
            meta["no_split"].append(not sub_kernel.split_axis)

        return meta
