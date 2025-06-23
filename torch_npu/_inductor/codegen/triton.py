import functools
import itertools
import operator
import os
import re
import textwrap
from enum import Enum
from typing import List, Set, Iterable, Callable, Sequence
from typing import (
    Optional,
    Union,
    Tuple,
    Any,
    cast,
    Dict
)
import sympy
import torch
from torch._inductor import config, ir
from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg,
    DeferredLine,
)
from torch._inductor.codegen.common import free_symbol_is_type
from torch._inductor.codegen.simd import CantSplit, DisableReduction, EnableReduction
from torch._inductor.codegen.triton import (
    IndexingOptions,
    triton_reshape,
    TritonCSEVariable,
    OpsHandler,
)
from torch._inductor.codegen.triton import (
    TritonKernel,
    TritonKernelOverrides,
    IterationRangesRoot,
    IterationRangesEntry,
    CSEVariable,
    gen_common_triton_imports,
    BlockPtrOptions,
    triton_acc_type,
    constant_repr,
    is_welford_reduction, FixedTritonConfig,
    prefix_is_reduction, upcast_acc_dtype,
    get_kernel_category_by_source_code,
    get_fused_kernel_name
)
from torch._inductor.codegen.triton_utils import config_of, signature_of, signature_to_meta
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import (
    Placeholder,
    get_bounds_index_expr,
    upcast_compute_type
)
from torch._inductor.utils import sympy_index_symbol, generate_assert
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import (
    V,
    StoreMode,
    ReductionType,
    _ops as ops,
)
from torch.utils import _pytree as pytree
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.symbol import SymT, symbol_is_type
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
import torch_npu._inductor.config as inductor_npu_config

from .kernel_analysis import IndexAnalysis, ReductionAnalysis
from .npu_kernel_features import NumelList
from ..runtime import NPUDeviceProperties


def flatten(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res


class NPUTritonKernelOverrides(TritonKernelOverrides):

    @staticmethod
    def exp(x):
        return f"tl_math.exp({x})"

    @staticmethod
    def sqrt(x):
        return f"tl_math.sqrt({x})"

    @staticmethod
    def tanh(x):
        return f"tl_math.tanh({x})"

    @staticmethod
    def rsqrt(x):
        return f"tl.rsqrt({x})"

    @staticmethod
    def floor(x):
        return f"tl_math.floor({x})"

    @staticmethod
    def erf(x):
        return f"tl_math.erf({x})"

    @staticmethod
    def ceil(x):
        return f"tl_math.ceil({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        indexing = V.kernel.indexing(expr, block_ptr=False, is_index_expr=True)
        if not isinstance(indexing, IndexingOptions):
            raise TypeError(f"not a IndexingOptions : {indexing}")

        # Our sympy expr printing casts to the current kernel index dtype.
        # we only respect non int32-int64 dtypes and otherwise use current kernel indexing dtype
        index_dtype = torch.int32 if V.kernel.index_dtype == "tl.int32" else torch.int64
        dtype = dtype if dtype not in (torch.int32, torch.int64) else index_dtype
        var = V.kernel.cse.generate(
            V.kernel.compute,
            indexing.index_str,
            bounds=get_bounds_index_expr(expr),
            dtype=dtype,
        )

        if dtype not in (torch.int32, torch.int64):
            var = V.kernel.cse.generate(
                V.kernel.compute,
                cls.to_dtype(var, dtype),
                dtype=upcast_compute_type(dtype),
            )
        else:
            # We are not always consistent in enforcing that the output of the index expr printing
            # results in the indexing dtype. So if we detect that we have an input which might type promote
            # to a dtype other than indexing dtype, add a cast.
            # Trying to avoid
            dtype = index_dtype
            for index_var in expr.free_symbols:
                if symbol_is_type(index_var, SymT.TMP):
                    dtype = torch.promote_types(
                        dtype, V.kernel.cse.varname_map[index_var.name].dtype
                    )

            if dtype != index_dtype:
                var = V.kernel.cse.generate(
                    V.kernel.compute,
                    cls.to_dtype(var, index_dtype),
                    dtype=index_dtype,
                )

        var.mask_vars = indexing.mask_vars
        return var


def group_fn(self, sizes):
    groups = list()
    for s in sizes:
        if not s:
            groups.append(1)
        elif isinstance(s, list):
            group = flatten(s)
            groups.append(NumelList(tuple(group)) if isinstance(group, list) else group)
        else:
            groups.append(s)
    return tuple(groups)


@staticmethod
def select_index_dtype(node_schedule, numel, reduction_numel):
    return "tl.int32"


class IterationRangesEntryNPUIndex(IterationRangesEntry):
    def __init__(
            self,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_tiling_axis = False
        self.is_split_axis = False
        self.indexing_code = IndentedBuffer()
        self.sorted_order = None
        self.tiling_order = None
        self.split_order = None
        self.var_directions = {}
        self.directions = []
        # don't use functools.lru_cache(None), so that previous indexing_code produdec by previous index,
        # could be overwritten
        self.codegen = self._codegen

    # axis mask
    def _codegen_mask(self):

        if self.is_tiling_axis:
            BLOCK_NAME = f"{self.name.upper()}BLOCK"
            upper = f"min({BLOCK_NAME}+{self.symbol()}_offset, {self.name}_numel)" if self.is_split_axis else f"{self.name}_numel"
            line = f"{self.name}_mask = {self.name} < {upper}"
            self.writeline(line)
            for var in self.var_directions.keys():
                line = f"{var.name}_mask = {var.name} < {upper}"
                self.writeline(line)
        else:
            pass

    def get_axis_direction(self):

        # assume self.golden_var_list is to be correct axis order

        if self.directions:
            return f"[{','.join(self.directions)}]"
        tiling_axis = [x.symbol() for x in self.kernel.tiling_axis]

        rev_orders = [x for x in self.kernel.golden_var_list if x in tiling_axis]
        self.directions = ["None"] * len(tiling_axis)
        if len(tiling_axis) != len(rev_orders):
            raise RuntimeError(f"assert tiling len={len(tiling_axis)}, not equal to golden varlist len ={len(rev_orders)}")
        var_orders = list(reversed(rev_orders))
        index = var_orders.index(self.symbol())
        self.directions[index] = ":"
        return f"[{','.join(self.directions)}]"

    # axis var, need to define var with diffent direction
    def _codegen(self):
        self.indexing_code.clear()
        index = None
        # for multiple reduce dims, don't need this
        if not self.is_tiling_axis:
            return self.name

        direction = self.get_axis_direction()
        index = f"{self.name} = {self.codegen_index(direction)}"
        for var, dir_index in self.var_directions.items():
            line = f"{var.name} = {self.codegen_index(dir_index)}"
            self.writeline(line)

        # reduction axis 
        if self.prefix == 'r':
            if V.kernel.inside_reduction and V.kernel.current_node \
                    and isinstance(V.kernel.current_node, SchedulerNode) \
                    and V.kernel.current_node.node \
                    and V.kernel.current_node.node.data \
                    and isinstance(V.kernel.current_node.node.data, ir.Reduction):
                reduction_type = V.kernel.current_node.node.data.reduction_type
                if reduction_type in {"argmax", "argmin"}:
                    self.writeline(f"{self.parent.prefix}index = "
                                   f"{self.codegen_index(None)}")
        if index:
            self.writeline(index)
            self._codegen_mask()
        return self.name

    def writeline(self, line):
        self.indexing_code.writeline(line)

    def is_1d_persisent_reduction(self):
        return len(V.kernel.tiling_axis) == 1 and V.kernel.persistent_reduction

    def codegen_index(self, direction):
        BLOCK_NAME = f"{self.name.upper()}BLOCK"
        BLOCK_NAME_SUB = f"{BLOCK_NAME}_SUB"
        index = None
        if self.prefix == 'r':
            if V.kernel.persistent_reduction:
                if self.is_1d_persisent_reduction():
                    index = f"tl.arange(0, {BLOCK_NAME_SUB})"
                else:
                    index = f"base_{self.name}"
            else:
                index = f"(loop_{self.name} * {BLOCK_NAME_SUB}) + base_{self.name}"
        else:
            if self.is_split_axis:
                offset = f"{self.symbol()}_offset"
                index = f"{offset} + (loop_{self.name} * {BLOCK_NAME_SUB}) + base_{self.name}"
            else:
                index = f"(loop_{self.name} * {BLOCK_NAME_SUB}) + base_{self.name}"

        if len(V.kernel.tiling_axis) > 1 and direction is not None:
            index += direction

        return index

    def codegen_header(self, code):
        # generate offset index loop
        lines = []
        BLOCK_NAME = f"{self.name.upper()}BLOCK"
        BLOCK_NAME_SUB = f"{BLOCK_NAME}_SUB"

        if self.is_1d_persisent_reduction():
            return

        if self.is_split_axis:
            lines.append(f"{self.symbol()}_offset = tl.program_id({self.split_order}) * {BLOCK_NAME}")

        if self.is_tiling_axis:
            lines.append(f"base_{self.name}= tl.arange(0, {BLOCK_NAME_SUB})")
            block = f"{BLOCK_NAME}" if self.is_split_axis else f"{self.symbol()}_numel"
            lines.append(f"loops_{self.name} = ({block} + {BLOCK_NAME_SUB} - 1) // {BLOCK_NAME_SUB}")

        else:
            pass

        code.writelines(lines)

    def precomputed_args(self):
        # for dynamic shapes, find parts of indexing expressions that have to be precomputed
        precomputed_args: List[sympy.Expr] = []
        if isinstance(self.expr, (sympy.Symbol, sympy.Integer)):
            return precomputed_args

        if not isinstance(self.expr, (FloorDiv, ModularIndexing)):
            raise RuntimeError("assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)")
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all(
                        symbol_is_type(s, SymT.SIZE) for s in symbols
                ):
                    precomputed_args.append(arg)
        return precomputed_args


class IterationRangesRootNPUIndex(IterationRangesRoot):
    def __init__(
            self,
            name: str,
            numel: sympy.Expr,
            prefix: str,
            index: int,
            kernel: TritonKernel,
            pid_cache=None,
            *,
            is_loop: bool,
            tensor_dim: Optional[int],
            grid_dim: Optional[int],
    ):
        super().__init__(name, numel, prefix, index, kernel, pid_cache, is_loop=is_loop, tensor_dim=tensor_dim,
                         grid_dim=grid_dim, has_zdim=False)

    def __repr__(self):
        return f"IterationRangesRootNPUIndex({self.name!r}, {self.numel}, ...)"

    def remove_entry(self, name):
        if name in self.var_ranges:
            del self.var_ranges[name]
        if name in self.var_list:
            del self.var_list[self.var_list.index(name)]
        if name in V.kernel.range_tree_nodes:
            V.kernel.range_tree_nodes_removed[name] = V.kernel.range_tree_nodes[name]
            del V.kernel.range_tree_nodes[name]
        if name in self.nodes:
            del self.nodes[name]

    def duplicated_check(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(sympy_index_symbol(f"{self.prefix}index"), divisor)
        else:
            expr = ModularIndexing(
                sympy_index_symbol(f"{self.prefix}index"), divisor, length
            )

        return expr not in self.nodes

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(sympy_index_symbol(f"{self.prefix}index"), divisor)
        else:
            expr = ModularIndexing(
                sympy_index_symbol(f"{self.prefix}index"), divisor, length
            )

        if expr not in self.nodes:
            node = IterationRangesEntryNPUIndex(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node

        return self.nodes[expr]


def is_compatible(groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]):
    try:
        groups = flatten(groups)
        NPUIndexTritonKernel._split_iteration_ranges(groups, lengths)
        return True
    except CantSplit:
        return False


class NPUIndexTritonKernel(TritonKernel):
    overrides = NPUTritonKernelOverrides

    def __init__(
            self,
            tiling: Dict[str, sympy.Expr],
            min_elem_per_thread=0,
            optimize_mask=True,
            fixed_config: Optional[FixedTritonConfig] = None,
            **kwargs, ):

        super().__init__(tiling=tiling,
                         min_elem_per_thread=min_elem_per_thread,
                         optimize_mask=optimize_mask,
                         fixed_config=fixed_config,
                         **kwargs)
        self.first_node = True
        self.inside_high_order_reduction = False
        self.low_dims = set()
        self.split_axis = []
        self.tiling_axis = []
        self.range_tree_nodes_removed: Dict[sympy.Symbol, IterationRangesEntry] = {}
        self.range_tree_nodes_substituted = {}
        self.expr_substituted = {}
        self.sorted_axis = []
        self.prefix: IndentedBuffer = IndentedBuffer()
        self.index_analysis = {}  # var_list -> indexAnalysis
        self.golden_var_list = None
        self.reduce_analysis = None
        self.load_store_indexing = None

    def gen_triton_ext_imports(self):
        imports = IndentedBuffer()
        imports.splice(
            """
            from torch._inductor.runtime import triton_helpers
            from torch_npu._inductor import npu_triton_heuristics
            from torch_npu._inductor import npu_triton_helpers
            from torch_npu._inductor.runtime import NPUDeviceProperties
            from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
            import torch
            import torch_npu
            """
        )
        return imports.getvalue()

    def patch_triton_hash(self):
        # remove this method once the original invocation is fixed
        import hashlib
        from triton.compiler.compiler import triton_key, make_backend
        from triton.runtime.driver import driver
        backend = make_backend(driver.active.get_current_target())
        key = f"{triton_key()}-{backend.hash()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def numof_tiling_axis(self):
        return len(self.tiling_axis)

    # do nothing in NpuTritonKernel
    def codegen_range_tree(self):
        pass

    def initialize_range_tree(self, pid_cache):
        self.total_numels = 0
        for k, x in self.numels.items():
            if not isinstance(x, sympy.Integer):
                x = x.subs(V.graph.sizevars.var_to_val)
                self.numels[k] = x
            if x > 1:
                self.total_numels += 1

        no_r_dim = not self.inside_reduction or self.numels["r"] == 1
        prefixes = "wvtzyxr"
        active_prefixes = prefixes[-len(self.numels):]
        # prefix can not be 's', 'u', 'ps' , 'i', 'z'
        # prefix can not be 'p' but can be 'z' since 2.6
        grid_dims = "xyztvw"
        if self.no_x_dim:
            tensor_dims = "r"
        elif no_r_dim:
            tensor_dims = "xyztvw"
        else:
            tensor_dims = "xyztvwr"
        tensor_dims = "".join(p for p in tensor_dims if p in active_prefixes)
        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix_is_reduction(prefix)
            tensor_dim = tensor_dims.find(prefix) if prefix in tensor_dims else None
            grid_dim = None if is_reduction else grid_dims.find(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRootNPUIndex(
                    f"{prefix}index",
                    self.numels[prefix],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim
                )
            )

    def get_axis_dtype(self, axis):
        dtype = None
        if axis is None:
            return None
        for node in self.node_schedule:
            if node in (EnableReduction, DisableReduction):
                continue
            if axis.symbol() in node._body.indexing_map:
                dtype = V.graph.get_dtype(node.node.name)
                break
        if dtype is None:
            should_break_all = False
            for node in self.node_schedule:
                if should_break_all:
                    break
                if node in (EnableReduction, DisableReduction):
                    continue
                for key, _ in node._body.indexing_map.items():
                    if key in self.range_tree_nodes:
                        dim = self.range_tree_nodes[key]
                    else:
                        dim = self.range_tree_nodes_removed[key]

                    if dim.parent == axis.parent:
                        dtype = V.graph.get_dtype(node.node.name)
                        should_break_all = True
                        break
        return dtype

    def create_inductor_meta(self):
        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                    mutation in self.args.inplace_buffers
                    and mutation not in V.graph.removed_buffers
                    and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)
        tiling_axis = [x.sorted_order for x in self.tiling_axis]
        split_axis = [x.sorted_order for x in self.split_axis]
        axis_names = [x.name for x in self.sorted_axis]
        split_axis_dtype = self.get_axis_dtype(self.split_axis[0]) if self.split_axis else None
        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,

            # Due to breaking change of triton 3.0, the original invocation is broken
            "backend_hash": self.patch_triton_hash(),  # torch.utils._triton.triton_hash_with_backend(),
            "split_axis": split_axis,
            "tiling_axis": tiling_axis,
            "axis_names": axis_names,
            "low_dims": self.low_dims,
            "numof_reduction_axis": self.numof_reduction_axis(),
            "split_axis_dtype": split_axis_dtype,
            "dual_reduction": self.numof_reduction_axis() > 1,
            "traced_graph_hash": "TRACED_GRAPH_HASH",
            "traced_graph_dir": "TRACED_GRAPH_DIR",
            "store_cubin": config.triton.store_cubin,
        }
        return inductor_meta

    # numels sent to autotune configs
    def get_size_hints(self):
        size_hints = []
        if (len(self.range_tree_nodes.values()) == 0):
            return [v for _, v in self.numels.items()]

        for _, node in enumerate(self.sorted_axis):
            if isinstance(node.expr, ModularIndexing):
                numel_expr = node.length
            else:
                numel_expr = node.expr.subs({sympy_index_symbol(r.name): r.numel for r in self.range_trees})

            numel_expr = V.graph.sizevars.symbolic_hint(numel_expr)

            size_hints.append(numel_expr)
        return size_hints

    # torch251 done
    def add_numel_to_call_args_and_grid(self, name, call_args, arg_types, grid):
        for node in self.sorted_axis:
            if isinstance(node.expr, ModularIndexing):
                numel_expr = node.length
            else:
                numel_expr = node.expr.subs({sympy_index_symbol(r.name): r.numel for r in self.range_trees})

            if isinstance(numel_expr, (sympy.Integer, sympy.Symbol)):
                expr = numel_expr
            else:
                expr = V.graph.wrapper_code.generate_node_numel_expr(name, node, numel_expr)
            call_args.append(expr)
            arg_types.append(type(expr))
            if node.parent.grid_dim is not None:
                grid.append(expr)

    def gen_numel_args(self, signature, triton_meta_signature, argdefs):
        for node in self.sorted_axis:
            arg_name = f"{node.name}_numel"
            if not inductor_npu_config.inductor_static_mode:
                sizearg = SizeArg(arg_name, node.length)
                signature.append(sizearg)
                triton_meta_signature[arg_name] = signature_of(
                    sizearg, size_dtype=self.index_dtype
                )
                argdefs.append(arg_name)
            else:
                argdefs.append(f"{arg_name}: tl.constexpr")
                self.triton_meta["constants"][arg_name] = node.length

    # BLOCK and SUB_BLOCK definitions
    def add_autotune_args(self, argdefs):
        for axis in self.split_axis:
            argdefs.append(f"{axis.name.upper()}BLOCK: tl.constexpr")

        for axis in self.tiling_axis:
            if axis.name[0] == 'r' and self.persistent_reduction:
                continue
            argdefs.append(f"{axis.name.upper()}BLOCK_SUB: tl.constexpr")

    def _get_heuristic(self):
        if self.persistent_reduction:
            if not self.inside_reduction:
                raise RuntimeError("assert self.inside_reduction to be true")
            return "persistent_reduction_npu_index"
        elif self.inside_reduction:
            return "reduction_npu_index"
        return "pointwise_npu_index"

    def get_kernel_name(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_category = get_kernel_category_by_source_code(src_code)[:3]
            kernel_name = "_".join(
                ["triton", kernel_category, fused_name, wrapper.get_next_kernel_suffix()]
            )
        return kernel_name

    # modify triton_meta, inductor_meta , etc.
    def codegen_kernel(self, name=None):
        code = IndentedBuffer()
        size_hints = self.get_size_hints()
        heuristics = self._get_heuristic()
        if name is None:
            code.splice(gen_common_triton_imports())
            # Note: add extra imports for extensions
            code.splice(self.gen_triton_ext_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()

        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        triton_meta_signature = signature_to_meta(signature, size_dtype=self.index_dtype, argdefs=argdefs)

        triton_meta = {
            "signature": triton_meta_signature,
            "device":
                NPUDeviceProperties.create(
                    V.graph.get_current_device_or_throw()
                ),
            "constants": {},
            # special config for NPU, specify compile target
            "mix_mode": "aiv",
        }

        inductor_meta = self.create_inductor_meta()
        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        self.triton_meta = triton_meta
        self.gen_numel_args(signature, triton_meta_signature, argdefs)

        # add in tiling args
        self.add_autotune_args(argdefs)
        # for scalar codegen
        if len(self.range_tree_nodes) == 0:
            self.write_scalar()
        else:
            self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        # Note: override original triton_heuristics
        if self.inside_reduction:
            reduction_hint = self.features.get_reduction_hint()
            heuristics_line = f"""
                @npu_triton_heuristics.{heuristics}(
                    size_hints={size_hints},
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
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @npu_triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()

    def codegen_static_numels(self, code):
        for symbol in self.reduction_axis_list():
            if symbol.name[0] != "r" or not self.persistent_reduction:
                continue

            node = self.range_tree_nodes[symbol]
            simplified_tree_numel = V.graph.sizevars.simplify(node.length)
            if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                val = int(simplified_tree_numel)
            else:
                continue
            val = next_power_of_2(val)
            code.writeline(f"{node.name.upper()}BLOCK_SUB: tl.constexpr = {val}")

    def lowest_axis_variable(self):
        if len(self.tiling_axis) == 0:
            return None
        return self.tiling_axis[-1]

    def is_isolated_symbol(self, input_str, range_val):
        patterns = [r'\b' + re.escape(range_val.name) + r'\b']
        for var in range_val.var_directions.keys():
            pattern = r'\b' + re.escape(var.name) + r'\b'
            patterns.append(pattern)

        for pattern in patterns:
            if re.search(pattern, input_str):
                return True
        return False

    def find_axis_in_load_store(self, range_val):
        if not range_val:
            return False
        for line in self.loads._lines:
            if line.find('tl.load') >= 0 and self.is_isolated_symbol(line, range_val):
                return True
        for line in self.compute._lines:
            if line.find('tl.load') >= 0 and self.is_isolated_symbol(line, range_val):
                return True
        for line in self.post_loop_store._lines:
            if line.find('tl.store') >= 0 and self.is_isolated_symbol(line, range_val):
                return True
        for line in self.stores._lines:
            if isinstance(line, DeferredLine):
                line = line.line
            if line.find('tl.store') >= 0 and self.is_isolated_symbol(line, range_val):
                return True
        return False

    def write_scalar(self):
        self.body.splice(self.indexing_code)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        self.body.splice(self.stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.post_loop_store.clear()
        self.prefix.clear()

    def codegen_body(self):
        if not (
                self.loads
                or self.stores
                or self.compute
                or self.post_loop_store
        ):
            return

        def write_pointwise():
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)

        def codegen_range(index):
            def is_1d_reduction():
                return self.numels["r"] > 1 and len(self.numels) == 1

            def loop_body(index, indexing_code, is_last_axis, do_indent=True):
                if do_indent:
                    self.body.do_indent()
                if indexing_code:
                    self.body.splice(indexing_code)
                if is_last_axis:
                    write_pointwise()
                else:
                    codegen_range(index + 1)
                if do_indent:
                    self.body.do_unindent()

            if index < 0 or index >= len(self.range_tree_nodes):
                return

            range_val = self.sorted_axis[index]
            numof_tilings = len(self.tiling_axis)
            last_tiling = range_val.is_tiling_axis and numof_tilings >= 1 and range_val.tiling_order == len(
                self.tiling_axis) - 1
            next_is_dual_reduction_tiling = index == len(
                self.sorted_axis) - numof_tilings - 1 and self.numof_reduction_axis()

            is_last_axis = index == len(self.sorted_axis) - 1
            indexing_code = getattr(range_val, "indexing_code")
            reduction_1d = is_1d_reduction()
            do_indent = False
            # do nothing except for writing porintwise
            if len(self.loads._lines) == 0 and len(self.stores._lines) == 0:
                do_indent = False
                indexing_code = None
            # tiling axis and last tiling
            if range_val.is_tiling_axis and last_tiling:
                do_indent = False
                need_axis_loop = self.find_axis_in_load_store(range_val)
                if not need_axis_loop:
                    indexing_code = None
                if (range_val.prefix != 'r' or not self.persistent_reduction) and need_axis_loop:
                    self.body.splice(self.prefix)
                    self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
                    do_indent = True
                loop_body(index, indexing_code, is_last_axis, do_indent)
                self.body.splice(self.post_loop_store)
                self.post_loop_store.clear()

            # tiling axis and but not last tiling
            elif range_val.is_tiling_axis:
                do_indent = False
                if len(self.loads._lines) == 0 and len(self.stores._lines) == 0:
                    do_indent = False
                    indexing_code = None
                if self.numof_reduction_axis() <= 1:
                    do_indent = True
                    self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
                loop_body(index, indexing_code, is_last_axis, do_indent=do_indent)

            elif not is_last_axis:
                do_indent = True
                if range_val.is_split_axis:
                    offset = f"{range_val.name}_offset"
                    self.body.writeline(f"for {range_val.name} in range({offset}, "
                                        f"min({offset} + {range_val.name.upper()}BLOCK, {range_val.name}_numel)):")
                else:
                    self.body.writeline(f"for {range_val.name} in range({range_val.name}_numel):")

                if not reduction_1d and self.persistent_reduction:
                    self.body.do_indent()
                    self.body.splice(self.prefix)
                    self.prefix.clear()
                    self.body.do_unindent()

                loop_body(index, indexing_code, is_last_axis, do_indent=do_indent)
            else:
                write_pointwise()

        if self.first_node:
            for node in self.sorted_axis:
                node.codegen_header(self.body)

        while True:
            if not self.sorted_axis[-1].is_tiling_axis:
                x = self.sorted_axis[-1]
                self.sorted_axis.pop(-1)
                self.sorted_axis.insert(0, x)
            else:
                break

        if self.first_node:
            codegen_range(0)
        else:
            last_axis_order = self.tiling_axis[-1].sorted_order
            if self.persistent_reduction and self.numof_reduction_axis() > 1:
                last_axis_order = last_axis_order - self.numof_reduction_axis() + 1
            for _ in range(last_axis_order):
                self.body.do_indent()
            codegen_range(last_axis_order)
            for _ in range(last_axis_order):
                self.body.do_unindent()

        self.cse.invalidate(self.outside_loop_vars)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.post_loop_store.clear()
        self.prefix.clear()
        self.first_node = False

    # for creat constant tensor, if have two axis, constant=tl.full([1,1]) else  tl.full([1])
    def triton_tensor_ndim(self):
        if self.numof_reduction_axis() > 1:
            return 1

        return len(self.tiling_axis)

    #  indexing.mask_str is None , see varmean_test.py
    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        if not self.inside_reduction:
            raise RuntimeError("assert self.inside_reduction")

        self.inside_reduction = False
        indexing = self.indexing(index, block_ptr=True)
        self.inside_reduction = True
        var = self.args.output(name)
        if isinstance(indexing, BlockPtrOptions):
            self.post_loop_store.writeline(
                DeferredLine(
                    name,
                    self.codegen_block_ptr_store_line(
                        name,
                        indexing,
                        indexing.format(var),
                        value,
                        f", boundary_check={indexing.boundary_check()!r}",
                    ),
                )
            )
        else:
            if not isinstance(indexing, IndexingOptions):
                raise RuntimeError("assert isinstance(indexing, IndexingOptions)")
            line = f"tl.store({var} + ({indexing.index_str} ), {value}, {indexing.mask_str})"
            if self.numof_reduction_axis() > 1:
                line = f"tl.store({var} + ({indexing.index_str} + tl.arange(0,1) ), {value}, {indexing.mask_str})"
            self.post_loop_store.writeline(
                DeferredLine(name, line)
            )

    # apply new var in case dim are permuted/broadcast
    def store(
            self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:

        var = self.args.output(name)
        original_index = index
        index_analyze = IndexAnalysis(self, index, is_store_index=True)
        index_analyze.analyze_index()
        indexing = self.indexing(index, dense_indexing=True, block_ptr=mode is None, index_analyze=index_analyze)
        index_str = indexing.index_str
        value_str = f"{value}"
        mask_str = indexing.mask_str

        if index_analyze.need_permute:
            value_str = value_str.replace(f"{value}", f"{value}{index_analyze.generate_statement()}")

        advance_block_ptr = None
        if isinstance(indexing, BlockPtrOptions):
            block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                name, var, indexing
            )
            # block_ptr stores don't do implicit casting
            line = self.codegen_block_ptr_store_line(
                name, indexing, block_ptr, value, other
            )
        elif mode is None:
            line = f"tl.store({var} + ({index_str}), {value_str}, {mask_str})"
            if self.numof_reduction_axis() > 1:
                line = f"tl.store({var} + ({index_str} + tl.arange(0,1) ), {value_str}, {indexing.mask_str})"

        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({index_str}), {value_str}, {indexing.mask_str})"
        else:
            raise NotImplementedError(f"store mode={mode}")

        self.stores.writeline(DeferredLine(name, line))
        if advance_block_ptr:
            self.stores.writeline(advance_block_ptr)

        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def find_reduction_node(self):
        node = self.current_node
        if node is not None and isinstance(node, SchedulerNode):
            reduction = node.node.data
            if reduction is not None and isinstance(reduction, ir.Reduction):
                return reduction

        for node in self.node_schedule:
            if node in (EnableReduction, DisableReduction):
                continue
            reduction = node.node.data
            if reduction is not None and isinstance(reduction, ir.Reduction):
                return reduction

        return None

    # select the golden varlist, from to which to deduce permute, broadcast shape 
    def select_golden_varlist(self):
        longest = None
        maximum_length = 0
        self.golden_var_list = None

        def all_tiling_in_var_list(var_list):
            return all([x in var_list for x in self.tiling_axis])

        # all are load indexings, select the longest as gold
        for index in self.load_store_indexing:
            index = index.subs(V.graph.sizevars.var_to_val)
            analyze = IndexAnalysis(self, index)
            if len(analyze.var_list) > maximum_length and all_tiling_in_var_list(analyze.var_list):
                longest = analyze.var_list
                maximum_length = len(longest)
        # this may cause problems
        if not longest:
            self.golden_var_list = tuple([x.symbol() for x in self.tiling_axis]) if self.tiling_axis else []
        else:
            self.golden_var_list = tuple([x for x in longest if x in self.tiling_axis]) if self.tiling_axis else []
        if self.golden_var_list is None:
            raise RuntimeError("assert self.golden_var_list is None")

        # to generate shape of the tile

    def dense_size_list(self) -> List[str]:
        if self.inside_reduction:
            if not self.reduce_analysis:
                self.reduce_analysis = ReductionAnalysis(self)
            return self.reduce_analysis.dense_size_list()

        if not self.golden_var_list:
            self.select_golden_varlist()

        golden_var_list = self.golden_var_list if self.golden_var_list else [x.symbol() for x in self.tiling_axis]
        if golden_var_list is None:
            raise RuntimeError("assert golden_var_list is None")
        sizes = [None for _ in golden_var_list]
        for i, var in enumerate(reversed(golden_var_list)):
            axis = self.range_tree_nodes[var]
            sizes[i] = f"{axis.name.upper()}BLOCK_SUB"
        return sizes

    def dense_size_str(self):
        if self.inside_reduction:
            if not self.reduce_analysis:
                self.reduce_analysis = ReductionAnalysis(self)
            return self.reduce_analysis.dense_size_str()
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

    # and add to shape to value
    def reduction_resize(self, value, dim):
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})"
        dense_list = self.dense_size_list()
        dense_list[dim] = "1"
        expand_str = ", ".join(dense_list)
        return f"{value}.reshape({expand_str})"

    # to determine reduction_dim
    def reduction_dim(self):
        if not self.reduce_analysis:
            self.reduce_analysis = ReductionAnalysis(self)
        return self.reduce_analysis.reduced_dim

    def filter_masks(self, mask_vars):
        for node in self.sorted_axis:
            if not (node.is_tiling_axis):
                mask_vars.discard(f"{node.name}_mask")

    def numof_reduction_axis(self):
        root = self.range_trees[-1]
        if root is None:
            return 0

        return len(root.var_list)

    def reduction_axis_list(self):
        root = self.range_trees[-1]
        if root is None:
            return []
        return root.var_list

    def reduction(
            self,
            dtype: torch.dtype,
            src_dtype: torch.dtype,
            reduction_type: ReductionType,
            value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        if not self.inside_reduction:
            raise RuntimeError("assert self.inside_reduction")
        masks = {f"{node.symbol()}_mask" for node in self.sorted_axis}
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix
        if not self.reduce_analysis:
            self.reduce_analysis = ReductionAnalysis(self)
        dense_size_str = self.dense_size_str()

        if len(dense_size_str) > 2:
            value = self._map_tuple_or_scalar(
                lambda v: self.cse.generate(
                    self.compute, f"tl.reshape({v}, {dense_size_str})", dtype=v.dtype,
                ),
                value,

            )

        dim: int
        root_op: str

        def final_reduction(value):
            module = "tl"  # use tl
            if reduction_type in {"max", "min"}:
                return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})", dim)
            return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})", dim)

        def final_argreduce(buffer, result_var, value, index):
            buffer.splice(
                f"""\
                _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f'{result_var}_tmp', dim)}
                """
            )

        def get_reduction_axis():
            return list(self.range_tree_nodes.values())[-1]

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        dim = self.reduction_dim()
        acc_type = triton_acc_type(src_dtype)
        torch_acc_type = upcast_acc_dtype(src_dtype)
        result_var: Any = self.cse.newvar(dtype=torch_acc_type)
        result_var.mask_vars = {var for var in masks if var[0] != "r"}
        cond = " & ".join(masks)

        def where_cond(tval, fval):
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)

        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(constant_repr, default)

            def _mask_value(value, default):
                return self.cse.generate(self.compute, where_cond(value, default), dtype=value.dtype)

            #  masked_value doesn't work dual reduction
            if self.numof_reduction_axis() == 1:
                if isinstance(value, tuple):
                    masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
                else:
                    masked_value = _mask_value(value, default)
            else:
                masked_value = value

            if reduction_type in {"argmax", "argmin", "max", "min"}:
                reduce_axis = get_reduction_axis()
                broadcast_string: str
                reshape_str = self.reduce_analysis.get_reduce_dim_reshape(reduce_axis)
                broadcast_string = f"tl.broadcast_to({reduce_axis.symbol()}.reshape({reshape_str}), {masked_value}.shape)"
                accumulator_index = str(
                    self.cse.generate(
                        self.compute,
                        broadcast_string,
                        dtype=torch.int64
                    )
                )
                if reduction_type == "argmax" or reduction_type == "argmin":
                    root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                    final_argreduce(
                        self.compute, result_var, masked_value, accumulator_index
                    )
                elif reduction_type == "max" or reduction_type == "min":
                    result_var = self.cse.generate(
                        self.compute, final_reduction(masked_value), dtype=masked_value.dtype,
                    )
            elif reduction_type == "welford_reduce":
                raise RuntimeError("assert False, welford_reduction and is not supported now..")
            elif reduction_type == "welford_combine":
                raise RuntimeError("assert False, welford_combine and is not supported now..")
            else:
                result_var = self.cse.generate(
                    self.compute, final_reduction(masked_value), dtype=masked_value.dtype,
                )
        else:
            accumulator = self.cse.namedvar(f"_{result_var}", dtype=torch_acc_type)
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(constant_repr, default)
            if not isinstance(default, tuple):
                self.prefix.writeline(
                    f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                )

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = f"_{result_var}_index"
                long_max = torch.iinfo(torch.int64).max
                self.prefix.writeline(
                    f"{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)"
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]

                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                    {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index
                )
                {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
                {accumulator_index} = {where_cond(f'{accumulator_index}_next', accumulator_index)}
                """
                )
                final_argreduce(self.post_loop_store, result_var, accumulator, accumulator_index)
            elif is_welford_reduction(reduction_type):
                raise RuntimeError("assert False, welford_reduction and is not supported now..")
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                self.compute.writeline(
                    f"{accumulator} = {where_cond(updated, accumulator)}"
                )

                if src_dtype == torch.bool:
                    accumulator = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.post_loop_store.writeline(
                        f"{result_var} = {final_reduction(accumulator)}.to({result_type})"
                    )
                else:
                    self.post_loop_store.writeline(
                        f"{result_var} = {final_reduction(accumulator)}"
                    )

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            self.outside_loop_vars |= set(result_var)
        else:
            self.outside_loop_vars.add(result_var)

        return result_var

    # broadcast, permute handling
    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        original_index = index
        store_cache = self.cse.store_cache
        if name in store_cache:
            result_var = store_cache[name]
            return result_var

        index_analyze = IndexAnalysis(self, index)
        index_analyze.analyze_index()
        indirect_indexing = self.is_indirect_indexing(index)
        indexing = self.indexing(index, block_ptr=True)
        has_rindex = indexing.has_rindex()
        has_tmpmask = indexing.has_tmpmask()
        is_coalesced = any(
            i == 1 for i in self.get_strides_of_load(original_index).values()
        )
        ep = ""
        if (
                (has_tmpmask or has_rindex)
                and V.graph.get_dtype(name) != torch.bool
                and indexing.has_mask()
        ):
            other = ", other=0.0"
        else:
            other = ""

        advance_block_ptr = None
        append_broadcast = None
        dtype = V.graph.get_dtype(name)

        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(indexing, BlockPtrOptions):
                block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                line = f"tl.load({block_ptr}{other}{ep})"
                # add needed size=1 dimensions
                line = triton_reshape(
                    line, indexing.block_shape, indexing.reshape_suffix
                )
            elif isinstance(original_index, sympy.Integer):
                line = f"tl.load({var} + ({original_index}))"
                full_list = ["1"] * (len(self.tiling_axis) if self.tiling_axis else 1)
                append_broadcast = f"[{', '.join(full_list)} ]"
            else:
                index_str = indexing.index_str
                mask_str = indexing.mask_str
                line = f"tl.load({var} + ({index_str}), {mask_str}{ep}{other})"

            dtype = V.graph.get_dtype(name)
            if dtype in (torch.bfloat16,):
                line += ".to(tl.float32)"
            if dtype == torch.bool and torch.version.hip is None:
                line += ".to(tl.int1)"
        if has_tmpmask:
            # Masked loads must come after the mask is computed
            load_buffer = self.compute
        elif (
                self.inside_reduction
                and self.range_trees[-1].is_loop
                and not indirect_indexing
                and not has_rindex
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            load_buffer = self.prefix

        else:
            load_buffer = self.loads

        result_var = self.cse.generate(load_buffer, line, dtype=dtype)
        if not (isinstance(result_var, TritonCSEVariable)):
            raise RuntimeError("assert isinstance(result_var, TritonCSEVariable)")
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast and append_broadcast != '[]':
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(load_buffer, line, dtype=dtype)
        # triton can handle broadcast
        elif index_analyze.need_permute:
            line = f"{result_var}{index_analyze.generate_statement()}"
            result_var = self.cse.generate(self.loads, line, dtype=dtype)

        if advance_block_ptr:
            load_buffer.writeline(advance_block_ptr)

        if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
            self.outside_loop_vars.add(result_var)

        return result_var

    # don't call symlify_indexing
    def prepare_indexing(
            self,
            index: sympy.Expr,
            index_analyze,
            is_index_expr=False
    ):
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)

        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                        symbol_is_type(s, (SymT.SIZE, SymT.PRECOMPUTED_SIZE))
                        for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        simp_index = index

        simp_index = (
            simp_index if not isinstance(simp_index, Identity) else simp_index.args[0]
        )

        # to generate range.var_directions for permuted axis
        index_analyze.analyze_index()
        return self.codegen_indexing(simp_index)

    def replace_index_vars(self, index, index_analyze):

        new_index = index
        if index_analyze.var_replacements:
            new_index = sympy_subs(index, index_analyze.var_replacements)
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        index = self.rename_indexing(index)
        return self.kexpr(index)  # type: ignore[call-arg]

    # 1. only remove the line which asserts index var should be in "xyr"
    # 2. don't do simplify_indexing, which combine continuous dims
    # 3. removed block_ptr, removed dense mask/broadcast support
    #  dense_mask_vars should be generated from sorted_axis
    # upgraded to torch251
    def indexing(
            self,
            index: sympy.Expr,
            *,
            copy_shape=None,
            dense_indexing=False,
            override_mask=None,
            block_ptr=False,
            index_analyze=None,
            is_index_expr=False
    ) -> Union[IndexingOptions, BlockPtrOptions]:
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        if not index_analyze:
            index_analyze = IndexAnalysis(self, index, is_index_expr=is_index_expr)
        index_analyze.analyze_index()

        index = self.prepare_indexing(index, index_analyze, is_index_expr)
        index_vars = index.free_symbols
        has_rindex = False
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                        s.name.startswith("s") or s.name.startswith("ps") for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        # if not self.inside_reduction :
        index = self.replace_index_vars(index, index_analyze)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars: Set[str] = set()
        for var in index_vars:
            if not (isinstance(var, sympy.Symbol)):
                raise RuntimeError("assert isinstance(var, sympy.Symbol)")

            has_rindex = has_rindex or var.name.startswith("r")
            if override_mask:
                pass
            elif var.name.startswith("tmp"):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif var.name.startswith(("s", "ps", "i")):
                pass
            else:
                # var is one of xN, yN or rN
                mask_vars.add(f"{var.name}_mask")

        expand_str = None
        index_str = self.index_to_str(index)

        if isinstance(index, sympy.Integer):
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            if (index != 0):
                index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            else:
                index_str = f"tl.arange(0,1)"
            return IndexingOptions(index_str, set(), "None", expand_str, has_rindex, index)

        if override_mask:
            mask_vars = {override_mask}
        if self._load_mask:
            mask_vars.add(self._load_mask)
        self.filter_masks(mask_vars)
        mask_str = " & ".join(sorted(map(str, mask_vars))) if mask_vars else "None"
        return IndexingOptions(index_str, mask_vars, mask_str, expand_str, has_rindex, index)  # type: ignore[arg-type]

    def codegen_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                # if indexing expression is complicated, we precompute it on the host side
                # and send the result as a kernel argument
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():  # type: ignore[index]
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(  # type: ignore[index]
                        self.range_tree_nodes[sym].expr, replacements  # type: ignore[index]
                    )
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        return expr

    #  when xindex(16) -> x2:2,x3:8, when new length:16 in , should return (x2,x3)
    def split_and_set_ranges(self, lengths: Sequence[Sequence[sympy.Expr]]):
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.S.One

        return self.map_kernel_groups_to_node_sizes(groups, lengths, self.set_ranges)

    # support split multiple ranges (instead of double) from one flatten range, triple-ranges are needed in mamba model
    @staticmethod
    def _split_iteration_ranges(
            groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
    ):
        sv = V.graph.sizevars
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        for i, group in enumerate(remaining):
            if isinstance(group, (list, tuple)):
                remaining[i] = NumelList(group).numels()

        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit()
            # guard on the last item out
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(strides, index_list):
            def getter(flat_vars):
                expr = sympy.Integer(0)
                for stride, index in zip(strides, index_list):
                    expr = stride * flat_vars[index] + expr
                return expr

            return getter

        def size_hints(group):
            if isinstance(group, (list, tuple)):
                return sv.size_hint(NumelList(group).numels())
            return sv.size_hint(group)

        def add_multiple_range(size, return_getters):
            # need to break size in multiple
            index_list = []
            stride_list = []
            group = current_group
            remained_size = size
            # Two checks:
            # 1. remaining sizes to be merged
            # 2. remained_size is already divided to 1
            while (group < len(remaining) and remaining[group] > 1) and (remained_size > 1):
                group_size = remaining[group]
                # size should be divisible by group_size
                if not sv.statically_known_multiple_of(remained_size, group_size):
                    raise CantSplit()
                index_list.append(add_range(group, group_size))
                remained_size = FloorDiv(remained_size, group_size)
                stride_list.append(remained_size)
                group = group + 1
            if remained_size != 1:
                raise CantSplit()
            return_getters.append(make_combined(stride_list, index_list))

        return_getters_groups = []
        current_group = 0

        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue

                while (
                        current_group < len(remaining)
                        and size_hints(remaining[current_group]) == 1
                ):
                    # scroll to next group with remaining elements
                    current_group += 1
                size_hint = sv.size_hint(size)
                if size_hint > size_hints(remaining[current_group]):
                    # add multiple ranges (two or more) to the list, as well as the getter funcs
                    add_multiple_range(size_hint, return_getters)
                else:
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size_hint))
                    )
            return_getters_groups.append(return_getters)

        if not (all(V.graph.sizevars.size_hint(s) == 1 for s in remaining)):
            raise RuntimeError("assert all(V.graph.sizevars.size_hint(s) == 1 for s in remaining)")

        return new_ranges, return_getters_groups

    # torch260 done
    # just to override load method of CSEProxy, however, CSEProxy is an inner which can not be monkey patched,
    # we need to override the whole inner class
    def __enter__(self):
        class CSEProxy:
            self.name = "CSEProxy"
            vr_analysis = ValueRangeAnalysis()

            @staticmethod
            def __getattr__(name: str) -> Callable[..., CSEVariable]:  # type: ignore[misc]
                def inner(*args, **kwargs):
                    bounds = CSEProxy._bound_variable(name, *args, **kwargs)

                    value = getattr(parent_handler, name)(*args, **kwargs)  # type: ignore[has-type]
                    dtype_handler = DtypePropagationOpsHandler()

                    output_idx = 0

                    def do_cse(v):
                        # cpp backend doesnt set current device
                        if V.graph.current_device is not None:
                            device_str = V.graph.get_current_device_or_throw().type
                            triton_backend = (
                                config.cpu_backend == "triton"
                                if device_str == "cpu"
                                else config.cuda_backend == "triton"
                            )
                        else:
                            triton_backend = False

                        # only triton backend tracks dtype currently
                        if triton_backend:
                            if name == "masked":
                                output_dtype = value.dtype
                            else:
                                output_dtype = getattr(
                                    dtype_handler,
                                    name,
                                )(*args, **kwargs)
                        else:
                            # cpp backend doesnt track dtype yet
                            output_dtype = None

                        csevar = V.kernel.cse.generate(
                            V.kernel.compute,
                            v,
                            bounds=bounds,
                            dtype=output_dtype,
                        )

                        nonlocal output_idx
                        if (
                                config.test_configs.runtime_triton_dtype_assert
                                and triton_backend
                        ):
                            from torch._inductor.codegen.triton import triton_type

                            # we tree_map over the output, so we need to fetch corresponding dtype
                            if isinstance(output_dtype, (list, tuple)):
                                output_dtype = output_dtype[output_idx]

                            V.kernel.compute.writeline(
                                f"tl.static_assert({csevar}.dtype == {triton_type(output_dtype)})"
                            )
                        output_idx += 1

                        csevar.update_on_args(name, args, kwargs)

                        return csevar

                    return pytree.tree_map(do_cse, value)

                return inner

            @staticmethod
            def _bound_variable(name, *args, **kwargs):
                """
                If the variable comes from an FX node, we forward the bound we have already computed
                Else, if the variable when codegen'ing another op, we try to compute its bounds
                """
                from torch._inductor.select_algorithm import TritonTemplateKernel

                if isinstance(V.kernel, TritonTemplateKernel):
                    return ValueRanges.unknown()

                fx_node = V.interpreter.current_node
                if fx_node.target == name and self.node_to_bounds is not None:
                    if not (isinstance(self.node_to_bounds, dict)):
                        raise RuntimeError("assert isinstance(self.node_to_bounds, dict)")

                    return self.node_to_bounds.get(fx_node, ValueRanges.unknown())
                elif config.compute_all_bounds and hasattr(ValueRangeAnalysis, name):
                    # These create lots of inner strings. We would need to compute the bounds at the ops
                    # We will also likely not get much from computing VRs on these nodes
                    if any(
                            s in fx_node.target
                            for s in ("set_indirect", "reduction", "scan")
                    ):
                        return ValueRanges.unknown()

                    # We assume that the inputs come from `ops.` and are not strings. If you want to generate
                    # intermediary strings, wrap them in CSE variables with properly initialised bounds.

                    # If there is no FX bound but we know how to compute one we do so
                    if (kwargs):
                        raise RuntimeError("assert not kwargs")

                    def arg_to_bound(x):
                        if isinstance(x, CSEVariable):
                            return x.bounds
                        elif isinstance(x, sympy.Expr):
                            return bound_sympy(x)
                        else:
                            return x

                    arg_bounds = list(map(arg_to_bound, args))
                    return getattr(CSEProxy.vr_analysis, name)(*arg_bounds)
                return ValueRanges.unknown()

            @staticmethod
            def indirect_indexing(
                    var: CSEVariable,
                    size: Union[sympy.Expr, int],
                    check: bool = True,
                    wrap_neg=True,
            ):
                if isinstance(size, int):
                    size = sympy.Integer(size)
                if not (isinstance(size, sympy.Expr)):
                    raise RuntimeError("assert isinstance(size, sympy.Expr), size")
                # Skip CSE since this doesn't return an expression

                if var.bounds.lower < 0:  # type: ignore[operator]
                    if wrap_neg:
                        stm = ops.add(var, ops.index_expr(size, torch.long))
                        # Mixed negative and non-negative
                        if var.bounds.upper >= 0:  # type: ignore[operator]
                            lt = ops.lt(var, 0)
                            stm = ops.where(lt, stm, var)
                    else:
                        stm = var

                    # Propagate bounds as we know how to compute them properly
                    new_bounds = ValueRanges.unknown()
                    if var.bounds != ValueRanges.unknown() and isinstance(
                            size, sympy.Number
                    ):
                        # Take the negative part of the bound and add size to it
                        # Then take union of that and the positive part
                        # This is a tighter bound than that of a generic ops.where, as we have info on the cond
                        neg_bounds = var.bounds & ValueRanges(-int_oo, -1)
                        new_bounds = ValueRanges(
                            neg_bounds.lower + size, neg_bounds.upper + size
                        )
                        # We don't have a good way of representing the empty range
                        if var.bounds.upper >= 0:  # type: ignore[operator]
                            pos = var.bounds & ValueRanges(0, int_oo)
                            new_bounds = new_bounds | pos

                    var = self.cse.generate(self.compute, stm, bounds=new_bounds)

                sympy_var = parent_handler.indirect_indexing(var, size, check)
                if generate_assert(check):
                    assert_lower = not (var.bounds.lower >= 0)
                    # value ranges cannot x < s when x and s are symbols
                    assert_upper = not isinstance(size, sympy.Number) or not (
                            var.bounds.upper < size
                    )
                    self.check_bounds(sympy_var, size, assert_lower, assert_upper)
                return sympy_var

            @staticmethod
            def check_bounds(
                    expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
            ):
                return self.check_bounds(expr, size, lower, upper)

            @staticmethod
            def load(name: str, index: sympy.Expr) -> CSEVariable:
                if name in self.cse.invalidated_stores:
                    # A load from an invalidated store requires us to
                    # keep the actual buffer around
                    V.kernel.must_keep_buffers.add(name)
                if free_symbol_is_type(index, SymT.TMP):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if name in store_cache:
                    return self.load(name, index)
                out = self.load(name, index)
                # count load that is not in the store_cache, and also not in the
                # cse cache.
                if out.use_count == 1:
                    self.num_load += 1
                return out

            @staticmethod
            def _update_store_cache(name: str, value: CSEVariable):
                self.cse.store_cache[name] = value
                if self.current_node and name in V.graph.name_to_buffer:
                    buf = self.current_node.get_output(name)
                    for other_name in buf.get_mutations():
                        self.cse.store_cache[other_name] = value

            @staticmethod
            def store(
                    name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
            ) -> None:
                self.store_buffer_names.add(name)
                if mode is None:
                    CSEProxy._update_store_cache(name, value)
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, mode=mode)
                return None  # type: ignore[return-value]

            @staticmethod
            def store_reduction(name: str, index: sympy.Expr, value: CSEVariable):
                self.store_buffer_names.add(name)
                CSEProxy._update_store_cache(name, value)

                if name not in V.graph.removed_buffers:
                    return self.store_reduction(name, index, value)
                raise RuntimeError("store_reduction")

            @staticmethod
            def reduction(
                    dtype: torch.dtype,
                    src_dtype: torch.dtype,
                    reduction_type: ReductionType,
                    value: Union[CSEVariable, Tuple[CSEVariable, ...]],
            ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
                self.num_reduction += 1
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def scan(
                    dtypes: Tuple[torch.dtype, ...],
                    combine_fn: Callable[
                        [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]],
                        Tuple[CSEVariable, ...],
                    ],
                    values: Tuple[CSEVariable, ...],
            ) -> Tuple[CSEVariable, ...]:
                return self.scan(dtypes, combine_fn, values)

            @staticmethod
            def sort(
                    dtypes: Tuple[torch.dtype, ...],
                    values: Tuple[CSEVariable, ...],
                    stable: bool,
                    descending: bool,
            ) -> Tuple[CSEVariable, ...]:
                return self.sort(dtypes, values, stable, descending)

            @staticmethod
            def bucketize(
                    values: CSEVariable,
                    boundaries: Tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
                    boundary_indices: CSEVariable,
                    indexing_dtype: torch.dtype,
                    right: bool,
                    sorter: Optional[Tuple[str, sympy.Expr]] = None,
                    sorter_indices: Optional[CSEVariable] = None,
            ) -> CSEVariable:
                return self.bucketize(
                    values,
                    boundaries,
                    boundary_indices,
                    indexing_dtype,
                    right,
                    sorter,
                    sorter_indices,
                )

        # Use mypy to check protocol implemented correctly
        def _typecheck_CSEProxy(h: CSEProxy) -> OpsHandler[CSEVariable]:
            return h

        super().__enter__()
        if not (self.overrides):
            raise RuntimeError("assert self.overrides")
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self
