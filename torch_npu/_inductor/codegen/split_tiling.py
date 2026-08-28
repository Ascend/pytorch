from functools import reduce
import sympy as sympy
import torch
from torch._inductor.codegen.simd import EnableReduction, DisableReduction
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.dependencies import MemoryDep
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import ModularIndexing, sympy_subs
from torch._inductor.virtualized import V

from .triton_utils import get_byte_per_numel
from .. import config as npu_config
from ..config import num_vector_core, log
from ..runtime.symbolic_grouping import GroupFeatureSpec, GroupedKernelMeta
from torch_npu._compat.inductor import get_sizevars_backed_var_to_val


_ELEMENTWISE_UNSUPPORTED_OPS = ("masked", "scan", "sort", "rand", "randn", "load_seed")
_NEUTRAL_CONSTANT_OPS = frozenset(("constant", "store", "output"))
_LARGE_GROUP_SYMBOLIC_AXIS = 128


def _add_wide_group_midpoints(boundaries):
    boundaries = sorted({int(boundary) for boundary in boundaries})
    expanded = []
    for index, boundary in enumerate(boundaries):
        if index and boundary // boundaries[index - 1] > 8:
            lower = boundaries[index - 1]
            upper_midpoint = next_power_of_2((boundary + 1) // 2)
            lower_limit = next_power_of_2(lower * 8)
            expanded.append(min(lower_limit, upper_midpoint))
        expanded.append(boundary)
    return tuple(expanded)


# split and tiling axis selector
class SplitTiling:
    def __init__(self, kernel: TritonKernel):
        self.kernel = kernel
        self.indexing = []  # load and store indexing  among all scheduler nodes
        kernel.sorted_axis = list(kernel.range_tree_nodes.values())
        kernel.sorted_axis.sort(reverse=True, key=self.key)
        for i, dim in enumerate(kernel.sorted_axis):
            dim.sorted_order = i

        self.find_lowest_dimension()
        self.should_outer_reduce = False
        self.contiguous_reduction = self.is_contiguous_reduction()


    def is_contiguous_reduction(self):
        def is_contiguous_axis(axis_list):
            axis_set = set(axis_list)
            return len(axis_set) == (max(axis_set) - min(axis_set) + 1)

        if self.kernel.numof_reduction_axis() > 1:
            stride_sorted_var_list = self.kernel.parse_golden_from_load_store_index()
            if not stride_sorted_var_list:
                if not self.kernel.golden_var_list:
                    self.kernel.select_golden_varlist()
                stride_sorted_var_list = list(self.kernel.golden_var_list) if self.kernel.golden_var_list else []
            reduction_dim_list = []
            for i, x in enumerate(reversed(stride_sorted_var_list)):
                if x.name[0] == 'r':
                    reduction_dim_list.append(i)
            return is_contiguous_axis(reduction_dim_list)
        return False

    @classmethod
    def key(cls, x):
        # to be higher than x and y
        if x.name[0] == 'w' or x.name[0] == 'v' or x.name[0] == 't':
            return "zz" + x.name
        # to be lower than floor_dir
        elif isinstance(x.expr, ModularIndexing):
            return x.name[0] + "0" + x.name[1:]
        else:
            return x.name

    @staticmethod
    def get_length_val(x):
        length_expr = x.length
        if not isinstance(length_expr, sympy.Integer):
            return length_expr.subs(get_sizevars_backed_var_to_val(V.graph.sizevars))
        else:
            return length_expr

    @classmethod
    def total_split_numels(cls, axis_list):
        numels = [cls.get_length_val(x) for x in axis_list]
        return reduce(lambda x, y: x * y, numels) if numels else 1

    # Split 原则1 ：先做维度合并，再切分 。通过维度合并降维降低split和tiling轴选择策略的复杂性 。
    # Split 原则2 : 切分轴尽量选择高维度的轴, 这样load/store 能够有比较好的线性度 ,
    # Split 原则3 : 规约轴和低维轴不应选为切分轴 。但如果高维规约类融合算子，而且高维尺寸非常大（ >= 64KB），其他维度不足以支持切分，可以考虑对规约轴切分。
    # Split 原则4 ：切分轴的总numel 要超过 aicore总数。切分轴的数量最好不要超过3个(triton 最多支持三维发射）， 因此 如果一点要超， 需要维度合并。
    # Split 原则5 ：Kernel如果包含cat算子，尾轴不做切分。
    def select_split_axis(self):
        self.kernel.split_axis.clear()

        # total numel exceed aicore or total split axis exceed 3
        def meet_stop_condition():
            sv = V.graph.sizevars
            current_numels = self.total_split_numels(self.kernel.split_axis)
            try:
                val = sv.optimization_hint(current_numels)
            except TypeError:
                if len(self.kernel.sorted_axis) <= 2:
                    return len(self.kernel.split_axis) >= 1
                else:
                    return len(self.kernel.split_axis) >= 2
            if val >= num_vector_core and val // ((val + num_vector_core - 1) // num_vector_core) > num_vector_core * 0.8:
                return True
            if len(self.kernel.split_axis) == 3:
                return True
            return False

        def select_one_split_axis(not_reduction=True, not_low_dims=True):
            for axis in self.kernel.sorted_axis:
                if not_reduction and axis.prefix == "r":
                    continue
                if not_low_dims and axis.sorted_order in self.kernel.low_dims:
                    continue
                if axis in self.kernel.split_axis:
                    continue
                axis.is_split_axis = True
                return axis
            return None

        count = 0
        while not meet_stop_condition():
            count += 1
            axis = select_one_split_axis(not_reduction=True, not_low_dims=True)
            if axis is not None:
                self.kernel.split_axis.append(axis)
                continue
            axis = select_one_split_axis(not_reduction=True, not_low_dims=False)
            if axis is not None:
                self.kernel.split_axis.append(axis)
                continue
            if count > 10:
                break

        if not self.kernel.split_axis and self.kernel.sorted_axis:
            only_axis = self.kernel.sorted_axis[0]
            # A dynamic full reduction (every axis is a reduction axis, at least one
            # dynamic) must keep grid==1: making a reduction axis a grid split axis
            # gives grid>1 with no cross-core combine (A5), which overwrites the
            # scalar output. Leave split_axis empty (loop the runtime numel with a
            # tail mask). Covers pure 1D ([n]) and full reduce with extra static
            # reduction dims ([n, 1024]).
            all_reduction = all(
                axis.prefix == "r" for axis in self.kernel.sorted_axis
            )
            any_dynamic = any(
                not isinstance(axis.length, sympy.Integer)
                for axis in self.kernel.sorted_axis
            )
            is_dynamic_full_reduction = all_reduction and any_dynamic
            if not is_dynamic_full_reduction:
                self.kernel.split_axis.append(only_axis)
                only_axis.is_split_axis = True

        self.kernel.split_axis.sort(reverse=True, key=self.key)
        for i, x in enumerate(self.kernel.split_axis):
            x.split_order = i

    # Tiling 原则1：load / store 中索引表达式的中的低维轴都要成为tiling轴.
    # Tiling 原则2：对于规约算子，规约轴要成为tiling轴。
    # Tiling 原则3: 多维规约， 只有规约轴可以被选择为tiling轴
    # Tiling 原则4: tiling轴 要覆盖 total numel 的 80%

    # two tiling axis might be insufficient when there're 3 or more low-dims in indexing
    def select_tiling_axis(self):
        self.kernel.tiling_axis.clear()

        #  cover the biggest axis and not exceed 3 axis
        def meet_stop_condition():
            total_numel = (
                reduce(
                    lambda x, y: x + y,
                    (self.get_length_val(x) for x in self.kernel.sorted_axis),
                )
                if self.kernel.sorted_axis
                else 1
            )
            tiling_numel = (
                reduce(
                    lambda x, y: x + y,
                    (self.get_length_val(x) for x in self.kernel.tiling_axis),
                )
                if self.kernel.tiling_axis
                else 1
            )

            # currently, the maximum dim that triton-ascend support is 2
            def can_stop():
                return self.kernel.numof_reduction_axis() > 1 and all(
                    self.kernel.range_tree_nodes[var].is_tiling_axis
                    for var in self.kernel.reduction_axis_list()
                ) and not self.contiguous_reduction

            if can_stop():
                return True
            return False

        def select_tiling(low_dim=True, reduction=True):
            for axis in reversed(self.kernel.sorted_axis):
                if (
                    low_dim
                    and axis.sorted_order in self.kernel.low_dims
                    and axis not in self.kernel.tiling_axis
                ):
                    axis.is_tiling_axis = True
                    self.kernel.tiling_axis.append(axis)
                if (
                    reduction
                    and axis.prefix == "r"
                    and axis not in self.kernel.tiling_axis
                ):
                    axis.is_tiling_axis = True
                    self.kernel.tiling_axis.append(axis)
                if low_dim or reduction:
                    continue
                    # using principle 4, select one longest
                longest = axis  # self.find_longest_dimension(check_in_tiling = True)
                if longest and longest not in self.kernel.tiling_axis:
                    self.kernel.tiling_axis.append(longest)
                    longest.is_tiling_axis = True
                if meet_stop_condition():
                    break

        select_tiling(low_dim=True, reduction=True)
        count = 0
        while not meet_stop_condition():
            select_tiling(low_dim=False, reduction=False)
            count += 1
            if count > 10:
                break
        self.kernel.tiling_axis.sort(reverse=True, key=self.key)
        for i, x in enumerate(self.kernel.tiling_axis):
            x.tiling_order = i

    # no_loop_axis 原则1：优先从low_dims tiling轴中选择
    # no_loop_axis 原则2：low_dims 轴仍未超过阈值，从tiling轴中选择其他轴
    # no_loop_axis 原则3：所有轴的所占空间预估小于等于4k时，无需loop
    # no_loop_axis 原则4：对于存在动态shape的轴，不做该优化
    def select_no_loop_axis(self):
        low_dims = [self.kernel.sorted_axis[dim] for dim in self.kernel.low_dims]

        def sort_key(dim):
            length_expr = self.get_length_val(dim)
            if isinstance(length_expr, (int, sympy.Integer)):
                return 0, length_expr
            return 1, sympy.default_sort_key(length_expr)

        sorted_low_dims = sorted(low_dims, key=sort_key)
        total_numels = 1
        axis_dtype = torch.float32
        if self.kernel.split_axis:
            axis_dtype = self.kernel.get_axis_dtype(self.kernel.split_axis[0])
        dtype_byte = get_byte_per_numel(axis_dtype)

        def stop_loop(axis, current_numels):
            is_reduce_or_split_axis = (axis.prefix == 'r' or axis.is_split_axis)
            if (is_reduce_or_split_axis or
                    not axis.is_tiling_axis or
                    axis.is_no_loop_axis):
                return False, current_numels
            if not isinstance(axis.length, sympy.Integer):
                return True, current_numels
            current_numels *= self.get_length_val(axis)
            over_flow = current_numels * dtype_byte > 4 * 1024
            if not over_flow:
                axis.is_no_loop_axis = True
            return over_flow, current_numels

        if self.kernel.persistent_reduction:
            for axis in self.kernel.sorted_axis:
                if axis.prefix == 'r':
                    total_numels *= self.get_length_val(axis)

        for axis in sorted_low_dims:
            overflow, total_numels = stop_loop(axis, total_numels)
            if overflow:
                return

        for axis in reversed(self.kernel.sorted_axis):
            overflow, total_numels = stop_loop(axis, total_numels)
            if overflow:
                return

    def select_split_tiling_axis(self):
        self.select_split_axis()
        self.select_tiling_axis()
        self.apply_grouped_rewrite_if_needed()

    def apply_grouped_rewrite_if_needed(self):
        if not npu_config.enable_symbolic_shape_group_autotune:
            self.kernel.grouped_autotune_meta = None
            return
        if not self._is_grouped_template_enabled():
            self.kernel.grouped_autotune_meta = None
            return
        self.kernel.grouped_autotune_meta = self._build_grouped_meta()

    def _grouped_template_name(self):
        if self.kernel.persistent_reduction:
            return "persistent_reduction"
        if self.kernel.inside_reduction:
            return "reduction"
        return "pointwise"

    def _is_grouped_template_enabled(self):
        template = self._grouped_template_name()
        return template in npu_config.symbolic_group_allow_templates

    def _classify_split_axes(self):
        dynamic_split_axes = []
        static_split_axes = []
        for axis in self.kernel.split_axis:
            if isinstance(axis.length, sympy.Integer):
                static_split_axes.append(axis)
            else:
                dynamic_split_axes.append(axis)
        return dynamic_split_axes, static_split_axes

    def _select_primary_group_axis(self, dynamic_split_axes):
        if not dynamic_split_axes:
            return None
        for axis in self.kernel.sorted_axis:
            if axis in dynamic_split_axes:
                return axis
        return dynamic_split_axes[0]


    def _dynamic_reduction_tiling_axis(self):
        """Sole dynamic reduction axis to bucket by runtime size while it stays a
        tiling axis (grid==1 over it). The reduction axis must NOT itself be a grid
        split axis (A5 has no cross-core combine -> grid>1 would overwrite the
        output); static non-reduction split axes are fine and drive the grid."""
        reduction_dynamic = [
            axis
            for axis in self.kernel.sorted_axis
            if axis.prefix == "r" and not isinstance(axis.length, sympy.Integer)
        ]
        if len(reduction_dynamic) != 1:
            return None
        axis = reduction_dynamic[0]
        if axis in self.kernel.split_axis:
            return None
        return axis

    def _dynamic_pointwise_tiling_axis(self):
        """Return the sole dynamic pointwise axis when it is tiled, not split.

        Transpose and broadcast kernels commonly keep the dynamic inner axis in
        the tiling space while a static outer axis drives the grid.  That axis can
        still select grouped compile variants even though it has no grid block.
        """
        if self.kernel.persistent_reduction or self.kernel.inside_reduction:
            return None
        dynamic_axes = [
            axis
            for axis in self.kernel.sorted_axis
            if not isinstance(axis.length, sympy.Integer)
        ]
        if len(dynamic_axes) != 1:
            return None
        axis = dynamic_axes[0]
        if axis in self.kernel.split_axis or axis not in self.kernel.tiling_axis:
            return None
        if self._pointwise_layout_kind() is None:
            return None
        return axis

    def _pointwise_layout_kind(self):
        """Classify transpose/broadcast from the kernel's memory indexings."""
        axis_symbols = tuple(axis.symbol() for axis in self.kernel.sorted_axis)
        if not axis_symbols or not self.indexing:
            return None

        stride_orders = set()
        has_broadcast = False
        for index in self.indexing:
            axis_strides = []
            unsupported_index = False
            for axis in axis_symbols:
                stride = sympy.expand(index).coeff(axis)
                if stride == 0 and axis in index.free_symbols:
                    unsupported_index = True
                    break
                if stride != 0:
                    axis_strides.append((axis, stride))
            if unsupported_index:
                continue

            present = [axis for axis, _ in axis_strides]
            if len(present) < len(axis_symbols):
                has_broadcast = True
                continue

            strides = []
            for axis, stride in axis_strides:
                try:
                    stride = int(stride)
                except (TypeError, ValueError):
                    try:
                        stride = int(V.graph.sizevars.size_hint(stride))
                    except (AttributeError, KeyError, TypeError, ValueError):
                        strides = []
                        break
                strides.append((stride, axis))
            if strides:
                stride_orders.add(
                    tuple(axis for _, axis in sorted(strides, key=lambda x: x[0]))
                )

        has_transpose = len(stride_orders) > 1
        if has_transpose and has_broadcast:
            return "transpose_broadcast"
        if has_transpose:
            return "transpose"
        if has_broadcast:
            return "broadcast"
        return None

    def non_reduction_axis_names(self):
        return tuple(axis.name for axis in self.kernel.sorted_axis if axis.prefix != "r")

    def all_axis_names(self):
        return tuple(axis.name for axis in self.kernel.sorted_axis)

    def reduction_axis_names(self):
        return tuple(axis.name for axis in self.kernel.sorted_axis if axis.prefix == "r")


    def _has_dynamic_axis(self, axis_names):
        names = set(axis_names)
        for axis in self.kernel.sorted_axis:
            if axis.name in names and not isinstance(axis.length, sympy.Integer):
                return True
        return False

    # Bucket boundaries on the group PRODUCT (reduction_product / outer_product).
    # A closed bucket tunes at its upper bound and the open tail at (max boundary *
    # 2). Kept to a single coarse boundary each. Adding closed buckets to the OUTER
    # (grid) axis backfires: the outer axis bakes XBLOCK from the representative, so
    # a mid-of-bucket runtime size (e.g. 257 in (256, 4096]) gets an XBLOCK tuned
    # for 4096 -> too few programs -> core underutilization. The single (256,) tail
    # (representative 512) is a better all-round compromise. Reduction likewise
    # stays a single boundary; the ~1e6 1D corner is out of scope.
    _REDUCTION_BUCKETS = (8192,)
    _OUTER_BUCKETS = (256,)

    def _axis_static_length(self, axis):
        try:
            return int(self.get_length_val(axis))
        except (TypeError, ValueError):
            try:
                return int(V.graph.sizevars.size_hint(axis.length))
            except (AttributeError, KeyError, TypeError, ValueError):
                return 1

    def _default_dynamic_axis_feature(
        self,
        dynamic_split_axes,
        static_split_axes,
        feature_name="pointwise",
        fallback_dynamic_axis=None,
    ):
        # The default policy is defined for one dynamic axis. Keep unsupported
        # multi-symbol split cases out of this feature construction.
        if len(dynamic_split_axes) > 1:
            return None

        dynamic_axis = (
            dynamic_split_axes[0]
            if dynamic_split_axes
            else fallback_dynamic_axis
        )
        if dynamic_axis is None or isinstance(dynamic_axis.length, sympy.Integer):
            return None
        axis_order = {
            axis.name: index for index, axis in enumerate(self.kernel.sorted_axis)
        }
        dynamic_order = axis_order[dynamic_axis.name]
        prefix_axes = [
            axis
            for axis in static_split_axes
            if axis_order[axis.name] < dynamic_order
        ]
        suffix_axes = [
            axis
            for axis in static_split_axes
            if axis_order[axis.name] > dynamic_order
        ]

        # Static split axes retain their split role. Axes before the dynamic
        # axis are part of the grouped outer-product feature. All split axes
        # are excluded from the tiling product used to classify the dynamic
        # workload, so the resulting bounds already describe that product.
        split_axis_names = {axis.name for axis in self.kernel.split_axis}
        tiling_product = 1
        for axis in self.kernel.tiling_axis:
            if axis.name not in split_axis_names and axis.name != dynamic_axis.name:
                tiling_product *= max(1, self._axis_static_length(axis))

        vector_core = int(num_vector_core)
        dtype_axis = (
            self.kernel.split_axis[0]
            if self.kernel.split_axis
            else dynamic_axis
        )
        axis_dtype = self.kernel.get_axis_dtype(dtype_axis)
        dtype_bytes = max(1, get_byte_per_numel(axis_dtype))
        base = max(1, (4096 * vector_core + tiling_product - 1) // tiling_product)
        lower = max(
            (4 * 1024) // dtype_bytes // max(1, tiling_product),
            next_power_of_2(2 * vector_core),
        )
        prefix_product = 1
        for axis in prefix_axes:
            prefix_product *= max(1, self._axis_static_length(axis))
        upper = max(
            prefix_product * _LARGE_GROUP_SYMBOLIC_AXIS,
            next_power_of_2(8 * vector_core),
            base,
        )
        boundaries = [lower, upper]
        if suffix_axes:
            boundaries.insert(0, next_power_of_2(max(1, vector_core // 2)))
        boundaries = _add_wide_group_midpoints(boundaries)

        feature_axis_names = tuple(axis.name for axis in (*prefix_axes, dynamic_axis))
        return GroupFeatureSpec(
            feature_name,
            "outer_product",
            feature_axis_names,
            boundaries,
        )

    def _build_group_features(
        self,
        workload,
        primary_axis,
        pointwise_layout=None,
        dynamic_split_axes=(),
        static_split_axes=(),
    ):
        if self.kernel.persistent_reduction or self.kernel.inside_reduction:
            outer_names = self.non_reduction_axis_names()
            reduction_names = self.reduction_axis_names()
            # Emit a feature only for a group with a dynamic axis: a static group
            # has a constant product (one reachable bucket), so bucketing it just
            # adds unreachable group ids. Works for the symbolic dim on either
            # side. Reduction axis stays grid==1; grid comes from outer split axes.
            features = []
            if outer_names and self._has_dynamic_axis(outer_names):
                features.append(
                    GroupFeatureSpec(
                        "outer", "outer_product", outer_names, self._OUTER_BUCKETS
                    )
                )
            if reduction_names and self._has_dynamic_axis(reduction_names):
                features.append(
                    GroupFeatureSpec(
                        "reduction",
                        "reduction_product",
                        reduction_names,
                        self._REDUCTION_BUCKETS,
                    )
                )
            return tuple(features)
        if workload == "elementwise":
            default_feature = self._default_dynamic_axis_feature(
                dynamic_split_axes,
                static_split_axes,
                feature_name="elementwise_numel",
                fallback_dynamic_axis=primary_axis,
            )
            if default_feature is not None:
                return (default_feature,)
            return (
                GroupFeatureSpec(
                    "elementwise_numel",
                    "outer_product",
                    self.all_axis_names(),
                    (num_vector_core * 4096,),
                ),
            )
        if pointwise_layout in ("broadcast", "transpose_broadcast"):
            return (
                GroupFeatureSpec(
                    "pointwise_broadcast_axis",
                    "axis",
                    (primary_axis.name,),
                    (16, 64, 256, 1024, 4096),
                ),
            )
        if pointwise_layout == "transpose":
            return (
                GroupFeatureSpec(
                    "pointwise_transpose_axis",
                    "axis",
                    (primary_axis.name,),
                    (64, 128, 256, 512),
                ),
            )
        default_feature = self._default_dynamic_axis_feature(
            dynamic_split_axes, static_split_axes
        )
        return (default_feature,) if default_feature is not None else ()

    @staticmethod
    def _alpha_rename_access_vars(dep):
        replacements = {
            var: sympy.Symbol(f"elementwise_dim_{idx}", integer=True, nonnegative=True)
            for idx, var in enumerate(dep.var_names)
        }
        return replacements

    def _make_access_signature(self, dep):
        if dep.is_indirect() or dep.mode is not None:
            return None
        normalized = dep.normalize()
        replacements = self._alpha_rename_access_vars(normalized)
        index = sympy_subs(
            normalized.index - normalized.get_offset(),
            replacements,
        )
        sizes = tuple(sympy_subs(size, replacements) for size in normalized.size)
        return (
            V.graph.sizevars.simplify(index),
            tuple(V.graph.sizevars.simplify(size) for size in sizes),
        )

    @staticmethod
    def _same_access_signature(left, right):
        left_index, left_sizes = left
        right_index, right_sizes = right
        if len(left_sizes) != len(right_sizes):
            return False
        sizevars = V.graph.sizevars
        return sizevars.statically_known_equals(left_index, right_index) and all(
            sizevars.statically_known_equals(left_size, right_size)
            for left_size, right_size in zip(left_sizes, right_sizes)
        )

    def _all_access_signatures_equal(self, signatures):
        if not signatures:
            return False
        reference = signatures[0]
        return all(
            self._same_access_signature(reference, signature)
            for signature in signatures[1:]
        )

    @staticmethod
    def _has_unsupported_elementwise_semantics(node):
        return any(node._body.has_op(op) for op in _ELEMENTWISE_UNSUPPORTED_OPS)

    @staticmethod
    def _is_constant_only_body(node):
        op_names = set(node._body.op_counts)
        return bool(node._body.op_counts.get("constant")) and op_names <= _NEUTRAL_CONSTANT_OPS

    def _classify_elementwise_node(self, node):
        if node.has_side_effects() or node.has_aliasing_or_mutation():
            return None
        if self._has_unsupported_elementwise_semantics(node):
            return None

        reads = tuple(node.read_writes.reads)
        writes = tuple(node.read_writes.writes)
        if not writes or node.read_writes.index_exprs:
            return None
        if not all(isinstance(dep, MemoryDep) for dep in writes):
            return None

        write_signatures = tuple(self._make_access_signature(dep) for dep in writes)
        if any(signature is None for signature in write_signatures):
            return None
        if not self._all_access_signatures_equal(write_signatures):
            return None
        reference = write_signatures[0]

        if not reads:
            # Constant-only producers are neutral only when the kernel also has
            # a direct external tensor read.
            if self._is_constant_only_body(node):
                return "neutral_constant", reference
            return None

        if not all(isinstance(dep, MemoryDep) for dep in reads):
            return None
        read_signatures = tuple(self._make_access_signature(dep) for dep in reads)
        if any(signature is None for signature in read_signatures):
            return None
        if not all(
            self._same_access_signature(reference, signature)
            for signature in read_signatures
        ):
            return None
        return "direct", reference

    @staticmethod
    def _neutral_outputs_are_consumed(classifications):
        # A neutral node must represent an internal value, not an independent
        # generated output fused horizontally into the same kernel.
        consumed_names = {
            dep.name
            for node, _, _ in classifications
            for dep in node.read_writes.reads
            if isinstance(dep, MemoryDep)
        }
        return all(
            dep.name in consumed_names
            for node, kind, _ in classifications
            if kind == "neutral_constant"
            for dep in node.read_writes.writes
        )

    @staticmethod
    def _has_external_direct_read(classifications):
        produced_names = {
            dep.name
            for node, _, _ in classifications
            for dep in node.read_writes.writes
            if isinstance(dep, MemoryDep)
        }
        return any(
            dep.name not in produced_names
            for node, kind, _ in classifications
            if kind == "direct"
            for dep in node.read_writes.reads
            if isinstance(dep, MemoryDep)
        )

    def _classify_group_workload(self, template):
        if template != "pointwise":
            return None
        nodes = tuple(self.kernel.features.scheduler_nodes())
        if not nodes:
            return None

        classifications = []
        for node in nodes:
            result = self._classify_elementwise_node(node)
            if result is None:
                return None
            kind, reference = result
            classifications.append((node, kind, reference))

        references = tuple(reference for _, _, reference in classifications)
        if not self._all_access_signatures_equal(references):
            return None
        if not self._neutral_outputs_are_consumed(classifications):
            return None
        if not self._has_external_direct_read(classifications):
            return None
        return "elementwise"

    def _build_grouped_meta(self):
        dynamic_split_axes, static_split_axes = self._classify_split_axes()
        template = self._grouped_template_name()
        workload = self._classify_group_workload(template)
        pointwise_layout = None
        if dynamic_split_axes:
            primary_axis = self._select_primary_group_axis(dynamic_split_axes)
            if primary_axis is None:
                return None
            secondary_axes = [axis for axis in dynamic_split_axes if axis is not primary_axis]
            self._downgrade_split_axes(secondary_axes)
            if template == "pointwise" and len(dynamic_split_axes) == 1:
                static_split_axes = self._downgrade_suffix_static_split_axes(
                    primary_axis, static_split_axes
                )
            static_names = tuple(axis.name for axis in static_split_axes)
            secondary_names = tuple(axis.name for axis in secondary_axes)
            runtime_block_arg_names = tuple(
                f"{axis.name.upper()}BLOCK" for axis in self.kernel.split_axis
            )
        else:
            if template in ("persistent_reduction", "reduction"):
                primary_axis = self._dynamic_reduction_tiling_axis()
            else:
                primary_axis = self._dynamic_pointwise_tiling_axis()
                pointwise_layout = self._pointwise_layout_kind()
            if primary_axis is None:
                return None
            static_names = tuple(axis.name for axis in static_split_axes)
            secondary_names = ()
            runtime_block_arg_names = tuple(
                f"{axis.name.upper()}BLOCK" for axis in self.kernel.split_axis
            )
        feature_specs = self._build_group_features(
            workload,
            primary_axis,
            pointwise_layout,
            dynamic_split_axes,
            static_split_axes,
        )
        if not feature_specs:
            return None
        return GroupedKernelMeta(
            enabled=True,
            template=template,
            workload=workload,
            primary_group_axis=primary_axis.name,
            static_split_axes=static_names,
            secondary_runtime_symbolic_axes=secondary_names,
            group_features=tuple(feature_specs),
            runtime_block_arg_names=runtime_block_arg_names,
        )

    def _downgrade_split_axes(self, axes):
        if not axes:
            return
        retained = []
        for axis in self.kernel.split_axis:
            if axis in axes:
                axis.is_split_axis = False
                continue
            retained.append(axis)
        self.kernel.split_axis = retained
        self.kernel.split_axis.sort(reverse=True, key=self.key)
        for i, axis in enumerate(self.kernel.split_axis):
            axis.split_order = i

    def _downgrade_suffix_static_split_axes(self, dynamic_axis, static_split_axes):
        axis_order = {
            axis.name: index for index, axis in enumerate(self.kernel.sorted_axis)
        }
        dynamic_order = axis_order[dynamic_axis.name]
        split_axes_through_dynamic = [
            axis
            for axis in self.kernel.split_axis
            if axis_order[axis.name] <= dynamic_order
        ]
        try:
            split_size_hint = V.graph.sizevars.size_hint(
                self.total_split_numels(split_axes_through_dynamic)
            )
        except TypeError:
            return static_split_axes
        if split_size_hint < num_vector_core:
            return static_split_axes

        suffix_axes = [
            axis
            for axis in static_split_axes
            if axis_order[axis.name] > dynamic_order
        ]
        self._downgrade_split_axes(suffix_axes)
        suffix_names = {axis.name for axis in suffix_axes}
        return [
            axis for axis in static_split_axes if axis.name not in suffix_names
        ]

    # the below logic doesn't work when there're two reduction axis, but only one need outer reduction
    def should_outer_reduce_me(self, x):
        should_outer = (
            self.kernel.is_higher_order_reduction(True)
            and SplitTiling.great_than(x.length, 32768)
            and x.is_loop
        )
        if should_outer:
            self.should_outer_reduce = True
            self.kernel.split_axis = x
            self.kernel.split_axis.is_split_axis = True
        return should_outer

    def find_longest_dimension(self, check_in_tiling=False):
        longest = None
        for axis in self.kernel.sorted_axis:
            not_tiling = not check_in_tiling or axis not in self.kernel.tiling_axis
            if (longest is None or axis.length > longest.length) and not_tiling:
                longest = axis
        return longest

    # return True when x is the low-dim in indexing
    def is_lowest_dimension(self, x):
        return x.sorted_order in self.kernel.low_dims

    def find_lowest_dimension(self):
        def construct_low_dim():
            low_dims = set()
            high_dims = set()
            for index in self.indexing:
                coefficients_dict = index.as_coefficients_dict()
                for key, value in coefficients_dict.items():
                    if not key.free_symbols:
                        continue
                    origin_key_free_symbols_len = len(list(key.free_symbols))
                    axis = None
                    for symbol_val in list(key.free_symbols):
                        if symbol_val in self.kernel.range_tree_nodes:
                            axis = self.kernel.range_tree_nodes[symbol_val]
                            break
                    if axis is None:
                        continue
                    if value == sympy.Integer(1) and origin_key_free_symbols_len == 1:
                        low_dims.add(axis.sorted_order)
                    else:
                        high_dims.add(axis.sorted_order)
            # Only add stride = 1 axis to low_dims in all indexing
            # eg: index0 = y0
            #     index1 = x0 + 128*y0
            #     x0 is valid low_dims
            self.kernel.low_dims = low_dims - high_dims
            if not self.kernel.low_dims:
                log.warning("%s low_dims is null, %s, %s", self.indexing, low_dims, high_dims)
                self.kernel.low_dims = low_dims

        # all read index should be considered
        buf_names = [
            node.node.name
            for node in self.kernel.node_schedule
            if node not in (EnableReduction, DisableReduction)
        ]
        for node in self.kernel.node_schedule:
            if node in (EnableReduction, DisableReduction):
                continue
            names = []

            for read in node._body.memory_usage[MemoryUsageType.LOAD]:
                name = read.index_name
                arg = read.buffer_name
                read_is_inptr = False if arg[:3] != 'arg' and arg in buf_names else True
                if read_is_inptr:
                    names.append(name)
            for key, index in node._body.indexing.items():
                if key in names and index not in self.indexing:
                    indirect_index = node._body.substitube_indirect_index(index)
                    self.indexing.append(indirect_index if indirect_index else index)

        if self.kernel.inside_reduction:
            construct_low_dim()

        # for non-reduction, write index should be considered
        for node in self.kernel.node_schedule:
            if node in (EnableReduction, DisableReduction):
                continue
            names = []
            for write in node._body.memory_usage[MemoryUsageType.STORE]:
                names.append(write.index_name)
            for write in node._body.memory_usage[MemoryUsageType.STORE_REDUCTION]:
                names.append(write.index_name)
            for key, index in node._body.indexing.items():
                if key in names and index not in self.indexing:
                    indirect_index = node._body.substitube_indirect_index(index)
                    self.indexing.append(indirect_index if indirect_index else index)

        construct_low_dim()

    @staticmethod
    def convert(x, y):
        xnumel = x
        ynumel = y
        if isinstance(xnumel, (sympy.Symbol, sympy.Expr)) and not isinstance(xnumel, sympy.Integer):
            xnumel = xnumel.subs(get_sizevars_backed_var_to_val(V.graph.sizevars))

        if isinstance(ynumel, (sympy.Symbol, sympy.Expr)) and not isinstance(ynumel, sympy.Integer):
            ynumel = ynumel.subs(get_sizevars_backed_var_to_val(V.graph.sizevars))

        if isinstance(xnumel, sympy.Integer) and isinstance(ynumel, int):
            ynumel = sympy.Integer(ynumel)

        if isinstance(ynumel, sympy.Integer) and isinstance(xnumel, int):
            xnumel = sympy.Integer(xnumel)

        return (xnumel, ynumel)

    @staticmethod
    def less_than(x, y):
        xnumel, ynumel = SplitTiling.convert(x, y)
        return xnumel < ynumel

    @staticmethod
    def great_than(x, y):
        xnumel, ynumel = SplitTiling.convert(x, y)
        return xnumel > ynumel

    @staticmethod
    def ge_than(x, y):
        xnumel, ynumel = SplitTiling.convert(x, y)
        return xnumel >= ynumel
