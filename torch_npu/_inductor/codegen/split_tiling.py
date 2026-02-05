from functools import reduce
import sympy as sympy
import torch
from torch._inductor.codegen.simd import EnableReduction, DisableReduction
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import ModularIndexing, sympy_subs
from torch._inductor.virtualized import V

from .kernel_analysis import IndexAnalysis
from .triton_utils import get_byte_per_numel
from ..config import num_vector_core, log


# split and tiling axis selector
class SplitTiling:
    def __init__(self, kernel: TritonKernel):
        self.kernel = kernel
        self.indexing = []  # load and store indexing  among all scheduler nodes
        kernel.sorted_axis = [x for x in kernel.range_tree_nodes.values()]
        kernel.sorted_axis.sort(reverse=True, key=self.key)
        for i, dim in enumerate(kernel.sorted_axis):
            dim.sorted_order = i

        self.find_lowest_dimension()
        self.should_outer_reduce = False
        self.contiguous_reduction = self.is_contiguous_reduction()


    def is_contiguous_reduction(self):
        def is_continugous_axis(axis_list):
            axis_set = set(axis_list)
            return len(axis_set) == (max(axis_set) - min(axis_set) + 1)

        if self.kernel.numof_reduction_axis() > 1:
            golden_var_list = self.kernel.parse_golden_from_load_store_index()
            reduction_dim_list = [] 
            for i, x in enumerate(reversed(golden_var_list)):
                if x.name[0] == 'r':
                    reduction_dim_list.append(i)
            return is_continugous_axis(reduction_dim_list)
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
            return length_expr.subs(V.graph.sizevars.var_to_val)
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
            if self.total_split_numels(self.kernel.split_axis) >= num_vector_core:
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
                if axis == self.kernel.sorted_axis[-1] and self.kernel.contains_cat_node():
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
            self.kernel.split_axis.append(self.kernel.sorted_axis[0])
            # todo: improve or remove this condition check
            if not self.kernel.contains_cat_node():
                self.kernel.sorted_axis[0].is_split_axis = True

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
                    map(lambda x: self.get_length_val(x), self.kernel.sorted_axis),
                )
                if self.kernel.sorted_axis
                else 1
            )
            tiling_numel = (
                reduce(
                    lambda x, y: x + y,
                    map(lambda x: self.get_length_val(x), self.kernel.tiling_axis),
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
        sorted_low_dims = sorted(low_dims, key=lambda x: self.get_length_val(x))
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
                    key = list(key.free_symbols)[0]
                    if key not in self.kernel.range_tree_nodes:
                        continue

                    axis = self.kernel.range_tree_nodes[key]
                    if value == sympy.Integer(1):
                        low_dims.add(axis.sorted_order)
                    else:
                        high_dims.add(axis.sorted_order)
            # Only add stride = 1 axis to low_dims in all indexing
            # eg: index0 = y0
            #     index1 = x0 + 128*y0
            #     x0 is valid low_dims
            self.kernel.low_dims = low_dims - high_dims
            if not self.kernel.low_dims:
                log.warning(f"{self.indexing} low_dims is null, {low_dims}, {high_dims}")
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
            xnumel = xnumel.subs(V.graph.sizevars.var_to_val)

        if isinstance(ynumel, (sympy.Symbol, sympy.Expr)) and not isinstance(ynumel, sympy.Integer):
            ynumel = ynumel.subs(V.graph.sizevars.var_to_val)

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
