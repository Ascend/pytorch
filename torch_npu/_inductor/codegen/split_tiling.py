from functools import reduce
import sympy as sympy
from torch._inductor.codegen.simd import (EnableReduction, DisableReduction)
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import ModularIndexing, sympy_subs
from torch._inductor.virtualized import V

from .kernel_analysis import IndexAnalysis
from .triton_utils import get_aligned_numel
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
        self.possible_need_permute = self.find_possible_permutes()

    def find_possible_permutes(self):
        if len(self.kernel.low_dims) <= 1:
            return False
        var_lists = []
        low_dims = [self.kernel.sorted_axis[x].symbol() for x in self.kernel.low_dims]
        for index in self.indexing:
            var_stride = [
                (key, coeff)
                for key, coeff in index.as_coefficients_dict().items()
                if not isinstance(key, sympy.Integer)
            ]
            var_stride.sort(key=lambda x: x[1])
            var_list = tuple([x[0] for x in var_stride if x[0] in low_dims])
            var_lists.append(var_list)
        for i, var_list in enumerate(var_lists):
            if len(var_list) < len(low_dims):
                continue
            for j, other in enumerate(var_lists):
                if i == j or len(other) < len(low_dims):
                    continue
                if var_list != other:
                    return True
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

    @classmethod
    def total_split_numels(cls, axis_list):
        numels = [x.length for x in axis_list]
        return reduce(lambda x, y: x * y, numels) if numels else 1

    # Split 原则1 ：先做维度合并，再切分 。通过维度合并降维降低split和tiling轴选择策略的复杂性 。
    # Split 原则2 : 切分轴尽量选择高维度的轴, 这样load/store 能够有比较好的线性度 ,
    # Split 原则3 : 规约轴和低维轴不应选为切分轴 。但如果高维规约类融合算子，而且高维尺寸非常大（ >= 64KB），其他维度不足以支持切分，可以考虑对规约轴切分。
    # Split 原则4 ：切分轴的总numel 要超过 aicore总数。切分轴的数量最好不要超过3个(triton 最多支持三维发射）， 因此 如果一点要超， 需要维度合并。
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

        self.kernel.split_axis.sort(reverse=True, key=self.key)
        for i, x in enumerate(self.kernel.split_axis):
            x.split_order = i

    # Tiling 原则1：load / store 中索引表达式的中的低维轴都要成为tiling 轴. 
    # Tiling 原则2：对于规约算子，规约轴要成为tiling轴。
    # Tiling 原则3: 多维规约， 只有规约轴可以被选择为tiling轴
    # Tiling 原则4: tiling轴 要覆盖 total numel 的 80%

    # two tiling axis might be insufficient when there're 3 or more low-dims in indexing
    def select_tiling_axis(self):
        self.kernel.tiling_axis.clear()

        #  cover the biggest axis and not exceed 3 axis
        def meet_stop_condition():
            total_numel = reduce(lambda x, y: x + y,
                                 map(lambda x: x.length, self.kernel.sorted_axis)) if self.kernel.sorted_axis else 1
            tiling_numel = reduce(lambda x, y: x + y,
                                  map(lambda x: x.length, self.kernel.tiling_axis)) if self.kernel.tiling_axis else 1
            if self.kernel.numof_reduction_axis() > 1 and all(
                    self.kernel.range_tree_nodes[var].is_tiling_axis for var in self.kernel.reduction_axis_list()):
                return True
                # currently, the maximum dim that triton-ascend support is 2
            max_transpose_dims = 2
            if (self.possible_need_permute or tiling_numel / total_numel >= 0.8) and \
                    len(self.kernel.tiling_axis) >= min(max_transpose_dims, len(self.kernel.sorted_axis)):
                return True
            return False

        def select_tiling(low_dim=True, reduction=True):
            for axis in reversed(self.kernel.sorted_axis):
                if low_dim and axis.sorted_order in self.kernel.low_dims and axis not in self.kernel.tiling_axis:
                    axis.is_tiling_axis = True
                    self.kernel.tiling_axis.append(axis)
                if reduction and axis.prefix == 'r' and axis not in self.kernel.tiling_axis:
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

    def select_split_tiling_axis(self):
        self.select_split_axis()
        self.select_tiling_axis()

    # the below logic doesn't work when there're two reduction axis, but only one need outer reduction
    def should_outer_reduce_me(self, x):
        should_outer = self.kernel.is_higher_order_reduction(True) and SplitTiling.great_than(x.length,
                                                                                              32768) and x.is_loop
        if should_outer:
            self.should_outer_reduce = True
            self.kernel.split_axis = x
            self.kernel.split_axis.is_split_axis = True
        return should_outer

    def find_longest_dimension(self, check_in_tiling=False):
        longest = None
        for axis in self.kernel.sorted_axis:
            if (longest is None or axis.length > longest.length) and \
                    (not check_in_tiling or axis not in self.kernel.tiling_axis):
                longest = axis
        return longest

    # return True when x is the low-dim in indexing
    def is_lowest_dimension(self, x):
        return x.sorted_order in self.kernel.low_dims

    def find_lowest_dimension(self):
        def construct_low_dim():
            for index in self.indexing:
                coefficients_dict = index.as_coefficients_dict()
                for key, value in coefficients_dict.items():
                    if not key.free_symbols:
                        continue
                    key = list(key.free_symbols)[0]
                    if key not in self.kernel.range_tree_nodes:
                        continue

                    if value == sympy.Integer(1):
                        axis = self.kernel.range_tree_nodes[key]
                        self.kernel.low_dims.add(axis.sorted_order)

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
                    self.indexing.append(index)

        if self.kernel.inside_reduction:
            construct_low_dim()
            return

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
                    self.indexing.append(index)

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
