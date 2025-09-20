from typing import List, Tuple
import sympy
from torch._inductor import ir
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import V


class IndexAnalysis:
    def __init__(self, kernel, raw_index, is_store_index=False, is_index_expr=False):
        self.index = raw_index.subs(V.graph.sizevars.var_to_val)
        self.kernel = kernel
        self.tiling_axis = [x.symbol() for x in self.kernel.tiling_axis]
        self.stride_list = None  # stride list [1,2,4,24]
        self.reshape_sizes = []  # [RBLOCK, 1, 1, XBLOCK_SUB]
        self.broadcast_sizes = []  # [RBLOCK, XBLOCK_SUB]
        self.permute_shape = []  # [0,2,1,3]
        self.var_replacements = {}  # r2 ->r2_0, etc
        self.var_directions = {}  # r2_0 -> [None,:,None]
        self.similar = None  # (r,x,z,y)
        self.need_permute = False
        self.need_broadcast = False
        self.need_reshape = False
        self.gold = kernel.golden_var_list  # tuple([x.symbol() for x in reversed(kernel.tiling_axis)])
        self.var_stride = [
            (key, coeff)
            for key, coeff in self.index.as_coefficients_dict().items()
            if not isinstance(key, sympy.Integer)
        ]
        # sort by stride 
        self.var_stride.sort(key=lambda x: x[1])
        # only contains tiing axis var
        self.var_list = tuple([x[0] for x in self.var_stride if x[0] in self.tiling_axis])
        self.stride_list = tuple([x[1] for x in self.var_stride if x[0] in self.tiling_axis])
        self.is_store_index = is_store_index
        self.is_index_expr = is_index_expr

    def get_most_similar_shape(self):
        matched_dims = 0
        self.similar = None
        for value in self.kernel.index_analysis.keys():
            if len(value) != len(self.gold):
                continue
            i = 0
            while i < len(self.var_list):
                if value[i] == self.var_list[i]:
                    i = i + 1
                else:
                    break

            if i > matched_dims:
                matched_dims = i
                self.similar = value
        return self.similar

    @classmethod
    def same_var_list(cls, var1, var2):
        if len(var1) != len(var2):
            return False
        for i, v in enumerate(var1):
            if v != var2[i]:
                return False
        return True

    def shrink_permute_shape(self, permute_shape):
        diff = len(self.gold) - len(self.kernel.tiling_axis)
        new_shape = [x for x in permute_shape if x - diff >= 0]
        return new_shape

    def analyze_permute_shape(self):
        if self.is_index_expr:
            return
        if self.gold == self.similar:
            self.need_permute = False
            return

        similar = tuple(reversed(self.similar))
        gold = tuple(reversed(self.gold))
        self.permute_shape = [None] * len(gold)

        if self.is_store_index:
            for i, x in enumerate(similar):
                if x != gold[i]:
                    index = gold.index(x)
                    self.permute_shape[i] = index
                    self.need_permute = True
                else:
                    self.permute_shape[i] = i
            return

        for i, x in enumerate(gold):
            if x != similar[i]:
                index = similar.index(x)
                self.permute_shape[i] = index
                self.need_permute = True
            else:
                self.permute_shape[i] = i

    def analyze_broadcast_sizes(self):
        if not self.need_reshape:
            self.need_broadcast = False
            return
        self.need_broadcast = True
        reversed_similar = reversed(self.similar)
        similar = [x for x in reversed_similar]
        self.broadcast_sizes = ["1"] * len(similar)
        for i, x in enumerate(similar):
            self.broadcast_sizes[i] = f"{x.name.upper()}BLOCK_SUB"

    def analyze_reshape_sizes(self):
        if all(x in self.var_list for x in self.tiling_axis):
            self.need_reshape = False
            return
        self.need_reshape = True
        reversed_similar = reversed(self.similar)
        similar = [x for x in reversed_similar]
        var_list = [x for x in reversed(self.var_list)]
        self.reshape_sizes = ["1"] * len(similar)
        for _, x in enumerate(var_list):
            index = similar.index(x)
            self.reshape_sizes[index] = f"{x.name.upper()}BLOCK_SUB"

    def analyze_var_direction(self):
        if self.var_list == self.gold:
            return
        var_list = self.var_list if len(self.var_list) == len(self.gold) else self.similar
        if var_list == self.gold:
            return
        if not var_list:
            return
        var_list = list(reversed(var_list))
        gold = list(tuple(reversed(self.gold)))
        if len(var_list) != len(gold):
            raise RuntimeError("assert var_list and gold must have same length")
        var_list = [x for x in var_list if x in self.kernel.tiling_axis]
        gold = [x for x in gold if x in self.kernel.tiling_axis]
        for i, x in enumerate(gold):
            index = var_list.index(x)
            if (index == i):
                continue
            new_var = sympy_index_symbol(f"{x}") if self.is_index_expr else sympy_index_symbol(f"{x}_{index}")
            if new_var in self.var_replacements:
                continue
            direction = ["None"] * len(gold)
            direction[index] = ":"
            direction_str = f"[{','.join(direction)}]"
            self.var_replacements[x] = new_var
            self.var_directions[new_var] = direction_str
            self.kernel.range_tree_nodes[x].var_directions[new_var] = direction_str

    def analyze_index(self):
        if isinstance(self.index, sympy.Integer):
            return
        if not self.kernel.golden_var_list:
            self.kernel.select_golden_varlist()
            self.gold = self.kernel.golden_var_list

        if self.gold is None:
            raise RuntimeError("assert gold must not be None")
        if len(self.gold) != len(self.tiling_axis):
            raise RuntimeError("assert gold must have same length as tiling_axis")

        def all_tiling_in_var_list():
            return all([x in self.var_list for x in self.tiling_axis])
            # 2 analyze permute shape for full_dim_len index

        if all_tiling_in_var_list():
            self.similar = self.var_list
            self.analyze_permute_shape()
            if self.var_list not in self.kernel.index_analysis:
                self.kernel.index_analysis[self.var_list] = self
        # 3. analyze reshape and broadcast sizes
        else:
            pass

        # 4 analyze var direction
        self.analyze_var_direction()

    def generate_statement(self):
        statement = ""
        if self.need_reshape:
            reshape_sizes = f"[{','.join(self.reshape_sizes)}]"
            statement = f".reshape({reshape_sizes})"
        if self.need_broadcast:
            broadcast_sizes = f"[{','.join(self.broadcast_sizes)}]"
            statement = f"{statement}.broadcast_to({broadcast_sizes})"
        if self.need_permute:
            statement = f"{statement}.permute({self.permute_shape})"
        return statement


class ReductionAnalysis:
    def __init__(self, kernel):
        self.kernel = kernel
        self.reduction = None
        self.reduced_dim = None
        if self.numof_reduction_axis() > 1:
            self.kernel.persistent_reduction = True
            self.reduced_dim = 0
            return

        reduction = self.kernel.find_reduction_node()
        if reduction is None or not isinstance(reduction, ir.Reduction):
            raise RuntimeError("failed to get one reduction node")
        if not hasattr(reduction, "reduced_idx"):
            raise RuntimeError("reduction node doesn't have attr reduced_idx")
        self.reduction = reduction
        self.reduced_dim = self.analyze_reduction_dim()

    def is_higher_order_reduction(self):
        return self.dim < len(self.kernel.tiling_axis) - 1

    def is_1d_reduction(self):
        return self.kernel.numels["r"] > 1 and len(self.kernel.numels) == 1

    def get_reduce_dim_reshape(self, reduce_axis):
        if self.is_1d_reduction():
            shape_str = f"[{reduce_axis.name.upper()}BLOCK_SUB]"
        else:
            shape = ["1"] * len(self.kernel.tiling_axis)
            shape[self.reduced_dim] = f"{reduce_axis.name.upper()}BLOCK_SUB"
            shape_str = f"[{','.join(shape)}]"
        return shape_str

    def dense_size_list(self) -> List[str]:
        sizes = [f"{x.name.upper()}BLOCK_SUB" for x in self.kernel.tiling_axis]
        if self.numof_reduction_axis() > 1:
            return sizes

        reduce_axis = self.kernel.tiling_axis[-1]
        sizes.pop(-1)
        sizes.insert(self.reduced_dim, f"{reduce_axis.name.upper()}BLOCK_SUB")
        return sizes

    def dense_size_str(self):
        sizes = self.dense_size_list()
        if self.numof_reduction_axis() > 1:
            return f"[{'* '.join(sizes)}]"
        return f"[{', '.join(sizes)}]"

    def numof_reduction_axis(self):
        return self.kernel.numof_reduction_axis()

    def reduction_axis_list(self):
        return self.kernel.reduction_axis_list()

    def analyze_reduction_dim(self):

        if self.numof_reduction_axis() > 1:
            self.kernel.persistent_reduction = True
            self.reduced_dim = 0
            return 0

        if not self.kernel.golden_var_list:
            self.kernel.select_golden_varlist()
        if self.kernel.golden_var_list is None:
            raise RuntimeError("assert self.kernel.golden_var_list is not None")

        dim = -1
        for i, x in enumerate(reversed(self.kernel.golden_var_list)):
            if x.name[0] == 'r':
                dim = i
                break
        return dim

    def analyze_reduction_dim1(self):
        if self.numof_reduction_axis() > 1:
            self.kernel.persistent_reduction = True
            self.reduced_dim = 0
            return 0
        reduction = self.reduction
        # kept = [0,1,3], reduced = [2]
        for i, x in enumerate(reduction.reduced_idx):
            if reduction.reduction_ranges[i] <= 1:
                continue
            reduced_idx = x
            break
            # the index (in reduction.ranges) of low_dims
        low_dims = [i for i, x in enumerate(reduction.kept_idx) if x > reduced_idx]
        if not low_dims:
            return len(self.kernel.tiling_axis) - 1
        elif len(low_dims) == len(reduction.kept_idx):
            return 0
        # reduction dim when low_dims are not meraged
        dim = len(reduction.kept_idx) - len(low_dims)

        tiling_axis = self.kernel.tiling_axis[:-1]
        merged = 1
        j = len(tiling_axis) - 1
        # remove all low_dims from tiling_axis
        # all axis before ahead of j are high-orders
        # then following is reduced dim 
        ranges = [x for x in reduction.ranges if x > 1]
        for i in reversed(low_dims):
            len_axis = tiling_axis[j].length
            len_reduction = ranges[i] * merged
            if len_reduction < len_axis:
                merged = merged * len_reduction
            elif len_reduction == len_axis:
                j = j - 1
                merged = 1
            else:
                raise RuntimeError(f"assert should not reach here low_dims({i})={len_reduction}, axis[{j}]=len)")
        dim = j + 1
        return dim
