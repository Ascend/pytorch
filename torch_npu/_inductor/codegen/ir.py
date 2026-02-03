from typing import List, Tuple, Dict, Any, Optional, cast
import os
import itertools
import sympy
import torch
from torch._inductor.ir import (ReductionHint, IRNode, ModularIndexing, FloorDiv)
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import sympy_subs, sympy_index_symbol, has_free_symbols
from torch._inductor.virtualized import V
from torch._inductor.loop_body import MemoryUsageType
from torch.utils._sympy.value_ranges import IntInfinity, ValueRanges
from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel
from .triton_utils import get_indirect_var, get_indirect_mem_var, NPUKernelType
from ..config import log, inductor_indirect_memory_mode


# NPU doesn't need to support ReductionHint.OUTER, and persistent reduction
def num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node: Optional[IRNode] = None,
):
    return ReductionHint.DEFAULT, 1


def detect_flattened_dims(kernel, index):
    new_vars = {}
    if not isinstance(index, (sympy.core.add.Add, ModularIndexing, FloorDiv)):
        return new_vars

    def detect_flattened_axis(expr):
        def init_new_vars(var, length):
            if var not in new_vars:
                new_vars[var] = {length: [None, None]}
            if length not in new_vars[var]:
                new_vars[var][length] = [None, None]

        if isinstance(expr, ModularIndexing):
            var, divisor, length = expr.args
            init_new_vars(var, length)
            new_vars[var][length][1] = (expr, divisor, length)
        elif isinstance(expr, FloorDiv):
            var, divisor = expr.args
            init_new_vars(var, divisor)
            # over than 1 node_schedule, var may be deleted in kernel.range_tree_nodes
            # it shoule be find in range_tree_nodes_removed dict
            if (var in kernel.range_tree_nodes):
                numel = kernel.range_tree_nodes[var].length
            else:
                numel = kernel.range_tree_nodes_removed[var].length

            length = expr.eval(numel, divisor)
            new_vars[var][divisor][0] = (expr, divisor, length)

        else:
            for x in expr.args:
                detect_flattened_axis(x)

    # add
    if isinstance(index, sympy.core.add.Add):
        for x in index.args:
            detect_flattened_axis(x)
    elif isinstance(index, (ModularIndexing, FloorDiv)):
        detect_flattened_axis(index)
    else:
        pass

    # make sure FloorDiv, MouldarIndexing must be in-pair
    for var, divisors in new_vars.items():
        if var in kernel.range_tree_nodes:
            parent_axis = kernel.range_tree_nodes[var]
        else:
            parent_axis = kernel.range_tree_nodes_removed[var]
        for divisor, pair in divisors.items():
            if not pair[0] and not pair[1]:
                pass
            # FloorDiv not inplace
            elif not pair[0]:
                _, _, length = pair[1]
                expr = FloorDiv(var, length)
                new_vars[var][divisor][0] = (expr, length, parent_axis.length // length)
            # ModularIndexing not inplace
            elif not pair[1]:
                expr = ModularIndexing(var, 1, divisor)
                new_vars[var][divisor][1] = (expr, 1, divisor)
            else:
                pass

    return new_vars


def rebuild_flattened_dims(indexing):
    def rebuild_flattened_dim(key, index, old_node, flatten_dim):
        for _, pair in flatten_dim.items():
            new_var_expr = sympy.Integer(0)
            origin_axis_length = 0
            pair_is_valid = True
            # don't create duplicated axis, e.g. y1:1024, y1 % 1024 is duplicated
            expr, divisor, length = pair[1]
            if not old_node.parent.duplicated_check(divisor, length):
                if expr not in V.kernel.expr_substituted:
                    V.kernel.expr_substituted[expr] = old_node.symbol()
                break

            for axis in pair:
                expr, divisor, length = axis
                # 3. try to rebuild the axis in kernel
                new_node = old_node.parent.lookup(divisor, length)

                # 4. substitute div/mod expression in indexing
                index = index.subs(expr, new_node.symbol())
                indexing[key] = index
                if isinstance(expr, FloorDiv):
                    new_var_expr = new_var_expr + new_node.symbol() * divisor
                    origin_axis_length = divisor * length
                elif isinstance(expr, ModularIndexing):
                    new_var_expr = new_var_expr + new_node.symbol()
                V.kernel.expr_substituted[expr] = new_node.symbol()

            if var not in V.kernel.range_tree_nodes_substituted:
                V.kernel.range_tree_nodes_substituted[var] = []
            V.kernel.range_tree_nodes_substituted[var].append((origin_axis_length, new_var_expr))

    def find_index_in_substitute(index, kernel):
        return any([index.find(key) for key in kernel.expr_substituted.keys()])

    kernel = V.kernel
    for key, index in indexing.items():
        # 1. try to find out flattened axis from indexing
        flatten_dims = detect_flattened_dims(kernel, index)
        # 2. try to rebuild these flattened dims
        for var, flatten_dim in flatten_dims.items():
            if (var in kernel.range_tree_nodes):
                old_node = kernel.range_tree_nodes[var]
            else:
                old_node = kernel.range_tree_nodes_removed[var]

            rebuild_flattened_dim(key, index, old_node, flatten_dim)

        if find_index_in_substitute(index, kernel):
            new_index = sympy_subs(index, kernel.expr_substituted)
            indexing[key] = new_index


def substituted_dims_in_indexing(self, indexing, kernel, range_tree_nodes_substituted):
    substituted = False
    for var, candidates in range_tree_nodes_substituted.items():
        if not (len(candidates) > 0):
            raise RuntimeError("assert len(candidates) > 0, candidates")
        exprs = sorted(candidates, reverse=True, key=lambda x: x[0])
        # the best candidate is with the longest numel
        numel = exprs[0][0]
        expr = exprs[0][1]
        node = kernel.range_tree_nodes[var]
        if node.length != numel:
            log.debug("sub nodes (expr%s, numel:%d) can not substitute parent node(%s:%d)",
                      expr, numel, node.symbol(), node.length)
            continue
        for key, index in indexing.items():
            if var in index.free_symbols:
                index = index.subs(var, expr)
                indexing[key] = index
                substituted = True

    return substituted


def generate_body_indexing(body, indices):
    index = list(itertools.chain.from_iterable(indices))
    if not (len(index) == len(body.var_ranges)):
        raise RuntimeError("assert len(index) == len(body.var_ranges), (index, body.var_ranges)")
    if not (all(v not in body.var_ranges for v in index)):
        raise RuntimeError("assert all(v not in body.var_ranges for v in index)")

    replacements = dict(zip(body.var_ranges.keys(), index))
    indexing_map = dict(zip(index, body.var_ranges.keys()))
    setattr(body, 'indexing_map', indexing_map)
    body.indexing = {
        name: sympy_subs(expr, replacements)
        for name, expr in body.indexing_exprs.items()
    }
    setattr(body, 'indirect_replacements', {})
    body.generate_indirect_replacements()


def remove_zero_terms_impl(expr, var_ranges):
    shape_env = V.graph.sizevars.shape_env
    var_to_range = dict(shape_env.var_to_range)
    var_to_range.update(
        {
            k: ValueRanges(
                0, max(0, v - 1) if not has_free_symbols([v]) else IntInfinity()
            )
            for k, v in var_ranges.items()
        }
    )
    for var in expr.free_symbols:
        if var not in var_to_range:
            var_to_range[var] = ValueRanges(0, IntInfinity())

    var_to_range_tuple = cast(
        tuple[tuple[sympy.Symbol, ValueRanges[sympy.Expr]]],
        tuple(var_to_range.items()),
    )

    axioms = []
    for var, upper_bound in var_ranges.items():
        axioms.append(0 <= var)
        axioms.append(var < upper_bound)
    axioms = tuple(axioms) + shape_env.get_axioms()

    def statically_known(expr):
        evaluated = shape_env._maybe_evaluate_static(
            expr,
            axioms=axioms,
            var_to_range=var_to_range_tuple,
        )
        return bool(evaluated)

    def _remove_zero_terms(base, divisor):
        if statically_known(base < divisor):
            return sympy.Integer(0)
        return FloorDiv(base, divisor)
    
    replacements = {}
    for sub_expr in expr.atoms(FloorDiv):
        base, divisor = sub_expr.args
        if statically_known(base < divisor):
            replacements[sub_expr] = sympy.Integer(0)
    
    if replacements:
        expr = expr.xreplace(replacements)
    
    return expr


# Eliminate terms such as 2560(((320p1 + p2)//2560)) when (320*p1 + p2)//2560 is constantly zero
def remove_zero_terms(indexing, var_ranges):
    for key, expr in indexing.items():
        if expr.has(FloorDiv):
            new_expr = remove_zero_terms_impl(expr, var_ranges)
            indexing[key] = new_expr


def transform_dims_in_indexing(self, indices):
    if self.indexing is None:
        remove_zero_terms(self.indexing_exprs, self.var_ranges)
        generate_body_indexing(self, indices)

    if V.kernel is not None and isinstance(V.kernel, NPUIndexTritonKernel):
        rebuild_flattened_dims(self.indexing)


# subsititude indirct var with real axis var 
def substitube_indirect_index(self, index):
    indirect_var = None 
    
    for symbol in index.free_symbols:
        indirect_var = get_indirect_var(str(symbol))
        if indirect_var:
            break

    if indirect_var:
        indirect_symbol = sympy_index_symbol(indirect_var)
        if not self.indirect_replacements.get(indirect_symbol, None):
            return None
        tmp_index = sympy_subs(index, self.indirect_replacements)
        return self.substitube_indirect_index(tmp_index)
    return index


def get_load_index_from_subblock(loop_body, subblock):
    node_map = {}
    for node in subblock.graph.nodes:
        node_map[node.name] = node
        load_index = get_indirect_index(loop_body, node, node_map)
        if load_index:
            return load_index
    
    return None


def analyze_all_index(all_indexs):
    if not all_indexs:
        return None

    if not all(all_indexs[0] == index for index in all_indexs):
        return None
    return all_indexs[0]


def get_indirect_index(loop_body, find_node, node_map):
    all_indexs = []
    for node in find_node.args:
        if not isinstance(node, torch.fx.node.Node):
            continue
        node = node_map[node.name]
        if 'get_index' in node.name:
            return node.args[0]
        if 'masked_subblock' in node.name:
            if node.name not in loop_body.subblocks:
                raise RuntimeError(f"can't find {node.name} in loopbody")
            load_index = get_load_index_from_subblock(loop_body, loop_body.subblocks[node.name])
        else:
            load_index = get_indirect_index(loop_body, node, node_map)
        if load_index:
            all_indexs.append(load_index)

    return analyze_all_index(all_indexs)


def define_npu_kernel_type(loop_body):
    '''
        For indirect load + sum pattern: simt_only is faster
    '''
    if inductor_indirect_memory_mode != str(NPUKernelType.SIMD_SIMT_MIX):
        return NPUKernelType(inductor_indirect_memory_mode)

    node_map = {}
    for node in loop_body.root_block.graph.nodes:
        node_map[node.name] = node
        if 'reduction' == node.name:
            reduction_type_pos = 3 # 3 is reduction_type_pos
            reduction_index_pos = 4 # 4 is reduction_index_pos
            reduction_type = node.args[reduction_type_pos]
            if reduction_type != 'sum':
                continue
            reduction_index = str(node.args[reduction_index_pos])
            if 'load' not in reduction_index:
                continue
            if reduction_index not in node_map:
                continue
            load_node = node_map[reduction_index]
            load_index_pos = 2 # 2 is load index position
            if load_node.args[load_index_pos].name not in node_map:
                continue
            get_load_index = node_map[load_node.args[load_index_pos].name]
            load_index = get_load_index.args[0]
            if load_index not in loop_body.indexing:
                continue
            if 'indirect' in str(loop_body.indexing[load_index]):
                return NPUKernelType.SIMT_ONLY

    return NPUKernelType(inductor_indirect_memory_mode)


def generate_indirect_replacements(self):
    '''
    ir:
        get_index=self.get_index('index0')
        load=ops.load('arg2_1', get_index)
        set_indirect0=ops.set_indirect(load)
    find { indirect0 : index0 }
    '''
    node_map = {}
    indirect_node_map = {}
    for node in self.root_block.graph.nodes:
        node_map[node.name] = node
        index_select_var = get_indirect_mem_var(node.name)
        if index_select_var:
            indirect_var = node.args[4]
            if indirect_var in indirect_node_map:
                indirect_node = indirect_node_map[indirect_var]
                indirect_node.meta['indirect_template'] = True
            indirect_var_symbol = sympy_index_symbol(indirect_var)
            origin_index = self.indirect_replacements.get(indirect_var_symbol, "")
            if 'indirect' in str(origin_index):
                node.meta['multi_indirect_index'] = True
            continue
        indirect_var = get_indirect_var(node.name)
        if indirect_var is None:
            continue
        indirect_var_symbol = sympy_index_symbol(indirect_var)
        if inductor_indirect_memory_mode:
            V.kernel.npu_kernel_type = define_npu_kernel_type(self)
        indirect_node_map[indirect_var] = node

        load_index = get_indirect_index(self, node, node_map)
        if load_index is None:
            continue

        origin_index = self.indexing[load_index]
        self.indirect_replacements[indirect_var_symbol] = origin_index


# select tiling axis, recover missing dimensions,
def loopbody__call__(self, *indices):
    if self.indexing is None:
        generate_body_indexing(self, indices)
    result = self.root_block()
    self.indexing = None
    return result


def loop_body_block_index_select(self, name: str, index: sympy.Expr, indirect_var, set_indirect, bound, index_select_type):
    index = self._simplify(index)
    index = self._add_index(index, MemoryUsageType.LOAD, buffer_name=name)
    return self._inner.index_select(name, index, indirect_var, str(set_indirect), bound, index_select_type)


def simplify_indexing_index_select(self, name: str, index: sympy.Expr, indirect_var, set_indirect, bound, index_select_type):
    return self._inner.index_select(name, self._simplify(index), indirect_var, str(set_indirect), bound, index_select_type)


def loop_body_block_gather_template(self, name: str, index: sympy.Expr, indirect_var, set_indirect, index_boundary):
    index = self._simplify(index)
    index = self._add_index(index, MemoryUsageType.LOAD, buffer_name=name)
    return self._inner.gather_template(name, index, indirect_var, str(set_indirect), index_boundary)


def simplify_indexing_gather_template(self, name: str, index: sympy.Expr, indirect_var, set_indirect, index_boundary):
    return self._inner.gather_template(name, self._simplify(index), indirect_var, str(set_indirect), index_boundary)


def loop_body_block_indexput_template(self, name, index, value, indirect_var, boundary):
    index = self._simplify(index)
    index = self._add_index(
        index, MemoryUsageType.STORE, buffer_name=name
    )
    return self._inner.indexput_template(name, index, value, str(indirect_var), boundary)


def simplify_indexing_indexput_template(self, name, index, value, indirect_var, boundary):
    return self._inner.indexput_template(name, self._simplify(index), value, str(indirect_var), boundary)


def loop_body_block_scatter_template(self, name, index, value, indirect_var, boundary):
    index = self._simplify(index)
    index = self._add_index(
        index, MemoryUsageType.STORE, buffer_name=name
    )
    return self._inner.scatter_template(name, index, value, str(indirect_var), boundary)


def simplify_indexing_scatter_template(self, name, index, value, indirect_var, boundary):
    return self._inner.scatter_template(name, self._simplify(index), value, str(indirect_var), boundary)


def loop_body_block_cat_store(self, dst, src, size, store_offset_index, output_buffer_index):
    store_offset_index = self._simplify(store_offset_index)
    store_offset_index = self._add_index(store_offset_index, MemoryUsageType.STORE, buffer_name=dst)
    output_buffer_index = self._simplify(output_buffer_index)
    output_buffer_index = self._add_index(output_buffer_index, MemoryUsageType.STORE, buffer_name=dst)
    return self._inner.cat_store(dst, src, size, store_offset_index, output_buffer_index)


def simplify_indexing_cat_store(self, dst, src, size, store_offset_index, output_buffer_index):
    return self._inner.cat_store(dst, src, size, self._simplify(store_offset_index), self._simplify(output_buffer_index))