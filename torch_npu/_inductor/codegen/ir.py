from typing import List, Tuple, Dict, Any, Optional, cast
import os
import itertools
from math import gcd
import sympy
from sympy import Integer
import torch
from torch._inductor.ir import (ReductionHint, IRNode, ModularIndexing, FloorDiv, sympy_product)
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import sympy_subs, sympy_index_symbol, has_free_symbols
from torch._inductor.virtualized import V
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.codegen.common import BackendFeature
from torch._inductor import config
from torch.utils._sympy.value_ranges import IntInfinity, ValueRanges
from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel
from .triton_utils import get_indirect_var, get_indirect_mem_var, NPUKernelType
from ..config import log, inductor_indirect_memory_mode


def reduction_split_factor(reduction_ranges):
    ranges = [num for num in reduction_ranges if num > 1]
    if len(ranges) == 0:
        return 1
    return min(ranges)


def num_splits(
    device,
    dst_dtype,
    src_dtype,
    inner_fn,
    ranges,
    reduction_ranges,
    reduction_type,
    reduction_numel,
    input_node=None,
):
    def _is_static(x: object) -> bool:
        return isinstance(x, (int, Integer))

    reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
    numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))
    if not (_is_static(reduction_numel_hint) and _is_static(numel_hint)):
        # We don't support unbacked symints
        return ReductionHint.DEFAULT, 1

    should_split = reduction_type == "scan" or (
        not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
        and reduction_type
        not in (
            "argmax",
            "argmin",
        )
        and config.split_reductions
    )

    if should_split:
        inner_reduction_splits = reduction_split_factor
    else:
        def inner_reduction_splits(reduction_ranges):
            return 1

    if numel_hint == 1:
        split = inner_reduction_splits(reduction_ranges)
        return ReductionHint.INNER, split
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


def analyze_expression(expr, range_tree_nodes):
    """
    Analyze the entire expression, identify all FloorDiv and ModularIndexing expressions
    and determine if they can be split
    
    Parameters:
        expr: sympy expression
        range_tree_nodes: dictionary mapping symbols to range nodes
        
    Returns:
        Dict: analysis result dictionary
    """
    result = {
        "original_expr": str(expr),
        "floordiv_expressions": [],
        "modular_expressions": [],
        "can_split_all": True,
        "split_details": {}
    }
    
    # Recursively collect all FloorDiv and ModularIndexing expressions
    def collect_expressions(sub_expr, path=""):
        """Recursively collect FloorDiv and ModularIndexing expressions"""
        nonlocal result
        
        # If it's a FloorDiv expression
        if isinstance(sub_expr, FloorDiv):
            analysis = analyze_floordiv_expression(sub_expr, range_tree_nodes)
            analysis["path"] = path
            result["floordiv_expressions"].append(analysis)
            
            # Update the can_split_all flag
            if not analysis.get("can_split", True):
                result["can_split_all"] = False
        
        # If it's a ModularIndexing expression
        elif isinstance(sub_expr, ModularIndexing):
            analysis = analyze_modular_expression(sub_expr, range_tree_nodes)
            analysis["path"] = path
            result["modular_expressions"].append(analysis)
            
            # Check if it can be split
            if not analysis.get("can_split", True):
                result["can_split_all"] = False
        
        # Recursively process sub-expressions
        if hasattr(sub_expr, 'args'):
            for i, arg in enumerate(sub_expr.args):
                collect_expressions(arg, f"{path}.args[{i}]")
    
    # Start collection
    collect_expressions(expr, "")
    return result


def calculate_max_remainder(coeff, length, divisor_or_mod):
    """
    Calculate the maximum remainder for term coeff * symbol, where symbol ∈ [0, length-1]
    
    Parameters:
        coeff: coefficient
        length: symbol's value range length (symbol ∈ [0, length-1])
        divisor_or_mod: divisor or modulus
        
    Returns:
        maximum remainder value
    """
    # Try to convert coefficient and divisor/modulus to integers
    coeff_int = int(coeff) if isinstance(coeff, sympy.Integer) else None
    divisor_int = int(divisor_or_mod) if isinstance(divisor_or_mod, sympy.Integer) else None
    
    # If not integers, use conservative estimate
    if coeff_int is None or divisor_int is None:
        return min(divisor_or_mod - 1, coeff * (length - 1))
    
    # If coefficient is 0, remainder is always 0
    if coeff_int == 0:
        return 0
    
    # Calculate greatest common divisor
    g = gcd(coeff_int, divisor_int)
    
    # If length-1 is large enough, can reach maximum remainder divisor_int - g
    # The remainder period is divisor_int/g
    period = divisor_int // g
    
    # If symbol's value range covers the entire period, then maximum remainder is divisor_int - g
    if length - 1 >= period - 1:
        return divisor_int - g
    else:
        # Cannot reach maximum remainder, need to calculate the maximum remainder within the range [0, length-1]
        max_k = length - 1
        
        # Remainder sequence: 0, coeff, 2*coeff, ... mod divisor_int
        # We need to find k ∈ [0, max_k] such that (coeff_int * k) % divisor_int is maximized
        
        # Calculate maximum possible value
        max_possible = coeff_int * max_k
        
        if max_possible < divisor_int:
            # If maximum possible value is less than divisor, then maximum remainder is the maximum possible value
            return max_possible
        else:
            # Calculate max_possible % divisor_int
            remainder = max_possible % divisor_int
            
            # Find the largest multiple of g that does not exceed remainder
            # Because all remainders are multiples of g
            max_remainder = remainder // g * g
            start_k = max(0, max_k - period + 1)
            best_remainder = max_remainder
            
            for k in range(start_k, max_k + 1):
                r = (coeff_int * k) % divisor_int
                if r > best_remainder:
                    best_remainder = r
            
            return best_remainder


def analyze_floordiv_expression(expr, range_tree_nodes: Dict) -> Dict:
    result = {
        "expression": str(expr),
        "type": "FloorDiv",
        "can_split": False,
        "reason": "",
        "details": {},
        "split_form": ""
    }
    
    if not isinstance(expr, FloorDiv):
        result["reason"] = "not FloorDiv expression"
        return result
    
    arg, divisor = expr.args[0], expr.args[1]
    free_symbols = arg.free_symbols
    num_symbols = len(free_symbols)
    
    result["details"]["divisor"] = divisor
    result["details"]["expr"] = arg
    result["details"]["num_symbols"] = num_symbols
    result["details"]["symbols"] = list(free_symbols)
    
    # Multi-dimensional memory access expressions (≥2 dimensions) require splitting;
    # unary(single-dimension) expressions do not.
    if num_symbols < 2:
        result["can_split"] = True
        result["reason"] = f"num_symbols {num_symbols} < 2, no need split"
        return result
    
    if not isinstance(arg, sympy.Add):
        result["reason"] = f"expr {arg} not sympy.Add expression, can not split"
        return result
    
    add_terms = arg.args
    max_remainder_sum = 0
    term_details = []
    
    for term in add_terms:
        term_info = {
            "term": term,
            "coeff": 1,
            "symbol": None,
            "length": None,
            "max_value": None,
            "max_remainder": None
        }
        term_details.append(term_info)

        coeff = 1
        symbol = None
        
        if isinstance(term, sympy.Symbol):
            symbol = term
        elif isinstance(term, sympy.Mul):
            constant_factors = []
            for factor in term.args:
                if isinstance(factor, sympy.Symbol):
                    symbol = factor
                elif factor.is_number:
                    constant_factors.append(factor)
            
            if constant_factors:
                coeff = 1
                for cf in constant_factors:
                    coeff *= cf
        
        term_info["coeff"] = coeff
        term_info["symbol"] = symbol
        
        if symbol is None:
            result["reason"] = f"term {term} with no symbol"
            result["details"]["term_details"] = term_details
            return result
        
        if symbol not in range_tree_nodes:
            result["reason"] = f"symbol {symbol} not in range_tree_nodes"
            result["details"]["term_details"] = term_details
            return result
        
        length = range_tree_nodes[symbol].length
        term_info["length"] = length
        
        max_term_value = coeff * (length - 1)
        term_info["max_value"] = max_term_value
        
        max_remainder = calculate_max_remainder(coeff, length, divisor)
        term_info["max_remainder"] = max_remainder
        
        max_remainder_sum += max_remainder
    
    result["details"]["term_details"] = term_details
    result["details"]["max_remainder_sum"] = max_remainder_sum
    
    if max_remainder_sum < divisor:
        result["can_split"] = True
        result["reason"] = f"expr can split, max_remainder_sum {max_remainder_sum} < divisor {divisor}"
        
        split_terms = []
        for term in add_terms:
            split_terms.append(f"({term} // {divisor})")
        
        result["split_form"] = " + ".join(split_terms)
    else:
        result["reason"] = f"expr can not split, max_remainder_sum {max_remainder_sum} >= divisor {divisor}"
    
    return result


def analyze_modular_expression(expr, range_tree_nodes: Dict) -> Dict:
    """
    Analyze a single ModularIndexing expression
    
    Parameters:
        expr: ModularIndexing expression
        range_tree_nodes: dictionary mapping symbols to range nodes
        
    Returns:
        Dict: analysis result dictionary
    """
    result = {
        "expression": str(expr),
        "type": "ModularIndexing",
        "can_split": False,
        "reason": "",
        "details": {},
        "split_form": ""
    }
    
    # Check number of arguments
    args = expr.args
    if len(args) != 3:
        result["reason"] = f"ModularIndexing must have 3 args, but {len(args)} found"
        return result
    
    expr_to_mod, lower, upper = args
    
    # Check free symbols in expr_to_mod
    free_symbols = expr_to_mod.free_symbols
    num_symbols = len(free_symbols)
    
    result["details"]["expr_to_mod"] = expr_to_mod
    result["details"]["lower"] = lower
    result["details"]["upper"] = upper
    result["details"]["num_symbols"] = num_symbols
    result["details"]["symbols"] = list(free_symbols)
    
    if num_symbols < 2:
        result["can_split"] = True
        result["reason"] = f"num_symbols {num_symbols} < 2, no need split"
        return result
    
    # Check if expr_to_mod is an addition
    if not isinstance(expr_to_mod, sympy.Add):
        result["can_split"] = True
        result["reason"] = f"expr {expr_to_mod} not sympy.Add expression, no need split"
        return result
    
    # For ModularIndexing, the split condition is:
    # (expr1 % mod) + (expr2 % mod) < mod
    # where mod = upper - lower + 1
    
    mod = upper - lower + 1
    
    # Calculate the maximum sum of remainders
    add_terms = expr_to_mod.args
    max_remainder_sum = 0
    term_details = []
    
    for term in add_terms:
        term_info = {
            "term": term,
            "coeff": 1,
            "symbol": None,
            "length": None,
            "max_value": None,
            "max_remainder": None,
            "gcd_val": None,
            "period": None
        }
        term_details.append(term_info)

        # Extract coefficient and symbol
        coeff = 1
        symbol = None
        
        if isinstance(term, sympy.Symbol):
            symbol = term
        elif isinstance(term, sympy.Mul):
            constant_factors = []
            for factor in term.args:
                if isinstance(factor, sympy.Symbol):
                    symbol = factor
                elif factor.is_number:
                    constant_factors.append(factor)
            
            if constant_factors:
                coeff = 1
                for cf in constant_factors:
                    coeff *= cf
        
        term_info["coeff"] = coeff
        term_info["symbol"] = symbol
        
        if symbol is None:
            result["reason"] = f"term {term} with no symbol"
            result["details"]["term_details"] = term_details
            return result
        
        # Get symbol's length
        if symbol not in range_tree_nodes:
            result["reason"] = f"symbol {symbol} not in range_tree_nodes"
            result["details"]["term_details"] = term_details
            return result
        
        node = range_tree_nodes[symbol]
        length = node.length
        term_info["length"] = length
        
        # Calculate maximum possible value
        max_term_value = coeff * (length - 1)
        term_info["max_value"] = max_term_value
        
        if coeff is not None and mod is not None:
            gcd_val = gcd(coeff, mod)
            term_info["gcd_val"] = gcd_val
            
            # Remainder period
            period = mod // gcd_val
            term_info["period"] = period
        
        max_remainder = calculate_max_remainder(coeff, length, mod)
        term_info["max_remainder"] = max_remainder
        
        max_remainder_sum += max_remainder
    
    result["details"]["term_details"] = term_details
    result["details"]["max_remainder_sum"] = max_remainder_sum
    result["details"]["mod"] = mod
    
    # Determine if it can be split
    if max_remainder_sum < mod:
        result["can_split"] = True
        result["reason"] = f"expr can split, max_remainder_sum {max_remainder_sum} < mod {mod}"
        
        # Generate the split expression form
        split_terms = []
        for term in add_terms:
            split_terms.append(f"ModularIndexing({term}, {lower}, {upper})")
        
        result["split_form"] = " + ".join(split_terms)
    else:
        result["reason"] = f"expr can not split, max_remainder_sum {max_remainder_sum} >= mod {mod}"
    
    return result


def extract_modular_indexing_coefficient(expr):
    """
    Extract coefficient from ModularIndexing expression
    
    Convert ModularIndexing(k*x, lower, upper) to k * ModularIndexing(x, lower, upper/k)
    Condition: k is a constant and can divide (upper - lower + 1)
    """
    if not isinstance(expr, ModularIndexing):
        return expr
    
    args = expr.args
    if len(args) != 3:
        return expr
    
    expr_to_mod, lower, upper = args
    
    # Check if expr_to_mod is a multiplication expression
    if isinstance(expr_to_mod, sympy.Mul):
        # Find constant coefficient
        coefficient = 1
        other_factors = []
        
        for factor in expr_to_mod.args:
            # Check if it's an integer constant
            if isinstance(factor, sympy.Integer) and factor.is_constant():
                coefficient = coefficient * factor
            else:
                other_factors.append(factor)
        
        # If a constant coefficient greater than 1 is found
        if coefficient != 1:
            # Calculate modulus range
            mod_range = upper - lower + 1
            
            # Check if coefficient can divide modulus range
            if mod_range % coefficient == 0:
                # Construct new ModularIndexing arguments
                if other_factors:
                    if len(other_factors) == 1:
                        new_expr = other_factors[0]
                    else:
                        new_expr = sympy.Mul(*other_factors)
                else:
                    # If no other factors, use 1
                    new_expr = sympy.Integer(1)
                
                # Calculate new upper bound
                new_upper = lower + mod_range // coefficient - 1
                
                # Create new ModularIndexing
                new_mod = ModularIndexing(new_expr, lower, new_upper)
                
                # Return coefficient multiplied by new ModularIndexing
                return coefficient * new_mod
    
    return expr


def eliminate_zero_term(term):
    expr, divisor = term.args
    if not isinstance(expr, sympy.Symbol):
        return term
    if (expr in V.kernel.range_tree_nodes):
        numel = V.kernel.range_tree_nodes[expr].length
    else:
        numel = V.kernel.range_tree_nodes_removed[expr].length
    
    length = term.eval(numel, divisor)
    if length == 0:
        return sympy.Integer(0)
    return term


def eliminate_modular(term):
    """
    Eliminate unnecessary ModularIndexing expressions
    
    When ModularIndexing(expr, lower, upper) has the same range as the entire range,
    it can be simplified to expr itself.
    
    Parameters:
        term: ModularIndexing expression
        
    Returns:
        Simplified expression or original expression
    """
    # If not a ModularIndexing expression, return directly
    if not isinstance(term, sympy.Function) or term.func.__name__ != 'ModularIndexing':
        return term
    
    # Get arguments
    expr, lower, upper = term.args
    
    # Get symbol's length information
    def get_symbol_length(symbol: sympy.Symbol) -> Optional[int]:
        """Get symbol's length (from range tree)"""
        if symbol in V.kernel.range_tree_nodes:
            return V.kernel.range_tree_nodes[symbol].length
        elif symbol in V.kernel.range_tree_nodes_removed:
            return V.kernel.range_tree_nodes_removed[symbol].length
        return None
    
    # Handle symbol expression
    if isinstance(expr, sympy.Symbol):
        numel = get_symbol_length(expr)
        if numel is not None:
            length = term.eval(numel, lower, upper)
            if length == numel:
                return expr
    
    # Handle multiplication expression
    elif isinstance(expr, sympy.Mul):
        # Try to extract coefficient and variable
        coeff = 1
        var = None
        
        for arg in expr.args:
            if isinstance(arg, sympy.Symbol):
                var = arg
            elif arg.is_number:
                coeff *= arg
            else:
                # Contains complex cases with non-symbols and non-numbers, not supported yet
                return term
        
        if var is not None:
            numel = get_symbol_length(var)
            if numel is not None:
                length = term.eval(numel, lower, upper)
                if length == numel:
                    return expr  # Return entire multiplication expression
    
    # Unsupported cases, return original expression
    return term


def split_expression(expr):
    """
    Split expression according to specified logic:
    1. If it's an Add/Mul expression, split into multiple args, recursively process each arg
    2. If it's a // expression (floor division), split it
    3. If it's a ModularIndexing expression, split it
    """
    # 1. If it's an Add expression
    if isinstance(expr, sympy.Add):
        # Recursively process each argument, then reconstruct Add
        new_args = [split_expression(arg) for arg in expr.args]
        return sympy.Add(*new_args)

    # 2. If it's a Mul expression
    elif isinstance(expr, sympy.Mul):
        # Recursively process each argument, then reconstruct Mul
        new_args = [split_expression(arg) for arg in expr.args]
        return sympy.Mul(*new_args)

    # 3. If it's a floor division expression
    elif isinstance(expr, FloorDiv):
        # Get floor arguments
        arg = expr.args[0]
        divisor = expr.args[1]
        
        # Check if arg is Add
        if isinstance(arg, sympy.Add):
            # Assume denominator is 1: floor(a+b) -> floor(a) + floor(b)
            split_terms = []
            for term in arg.args:
                new_term = FloorDiv(term, divisor)
                new_term = eliminate_zero_term(new_term)
                split_terms.append(new_term)
            return sympy.Add(*split_terms)
        
        # Cannot split, return original expression
        return expr
    
    # 4. If it's a ModularIndexing expression
    elif isinstance(expr, ModularIndexing):
        args = expr.args
        expr_to_mod, lower, upper = args

        # If first argument is Add, split it
        if isinstance(expr_to_mod, sympy.Add):
            # Split: ModularIndexing(a+b, lower, upper) -> 
            # ModularIndexing(a, lower, upper) + ModularIndexing(b, lower, upper)
            split_terms = []
            for term in expr_to_mod.args:
                new_mod = ModularIndexing(term, lower, upper)
                # ModularIndexing(16*z0, 1, 128) -> 16*ModularIndexing(z0, 1, 8)
                new_mod = eliminate_modular(new_mod)
                new_mod = extract_modular_indexing_coefficient(new_mod)
                split_terms.append(new_mod)
            return sympy.Add(*split_terms)
        else:
            new_mod = ModularIndexing(expr_to_mod, lower, upper)
            new_mod = eliminate_modular(new_mod)
            new_mod = extract_modular_indexing_coefficient(new_mod)
            return new_mod

    # 5. Other types of expressions, return directly
    else:
        return expr


def transform_dims_in_indexing(self, indices):
    if self.indexing is None:
        remove_zero_terms(self.indexing_exprs, self.var_ranges)
        generate_body_indexing(self, indices)
    
    log.debug(f"[Linear] ori indexing:{self.indexing}\nV.kernel.range_tree_nodes:{V.kernel.range_tree_nodes}")
    for key, index_expr in self.indexing.items():
        analyse_res = analyze_expression(index_expr, V.kernel.range_tree_nodes)
        log.debug(f"[Linear] linear analyse res:{analyse_res}")
        if not analyse_res["can_split_all"]:
            raise ValueError(f"Can not split expression:{self.indexing}"\
                             f"\nrange_tree_nodes:{V.kernel.range_tree_nodes}"\
                             f"\nanalyse_res:{analyse_res}")
        self.indexing[key] = split_expression(index_expr)

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
    pointwise_op_list = [
        'mul',
        'add'
    ]

    def check_pointwise_op(reduction_index):
        for pointwise_op in pointwise_op_list:
            if pointwise_op in reduction_index:
                return True
        return False
    if inductor_indirect_memory_mode != str(NPUKernelType.SIMD_SIMT_MIX):
        return NPUKernelType(inductor_indirect_memory_mode)

    node_map = {}
    for node in loop_body.root_block.graph.nodes:
        node_map[node.name] = node
        if 'reduction' == node.name:
            reduction_type_pos = 3 # 3 is reduction_type_pos
            reduction_index_pos = 4 # 4 is reduction_index_pos
            load_index_pos = 2 # 2 is load index position
            reduction_type = node.args[reduction_type_pos]
            if reduction_type != 'sum':
                continue
            reduction_index = str(node.args[reduction_index_pos])
            if 'load' in reduction_index:
                if reduction_index not in node_map:
                    continue
                load_node = node_map[reduction_index]
                if load_node.args[load_index_pos].name not in node_map:
                    continue
                get_load_index = node_map[load_node.args[load_index_pos].name]
                load_index = get_load_index.args[0]
                if load_index not in loop_body.indexing:
                    continue
                if 'indirect' in str(loop_body.indexing[load_index]):
                    return NPUKernelType.SIMT_ONLY
                    
            elif check_pointwise_op(reduction_index):
                pointwise_node = node_map.get(reduction_index, None)
                if pointwise_node is None:
                    continue           
                pointwise_inputs = pointwise_node.args
                for pointwise_input in pointwise_inputs:
                    if 'load' not in pointwise_input.name:
                        continue
                    load_node = node_map.get(pointwise_input.name, None)
                    if load_node is None:
                        continue
                    get_load_index = node_map.get(load_node.args[load_index_pos].name, None)
                    if get_load_index is None:
                        continue                 
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