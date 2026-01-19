import os
import importlib
import inspect
import pkgutil


__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

from . import ir
from . import lowering as npu_lowering
from torch._inductor import lowering
import sys

def get_functions_from_module(module):
    functions = {}
    members = inspect.getmembers(module, inspect.isfunction)

    for name, func in members:
        if inspect.getmodule(func) == module:
            functions[name] = func
    
    return functions

npu_functions = get_functions_from_module(npu_lowering)
functions = get_functions_from_module(lowering)
for name, _ in functions.items():
    if name in npu_functions:
        setattr(lowering, name, npu_functions[name])

extra_lowerings = set(lowering.lowerings.keys()) - set(npu_lowering.lowerings.keys())
npu_lowering.lowerings.update({k: lowering.lowerings[k] for k in extra_lowerings})
lowering.lowerings = npu_lowering.lowerings
lowering._maybe_layout_constraints = npu_lowering._maybe_layout_constraints
lowering.fallbacks = npu_lowering.fallbacks
lowering.needs_realized_inputs = npu_lowering.needs_realized_inputs
lowering.foreach_ops = npu_lowering.foreach_ops
lowering.inplace_foreach_ops = npu_lowering.inplace_foreach_ops
lowering.inplaceable_foreach_ops = npu_lowering.inplaceable_foreach_ops

from torch._inductor import graph
importlib.reload(graph)