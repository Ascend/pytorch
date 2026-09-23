"""DVM v2 adapter helpers for the torch.fx backend."""

import ctypes
import os
from pathlib import Path
from typing import List, Optional

from torch.fx.node import Node

from fxrt.ir import Op
from fxrt.compiled_kernel_adapter import (
    get_compiled_kernel_index,
    get_compiled_kernel_mutation_args,
    get_compiled_kernel_mutation_output_nodes,
    is_compiled_kernel_wrapper_mutation_target,
    map_hop_args,
    resolve_compiled_kernel,
)


# Registry for DVM V2 callable kernels. FXRT IR carries only the string
# handle; the Python registry owns the materialized kobj so C++ operators do
# not hold Python objects until interpreter teardown.
_DVM_FUNC_REGISTRY = {}
_DVM_KERNEL_REGISTRY = {}
_DVM_FUNC_COUNTER = 0
_DVM_COMPILED_KERNEL_NAME_PREFIX = "dvm_"
_DVM_V2_LIBRARY_NAME = "libops_ascend_dvm_v2.so"
_DVM_V2_LOADED = False
_DVM_V2_LOAD_ERROR = None
_DVM_V2_CDLL = None
_DVM_TEMPLATE_REQUIRED_ATTRS = ("is_split", "kernel_type", "kernel_flags")
_DVM_KERNEL_REQUIRED_ATTRS = (
    "kernel",
    "relocs",
    "loads",
    "stores",
    "num_tensor_inputs",
    "num_outputs",
    "workspace_size",
    "is_dynamic",
    "is_split",
)


def _dvm_v2_library_path() -> Path:
    import fxrt  # pylint: disable=import-outside-toplevel

    return Path(fxrt.__file__).resolve().parent / "lib" / _DVM_V2_LIBRARY_NAME


def ensure_dvm_v2_runtime_available():
    """Load the optional DVM v2 op library only when a DVM graph needs it."""
    global _DVM_V2_LOADED, _DVM_V2_LOAD_ERROR, _DVM_V2_CDLL  # pylint: disable=global-statement
    if _DVM_V2_LOADED:
        return

    library_path = _dvm_v2_library_path()
    if not library_path.exists():
        _DVM_V2_LOAD_ERROR = (
            f"DVM v2 support is not built in this fxrt package: "
            f"{library_path} does not exist"
        )
        raise RuntimeError(_DVM_V2_LOAD_ERROR)

    mode = getattr(os, "RTLD_LAZY", 1) | getattr(os, "RTLD_LOCAL", 0)
    try:
        _DVM_V2_CDLL = ctypes.CDLL(str(library_path), mode=mode)
    except OSError as exc:
        _DVM_V2_LOAD_ERROR = f"Failed to load DVM v2 op library {library_path}: {exc}"
        raise RuntimeError(_DVM_V2_LOAD_ERROR) from exc

    _DVM_V2_LOADED = True
    _DVM_V2_LOAD_ERROR = None


def _require_attrs(obj, attrs, what: str):
    missing = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing:
        raise RuntimeError(
            f"Current torch-npu DVM kernel object does not provide FXRT DVM v2 ABI for {what}; "
            f"missing attributes: {', '.join(missing)}"
        )


def _materialize_dvm_kernel(dvm_func):
    """Instantiate and setup an isolated DVM kernel object for one FXRT op."""
    template = getattr(dvm_func, "kobj", None)
    builder = getattr(dvm_func, "__wrapped__", None)
    if template is None or builder is None:
        raise RuntimeError(
            "DVM function must be a torch_npu dvm.kernel decorated function "
            "with both 'kobj' and '__wrapped__' attributes"
        )
    _require_attrs(template, _DVM_TEMPLATE_REQUIRED_ATTRS, "kernel template")

    kernel_cls = type(template)
    if template.is_split():
        kobj = kernel_cls()
    else:
        kobj = kernel_cls(template.kernel_type(), template.kernel_flags())
    builder(kobj)
    kobj.setup()
    if kobj is None:
        raise RuntimeError("DVM function did not produce a kernel object")
    _require_attrs(kobj, _DVM_KERNEL_REQUIRED_ATTRS, "materialized kernel")
    return kobj


def register_dvm_func(dvm_func) -> str:
    """Register a decorated dvm.kernel function and return its handle."""
    global _DVM_FUNC_COUNTER  # pylint: disable=global-statement
    ensure_dvm_v2_runtime_available()
    handle = f"dvm_func_{_DVM_FUNC_COUNTER}"
    _DVM_FUNC_COUNTER += 1
    _DVM_FUNC_REGISTRY[handle] = dvm_func
    _DVM_KERNEL_REGISTRY[handle] = _materialize_dvm_kernel(dvm_func)
    return handle


def get_dvm_kernel_obj(handle: str):
    """Get a materialized dvm.kernel object by handle."""
    return _DVM_KERNEL_REGISTRY.get(handle)


def _get_compiled_kernel_name(compiled_kernel) -> Optional[str]:
    """Resolve the generated kernel name from a compiled-kernel wrapper."""
    if compiled_kernel is None:
        return None
    for candidate in (compiled_kernel, getattr(compiled_kernel, "_compiled", None)):
        if candidate is None:
            continue
        kernel_name = getattr(candidate, "_kernel_name", None)
        if isinstance(kernel_name, str) and kernel_name:
            return kernel_name
        kernel_name = getattr(candidate, "__name__", None)
        if isinstance(kernel_name, str) and kernel_name:
            return kernel_name
    return None


def _extract_dvm_func_from_compiled_kernel(compiled_kernel):
    """Return the decorated DVM function from a compiled-kernel wrapper."""
    kernel_name = _get_compiled_kernel_name(compiled_kernel)
    if kernel_name is None or not kernel_name.startswith(_DVM_COMPILED_KERNEL_NAME_PREFIX):
        return None

    candidates = [compiled_kernel, getattr(compiled_kernel, "_compiled", None)]
    for candidate in candidates:
        if candidate is None or not callable(candidate):
            continue
        if hasattr(candidate, "kobj"):
            return candidate
    for candidate in candidates:
        if candidate is None or not callable(candidate):
            continue
        return candidate
    return None


def get_dvm_func_from_node(node: Node):
    """Resolve the decorated DVM function represented by an FX node."""
    if not is_compiled_kernel_wrapper_mutation_target(node.target):
        return None

    compiled_kernel = resolve_compiled_kernel(get_compiled_kernel_index(node))
    dvm_func = _extract_dvm_func_from_compiled_kernel(compiled_kernel)
    if dvm_func is None:
        return None

    return dvm_func


def _prepare_dvm_call_v2_args(node: Node, dvm_func, executor, env, sym_mgr) -> List[Node]:
    """Build FXRT dvm_call_v2 inputs from a compiled-kernel mutation HOP node."""
    hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)
    mutated_arg_index_set = set(mutated_arg_indices)
    flat_node_args = [
        arg for idx, arg in enumerate(hop_args) if idx not in mutated_arg_index_set
    ]
    handle = register_dvm_func(dvm_func)
    return map_hop_args([handle] + flat_node_args, env, executor, sym_mgr)


def lower_compiled_kernel_dvm_node(
    node,
    executor,
    env,
    sym_mgr,
    get_node_meta_value,
    add_tuple_getitem_node,
):
    """Lower a DVM compiled-kernel mutation HOP node into an FXRT dvm_call_v2 op."""
    dvm_func = get_dvm_func_from_node(node)
    if dvm_func is None:
        return False

    output_nodes = get_compiled_kernel_mutation_output_nodes(node)
    if not output_nodes:
        raise RuntimeError("DVM compiled_kernel_wrapper_mutation requires at least one mutated output buffer")

    input_nodes = _prepare_dvm_call_v2_args(node, dvm_func, executor, env, sym_mgr)

    output_examples = []
    for output_node in output_nodes:
        example_value = get_node_meta_value(output_node)
        if example_value is None:
            raise RuntimeError(
                f"DVM compiled kernel output buffer node '{output_node.name}' is missing example_value/val metadata"
            )
        output_examples.append(example_value)

    if len(output_examples) == 1:
        output_value = sym_mgr.from_torch_with_sym(output_examples[0])
        dvm_node = executor.add_op_node(Op.dvm_call_v2, input_nodes, output_value)
        env[node] = dvm_node
        env[output_nodes[0]] = dvm_node
        return True

    tuple_output = sym_mgr.from_torch_with_sym(tuple(output_examples))
    tuple_node = executor.add_op_node(Op.dvm_call_v2, input_nodes, tuple_output)
    env[node] = tuple_node
    for idx, (output_node, output_example) in enumerate(zip(output_nodes, output_examples)):
        output_value = sym_mgr.from_torch_with_sym(output_example)
        env[output_node] = add_tuple_getitem_node(executor, sym_mgr, tuple_node, idx, output_value)
    return True
