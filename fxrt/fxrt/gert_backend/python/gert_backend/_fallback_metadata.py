"""Describe fallback output alias contracts without inspecting device data."""

import torch
from torch.fx import map_arg

from fxrt import fx_backend as fxb
from fxrt.ir import Op


def _proven_contiguous(value):
    if not isinstance(value, torch.Tensor):
        return False
    try:
        return value.is_contiguous()
    except (RuntimeError, TypeError, ValueError):
        # Unbacked/data-dependent sizes need the runtime fallback protocol.
        # Their lack of a compile-time dense proof must not add a shape guard.
        return False


# The converter already uses fx_backend's exact schema selection for custom_call.
# pylint: disable=protected-access
def fallback_output_contract(node):
    """Return schema read aliases and independently allocated dense outputs.

    Read aliases require both the dispatcher schema and FakeTensor to agree.
    The V3 bridge does not export arbitrary output strides; only dense views
    starting at the same address can reuse the GE input allocation here.
    """
    name, arguments = fxb._flatten_args(Op.custom_call, node)
    _, schemas = fxb._get_op_schemas(fxb._convert_operator_to_torch_op(node.target))
    schema = next((candidate for candidate in schemas or ()
                   if f"{candidate.name}.{candidate.overload_name or 'default'}" == name), None)
    values = map_arg(arguments, fxb._get_node_meta_value)
    output = fxb._get_node_meta_value(node)
    outputs = output if isinstance(output, (tuple, list)) else (output,)
    tensors = []

    def collect(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(values)
    fresh = tuple(_proven_contiguous(value)
                  and str(value.storage_offset()) == "0"
                  and not any(torch._C._is_alias_of(value, source) for source in tensors)
                  for value in outputs)
    aliases = []
    if schema is None or len(schema.returns) != len(outputs):
        return (), fresh
    for index, (result, value) in enumerate(zip(schema.returns, outputs)):
        if result.alias_info is None or not _proven_contiguous(value):
            continue
        for arg_index, (argument, source) in enumerate(zip(schema.arguments, values)):
            info = argument.alias_info
            if (info is None or info.is_write or not isinstance(source, torch.Tensor)
                    or not _proven_contiguous(source)
                    or not (info.before_set & result.alias_info.before_set)
                    or not torch._C._is_alias_of(value, source)):
                continue
            # Compare expressions, without introducing a new sample-dependent
            # condition. Offset-bearing aliases need the richer output ABI.
            if str(value.storage_offset()) != str(source.storage_offset()):
                continue
            aliases.append((arg_index, index))
            break
    return tuple(aliases), fresh
