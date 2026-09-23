"""
utils for converting between torch and fxrt.ir.
"""
from typing import Any, List, Tuple, Optional

import torch
from torch import distributed as dist
from torch._C._distributed_c10d import _resolve_process_group
from torch.fx.node import Node

from fxrt import _fxrt_torch
from fxrt._fxrt_api import is_custom_op_registered
from fxrt.ir import (
    Value,
    Tuple as FxrtTuple,
    DataType,
    SymbolicVar,
)
from fxrt._fxrt_collective import CollectiveManager

_INT64_MAX = (1 << 63) - 1

# pylint: disable=protected-access
_DIST_OP_LIST = [
    torch.ops._c10d_functional.all_gather_into_tensor,
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.reduce_scatter_tensor,
    torch.ops._c10d_functional.all_to_all_single,
]


def _extract_global_comm_info():
    """Extract distributed communication information (rank, world_size)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()

    CollectiveManager.instance().set_global_rank_id(rank)
    # TODO: Multi-machine scenario needs verification, current implementation only supports single machine with 8 NPUs
    CollectiveManager.instance().set_local_rank_id(torch.npu.current_device())
    CollectiveManager.instance().set_global_rank_size(world_size)


def _set_communication_info(ptd):
    """Get communication info from torch and set to CollectiveManager for a given process group."""
    pg = _resolve_process_group(ptd)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()

    group_rank = dist.get_rank(pg)
    rank_list = dist.get_process_group_ranks(pg)

    hccl_comm_handle = pg._get_backend(torch.device("npu")).get_hccl_comm(rank)

    CollectiveManager.instance().set_global_rank_id(rank)
    CollectiveManager.instance().set_local_rank_id(torch.npu.current_device())
    CollectiveManager.instance().set_global_rank_size(world_size)

    CollectiveManager.instance().create_communication_group(
        f"{ptd}", rank_list, group_rank, hccl_comm_handle
    )


def _extract_and_setup_comm_groups(node_args):
    ptd_arg = node_args[-1]
    if CollectiveManager.instance().is_group_exist(f"{ptd_arg}"):
        return
    _set_communication_info(ptd_arg)


def get_collective_info_from_torch(gm: torch.fx.GraphModule):
    """
    Extract communication info from fx graph and set to CollectiveManager.
    """
    if dist.is_initialized():
        _extract_global_comm_info()
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method"):
                if node.target in _DIST_OP_LIST:
                    _extract_and_setup_comm_groups(node.args)


def from_torch(obj: Any) -> Value:
    """
    Convert a torch object to fxrt.ir.Value.
    """
    if isinstance(obj, Value):
        return obj
    if isinstance(obj, torch.SymInt):
        return Value(SymbolicVar(str(obj)))
    if isinstance(obj, torch.SymFloat):
        return Value(float(obj))
    if isinstance(obj, (list, tuple)):
        return Value(FxrtTuple([from_torch(e) for e in obj]))
    # pylint: disable=protected-access
    if isinstance(obj, torch._subclasses.FakeTensor):
        return Value(_fxrt_torch.from_torch(obj, is_fake=True))
    if isinstance(obj, torch.Tensor):
        return Value(_fxrt_torch.from_torch(obj))
    if isinstance(obj, (int, float, bool, str)):
        return Value(obj)
    if isinstance(obj, torch.device):
        # Format: "device:index", device in ("cpu", "npu"), index must be int
        type_str = str(obj.type).lower()
        if type_str == "privateuse1":
            type_str = "npu"
        if type_str not in ("cpu", "npu"):
            raise ValueError(
                f"from_torch(torch.device): device must be 'cpu' or 'npu', got '{type_str}'"
            )
        index = obj.index if obj.index is not None else 0
        if not isinstance(index, int):
            raise ValueError(
                f"from_torch(torch.device): device index must be int, got {type(index).__name__}"
            )
        return Value(f"{type_str}:{index}")
    if isinstance(obj, torch.dtype):
        dtype_str = str(obj).rsplit(".", maxsplit=1)[-1]  # "torch.float32" -> "float32"
        return Value(DataType.convert_str_to_int(dtype_str))
    if isinstance(obj, torch.layout):
        return Value()
    if isinstance(obj, torch.memory_format):
        return Value()
    if obj is None:
        return Value()
    raise TypeError(
        f"Unsupported python type for conversion to fxrt.ir.Value: {type(obj)}"
    )


def get_tensor_arg_dtype(arg):
    """Resolve dtype from a tensor argument or FX tensor node."""
    if isinstance(arg, Node):
        example_value = arg.meta.get("example_value", None)
        return getattr(example_value, "dtype", None)
    return getattr(arg, "dtype", None)


def to_torch(value: Value) -> Any:
    """
    Convert a fxrt.ir.Value to torch object.
    """
    if not isinstance(value, Value):
        return value
    if value.is_none():
        return None
    if value.is_tensor():
        return _fxrt_torch.to_torch(value.to_tensor())
    if value.is_tuple():
        return tuple(to_torch(item) for item in value.to_tuple())
    if value.is_int() or value.is_symbol():
        return value.to_int()
    if value.is_double():
        return value.to_double()
    if value.is_bool():
        return value.to_bool()
    if value.is_string():
        return value.to_string()
    raise TypeError(
        f"Unsupported fxrt.ir.Value for conversion to python object: {value}"
    )


def set_device_context():
    _fxrt_torch.set_device_context()


def update_runtime_inputs(
    param_nodes: List[Any],
    new_inputs: Tuple[Any, ...],
    input_is_parameter: Optional[List[bool]] = None,
    graph_key: Optional[int] = None,
    non_parameter_tensor_indices: Optional[List[int]] = None,
) -> None:
    """
    Update placeholder nodes with runtime input values.

    When AclGraph is enabled, non-parameter tensor inputs are staticized
    (cloned during capture, copy_'d during replay) so the captured graph
    always sees stable device addresses.
    """
    _fxrt_torch.batch_update_runtime_inputs(
        param_nodes,
        new_inputs,
        input_is_parameter,
        graph_key,
        non_parameter_tensor_indices,
    )


def _dim_size_or_open(dim_size):
    """Return concrete int for static dims, INT64_MAX for dynamic SymInt dims."""
    if isinstance(dim_size, int):
        return dim_size
    if isinstance(dim_size, torch.SymInt):
        return _INT64_MAX
    try:
        return int(dim_size)
    except (TypeError, ValueError):
        return _INT64_MAX


def tuple_indices_to_slice_arg(indices: Tuple[int, ...], shape: Tuple[int, ...]):
    """
    Convert tuple indices to slice arguments.

    Open ends use INT64_MAX only for dynamic SymInt dims; concrete static dims
    keep the real size so runtime aclnn paths (e.g., getitem_slice) can use it
    directly, while Path C Slice expand resolves INT64_MAX for dynamic shapes.
    """
    num_dims = len(shape)
    begin = []
    end = []
    axes = []
    steps = []
    processed_indices = []
    none_nums = 0
    # None in indices means expanding a dimension at the corresponding shape position, with size 1
    for idx in indices:
        if idx is None:
            none_nums += 1
    for idx in indices:
        if idx is Ellipsis:
            # Insert missing dimensions after ellipsis
            missing_dims = num_dims - len(indices) + 1 + none_nums
            processed_indices.extend(["ellipsis"] * missing_dims)
        else:
            processed_indices.append(idx)

    axis = 0
    for idx in processed_indices:
        if isinstance(idx, slice):
            begin.append(idx.start if idx.start is not None else 0)
            end.append(idx.stop if idx.stop is not None else _dim_size_or_open(shape[axis]))
            steps.append(idx.step if idx.step is not None else 1)
            axes.append(axis)
        elif idx == "ellipsis":
            begin.append(0)
            end.append(_dim_size_or_open(shape[axis]))
            steps.append(1)
            axes.append(axis)
        elif idx is None:
            continue
        else:
            begin.append(idx)
            end.append(idx + 1)
            steps.append(1)
            axes.append(axis)
        axis += 1
    return begin, end, steps, axes


def is_op_registered_by_custom_or_torch(full_op_name: str) -> bool:
    """
    Check if the full_op_name is registered in the custom operator registry or torch.ops registry.
    """
    if full_op_name is None:
        return False

    if "." in full_op_name:
        op_namespace, op_name = full_op_name.rsplit(".", 1)
    elif "::" in full_op_name:
        op_namespace, op_name = full_op_name.rsplit("::", 1)
    else:
        op_namespace, op_name = None, full_op_name

    # Check if the op_name is registered in the custom operator registry.
    if is_custom_op_registered(op_name):
        return True

    # Check if the op_name is registered in the torch.ops registry.
    torch_ns = getattr(torch.ops, op_namespace)
    if hasattr(torch_ns, op_name):
        return True

    return False
