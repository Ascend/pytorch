import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error

__all__ = ["npu_combine_tensors", "get_part_combined_tensor", "is_combined_tensor_valid"]


def npu_combine_tensors(list_of_tensor, require_copy_value=True):
    if len(list_of_tensor) == 0:
        return None
    if None in list_of_tensor:
        raise RuntimeError("Tensors to combine must not have `None`." + pta_error(ErrCode.PARAM))

    total_numel = 0
    dtype = list_of_tensor[0].dtype
    for tensor in list_of_tensor:
        if tensor.dtype != dtype:
            raise RuntimeError("Tensors to combine must have the same dtype." + pta_error(ErrCode.TYPE))
        if tensor.device.type != "npu":
            raise RuntimeError("Tensors to combine must be on NPU, got {}.".format(tensor.device.type) +
                               pta_error(ErrCode.VALUE))
        total_numel += torch_npu.get_storage_size(tensor)

    if total_numel == 0:
        return None

    combined_tensor = torch.zeros(total_numel, dtype=dtype).npu()

    idx = 0
    if require_copy_value:
        for tensor in list_of_tensor:
            temp = tensor.clone()
            torch_npu.npu_change_data_ptr(tensor, combined_tensor, idx)
            tensor.copy_(temp)
            idx += torch_npu.get_storage_size(tensor)
    else:
        for tensor in list_of_tensor:
            torch_npu.npu_change_data_ptr(tensor, combined_tensor, idx)
            idx += torch_npu.get_storage_size(tensor)

    return combined_tensor


def get_part_combined_tensor(combined_tensor, index, size):
    if combined_tensor is None or size == 0:
        return None

    if (index + size) > torch_npu.get_storage_size(combined_tensor):
        raise RuntimeError("(index + size) ({}) > torch_npu.get_storage_size(combined_tensor) ({})".format(
                           index + size, torch_npu.get_storage_size(combined_tensor)) + pta_error(ErrCode.VALUE))

    part_tensor = torch.zeros(size, dtype=combined_tensor.dtype).npu()
    torch_npu.npu_change_data_ptr(part_tensor, combined_tensor, index)

    return part_tensor


def is_combined_tensor_valid(combined_tensor, list_of_tensor):
    if len(list_of_tensor) == 0:
        return True
    if combined_tensor is None:
        return False

    combined_tensor_start_addr = combined_tensor.data_ptr()
    combined_tensor_end_addr = combined_tensor_start_addr + \
        torch_npu.get_storage_size(combined_tensor) * combined_tensor.element_size()

    for tensor in list_of_tensor:
        if tensor is None or \
                tensor.data_ptr() < combined_tensor_start_addr or \
                tensor.data_ptr() >= combined_tensor_end_addr:
            return False

    return True
