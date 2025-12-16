import torch


def npu_wait_tensor(self: torch.Tensor, dependency: torch.Tensor):
    from torch_npu.dynamo.torchair import scope
    return scope.npu_wait_tensor(self, dependency)


def npu_stream_switch(stream_tag: str):
    from torch_npu.dynamo.torchair import scope
    return scope.npu_stream_switch(stream_tag, 0)


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    from torch_npu.dynamo.torchair import scope
    return scope.limit_core_num(op_aicore_num, op_vectorcore_num)