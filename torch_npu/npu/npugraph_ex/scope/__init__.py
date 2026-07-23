__all__ = ["limit_core_num", "deterministic"]


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int, stream=None):
    from torch_npu.dynamo.npugraph_ex import scope
    return scope.limit_core_num(op_aicore_num, op_vectorcore_num, stream=stream)


def deterministic(level: int):
    from torch_npu.dynamo.npugraph_ex import scope
    return scope.deterministic(level)
