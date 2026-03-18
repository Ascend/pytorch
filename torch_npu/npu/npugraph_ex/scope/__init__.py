__all__ = ["limit_core_num"]


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    from torch_npu.dynamo.npugraph_ex import scope
    return scope.limit_core_num(op_aicore_num, op_vectorcore_num)
