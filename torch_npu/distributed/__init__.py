# import torch_npu


# __all__ = [
#     "Backend", "_backend", "group", "GroupMember", "is_hccl_available", "is_initialized",
#     "get_backend", "init_process_group", "destroy_process_group", "get_rank", "get_world_size",
#     "isend", "irecv", "send", "recv", "P2POp", "batch_isend_irecv", "broadcast", "all_reduce",
#     "all_reduce_coalesced", "reduce",  "all_gather", "all_gather_coalesced", "gather", "scatter",
#     "reduce_scatter", "all_to_all_single", "all_to_all", "barrier", "new_group", "ProcessGroupHCCL",
#     "Reducer", "_DEFAULT_FIRST_BUCKET_BYTES", "_register_comm_hook", "_register_builtin_comm_hook",
#     "_broadcast_coalesced", "_compute_bucket_assignment_by_size", "_set_construction_logging_data",
#     "_get_ddp_logging_data", "_get_global_rank"
# ]


# def is_available():
#     """
#     Returns ``True`` if the distributed package is available. Otherwise,
#     ``torch.distributed`` does not expose any other APIs. Currently,
#     ``torch.distributed`` is available on Linux, MacOS and Windows. Set
#     ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
#     Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
#     ``USE_DISTRIBUTED=0`` for MacOS.
#     """
#     return hasattr(torch_npu._C, "_c10d_init")


# if is_available() and not torch_npu._C._c10d_init():
#     raise RuntimeError("Failed to initialize torch_npu.distributed")


# from torch_npu._C._distributed_c10d import (
#     Reducer,
#     _DEFAULT_FIRST_BUCKET_BYTES,
#     _register_comm_hook,
#     _register_builtin_comm_hook, 
#     _broadcast_coalesced,
#     _compute_bucket_assignment_by_size,
#     _set_construction_logging_data,
#     _get_ddp_logging_data
# )
# from .distributed_c10d import Group as group
# from .distributed_c10d import (
#     _backend, Backend, GroupMember, is_hccl_available, is_initialized, get_backend,
#     init_process_group, destroy_process_group, get_rank, get_world_size, isend,
#     irecv, send, recv, P2POp, batch_isend_irecv, broadcast, all_reduce, all_reduce_coalesced,
#     reduce, all_gather, all_gather_coalesced, gather, scatter, reduce_scatter,
#     all_to_all_single, all_to_all, barrier, new_group, ProcessGroupHCCL, _get_global_rank
# )
