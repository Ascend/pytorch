from datetime import timedelta
from typing import Optional
import warnings
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.distributed.distributed_c10d import _get_default_group, get_group_rank, _check_single_tensor, \
    _check_tensor_list, _coalescing_manager, _ensure_all_tensors_same_dtype, get_rank, _rank_not_in_group, \
    _warn_not_in_group, GatherOptions, _validate_output_list_for_rank, GroupMember, _get_group_size,\
    _get_pg_default_device, _object_to_tensor, get_world_size, _tensor_to_object, all_gather, Backend,\
    get_backend, GatherOptions, _update_default_pg, _world, _unregister_all_process_groups, _pg_map,\
    ProcessGroup, default_pg_timeout, _unregister_process_group

__all__ = ["batch_isend_irecv", "gather", "gather_object", "is_hccl_available", "reinit_process_group"]


def batch_isend_irecv(p2p_op_list):
    group = p2p_op_list[0].group
    device = p2p_op_list[0].tensor.device
    is_multi_pg = True
    if device.type == "cuda":
        with _coalescing_manager(group, device, async_ops=True) as cm:
            for p2p_op in p2p_op_list:
                p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        return cm.works
    elif device.type == "npu":
        if group is None:
            group = _get_default_group()
            is_multi_pg = False
        _group = group._get_backend(device)
        op_type = []
        tensors = []
        remote_rank_list = []
        for p2p_op in p2p_op_list:
            op_type.append(p2p_op.op.__name__)
            tensors.append(p2p_op.tensor)
            rank_for_op = get_group_rank(group, p2p_op.peer) if is_multi_pg else p2p_op.peer
            remote_rank_list.append(rank_for_op)
        return [_group.batch_isend_irecv(op_type, tensors, remote_rank_list)]
    else:
        # Backward support for Gloo
        reqs = []
        for p2p_op in p2p_op_list:
            work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
            if work:
                reqs.append(work)
    return reqs


def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    Note:
        Npu doesn't support gather currently, replaced with all_gather.
    """

    _check_single_tensor(tensor, "tensor")

    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        gather_list = []
    _ensure_all_tensors_same_dtype(tensor, gather_list)

    if _rank_not_in_group(group):
        _warn_not_in_group("gather")
        return None
    my_rank = get_rank()

    _validate_output_list_for_rank(my_rank, dst, gather_list)
    group_size = _get_group_size(group)
    recv_size_list = [None for _ in range(group_size)] if my_rank != dst else \
        [tensor.size() for tensor in gather_list]

    input_tensors = [tensor]
    opts = GatherOptions()
    opts.rootRank = dst
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        backend_str = get_backend(default_pg)
        if 'hccl' in backend_str:
            if my_rank == dst:
                warnings.warn("HCCL doesn't support gather at the moment. Implemented with allgather instead.")
            # To handle tensors of different shape on each rank, update recv shape first.
            dist.broadcast_object_list(recv_size_list, dst, group)
            if not gather_list:
                gather_list = [torch.empty(tensor_size, dtype=tensor.dtype).npu() for tensor_size in recv_size_list]

            output_tensors = [gather_list]
            _group = default_pg._get_backend(torch.device("npu"))
            work = _group.allgather(output_tensors, input_tensors)
        else:
            output_tensors = [gather_list] if dst == my_rank else []
            default_pg = _get_default_group()
            work = default_pg.gather(output_tensors, input_tensors, opts)
    else:
        backend_str = get_backend(group)
        if 'hccl' in backend_str:
            if my_rank == dst:
                warnings.warn("HCCL doesn't support gather at the moment. Implemented with allgather instead.")
            # To handle tensors of different shape on each rank, update recv shape first.
            dist.broadcast_object_list(recv_size_list, dst, group)
            if not gather_list:
                gather_list = [torch.empty(tensor_size, dtype=tensor.dtype).npu() for tensor_size in recv_size_list]

            output_tensors = [gather_list]
            _group = group._get_backend(torch.device("npu"))
            work = _group.allgather(output_tensors, input_tensors)
        else:
            group_dst_rank = get_group_rank(group, dst)
            output_tensors = [gather_list] if dst == my_rank else []
            opts.rootRank = group_dst_rank
            work = group.gather(output_tensors, input_tensors, opts)
    if async_op:
        return work
    else:
        work.wait()
        return None


def gather_object(obj, object_gather_list=None, dst=0, group=None):
    """
    Note:
    Avoid gather_object to use gather func defined in origin distributed_c10d.
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("gather_object")
        return

    # Ensure object_gather_list is specified appropriately.
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    current_device = _get_pg_default_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=current_device
        )
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,  # type: ignore[possibly-undefined]
        dst=dst,
        group=group,
    )
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size, group)


def is_hccl_available():
    return "hccl" in Backend.backend_list


def _clear_pg_cache_in_torch(group: ProcessGroup):
    if _world.pg_map.get(group) is not None:
        del _world.pg_map[group]
    if _world.pg_names.get(group) is not None:
        del _world.pg_names[group]
    if _world.pg_group_ranks.get(group) is not None:
        del _world.pg_group_ranks[group]
    if _world.pg_backend_config.get(group) is not None:
        del _world.pg_backend_config[group]
    if _world.pg_to_tag.get(group) is not None:
        del _world.pg_to_tag[group]
    tags_list = [key for key, value in _world.tags_to_pg.items() if group in value]
    if len(tags_list) > 0:
        for tag in tags_list:
            del _world.tags_to_pg[tag]
    if _world.pg_default_device.get(group) is not None:
        del _world.pg_default_device[group]
    _unregister_process_group(group.group_name)


def _reinit_process_group(group: ProcessGroup):
    # Prepare parameters
    backend = dist_c10d.Backend(_world.pg_map[group][0])
    init_method = dist_c10d._default_pg_init_method if dist_c10d._default_pg_init_method else "env://"
    rank = group.rank()
    world_size = dist_c10d._get_group_size(group)
    device_id = None
    pg_options = None
    pg_tag = None
    timeout = default_pg_timeout
    group_name = _world.pg_names[group]
    # clear processgroup and map cache
    _world.default_pg = None
    _clear_pg_cache_in_torch(group)
    group._get_backend(torch.device('npu')).abort_hccl_comm("reinit")
    group = None
    # init store
    rendezvous_iterator = dist_c10d.rendezvous(
        init_method, rank, world_size, timeout=timeout
    )
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)
    store = dist_c10d.PrefixStore("default_pg", store)
    if store.num_keys() > 0:
        # "0" is a key for hccl masterID.
        # We need to refresh this value in reinit process.
        store.delete_key("0")
    # new group
    default_pg, _ = dist_c10d._new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name,
        pg_options,
        timeout,
        pg_tag,
        device_id
    )
    # update info
    dist_c10d._update_default_pg(default_pg)
    _world.pg_group_ranks[dist_c10d.GroupMember.WORLD] = {i: i for i in range(dist_c10d.GroupMember.WORLD.size())}
    dist_c10d._backend = _world.pg_map[dist_c10d.GroupMember.WORLD][0]
    dist_c10d._default_pg_init_method = init_method
    _world.pg_default_device[default_pg] = torch.device('npu')
    group = default_pg


def reinit_process_group(group=None, rebuild_link=True):
    if not rebuild_link:
        device_id = torch.npu.current_device()
        npu_device = torch.device('npu')
        for pg in _pg_map:
            if (npu_device in pg._device_types):
                pg._get_backend(npu_device).resume_hccl_comm(device_id)
    else:
        group = _world.default_pg
        _reinit_process_group(group)


def _destructor_process_group():
    _update_default_pg(None)
    _world.pg_map.clear()
    _world.pg_names.clear()
    _world.pg_group_ranks.clear()
    _world.pg_backend_config.clear()
    _world.pg_to_tag.clear()
    _world.tags_to_pg.clear()
    _world.pg_coalesce_state.clear()
    _world.pg_default_device.clear()
    _unregister_all_process_groups()
    _world.group_count = 0