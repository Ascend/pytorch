__all__ = ["is_hccl_available", "reinit_process_group"]

import os
from datetime import timedelta
from typing import Optional
from functools import wraps
import warnings
import logging
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.distributed_c10d import _get_default_group, get_group_rank, _check_single_tensor, \
    _check_tensor_list, _coalescing_manager, _ensure_all_tensors_same_dtype, get_rank, _rank_not_in_group, \
    _warn_not_in_group, GatherOptions, _validate_output_list_for_rank, GroupMember, _get_group_size, \
    _get_object_coll_device, _object_to_tensor, get_world_size, _tensor_to_object, all_gather, Backend, \
    get_backend, GatherOptions, _update_default_pg, _world, _unregister_all_process_groups, _pg_map, \
    ProcessGroup, default_pg_timeout, ReduceScatterOptions, _unregister_process_group, _check_valid_timeout, \
    _find_pg_by_ranks_and_tag, is_initialized, _get_split_source, BackendConfig, is_mpi_available, is_gloo_available, \
    is_nccl_available, is_ucc_available, is_xccl_available, get_debug_level, DebugLevel, _process_group_color, \
    _create_process_group_wrapper, _GLOO_AVAILABLE

from torch._C._distributed_c10d import PrefixStore, _register_process_group, _DistributedBackendOptions

from torch_npu.utils._error_code import ErrCode, dist_error

if is_mpi_available():
    from torch.distributed.distributed_c10d import ProcessGroupMPI
if is_nccl_available():
    from torch.distributed.distributed_c10d import ProcessGroupNCCL
if is_gloo_available():
    from torch._C._distributed_c10d import _ProcessGroupWrapper
    from torch.distributed.distributed_c10d import ProcessGroupGloo
if is_ucc_available():
    from torch.distributed.distributed_c10d import ProcessGroupUCC
if is_xccl_available():
    from torch.distributed.distributed_c10d import ProcessGroupXCCL



logger = logging.getLogger("torch.distributed")
origin_get_sequence_number_for_group = ProcessGroup._get_sequence_number_for_group


def _batch_isend_irecv(p2p_op_list):
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
            if p2p_op.tensor.device.type != "npu":
                deviceType = p2p_op.tensor.device.type
                raise RuntimeError(f"No backend type associated with device type {deviceType}" + dist_error(ErrCode.PARAM))
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


def _gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
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
        if tensor.device.type == 'npu':
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
        if tensor.device.type == 'npu':
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


def _gather_object(obj, object_gather_list=None, dst=0, group=None):
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
    current_device = _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0)
        for i in range(group_size)
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
    _gather(
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
    _unregister_process_group(group.group_name)


def reinit_process_group(group=None, rebuild_link=True):
    if group is None:
        group = _world.default_pg
    if not rebuild_link:
        device_id = torch.npu.current_device()
        npu_device = torch.device('npu')
        for pg in _pg_map:
            if (npu_device in pg._device_types):
                pg._get_backend(npu_device).resume_hccl_comm(device_id)
        return None
    else:
        backend = dist_c10d.Backend(_world.pg_map[group][0])
        if 'hccl' in backend:
            group._get_backend(torch.device('npu'))._delete_tcpstore_key()
            group._get_backend(torch.device('npu')).abort_hccl_comm("reinit")
        return group


def _comm_switch_nic(ranks, useBackup):
    nRanks = len(ranks)
    npu_device = torch.device('npu')
    rankid = int(os.environ['RANK'])
    result = True
    for pg in _pg_map:
        if (npu_device in pg._device_types):
            presult = pg._get_backend(npu_device)._set_switch_nic_comm(rankid, nRanks, ranks, useBackup)
            if not presult:
                result = False
    return result


def _reduce_scatter_tensor_uneven(output, input, input_split_sizes=None, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter_tensor_uneven")
        return None

    if output.device.type != 'npu' or input.device.type != 'npu': 
        warnings.warn("Support for Tensors is limited to those of type npu") 
        return None

    if group is None or group is GroupMember.WORLD:
        group = _get_default_group()
    group = group._get_backend(torch.device("npu"))

    opts = ReduceScatterOptions()
    opts.reduceOp = op
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes
    
    work = group.reduce_scatter_tensor_uneven(output, input, input_split_sizes, opts)

    if async_op:
        return work
    else:
        work.wait()
        return None


def _all_gather_into_tensor_uneven(output, input, output_split_sizes=None, group=None, async_op=False):
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_into_tensor_uneven")
        return None
    
    if output.device.type != 'npu' or input.device.type != 'npu':
        warnings.warn("Support for Tensors is limited to those of type npu") 
        return None
    
    if group is None or group is GroupMember.WORLD:
        group = _get_default_group()
    group = group._get_backend(torch.device("npu"))

    output_split_sizes = [] if output_split_sizes is None else output_split_sizes

    work = group.all_gather_into_tensor_uneven(output, input, output_split_sizes)

    if async_op:
        return work
    else:
        work.wait()
        return None


def _trigger__get_addr_and_port_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Only supports obtaining the master_addr and master_port through the endpoint when the backend is static.
        if len(args) > 0 and isinstance(args[0], RendezvousParameters) and args[0].backend == "parallel":
            args[0].backend = "static"
            master_addr, master_port = func(*args, **kwargs)
            args[0].backend = "parallel"
            return master_addr, master_port
        else:
            return func(*args, **kwargs)
    return wrapper


def _trigger_rendezvous_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        use_parallel = os.getenv("TORCH_NPU_USE_PARALLEL_TCPSTORE", "False")
        if use_parallel == "True":
            if len(args) > 0 and args[0] == "env://":
                master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
                master_port = os.getenv("MASTER_PORT", "29500")
                args = (f"parallel://{master_addr}:{master_port}",) + args[1:]
                logger.info(f"torch_npu_run change the rendezvous url from env:// to {args[0]}")
        return func(*args, **kwargs)
    return wrapper


def _destructor_process_group():
    _update_default_pg(None)
    _world.pg_map.clear()
    _world.pg_names.clear()
    _world.pg_group_ranks.clear()
    _world.pg_backend_config.clear()
    _world.pg_to_tag.clear()
    _world.tags_to_pg.clear()
    _world.pg_coalesce_state.clear()
    _unregister_all_process_groups()
    _world.group_count = 0


def _hccl_get_sequence_number_for_group(self):
    backend = torch.distributed.get_backend_config(self)
    if backend == "hccl" or backend == "npu:hccl":
        return self._get_backend(torch.device("npu"))._get_sequence_number_for_group()
    else:
        return origin_get_sequence_number_for_group(self)


def _patched_new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    backend_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    global _world

    if group_name in _world.pg_names.values():
        raise ValueError(
            "The specified group name has already been "
            "created, please use a different group name"
        )

    if device_id is not None and (device_id.index is None or device_id.type == "cpu"):
        raise ValueError(
            "init_process_group device_id parameter must be an accelerator with an index"
        )

    # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
    _check_valid_timeout(timeout)

    if pg_tag not in [None, ""]:
        # creating with the same tag and rank set results in the same underlying PG
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = _world.pg_map[existing_group]
            return existing_group, prefix_store

    group_desc = "undefined" if group_desc is None else group_desc

    # The list of group ranks is empty if we're creating the default group.
    is_default_group = len(global_ranks_in_group) == 0

    # nccl and potentially other backends allow creation of
    # communicators based on pre-existing ones, which can save
    # initialization time.  Due to lazy initialization of
    # communicators in some backends, we have to be careful and only
    # split when we *know* the default PG has already started communicator initialization.
    # We know this if we have bound a device id to the default pg (eager initialized).
    if is_initialized() and _get_default_group().bound_device_id:
        split_from = _get_split_source(_get_default_group())
    else:
        split_from = None

    # If this is a subgroup (which means group_ranks is specified),
    # we check if the current process is a member of the new group.
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            # If we are using `ncclCommSplit` (or similar split from
            # other APIs) to create the communicator, we will need to
            # call `ncclCommSplit` on *all* ranks in this new group's
            # parent group, even those not in the new group.  This is
            # a requirement of the NCCL API as otherwise we would get
            # out of sync.
            if split_from:
                split_from.perform_nocolor_split(_get_default_group().bound_device_id)
            return GroupMember.NON_GROUP_MEMBER, None

    prefix_store = PrefixStore(f"{group_name}/", store)
    # The backend for PG will be set later based on what's inside BackendConfig
    # and timeout are set in each backend's option.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    backend_config = BackendConfig(backend)
    # Set the default backend when single backend is passed in.
    if "," not in str(backend) and ":" not in str(backend):
        assert backend in Backend.backend_type_map, f"Unknown backend type {backend}"
        if backend == Backend.UNDEFINED:
            # Currently when backend is UNDEFINED, both ``gloo`` and ``nccl`` backends
            # will be created, we use nccl(if cuda is available) or gloo as default
            # backend so we can correctly call getDefaultBackend which in ProcessGroup.
            if Backend.NCCL in backend_config.get_device_backend_map().values():
                pg._set_default_backend(ProcessGroup.BackendType.NCCL)
            elif Backend.HCCL in backend_config.get_device_backend_map().values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
            else:
                pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        else:
            pg._set_default_backend(Backend.backend_type_map[backend])
    # In order to correctly call pg._has_hooks(), we should set the default backend
    # when multi backend is passed in
    else:
        if Backend.NCCL in backend_config.device_backend_map.values():
            pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        elif Backend._plugins.keys():
            custom_backend = next(iter(Backend._plugins.keys()))
            if custom_backend in backend_config.device_backend_map.values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        else:
            pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    if device_id:
        pg.bound_device_id = device_id
    backend_class: torch._C._distributed_c10d.Backend
    for device, backend_str in backend_config.get_device_backend_map().items():
        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError(
                    "Distributed package doesn't have MPI built in."
                    " MPI is only included if you build PyTorch from"
                    " source on a host that has MPI installed."
                )
            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER, None
            # create new process group with accurate rank and size
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(
                    backend_prefix_store,
                    backend_class.rank(),
                    backend_class.size(),
                )
                pg._set_default_backend(backend_type)
        elif backend_str == Backend.GLOO:
            # TODO: remove this check after lazy initialization is supported
            # if pg_options is not None:
            #     raise RuntimeError("GLOO options not supported")
            if not is_gloo_available():
                raise RuntimeError("Distributed package doesn't have Gloo built in")
            backend_class = ProcessGroupGloo(
                backend_prefix_store, group_rank, group_size, timeout=timeout
            )
            backend_class.options.global_ranks_in_group = global_ranks_in_group
            backend_class.options.group_name = group_name
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            if backend_options is not None:
                assert isinstance(backend_options, ProcessGroupNCCL.Options), (
                    "Expected backend_options argument to be of type ProcessGroupNCCL.Options"
                )
                if backend_options._timeout != timeout:
                    warnings.warn(
                        "backend_options._timeout was specified, "
                        "but timeout kwarg has a default value that will always override it. "
                    )
            else:
                # default backend_options for NCCL
                backend_options = ProcessGroupNCCL.Options()
                backend_options.is_high_priority_stream = False
            backend_options._timeout = timeout

            if split_from:
                backend_options.split_from = split_from
                backend_options.split_color = _process_group_color(
                    global_ranks_in_group
                )
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            backend_class = ProcessGroupNCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            # TODO: once UCC plugin is fully deprecated, remove
            # is_ucc_available() from above elif-condition and raise
            # RuntimeError if is_ucc_available() returns false.

            backend_class = ProcessGroupUCC(
                backend_prefix_store, group_rank, group_size, timeout=timeout
            )
            backend_type = ProcessGroup.BackendType.UCC
        elif backend_str == Backend.XCCL:
            if not is_xccl_available():
                raise RuntimeError("Distributed package doesn't have XCCL built in")
            backend_class = ProcessGroupXCCL(
                backend_prefix_store, group_rank, group_size
            )
            backend_type = ProcessGroup.BackendType.XCCL
        else:
            assert backend_str.upper() in Backend._plugins, (
                f"Unknown c10d backend type {backend_str.upper()}"
            )

            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM

            if not extended_api:
                backend_class = creator_fn(
                    backend_prefix_store, group_rank, group_size, timeout
                )
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                backend_class = creator_fn(dist_backend_opts, backend_options)

        # Set sequence numbers for gloo and nccl backends.
        if backend_str == Backend.GLOO:
            assert isinstance(backend_class, ProcessGroupGloo)
            backend_class._set_sequence_number_for_group()
        elif backend_str == Backend.NCCL:
            assert isinstance(backend_class, ProcessGroupNCCL)
            backend_class._set_sequence_number_for_group()

        # If the type is a subclass of ProcessGroup then return this process group immediately
        # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
        # ProcessGroup instance
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class  # type: ignore[assignment]
            break

        # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
        if (
            backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]
            or backend_str.upper() in Backend._plugins
        ):
            # In debug mode and if GLOO is available, wrap in a wrapper PG that
            # enables enhanced collective checking for debuggability.
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info(
                        """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                GLOO is not available. Build with Gloo to
                                create a wrapper process group in debug mode
                                to aid collective desynchronization debugging."""
                    )
                else:
                    backend_class = _create_process_group_wrapper(
                        wrapped_pg=backend_class,
                        store_prefix=group_name,
                        store=backend_prefix_store,
                        rank=group_rank,
                        world_size=group_size,
                        timeout=timeout,
                    )

        # register only a single backend when all get_device_backend_map values are the same
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map().keys():
                pg._register_backend(torch.device(device), backend_type, backend_class)

            # break out of outer loop to not create any more backends
            break

        pg._register_backend(torch.device(device), backend_type, backend_class)

    # set group_name and group_dsec to backend
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    if device_id and pg._get_backend(device_id).supports_splitting:
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)

    # update global state
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)

    _world.pg_backend_config[pg] = str(backend_config)
    # "" is the default tag for user PGs
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return pg, prefix_store

torch.distributed.distributed_c10d._new_process_group_helper = _patched_new_process_group_helper