__all__ = []

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch._C import _get_privateuse1_backend_name

from torch.distributed.rpc import api
from torch.distributed.rpc import constants as rpc_constants

import torch_npu._C
from torch_npu.utils._error_code import ErrCode, dist_error


def _get_device_count_info():
    # Function used to replace torch.cuda.device_count in torch_npu
    device_count = dict()
    # Whether a third-party device is registered, and if so,
    # the information of the third-party device is also stored in device_count dictionary
    custom_backend_name = _get_privateuse1_backend_name()
    if hasattr(torch, custom_backend_name):
        custom_device_count_func = torch.utils.backend_registration._get_custom_mod_func("device_count")
        custom_device_count = custom_device_count_func() if custom_device_count_func else 0
        device_count[custom_backend_name] = custom_device_count
    return device_count


def _init_device_state(custom_backend_name):
    # Function used to replace torch.cuda.init in torch_npu
    if getattr(torch, custom_backend_name).is_available():
        getattr(torch, custom_backend_name).init()


def _tensorpipe_validate_devices(devices, device_count):
    return all(
        d.type == "cpu" or (0 <= d.index < device_count.get(d.type, 0))
        for d in devices
    )


def _validate_device_maps(
    all_names, all_device_counts, all_device_maps, all_devices, is_static_group=True
):
    for node in all_names:
        devices = all_devices[node]
        if len(set(devices)) != len(devices):
            raise ValueError(
                f"Node {node} has duplicated devices\n"
                f"devices = {devices}" + dist_error(ErrCode.VALUE)
            )
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(
                f"Node {node} has devices with invalid indices\n"
                f"devices = {devices}\n"
                f"device count = {all_device_counts[node]}" + dist_error(ErrCode.VALUE)
            )

    for source_node in all_names:
        # For dynamic group (non-static) do not check the target node name since it may not have joined yet
        if is_static_group and not set(all_device_maps[source_node].keys()).issubset(all_names):
            raise ValueError(
                f"Node {source_node} has invalid target node names in its device maps\n"
                f"device maps = {all_device_maps[source_node].keys()}\n"
                f"node names = {all_names}" + dist_error(ErrCode.VALUE)
            )
        for target_node, map_ in all_device_maps[source_node].items():
            if len(set(map_.values())) != len(map_):
                raise ValueError(
                    f"Node {source_node} has duplicated target devices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}" + dist_error(ErrCode.VALUE)
                )
            if all_devices[source_node]:
                if not set(map_.keys()).issubset(all_devices[source_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected source devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[source_node]}" + dist_error(ErrCode.VALUE)
                    )
            elif not _tensorpipe_validate_devices(
                map_.keys(), all_device_counts[source_node]
            ):
                raise ValueError(
                    f"Node {source_node} has source devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[source_node]}" + dist_error(ErrCode.VALUE)
                )
            if all_devices.get(target_node, []):
                if not set(map_.values()).issubset(all_devices[target_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected target devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[target_node]}" + dist_error(ErrCode.VALUE)
                    )
            elif target_node in all_device_counts and not _tensorpipe_validate_devices(
                map_.values(), all_device_counts[target_node]
            ):
                raise ValueError(
                    f"Node {source_node} has target devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[target_node]}" + dist_error(ErrCode.VALUE)
                )


def _get_device_infos():
    from torch_npu._C._distributed_rpc import TensorPipeAgent
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    opts = agent._get_backend_options()
    device_count = _get_device_count_info()
    if opts.devices:
        _init_device_state(opts.devices[0].type)
    return device_count, opts.device_maps, opts.devices


def _tensorpipe_exchange_and_check_all_device_maps(
    my_name, my_device_count, my_device_maps, my_devices, group
):
    gathered: List[Tuple[
        str, int, Dict[str, Dict[torch.device, torch.device]], List[torch.device]
    ]] = [("", 0, {}, []) for _ in range(group.size())]
    dist.all_gather_object(
        gathered, (my_name, my_device_count, my_device_maps, my_devices), group
    )
    all_names = [name for name, _, _, _ in gathered]
    all_device_counts = {name: count for name, count, _, _ in gathered}
    all_device_maps = {name: map_ for name, _, map_, _ in gathered}
    all_devices = {name: devices for name, _, _, devices in gathered}

    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)

    # passed all checked, construct reverse mapping and get list of devices handled by this agent
    reverse_device_maps = rpc.backend_registry._create_reverse_mapping(my_name, all_names, all_device_maps)
    my_devices = rpc.backend_registry._create_device_list(my_devices, my_device_maps, reverse_device_maps)
    return reverse_device_maps, my_devices


def _set_devices_and_reverse_device_map(agent):
    from torch_npu._C._distributed_rpc import TensorPipeAgent
    agent = cast(TensorPipeAgent, agent)
    # Group state is retrieved from local agent
    # On initialization, tensorpipe agent retrieves information from all existing workers, so group state is valid
    my_worker_info = agent.get_worker_info()
    my_name = my_worker_info.name
    all_worker_infos = agent.get_worker_infos()
    # One round to get device_maps of all workers and construct reverse device maps
    all_device_counts, all_device_maps, all_devices, all_names = {}, {}, {}, []
    for worker_info in all_worker_infos:
        worker_name = worker_info.name
        if worker_name != my_name:
            device_count, device_map, devices = api.rpc_sync(worker_name, _get_device_infos)
        else:
            opts = agent._get_backend_options()
            device_map, devices = opts.device_maps, opts.devices
            device_count = _get_device_count_info()
        all_device_counts[worker_name] = device_count
        all_device_maps[worker_name] = device_map
        all_devices[worker_name] = devices
        all_names.append(worker_name)

    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices, is_static_group=False)
    reverse_device_maps = rpc.backend_registry._create_reverse_mapping(my_name, all_names, all_device_maps)

    # Perform RPC call to all workers, including itself, to include newly joined worker information and device maps
    for worker_name in all_names:
        # Set device list for each worker
        all_devices[worker_name] = rpc.backend_registry._create_device_list(all_devices[worker_name], all_device_maps[worker_name], reverse_device_maps)
        api.rpc_sync(worker_name, _update_group_membership,
                     args=(my_worker_info, all_devices[worker_name], reverse_device_maps, True))


def _backend_type_repr(self):
    return "BackendType." + self.name


def _construct_rpc_backend_options(
    backend,
    rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
    init_method=rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs
):

    return backend.value.construct_rpc_backend_options_handler(
        rpc_timeout, init_method, **kwargs
    )


def _init_backend(backend, *args, **kwargs):
    return backend.value.init_backend_handler(*args, **kwargs)


# Backend Option Handler
def _npu_tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads=rpc.constants.DEFAULT_NUM_WORKER_THREADS,
    _transports=None,
    _channels=None,
    **kwargs
):
    from .options import NPUTensorPipeRpcBackendOptions

    return NPUTensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_worker_threads=num_worker_threads,
        _transports=_transports,
        _channels=_channels,
    )


# Backend Handler
def _npu_tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from torch_npu._C._distributed_rpc import TensorPipeAgent
    from .options import NPUTensorPipeRpcBackendOptions

    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}" + dist_error(ErrCode.TYPE))

    if not isinstance(
        rpc_backend_options, NPUTensorPipeRpcBackendOptions
    ):
        raise TypeError(
            f"`rpc_backend_options` must be a `NPUTensorPipeRpcBackendOptions`. {rpc_backend_options}" +
            dist_error(ErrCode.TYPE)
        )

    device_count = _get_device_count_info()

    is_static_group = True if world_size else False
    if is_static_group:
        group = rpc.backend_registry._init_process_group(store, rank, world_size)

        reverse_device_maps, devices = _tensorpipe_exchange_and_check_all_device_maps(
            name,
            device_count,
            rpc_backend_options.device_maps,
            rpc_backend_options.devices,
            group,
        )

        if devices:
            _init_device_state(devices[0].type)

        agent = TensorPipeAgent(
            store,
            name,
            rank,
            world_size,
            rpc_backend_options,
            reverse_device_maps,
            devices,
        )

        api._init_rpc_states(agent)
        api._all_gather(None, timeout=rpc_backend_options.rpc_timeout)
        group.barrier().wait()

        return agent
    else:
        with _group_membership_management(store, name, True):
            agent = TensorPipeAgent(
                store,
                name,
                rank,
                world_size,
                rpc_backend_options,
                {},
                [],
            )
            api._init_rpc_states(agent)

            try:
                _set_devices_and_reverse_device_map(agent)
                pass
            except Exception as e:
                api.shutdown()
                e.msg += dist_error(ErrCode.INTERNAL)
                raise
            return agent


def _rpc_backend_registry():
    if hasattr(torch_npu._C, "_rpc_npu_init"):
        torch_npu._C._rpc_npu_init()
        rpc.backend_registry.register_backend(
            "NPU_TENSORPIPE",
            _npu_tensorpipe_construct_rpc_backend_options_handler,
            _npu_tensorpipe_init_backend_handler,
        )
