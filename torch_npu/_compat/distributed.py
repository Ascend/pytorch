from torch_npu._compat.version import CURRENT_VERSION

__all__ = [
    "register_op_strategy",
    "_mm_like_strategy",
    "_add_ephemeral_timeout_for_all_pgs",
    "_new_process_group_helper",
]

# register_op_strategy moved from _ops.registration to _ops.utils in PyTorch 2.11.
from torch.distributed.tensor._ops.utils import register_op_strategy


# COMPAT(>= 2.14): upstream pytorch#186667 removed the helper
#   torch.distributed.tensor._ops._matrix_ops._mm_like_strategy as part of the
#   matrix_ops "single dim strategies" refactor. All three helpers the old
#   implementation depended on (gen_einsum_strategies, is_tensor_shardable,
#   generate_redistribute_costs) are still exported, so we replay the
#   pre-refactor body inline. This keeps torch_npu's custom_bmm_strategy
#   callsite (which still uses the old @register_op_strategy pipeline)
#   untouched, and does not require migrating to the new
#   register_single_dim_strategy interface.
# CAN REMOVE else branch when MIN_SUPPORTED >= (2, 14)
if CURRENT_VERSION >= (2, 14):
    from torch.distributed.tensor._op_schema import OpStrategy
    from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
    from torch.distributed.tensor._ops.utils import (
        is_tensor_shardable,
        generate_redistribute_costs,
    )

    def _mm_like_strategy(mm_equation, mesh, op_schema):
        self_strategy, mat2_strategy = op_schema.args_schema
        if not isinstance(self_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(self_strategy)}")
        if not isinstance(mat2_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(mat2_strategy)}")
        mm_strategy = gen_einsum_strategies(mm_equation, mesh)
        filtered_strategies = []
        for strtg in mm_strategy.strategies:
            if strtg.input_specs is None:
                raise AssertionError(
                    f"Expected input_specs to be not None, got {strtg.input_specs}"
                )
            self_spec = strtg.input_specs[0]
            mat2_spec = strtg.input_specs[1]
            if is_tensor_shardable(
                self_strategy.shape, self_spec, allow_unbacked_sharding=True
            ) and is_tensor_shardable(
                mat2_strategy.shape, mat2_spec, allow_unbacked_sharding=True
            ):
                strtg.redistribute_cost = [
                    generate_redistribute_costs(self_strategy, self_spec),
                    generate_redistribute_costs(mat2_strategy, mat2_spec),
                ]
                filtered_strategies.append(strtg)
        mm_strategy.strategies = filtered_strategies
        return mm_strategy
else:
    from torch.distributed.tensor._ops._matrix_ops import _mm_like_strategy


# COMPAT(< 2026-08-04 torch nightly): the upstream
#   ``_add_ephemeral_timeout_for_all_pgs`` only became backend-generic
#   (dispatching through ``ProcessGroup::addEphemeralTimeout``, NPU included)
#   on the torch nightly of 2026-08-04 (pytorch#191980). On torch 2.13 and
#   earlier 2.14 nightlies it is CUDA/NCCL-only and a no-op on NPU, so the
#   torch_npu implementation must be used there.
# CAN REMOVE else branch when MIN_SUPPORTED >= the 2026-08-04 nightly.
if CURRENT_VERSION >= (2, 14):
    import inspect

    from torch.distributed import distributed_c10d as c10d

    _UPSTREAM_EPHEMERAL_TIMEOUT_IS_BACKEND_GENERIC = (
        "pg._add_ephemeral_timeout" in inspect.getsource(
            c10d._add_ephemeral_timeout_for_all_pgs)
    )
else:
    _UPSTREAM_EPHEMERAL_TIMEOUT_IS_BACKEND_GENERIC = False

if _UPSTREAM_EPHEMERAL_TIMEOUT_IS_BACKEND_GENERIC:
    from torch.distributed.distributed_c10d import _add_ephemeral_timeout_for_all_pgs
else:
    from torch.distributed import distributed_c10d as c10d
    from datetime import timedelta

    def _add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
        """
        This API adds an ephemeral timeout extension for all PGs locally
        on one rank. The timeout gets reset when the first collective issued
        after API called finished.
        NOTE: We only support to set timeout for hccl backends for now.
        NOTE: While this feature provides flexibility in specific scenarios,
        it introduces statefulness
        to timeout setting. Therefore, it is advisable to use this API sparingly
        and consider alternative approaches, such as directly setting the timeout
        or utilizing a barrier collective (one can set any timeout to the barrier),
        whenever feasible.

        Args:
            timeout (timedelta): The delta of timeout to extend.

        Returns:
            None.
        """
        import torch

        from torch_npu.distributed.distributed_c10d import is_hccl_available

        if not is_hccl_available():
            return

        try:
            from torch_npu._C._distributed_c10d import ProcessGroupHCCL
        except ImportError:
            return

        for pg in c10d._world.pg_map:
            devices = pg._device_types
            if torch.device("npu") in devices:
                backend = pg._get_backend(torch.device("npu"))
                if isinstance(backend, ProcessGroupHCCL) and hasattr(
                    backend, "_add_ephemeral_timeout"
                ):
                    backend._add_ephemeral_timeout(timeout)
    c10d._add_ephemeral_timeout_for_all_pgs = _add_ephemeral_timeout_for_all_pgs


# COMPAT(< 2.14): upstream _new_process_group_helper only resolves an NPU
#   "undefined" backend natively after pytorch#179901 (2.14.0.dev mid-July
#   2026 nightly). torch_npu used to replace the helper with a patched copy
#   adding an "HCCL -> BackendType.CUSTOM" branch; restore it on old torch so
#   init_process_group() without an explicit backend keeps working on NPU.
# CAN REMOVE when MIN_SUPPORTED_VERSION >= (2, 14) and every 2.14 nightly
#   includes pytorch#179901: then _upstream_supports_npu_default_backend()
#   always returns True and this whole block is dead code.
def _upstream_supports_npu_default_backend() -> bool:
    """Whether upstream _new_process_group_helper natively resolves an NPU
    "undefined" backend to BackendType.CUSTOM.

    torch_npu registers hccl via Backend.register_backend(..., devices=["npu"]),
    so upstream resolves it natively only when the UNDEFINED branch scans for
    CUSTOM-type backends. Older torch (2.13.x, early 2.14 nightlies) falls
    back to GLOO, which loses the hccl default backend on NPU and needs the
    patch below.
    """
    import inspect
    import torch.distributed.distributed_c10d as dist_c10d

    src = inspect.getsource(dist_c10d._new_process_group_helper)
    return (
        "_get_default_backend_type_for_backend_config" in src
        or "backend_type_map.get(str(backend))" in src
    )

if CURRENT_VERSION < (2, 14) or not _upstream_supports_npu_default_backend():
    # Only old torch (2.13.x, early 2.14 nightlies) reaches here; load the
    # imports and the patched helper lazily so supported torch versions pay
    # no cost and do not expose the patch.
    import logging
    import warnings

    import torch
    import torch.distributed.distributed_c10d as dist_c10d
    from torch._C._distributed_c10d import _DistributedBackendOptions
    from torch.distributed.distributed_c10d import (
        Backend,
        BackendConfig,
        DebugLevel,
        GroupMember,
        PrefixStore,
        ProcessGroup,
        _GLOO_AVAILABLE,
        _check_valid_timeout,
        _create_process_group_wrapper,
        _find_pg_by_ranks_and_tag,
        _get_default_group,
        _get_split_source,
        _process_group_color,
        _register_process_group,
        get_debug_level,
        is_gloo_available,
        is_initialized,
        is_mpi_available,
        is_nccl_available,
        is_ucc_available,
        is_xccl_available,
    )

    if is_mpi_available():
        from torch.distributed.distributed_c10d import ProcessGroupMPI
    if is_nccl_available():
        from torch.distributed.distributed_c10d import ProcessGroupNCCL
    if is_gloo_available():
        from torch.distributed.distributed_c10d import ProcessGroupGloo
    if is_ucc_available():
        from torch.distributed.distributed_c10d import ProcessGroupUCC
    if is_xccl_available():
        from torch.distributed.distributed_c10d import ProcessGroupXCCL

    logger = logging.getLogger("torch.distributed")


    def _new_process_group_helper(
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
        enable_reconfigure=False,  # noqa: ARG001  accepted for 2.14+ init_process_group signature; HCCL does not consume it
    ):
        """
        Create a new distributed process group.

        This function must be called by ALL processes in the global group, even if
        the calling process is not part of the newly created group. In that case,
        this function returns GroupMember.NON_GROUP_MEMBER.

        This function is called with ``global_ranks_in_group == []`` for the default group.
        """

        if group_name in dist_c10d._world.pg_names.values():
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
                _, prefix_store = dist_c10d._world.pg_map[existing_group]
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
            if backend not in Backend.backend_type_map:
                raise AssertionError(f"Unknown backend type {backend}")
            if backend == Backend.UNDEFINED:
                # Currently when backend is UNDEFINED, only one backend will be initialized
                # we use nccl (if cuda is available) or gloo as default backend
                # so we can correctly call getDefaultBackend which in ProcessGroup.
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
                    backend_prefix_store,
                    group_rank,
                    group_size,
                    # pyrefly: ignore [bad-argument-type]
                    timeout=timeout,
                )
                backend_class.options.global_ranks_in_group = global_ranks_in_group
                backend_class.options.group_name = group_name
                backend_type = ProcessGroup.BackendType.GLOO
            elif backend_str == Backend.NCCL:
                if not is_nccl_available():
                    raise RuntimeError("Distributed package doesn't have NCCL built in")
                if backend_options is not None:
                    if not isinstance(backend_options, ProcessGroupNCCL.Options):
                        raise AssertionError(
                            "Expected backend_options argument to be of type ProcessGroupNCCL.Options"
                        )
                    if backend_options._timeout != timeout:
                        warnings.warn(
                            "backend_options._timeout was specified, "
                            "but timeout kwarg has a default value that will always override it. ",
                            stacklevel=2,
                        )
                else:
                    # default backend_options for NCCL
                    backend_options = ProcessGroupNCCL.Options()
                    backend_options.is_high_priority_stream = False
                # pyrefly: ignore [bad-argument-type]
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
                    backend_prefix_store,
                    group_rank,
                    group_size,
                    # pyrefly: ignore [bad-argument-type]
                    timeout=timeout,
                )
                backend_type = ProcessGroup.BackendType.UCC
            elif backend_str == Backend.XCCL:
                if not is_xccl_available():
                    raise RuntimeError("Distributed package doesn't have XCCL built in")
                backend_options = ProcessGroupXCCL.Options()
                backend_options.global_ranks_in_group = global_ranks_in_group
                backend_options.group_name = group_name
                # pyrefly: ignore [bad-argument-type]
                backend_options._timeout = timeout
                backend_class = ProcessGroupXCCL(
                    backend_prefix_store, group_rank, group_size, backend_options
                )
                backend_type = ProcessGroup.BackendType.XCCL
            else:
                if backend_str.upper() not in Backend._plugins:
                    raise AssertionError(f"Unknown c10d backend type {backend_str.upper()}")

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
                    # pyrefly: ignore [bad-argument-type]
                    dist_backend_opts.timeout = timeout
                    dist_backend_opts.group_id = group_name
                    dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                    backend_class = creator_fn(dist_backend_opts, backend_options)

            # Set sequence numbers for gloo and nccl backends.
            if backend_str == Backend.GLOO:
                if not isinstance(backend_class, ProcessGroupGloo):
                    raise AssertionError(
                        f"Expected ProcessGroupGloo, got {type(backend_class)}"
                    )
                backend_class._set_sequence_number_for_group()
            elif backend_str == Backend.NCCL:
                if not isinstance(backend_class, ProcessGroupNCCL):
                    raise AssertionError(
                        f"Expected ProcessGroupNCCL, got {type(backend_class)}"
                    )
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
                            # pyrefly: ignore [bad-argument-type]
                            timeout=timeout,
                        )

            # register only a single backend when all get_device_backend_map values are the same
            if len(set(backend_config.get_device_backend_map().values())) == 1:
                for device in backend_config.get_device_backend_map():
                    pg._register_backend(torch.device(device), backend_type, backend_class)

                # break out of outer loop to not create any more backends
                break

            pg._register_backend(torch.device(device), backend_type, backend_class)

        # set group_name and group_dsec to backend
        if group_name is None:
            raise AssertionError("group_name must not be None")
        if group_desc is None:
            raise AssertionError("group_desc must not be None")
        pg._set_group_name(group_name)
        pg._set_group_desc(group_desc)

        if device_id and pg._get_backend(device_id).supports_splitting:
            eager_backend = pg._get_backend(device_id)
            eager_backend.eager_connect_single_device(device_id)

        # update global state
        dist_c10d._world.pg_map[pg] = (backend, prefix_store)
        dist_c10d._world.pg_names[pg] = group_name
        _register_process_group(group_name, pg)

        dist_c10d._world.pg_backend_config[pg] = str(backend_config)
        # "" is the default tag for user PGs
        if pg_tag in [None, ""]:
            pg_tag = f"ptd:{group_name}"
            dist_c10d._world.tags_to_pg.setdefault("", []).append(pg)
        else:
            pg_tag = f"user:{pg_tag}"

        dist_c10d._world.tags_to_pg.setdefault(pg_tag, []).append(pg)
        dist_c10d._world.pg_to_tag[pg] = pg_tag
        return pg, prefix_store

    dist_c10d._new_process_group_helper = _new_process_group_helper
else:
    # Upstream resolves it natively: alias the upstream helper.
    from torch.distributed.distributed_c10d import _new_process_group_helper


# COMPAT(< 2.14): upstream ShardedTensor.cuda()/to() are CUDA-hardcoded on
#   torch < 2.14 (torch.cuda.current_device(), shard.tensor.cuda(), device
#   allowlist {"cuda", "xpu"}), so torch_npu patches a npu() method onto
#   ShardedTensor. Upstream pytorch#187939 (shipped in 2.14) makes cuda()/to()
#   hardware-agnostic via torch.accelerator, so the patch is not needed and
#   must not be applied on torch >= 2.14.
# CAN REMOVE this block when MIN_SUPPORTED >= (2, 14)
if CURRENT_VERSION < (2, 14):
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch_npu.distributed.tensor._sharded_tensor_patch import _patched_sharded_tensor_npu

    # Add the patched npu() method if it doesn't exist.
    if not hasattr(ShardedTensor, "npu"):
        ShardedTensor.npu = _patched_sharded_tensor_npu
