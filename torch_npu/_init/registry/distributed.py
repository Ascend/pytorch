import torch


def _new_process_group_hccl_helper(dist_backend_opts, pg_options):
    import torch_npu

    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    if pg_options is None or not isinstance(
        pg_options, torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options
    ):
        pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    pg_options.is_high_priority_stream = False
    pg_options._timeout = dist_backend_opts.timeout
    pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    pg_options.group_id = dist_backend_opts.group_id
    return torch_npu._C._distributed_c10d.ProcessGroupHCCL(
        store, group_rank, group_size, pg_options
    )


def _new_process_group_lccl_helper(dist_backend_opts, pg_options):
    import torch_npu

    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    return torch_npu._C._distributed_c10d.ProcessGroupLCCL(
        store, group_rank, group_size
    )


def register_distributed_backend_for_npu():
    # init and register hccl backend
    # Note: Since torch 2.8, the hccl backend must be registered at first to keep a right default_device_backend_map
    torch.distributed.Backend.register_backend(
        "hccl",
        lambda dist_backend_opts, pg_options: _new_process_group_hccl_helper(
            dist_backend_opts, pg_options
        ),
        extended_api=True,
        devices=["npu"],
    )

    # init and register lccl backend
    torch.distributed.Backend.register_backend(
        "lccl",
        lambda dist_backend_opts, pg_options: _new_process_group_lccl_helper(
            dist_backend_opts, pg_options
        ),
        extended_api=True,
        devices=["npu"],
    )
