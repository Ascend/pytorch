import os
import stat
import queue
import threading
import collections
from typing import Dict, List, Sequence, cast
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp

from torch.distributed.checkpoint.optimizer import (
    _get_state_dict_2d_layout,
    _create_colwise_spec,
    _ReaderWithOffset
)

from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import (
    ChunkShardingSpec
)

from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    STATE_DICT_TYPE,
)

from torch.distributed.remote_device import _remote_device
from torch.distributed._shard.api import _shard_tensor
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.utils import (
    _normalize_device_info
)

from torch._utils import _get_device_module
from torch.futures import Future
from torch.distributed.checkpoint.storage import (
    WriteResult,
)

from torch.distributed.checkpoint.planner import (
    SavePlan,
    SavePlanner,
    WriteItemType,
)

from torch.distributed.checkpoint.filesystem import (
    _TensorLoader,
    _SerialCpuLoader,
    _StoragePrefix,
    _OverlappingCpuLoader,
    _item_size,
    _write_item,
    _split_by_size_and_type
)

DEFAULT_SUFFIX = ".distcp"


def _dcp_loader_init(self, resolve_fun, stream=None, inflight_threshhold=1_000_000):
    self.resolve_fun = resolve_fun
    self.items = []
    self.inflight_threshhold = inflight_threshhold
    self.in_flight_data = 0
    self.current_items: collections.deque = collections.deque()
    self.idx = 0
    self.started = False
    self.device_type = stream.device_type if stream else torch.device("npu").type
    self.device_module = _get_device_module(self.device_type)
    self.stream = stream or self.device_module.current_stream()
    if self.stream != self.device_module.current_stream():
        self.stream.wait_stream(self.device_module.current_stream())


def _alloc_tensor(props: TensorProperties, size: Sequence[int], device_type: str = "npu") -> torch.Tensor:
    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device="npu"
    )


def _load_sharded_optimizer_state_dict(
    model_state_dict: STATE_DICT_TYPE,
    optimizer_key: str,
    storage_reader: dist_cp.StorageReader,
) -> STATE_DICT_TYPE:
    
    metadata = storage_reader.read_metadata()

    layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
    dp_pg_device_type = dist.distributed_c10d._get_pg_default_device(dp_pg).type
    device_module = _get_device_module(dp_pg_device_type)

    if dp_pg is None:
        placements = []
        for i in range(dist.get_world_size()):
            device_info = _normalize_device_info(dp_pg_device_type, i % device_module.device_count())
            placements.append(f"rank:{i}/{device_info}")
        sharding_spec = ChunkShardingSpec(dim=0, placements=placements)  # type: ignore[arg-type]
    else:
        sharding_spec = _create_colwise_spec(dp_pg)

    # Create a state_dict for optimizer state
    state_dict: STATE_DICT_TYPE = {}

    fqn_to_offset: Dict[str, Sequence[int]] = {}
    for key, value in metadata.state_dict_metadata.items():
        key_path = metadata.planner_data[key]
        if key_path[0] != optimizer_key:
            continue

        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
            continue

        # value: TensorStorageMetadata
        if value.size.numel() == 1:
            state_dict[key] = _alloc_tensor(value.properties, value.size, dp_pg_device_type)
        elif dp_pg is None:
            state_dict[key] = _shard_tensor(
                _alloc_tensor(value.properties, value.size, dp_pg_device_type), sharding_spec
            )
        else:
            spec_key = key_path[2]
            alloc_size = layout_specs.get(spec_key, (None, value.size))[1]

            st_md = sharding_spec.build_metadata(
                torch.Size(alloc_size), value.properties
            )
            local_shards = []
            current_rank = dist.get_rank(dp_pg)
            for shard_md in st_md.shards_metadata:
                if (
                    cast(_remote_device, shard_md.placement).rank()
                    != current_rank
                ):
                    continue
                local_shards.append(
                    Shard(
                        tensor=_alloc_tensor(
                            value.properties, shard_md.shard_sizes, dp_pg_device_type
                        ),
                        metadata=shard_md,
                    )
                )

            st = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

            if (
                spec_key in layout_specs
                and layout_specs[spec_key][0] is not None
            ):
                fqn_to_offset[key] = cast(
                    Sequence[int], layout_specs[spec_key][0]
                )

            state_dict[key] = st

    # Whether we unflatten before or after doesn't matter
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else None,
    )

    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

    return state_dict


def _write_files_from_queue(
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
):
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            if torch.npu.is_available() and inflight_threshhold > 0:
                loader = _OverlappingCpuLoader(
                    lambda x: planner.resolve_data(x),
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    lambda x: planner.resolve_data(x),
                )

            tensor_w = [
                wi for wi in write_items if wi.type != WriteItemType.BYTE_IO
            ]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [
                wi for wi in write_items if wi.type == WriteItemType.BYTE_IO
            ]
            write_results = []

            with os.fdopen(os.open(file_name, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR), "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(stream, data, write_item, storage_key)
                    )

                for tensor, write_item in loader.values():
                    torch._check(
                        tensor.is_cpu,
                        lambda: "tensor must be cpu tensor ",
                    )
                    write_results.append(
                        _write_item(stream, tensor, write_item, storage_key)
                    )

                if use_fsync:
                    os.fsync(stream.fileno())
            result_queue.put(write_results)
    except queue.Empty:
        pass


def _write_data(
    self,
    plan: SavePlan,
    planner: SavePlanner,
) -> Future[List[WriteResult]]:
    storage_plan: _StoragePrefix = plan.storage_data
    file_count = 0

    def gen_file():
        nonlocal file_count
        file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
        file_count += 1
        return file_name

    file_queue: queue.Queue = queue.Queue()
    if self.single_file_per_rank:
        for bucket in _split_by_size_and_type(
            self.thread_count, plan.items
        ):
            file_name = gen_file()
            file_queue.put((self.path / file_name, file_name, bucket))
    else:
        for item in plan.items:
            file_name = gen_file()
            file_queue.put((self.path / file_name, file_name, [item]))

    result_queue: queue.Queue = queue.Queue()

    threads = []
    for _ in range(1, self.thread_count):
        t = threading.Thread(
            target=_write_files_from_queue,
            args=(
                file_queue,
                result_queue,
                planner,
                self.per_thread_copy_ahead,
                self.sync_files,
            ),
        )
        t.start()
        threads.append(t)

    _write_files_from_queue(
        file_queue=file_queue,
        result_queue=result_queue,
        planner=planner,
        inflight_threshhold=self.per_thread_copy_ahead,
        use_fsync=self.sync_files,
    )

    for t in threads:
        t.join()

    res = []
    try:
        while True:
            res += result_queue.get_nowait()
    except queue.Empty:
        pass

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(res)
        return fut


def _apply_dcp_patch():
    _OverlappingCpuLoader.__init__ = _dcp_loader_init
    torch.distributed.checkpoint.optimizer.load_sharded_optimizer_state_dict = _load_sharded_optimizer_state_dict
    torch.distributed.checkpoint.FileSystemWriter.write_data = _write_data
