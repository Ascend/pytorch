"""Test Gather receive views, AllGather results, and NPU allocations.

Check that Gather allocates one shared receive buffer and that AllGather
allocates its communication buffer without padding each output tensor.
"""

from datetime import timedelta
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch_npu.distributed.distributed_c10d import _create_coalesced_output_tensors


def _reference(rank, shape, dtype):
    values = torch.arange(torch.Size(shape).numel()).reshape(shape)
    if dtype == torch.bool:
        return (values + rank) % 3 == 0
    return (values % 23 + rank * 32).to(dtype)


def _check_gather(rank, group, peers, dtype, shape, async_op):
    dst = peers[-1]
    source = _reference(rank, shape, dtype).npu()
    outputs = None
    if rank == dst:
        outputs = _create_coalesced_output_tensors([shape] * len(peers), dtype, source.device)
    work = dist.gather(source, gather_list=outputs, dst=dst, group=group, async_op=async_op)
    if async_op and rank in peers:
        pressure = torch.empty(max(1, len(peers) * source.numel()), dtype=dtype, device=source.device)
        pressure.fill_(1)
        work.wait()
    if rank == dst:
        for peer, output in zip(peers, outputs):
            torch.testing.assert_close(output.cpu(), _reference(peer, shape, dtype), rtol=0, atol=0)


def _check_allgather(rank, group, peers, dtype):
    shape = (3, 7)
    source = _reference(rank, shape, dtype).npu()
    outputs = [torch.empty((7, 3), dtype=dtype, device=source.device).t() for _ in peers]
    dist.all_gather(outputs, source, group=group, async_op=True).wait()
    for peer, output in zip(peers, outputs):
        torch.testing.assert_close(output.cpu(), _reference(peer, shape, dtype), rtol=0, atol=0)


def _trace(fn):
    torch.npu.synchronize()
    torch.npu.memory._record_memory_history(enabled="all", context="all", stacks="all", max_entries=50000)
    try:
        output = fn()
        torch.npu.synchronize()
        snapshot = torch.npu.memory._snapshot()
    finally:
        torch.npu.memory._record_memory_history(enabled=None)
    events = snapshot["device_traces"][torch.npu.current_device()]
    assert events and len(events) < 49000, "empty or truncated allocator trace"
    return output, [event["size"] for event in events if event["action"] == "alloc"]


def _check_allocations(rank, world_size):
    # Trace Python output allocation and the C++ AllGather call separately.
    # Their buffers can have the same size on A5.
    size = 536930
    outputs, sizes = _trace(lambda: _create_coalesced_output_tensors(
        [(size,)] * world_size, torch.uint8, torch.device("npu")))
    assert sizes == [world_size * size], sizes
    del outputs
    source = torch.full((size,), rank, dtype=torch.uint8, device="npu")
    outputs = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(outputs, source)
    _, sizes = _trace(lambda: dist.all_gather(outputs, source))
    no_padding = torch_npu._C._npu_get_soc_version() >= 260  # SocVersion::Ascend950
    aligned = size if no_padding else size + (-size % 512)
    expected = ([] if no_padding else [aligned]) + [world_size * aligned]
    assert sorted(sizes) == sorted(expected), (sizes, expected)
    for peer, output in enumerate(outputs):
        torch.testing.assert_close(output.cpu(), torch.full((size,), peer, dtype=torch.uint8), rtol=0, atol=0)


def _worker(rank, world_size, port, allocation_only):
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
                       "LOCAL_WORLD_SIZE": str(world_size), "GROUP_RANK": "0"})
    torch.npu.set_device(rank)
    torch.npu.use_compatible_impl(False)
    dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=180))
    try:
        if allocation_only:
            _check_allocations(rank, world_size)
        else:
            peers = list(range(world_size))
            for dtype in (torch.uint8, torch.float16, torch.float32, torch.int64, torch.bool):
                for shape in ((513,), (512,), (1,), (), (0,)):
                    for async_op in (False, True):
                        _check_gather(rank, None, peers, dtype, shape, async_op)
                _check_allgather(rank, None, peers, dtype)
            peers = [1, world_size - 1] if world_size > 2 else [1]
            group = dist.new_group(peers)
            if rank in peers:
                _check_gather(rank, group, peers, torch.uint8, (513,), True)
                _check_allgather(rank, group, peers, torch.float32)
                dist.destroy_process_group(group)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class HcclCoalescedGatherTest(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU unavailable")

    def _spawn(self, allocation_only):
        world_size = int(os.environ.get("ALL_GATHER_TEST_WORLD_SIZE", "2"))
        self.assertGreaterEqual(world_size, 2)
        if torch.npu.device_count() < world_size:
            self.skipTest(f"requires {world_size} visible NPUs")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(world_size, port, allocation_only), nprocs=world_size, join=True)

    def test_output_views(self):
        torch.npu.set_device(0)
        shapes = [(2, 3), (), (0,), (2, 0, 3), (4,)]
        outputs = _create_coalesced_output_tensors(shapes, torch.float32, torch.device("npu"))
        self.assertEqual([tuple(t.shape) for t in outputs], shapes)
        self.assertEqual([t.storage_offset() for t in outputs], [0, 6, 7, 7, 7])
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in outputs}), 1)
        for index, output in enumerate(outputs):
            output.fill_(index)
        for index, output in enumerate(outputs):
            torch.testing.assert_close(output.cpu(), torch.full(shapes[index], index, dtype=torch.float32))

    def test_gather_and_allgather_correctness(self):
        self._spawn(False)

    def test_allocator_regression(self):
        self._spawn(True)


if __name__ == "__main__":
    unittest.main()
