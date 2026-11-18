"""Test Scatter with shared storage and object packing.

Check what each rank receives and whether shared buffers stay valid when
asynchronous Gather, AllGather, and Scatter outlive the caller's input tensors.
"""

from datetime import timedelta
import gc
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


def _object(rank):
    return {"rank": rank, "text": "汉字" * (rank + 1), "data": bytes([rank]) * (rank * 17),
            "nested": [None, True, (rank, rank + 1)], "tensor": torch.arange(7) + rank}


def _assert_object(actual, rank):
    expected = _object(rank)
    torch.testing.assert_close(actual.pop("tensor"), expected.pop("tensor"), rtol=0, atol=0)
    assert actual == expected, (actual, expected)


class _CountedObject:
    def __init__(self, rank):
        self.rank = rank
        self.calls = 0

    def __reduce__(self):
        self.calls += 1
        return tuple, ([self.rank],)


def _tensor_case(rank, peers, group, shape, dtype, asynchronous, group_src=False, source_index=-1):
    root = peers[source_index]
    output = torch.empty(shape, dtype=dtype, device="npu")
    inputs = None
    if rank == root:
        inputs = _create_coalesced_output_tensors([shape] * len(peers), dtype, output.device)
        for peer, tensor in zip(peers, inputs):
            tensor.copy_(_reference(peer, shape, dtype))
        del tensor
    kwargs = {"group_src": peers.index(root)} if group_src else {"src": root}
    work = dist.scatter(output, inputs, group=group, async_op=asynchronous, **kwargs)
    del inputs
    if asynchronous:
        pressure = torch.full((max(1, len(peers) * output.numel()),), 7, dtype=dtype, device="npu")
        work.wait()
        torch.testing.assert_close(pressure.cpu(), torch.full(pressure.shape, 7, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), _reference(rank, shape, dtype), rtol=0, atol=0)


def _object_case(rank, peers, group, source_index, group_src):
    root = peers[source_index]
    inputs = tuple(_object(peer) for peer in peers) if rank == root else None
    output = [None, "preserve"]
    kwargs = {"group_src": peers.index(root)} if group_src else {"src": root}
    result = dist.scatter_object_list(output, inputs, group=group, **kwargs)
    assert result is None
    if rank in peers:
        _assert_object(output[0], rank)
    else:
        assert output[0] is None
    assert output[1] == "preserve"


def run_protocol(rank, world_size, check, optimized=True):
    peers = list(range(world_size))
    torch.npu.use_compatible_impl(False)
    for dtype in (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.int64, torch.bool):
        edge = 512 // torch.empty((), dtype=dtype).element_size()
        for shape in ((), (0,), (1,), (edge - 1,), (edge,), (edge + 1,), (3, 7)):
            for asynchronous in (False, True):
                check(f"scatter/{dtype}/{shape}/async{asynchronous}",
                      lambda: _tensor_case(rank, peers, None, shape, dtype, asynchronous))
    for source_index in (0, -1, 0):
        for group_src in (False, True):
            check(f"object/source{source_index}/group_src{group_src}",
                  lambda: _object_case(rank, peers, None, source_index, group_src))
    counted = [_CountedObject(peer) for peer in peers] if rank == 0 else None
    output = [None]
    dist.scatter_object_list(output, iter(counted) if counted is not None else None, src=0)
    assert output == [(rank,)]
    if counted is not None:
        assert [obj.calls for obj in counted] == [1] * world_size
    check("serialize_once_iterable_input", lambda: None)

    def changing_objects():
        shared = {}
        for peer in peers:
            shared["rank"] = peer
            yield shared

    dist.scatter_object_list(output, changing_objects() if rank == 0 else None, src=0)
    assert output == [{"rank": rank}]
    check("serialize_mutable_object_before_advancing_iterator", lambda: None)

    subgroup_peers = [1, world_size - 1] if world_size > 2 else [world_size - 1]
    subgroup = dist.new_group(subgroup_peers)
    check("subgroup_members_and_nonmembers",
          lambda: _object_case(rank, subgroup_peers, subgroup, -1, True))
    if rank in subgroup_peers:
        check("subgroup_tensor_group_src",
              lambda: _tensor_case(rank, subgroup_peers, subgroup, (513,), torch.uint8, True, True))
        dist.destroy_process_group(subgroup)
    dist.barrier()
    cpu_group = dist.new_group(backend="gloo")
    check("gloo_fallback", lambda: _object_case(rank, peers, cpu_group, -1, True))
    dist.destroy_process_group(cpu_group)

    case = unittest.TestCase()
    with case.assertRaises(ValueError):
        dist.scatter_object_list([], src=0)
    with case.assertRaises(ValueError):
        dist.scatter_object_list([None], src=0, group_src=0)
    wrong_output = torch.empty(1, dtype=torch.float32, device="npu")
    wrong_input = torch.empty(1, dtype=torch.int64, device="npu")
    with case.assertRaisesRegex(ValueError, "different dtypes"):
        dist.all_gather([wrong_output] * world_size, wrong_input)
    if rank == 0:
        with case.assertRaisesRegex(ValueError, "different dtypes"):
            dist.scatter(wrong_output, [wrong_input] * world_size, src=0)
    check("mismatched_dtype_rejected_before_collective", lambda: None)
    if rank == 0:
        with case.assertRaises(ValueError):
            dist.scatter_object_list([None], None, src=0)
        if optimized:
            with case.assertRaisesRegex(ValueError, "one object per group rank"):
                dist.scatter_object_list([None], [None] * (world_size + 1), src=0)
    dist.barrier()
    check("valid_collective_after_invalid_arguments", lambda: _object_case(rank, peers, None, 0, False))
    compatible_soc = torch.npu.get_device_name().startswith(("Ascend910B", "Ascend910_93"))
    if compatible_soc:
        torch.npu.use_compatible_impl(True)
        try:
            for shape in ((), (0,), (513,)):
                check(f"compatible_scatter/{shape}",
                      lambda: _tensor_case(rank, peers, None, shape, torch.float32, True))
        finally:
            torch.npu.use_compatible_impl(False)


def run_lifetime(rank, world_size, check):
    # Allocate replacement tensors on the input's stream after dropping the
    # input reference to check for allocator reuse while HCCL still reads it.
    producer = torch.npu.Stream()
    consumer = torch.npu.Stream()
    with torch.npu.stream(consumer):
        delay_input = torch.ones((4096, 4096), device="npu")
        delay_output = torch.empty_like(delay_input)
        torch.mm(delay_input, delay_input, out=delay_output)
    torch.npu.synchronize()
    for operation, future_only in ((op, future) for op in ("allgather", "scatter", "gather") for future in (False, True)):
        size = 32 * 1024 * 1024 if operation == "gather" else 4 * 1024 * 1024
        with torch.npu.stream(producer):
            source = torch.full((size,), rank + 1, dtype=torch.uint8, device="npu")
            parent = torch.empty((world_size * size + 1024,), dtype=torch.uint8, device="npu")
            parent.fill_(219)
            outputs = list(parent[512:-512].split(size))
            inputs = None
            if rank == 0:
                inputs = _create_coalesced_output_tensors([(size,)] * world_size, source.dtype, source.device)
                for peer, tensor in enumerate(inputs):
                    tensor.fill_(peer + 1)
                del tensor
        consumer.wait_stream(producer)
        with torch.npu.stream(consumer):
            if operation != "gather":
                # Queue matmuls to keep communication pending when the test
                # drops the input references.
                for _ in range(16):
                    torch.mm(delay_input, delay_input, out=delay_output)
            if operation == "allgather":
                work = dist.all_gather(outputs, source, async_op=True)
                expected = list(range(1, world_size + 1))
            elif operation == "scatter":
                work = dist.scatter(outputs[0], inputs, src=0, async_op=True)
                outputs = outputs[:1]
                expected = [rank + 1]
            else:
                if rank != world_size - 1:
                    outputs = None
                work = dist.gather(source, outputs, dst=world_size - 1, async_op=True)
                expected = list(range(1, world_size + 1)) if outputs is not None else []
            inflight = not work.is_completed()
            del source, inputs
            # View references keep the destination storage alive after deleting
            # its parent tensor. Non-root gather's automatic outputs have no
            # caller-owned reference and must be protected by the backend.
            left, right = parent[:512], parent[-512:]
            del parent
        pressure = []
        for stream in (producer, consumer):
            with torch.npu.stream(stream):
                for length in (size, world_size * size):
                    value = torch.full((length,), 173, dtype=torch.uint8, device="npu")
                    pressure.append(value)
        with torch.npu.stream(consumer):
            if future_only:
                work.get_future().wait()
            else:
                work.wait()
            consumer.wait_stream(producer)
            if outputs is not None:
                for output, value in zip(outputs, expected):
                    torch.testing.assert_close(output.cpu(), torch.full((size,), value, dtype=torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(left.cpu(), torch.full((512,), 219, dtype=torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(right.cpu(), torch.full((512,), 219, dtype=torch.uint8), rtol=0, atol=0)
        for value in pressure:
            torch.testing.assert_close(value.cpu(), torch.full(value.shape, 173, dtype=torch.uint8), rtol=0, atol=0)
        assert inflight, f"{operation}: operation completed before early-release pressure; no in-flight coverage"
        check(f"nondefault_stream_early_release/{operation}/future_only{future_only}", lambda: None)
        del outputs, pressure, left, right, value, work
        gc.collect()


def _worker(rank, world_size, port, suite):
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
                       "LOCAL_WORLD_SIZE": str(world_size), "GROUP_RANK": "0"})
    torch_npu.npu.set_device(rank)
    torch.npu.use_compatible_impl(False)
    dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=240))
    results = []

    def check(name, fn):
        fn()
        results.append(name)

    try:
        {"protocol": run_protocol, "lifetime": run_lifetime}[suite](rank, world_size, check)
        assert results
        dist.barrier()
    finally:
        dist.destroy_process_group()


class HcclCoalescedScatterTest(unittest.TestCase):
    def _spawn(self, suite):
        world_size = int(os.environ.get("ALL_GATHER_TEST_WORLD_SIZE", "2"))
        if not torch.npu.is_available() or torch.npu.device_count() < world_size:
            self.skipTest(f"requires {world_size} visible NPUs")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(world_size, port, suite), nprocs=world_size, join=True)

    def test_protocol(self):
        self._spawn("protocol")

    def test_nondefault_stream_lifetime(self):
        self._spawn("lifetime")


if __name__ == "__main__":
    unittest.main()
