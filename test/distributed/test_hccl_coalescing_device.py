import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from torch.distributed.distributed_c10d import _coalescing_manager
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests

OPS = ("p2p", "allreduce")


def _op(operator, rank, ranks, group, device, iteration, different=False):
    group_rank, size = ranks.index(rank), len(ranks)
    value = iteration * size + group_rank + 1
    tensors = [torch.full((32,), value + i, dtype=torch.int32, device=device)
               for i in range(2)]
    recv = torch.full_like(tensors[0], -1)
    torch.npu.set_device((device.index + 1) % torch.npu.device_count()
                         if different else device)
    with _coalescing_manager(group=group, device=device, async_ops=True) as cm:
        if operator == "p2p":
            dist.isend(tensors[0], dst=ranks[(group_rank + 1) % size], group=group)
            dist.irecv(recv, src=ranks[(group_rank - 1) % size], group=group)
        else:
            for tensor in tensors:
                dist.all_reduce(tensor, group=group)
    if operator == "p2p":
        assert cm.works
    for work in cm.works:
        if work is not None:
            work.wait()
    torch.npu.set_device(device)
    torch.npu.synchronize()
    if operator == "p2p":
        expected = iteration * size + (group_rank - 1) % size + 1
        torch.testing.assert_close(recv.cpu(), torch.full((32,), expected,
                                                          dtype=torch.int32), rtol=0, atol=0)
    else:
        for i, tensor in enumerate(tensors):
            expected = size * (iteration * size + i) + size * (size + 1) // 2
            torch.testing.assert_close(tensor.cpu(), torch.full((32,), expected, dtype=torch.int32),
                                       rtol=0, atol=0)


def _worker(rank, world, port, mapping, bind, iterations, operator, different=False,
            local_rank=None):
    if local_rank is None:
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
        local_rank = rank
    device = torch.device("npu", mapping[local_rank])
    torch.npu.set_device(device)
    kwargs = {"device_id": device} if bind else {}
    dist.init_process_group("hccl", rank=rank, world_size=world,
                            timeout=timedelta(seconds=60), **kwargs)
    try:
        ranks = list(range(world))
        for iteration in range(iterations):
            _op(operator, rank, ranks, None, device, iteration, different)
    finally:
        dist.destroy_process_group()


class HcomCoalescingDeviceTest(TestCase):
    def _spawn(self, mapping=(0, 1), bind=False, operator="p2p", different=False,
               iterations=3):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        for key, value in (("TASK_QUEUE_ENABLE", "1"), ("PER_STREAM_QUEUE", "0"),
                           ("ASCEND_LAUNCH_BLOCKING", "0")):
            os.environ.setdefault(key, value)
        mp.spawn(_worker, args=(len(mapping), port, mapping, bind, iterations, operator,
                                different), nprocs=len(mapping), join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_reversed_mapping(self):
        for operator in OPS:
            with self.subTest(operator=operator):
                self._spawn(mapping=(1, 0), operator=operator)

    @skipIfUnsupportMultiNPU(2)
    def test_bound_overrides_current(self):
        self._spawn(mapping=(1, 0), bind=True, different=True)


def _torchrun():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-offset", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--operator", choices=OPS, default="p2p")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--bind-device", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if world < 2 or args.iterations < 1:
        raise ValueError("Need at least two processes and one iteration")
    if not 0 <= args.device_offset <= torch.npu.device_count() - local_world:
        raise ValueError("device-offset + local world size must fit visible devices")
    mapping = tuple(range(args.device_offset, args.device_offset + local_world))
    _worker(rank, world, None, mapping[::-1] if args.reverse else mapping,
            args.bind_device, args.iterations, args.operator, local_rank=local_rank)


if __name__ == "__main__":
    if os.environ.get("HCCL_TEST_TORCHRUN") == "1" or "LOCAL_RANK" in os.environ:
        _torchrun()
    else:
        run_tests()
