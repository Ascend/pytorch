import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


def _run_worker(rank, world_size, store_path, messages_per_peer):
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method="file://" + store_path,
                            rank=rank, world_size=world_size)
    try:
        peers = range(world_size)  # Include this rank.
        values = torch.arange(8 * messages_per_peer, dtype=torch.float32).reshape(-1, 8)
        send_tensors = (values + rank * values.numel()).to("npu")
        recv_tensors = [torch.full_like(send_tensors, -1) for _ in peers]
        # Submit all sends before all receives in one call.
        sends, recvs = [], []
        for peer, recv_buffer in zip(peers, recv_tensors):
            for send, recv in zip(send_tensors, recv_buffer):
                sends.append(dist.P2POp(dist.isend, send, peer))
                recvs.append(dist.P2POp(dist.irecv, recv, peer))
        works = torch_npu.distributed.fused_isend_irecv(sends + recvs)
        for work in works:
            work.wait()
        for peer, recv_buffer in zip(peers, recv_tensors):
            received = recv_buffer.cpu()
            expected = values + peer * values.numel()
            if not torch.equal(received, expected):
                raise AssertionError(f"Rank {rank} received incorrect data from rank {peer}")
    finally:
        dist.destroy_process_group()


class TestNpuFusedIsendIrecv(TestCase):
    def _run_test(self, world_size, messages_per_peer=1):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_run_worker, args=(world_size, directory + "/store", messages_per_peer),
                     nprocs=world_size, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_send_then_recv_2p(self):
        self._run_test(2)

    @skipIfUnsupportMultiNPU(2)
    def test_200_messages_2p(self):
        self._run_test(2, messages_per_peer=100)

    @skipIfUnsupportMultiNPU(4)
    def test_all_ranks_4p(self):
        self._run_test(4)


if __name__ == "__main__":
    run_tests()
