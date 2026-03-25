import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _world

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class P2PStreamIdTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        """Initialize HCCL distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_p2p_stream_id_batched(cls, rank, world_size, init_pg, c2p, p2c):
        """Test P2P stream ID in batched mode (is_batched=1)."""
        dist_group = init_pg(rank, world_size)

        # Get backend for stream operations
        backend = _world.default_pg._get_backend(torch.device('npu'))

        # Perform P2P operations to trigger stream creation
        send_tensor = torch.ones(2, 2).to(f"npu:{rank}") * (rank + 1)
        recv_tensor = torch.ones(2, 2).to(f"npu:{rank}") * -1

        # Test P2P communication between ranks
        peer = 1 - rank
        p2p_op_list = [
            dist.P2POp(dist.isend, send_tensor, peer),
            dist.P2POp(dist.irecv, recv_tensor, peer),
        ]

        reqs = dist.batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

        device = torch.device(f'npu:{rank}')
        p2p_stream_id = backend.get_p2p_stream_id(device, peer, 1)  # is_batched=1

        # Verify P2P Stream ID is valid
        assert0 = False
        if p2p_stream_id != -1:
            # Create stream object to verify validity
            stream = torch.npu.Stream(stream_id=p2p_stream_id, device_type=20, device_index=device.index)
            assert0 = (stream.npu_stream is not None)

        c2p.put(assert0)
        p2c.get()

    @classmethod
    def _test_p2p_stream_id_non_batched(cls, rank, world_size, init_pg, c2p, p2c):
        """Test P2P stream ID in non-batched mode (is_batched=0)."""
        dist_group = init_pg(rank, world_size)

        # Get backend for stream operations
        backend = _world.default_pg._get_backend(torch.device('npu'))

        # Perform P2P operations to trigger stream creation
        send_tensor = torch.ones(2, 2).to(f"npu:{rank}") * (rank + 1)
        recv_tensor = torch.ones(2, 2).to(f"npu:{rank}") * -1

        # Rank 0 sends to Rank 1, Rank 1 recvs from Rank 0
        if rank == 0:
            dist_group.send(send_tensor, 1)
        else:
            dist_group.recv(recv_tensor, 0)

        device = torch.device(f'npu:{rank}')
        peer = 1 - rank
        p2p_stream_id = backend.get_p2p_stream_id(device, peer, 0)

        assert0 = False
        if p2p_stream_id != -1:
            stream = torch.npu.Stream(stream_id=p2p_stream_id, device_type=20, device_index=device.index)
            assert0 = (stream.npu_stream is not None)

        c2p.put(assert0)
        p2c.get()

    @classmethod
    def _test_p2p_stream_id_invaild(cls, rank, world_size, init_pg, c2p, p2c):
        """Test P2P stream ID without establishing communication."""
        dist_group = init_pg(rank, world_size)

        # Get backend for stream operations
        backend = _world.default_pg._get_backend(torch.device('npu'))

        device = torch.device(f'npu:{rank}')
        peer = 1 - rank
        p2p_stream_id = backend.get_p2p_stream_id(device, peer, 0)

        # Verify P2P Stream ID is invaild
        assert0 = True if p2p_stream_id == -1 else False

        c2p.put(assert0)
        p2c.get()

    def _run_multiprocess_test(self, test_func, init_pg, world_size, test_name=""):
        """Helper method to run multi-process tests."""
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)

        ps = []
        for rank in range(world_size):
            p = ctx.Process(target=test_func, args=(rank, world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            output = c2p.get()
            self.assertEqual(True, output)

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_p2p_stream_id_batched(self):
        """Test batched mode P2P stream ID with same device."""
        print("\n=== Testing Batched P2P Stream ID ===")
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "1"}):
            ranks = [2]
            for world_size in ranks:
                self._run_multiprocess_test(
                    P2PStreamIdTest._test_p2p_stream_id_batched,
                    P2PStreamIdTest._init_dist_hccl,
                    world_size,
                    "Batched P2P Stream ID Test"
                )

    @skipIfUnsupportMultiNPU(2)
    def test_p2p_stream_id_non_batched(self):
        """Test non-batched mode P2P stream ID with cross-rank communication."""
        print("\n=== Testing Non-batched P2P Stream ID ===")
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "1"}):
            ranks = [2]
            for world_size in ranks:
                self._run_multiprocess_test(
                    P2PStreamIdTest._test_p2p_stream_id_non_batched,
                    P2PStreamIdTest._init_dist_hccl,
                    world_size,
                    "Non-batched P2P Stream ID Test"
                )

    @skipIfUnsupportMultiNPU(2)
    def test_p2p_stream_id_invalid(self):
        """Test P2P stream ID without establishing communication."""
        print("\n=== Testing Non P2P Stream ID ===")
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "1"}):
            ranks = [2]
            for world_size in ranks:
                self._run_multiprocess_test(
                    P2PStreamIdTest._test_p2p_stream_id_invaild,
                    P2PStreamIdTest._init_dist_hccl,
                    world_size,
                    "Non P2P Stream ID Test"
                )


if __name__ == '__main__':
    run_tests()
