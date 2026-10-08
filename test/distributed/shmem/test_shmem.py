import ctypes
from unittest import skipIf

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


# So that tests are written in device-agnostic way
device_type = "npu"
device_module = torch.get_device_module(device_type)


def _shmem_available() -> bool:
    """Return whether the NPU and CANN SHMEM runtime are available."""
    if not torch.npu.is_available():
        return False
    try:
        ctypes.CDLL("libshmem.so", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


@instantiate_parametrized_tests
@skipIf(not _shmem_available(), "CANN SHMEM is not available")
class NPUSHMEMSymmetricMemoryTest(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # Testing with HCCL backend
        return "hccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.npu.device_count()
        cls.device = torch.device(f"npu:{dev_id}")

    def _init_device(self) -> None:
        device_module.set_device(self.device)
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _make_get_buffer_fixture(self):
        """Create one two-rank symmetric allocation with distinguishable values."""
        rank = self.rank
        peer = 1 - rank
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # rank 0 保存 [0..15]，rank 1 保存 [100..115]。
        tensor = symm_mem.empty(16, dtype=torch.int32, device=self.device)
        tensor.copy_(torch.arange(16, dtype=torch.int32, device=self.device) + rank * 100)
        handle = symm_mem.rendezvous(tensor, group=group_name)
        dist.barrier(device_ids=[rank])
        return rank, peer, tensor, handle

    @skipIfUnsupportMultiNPU(2)
    def test_alloc(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skipIfUnsupportMultiNPU(2)
    def test_alloc_free(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)
        del out

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_copy(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        shape = (512, 512)

        tensor = torch.randn(shape, dtype=dtype, device=self.device)

        shmem_tensor = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor.copy_(tensor)
        self.assertEqual(shmem_tensor, tensor)

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_matmul(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        shape = (512, 512)

        tensor = torch.randn(shape, dtype=dtype, device=self.device)
        tensor1 = torch.randn(shape, dtype=dtype, device=self.device)

        matmul = torch.matmul(tensor, tensor1)

        shmem_tensor = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor.copy_(tensor)

        shmem_matmul = torch.matmul(shmem_tensor, tensor1)
        self.assertEqual(shmem_matmul, matmul)

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_matmul1(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        shape = (512, 512)

        tensor = torch.randn(shape, dtype=dtype, device=self.device)
        tensor1 = torch.randn(shape, dtype=dtype, device=self.device)

        matmul = torch.matmul(tensor, tensor1)

        shmem_tensor = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor.copy_(tensor)
        shmem_tensor1 = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor1.copy_(tensor1)

        shmem_matmul = torch.matmul(shmem_tensor, shmem_tensor1)
        self.assertEqual(shmem_matmul, matmul)

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_put(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_put(tensor, 1)
            dist.barrier(device_ids=[self.rank])
        elif self.rank == 1:
            dist.barrier(device_ids=[self.rank])
            torch.testing.assert_close(
                tensor, torch.zeros(numel, dtype=dtype, device=self.device)
            )

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_get(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_get(tensor, 1)
            dist.barrier(device_ids=[self.rank])
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        elif self.rank == 1:
            dist.barrier(device_ids=[self.rank])

    @skipIfUnsupportMultiNPU(2)
    def test_get_buffer(self) -> None:
        """Validate mapped buffers, explicit offsets and bounds checking."""
        self._init_device()
        rank, peer, tensor, handle = self._make_get_buffer_fixture()

        self.assertEqual(handle.buffer_size, tensor.numel() * tensor.element_size())

        local = handle.get_buffer(rank, (16,), torch.int32)
        remote = handle.get_buffer(peer, (16,), torch.int32)
        self.assertEqual(local.data_ptr(), tensor.data_ptr())
        self.assertEqual(remote.data_ptr(), handle.buffer_ptrs[peer])
        self.assertEqual(local.untyped_storage().size(), handle.buffer_size)
        torch.testing.assert_close(local, tensor)
        torch.testing.assert_close(
            remote, torch.arange(16, dtype=torch.int32, device=self.device) + peer * 100
        )

        # storage_offset=4 表示跳过 4 个 int32，即 16 字节。
        remote_slice = handle.get_buffer(peer, (4,), torch.int32, 4)
        self.assertEqual(
            remote_slice.data_ptr(), handle.buffer_ptrs[peer] + 4 * tensor.element_size()
        )
        torch.testing.assert_close(
            remote_slice, torch.arange(4, 8, dtype=torch.int32, device=self.device) + peer * 100
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds the allocated size"):
            handle.get_buffer(peer, (4,), torch.int32, 14)
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    def test_get_buffer_alias(self) -> None:
        """Validate local aliasing and writes through a peer mapping."""
        self._init_device()
        rank, peer, tensor, handle = self._make_get_buffer_fixture()
        local = handle.get_buffer(rank, (16,), torch.int32)
        remote = handle.get_buffer(peer, (16,), torch.int32)

        local[3:4].fill_(1000 + rank)
        torch.npu.synchronize()
        self.assertEqual(tensor[3].item(), 1000 + rank)
        dist.barrier(device_ids=[rank])

        if rank == 0:
            remote[7:8].fill_(12345)
            torch.npu.synchronize()
        dist.barrier(device_ids=[rank])
        if rank == 1:
            self.assertEqual(tensor[7].item(), 12345)
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    def test_get_buffer_allocation_lifetime(self) -> None:
        """Validate that a live handle keeps the SHMEM allocation alive."""
        self._init_device()
        rank, peer, tensor, handle = self._make_get_buffer_fixture()
        local_ptr = handle.buffer_ptrs[rank]
        del tensor
        dist.barrier(device_ids=[rank])

        local = handle.get_buffer(rank, (16,), torch.int32)
        remote = handle.get_buffer(peer, (16,), torch.int32)
        self.assertEqual(local.data_ptr(), local_ptr)
        torch.testing.assert_close(
            local, torch.arange(16, dtype=torch.int32, device=self.device) + rank * 100
        )
        torch.testing.assert_close(
            remote, torch.arange(16, dtype=torch.int32, device=self.device) + peer * 100
        )
        dist.barrier(device_ids=[rank])

if __name__ == "__main__":
    run_tests()
