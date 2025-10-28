import os
from unittest import skip
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcContinousTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, run_tests
import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


# So that tests are written in device-agnostic way
device_type = "npu"
device_module = torch.get_device_module(device_type)


@instantiate_parametrized_tests
@skip("request shmem")
class NPUSHMEMSymmetricMemoryTest(MultiProcContinousTest):
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


if __name__ == "__main__":
    run_tests()
