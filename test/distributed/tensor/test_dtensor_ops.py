import torch
from torch.distributed._tensor import distribute_tensor, Replicate
from torch_npu.testing.testcase import run_tests
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class TestDTensorOps(NPUDTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_nn_functional_one_hot(self):
        """Test torch.nn.functional.one_hot with DTensor on NPU"""
        # Build device mesh for distributed tensor operations on NPU
        device_mesh = self.build_device_mesh()

        input_tensor = torch.randint(0, 4, (5,), device="npu")
        num_classes = 5

        x = distribute_tensor(input_tensor, device_mesh, [Replicate()])

        # Apply one-hot encoding on distributed tensor
        one_hot_tensor = torch.nn.functional.one_hot(x, num_classes)

        # Verify the output
        self.assertTrue(hasattr(one_hot_tensor, "to_local"))
        self.assertEqual(one_hot_tensor.shape, (5, 5))
        self.assertEqual(one_hot_tensor.placements, (Replicate(),))


if __name__ == "__main__":
    run_tests()
