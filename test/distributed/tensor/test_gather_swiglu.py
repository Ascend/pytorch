import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class TestRegisterSharding(NPUDTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_swiglu(self):
        mesh = self.build_device_mesh()
        
        input_tensor = torch.randn(1024, 1024, device="npu", requires_grad=True)
        grad_tensor = torch.randn(1024, 512, device="npu")
        dim = -1
        out_tensor = torch_npu.npu_swiglu(input_tensor, dim)
        out_tensor.backward(grad_tensor)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        grad_dtensor = distribute_tensor(grad_tensor, mesh, [Replicate()])

        output = torch_npu.npu_swiglu(input_dtensor, dim)
        output.backward(grad_dtensor)
        self.assertEqual(output.full_tensor(), out_tensor)
        self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)
    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_swiglu_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(1024, 1024, device="npu", requires_grad=True)
        grad_tensor = torch.randn(1024, 512, device="npu")
        dim = -1
        out_tensor = torch_npu.npu_swiglu(input_tensor, dim)
        out_tensor.backward(grad_tensor)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(0)])

        output = torch_npu.npu_swiglu(input_dtensor, dim)
        grad_dtensor = distribute_tensor(grad_tensor, mesh, output.placements)
        output.backward(grad_dtensor)
        self.assertEqual(output.full_tensor(), out_tensor)
        self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_swiglu_shard1(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(1024, 1024, device="npu", requires_grad=True)
        grad_tensor = torch.randn(1024, 512, device="npu")
        dim = -1  
        out_tensor = torch_npu.npu_swiglu(input_tensor, dim)
        out_tensor.backward(grad_tensor)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(1)])

        output = torch_npu.npu_swiglu(input_dtensor, dim)
        grad_dtensor = distribute_tensor(grad_tensor, mesh, output.placements)
        output.backward(grad_dtensor)
        self.assertEqual(output.full_tensor(), out_tensor)
        self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)
    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_gather(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(3, 2, device="npu", requires_grad=True)
        index_tensor = torch.tensor([[0, 1], [1, 0], [0, 1]], device="npu")
        grad_tensor = torch.randn(3, 2, device="npu")

        dim = 1

        out_tensor = torch.gather(input=input_tensor, dim=1, index=index_tensor)
        out_tensor.backward(grad_tensor)

        x = distribute_tensor(input_tensor, mesh, [Replicate()])
        index = distribute_tensor(index_tensor, mesh, [Replicate()])
        grad = distribute_tensor(grad_tensor, mesh, [Replicate()])

        out = torch.gather(input=x, dim=dim, index=index)

        out.backward(grad)
        self.assertEqual(out.full_tensor(), out_tensor)
        self.assertEqual(x.grad.full_tensor(), input_tensor.grad)
    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_gather_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(3, 2, device="npu", requires_grad=True)
        index_tensor = torch.tensor([[0, 1], [1, 0], [0, 1]], device="npu")
        grad_tensor = torch.randn(3, 2, device="npu")

        dim = 1

        out_tensor = torch.gather(input=input_tensor, dim=1, index=index_tensor)
        out_tensor.backward(grad_tensor)

        x = distribute_tensor(input_tensor, mesh, [Shard(0)])
        index = distribute_tensor(index_tensor, mesh, [Shard(0)])
        
        out = torch.gather(input=x, dim=dim, index=index)
        grad = distribute_tensor(grad_tensor, mesh, out.placements)

        out.backward(grad)
        self.assertEqual(out.full_tensor(), out_tensor)
        self.assertEqual(x.grad.full_tensor(), input_tensor.grad)
    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_gather_shard1(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(3, 2, device="npu", requires_grad=True)
        index_tensor = torch.tensor([[0, 1], [1, 0], [0, 1]], device="npu")
        grad_tensor = torch.randn(3, 2, device="npu")

        dim = 1

        out_tensor = torch.gather(input=input_tensor, dim=1, index=index_tensor)
        out_tensor.backward(grad_tensor)

        x = distribute_tensor(input_tensor, mesh, [Shard(1)])
        index = distribute_tensor(index_tensor, mesh, [Shard(1)])
        
        out = torch.gather(input=x, dim=dim, index=index)
        grad = distribute_tensor(grad_tensor, mesh, out.placements)

        out.backward(grad)
        self.assertEqual(out.full_tensor(), out_tensor)
        self.assertEqual(x.grad.full_tensor(), input_tensor.grad)

if __name__ == "__main__":
    run_tests()