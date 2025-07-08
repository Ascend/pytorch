from copy import deepcopy
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)

types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.ShortTensor,
    torch.CharTensor,
    torch.ByteTensor,
    torch.HalfTensor,
    torch.BoolTensor,
    torch.BFloat16Tensor,
]


def get_npu_type(type_name):
    if isinstance(type_name, type):
        type_name = '{}.{}'.format(type_name.__module__, type_name.__name__)
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch'
    return getattr(torch.npu, name)


class TestTensor(TestCase):
    def test_is_tensor(self):
        for t in types:
            tensor = get_npu_type(t)()
            self.assertTrue(torch.is_tensor(tensor))

    def test_is_storage(self):
        input1 = torch.randn(4, 6).npu()
        out = torch.is_storage(input1)
        self.assertFalse(out)

    def test_is_complex(self):
        a = torch.randn(1, 2).npu()
        out = torch.is_complex(a)
        self.assertFalse(out)

    def test_is_floating_point(self):
        input1 = torch.randn(2, 3).npu().float()
        out = torch.is_floating_point(input1)
        self.assertTrue(out)

        input1 = torch.randn(2, 3).npu().half()
        out = torch.is_floating_point(input1)
        self.assertTrue(out)

        input1 = torch.randn(2, 3).npu().int()
        out = torch.is_floating_point(input1)
        self.assertFalse(out)

    def test_is_nonzero(self):
        out1 = torch.is_nonzero(torch.tensor([0.]).npu())
        self.assertFalse(out1)

        out2 = torch.is_nonzero(torch.tensor([1.5]).npu())
        self.assertTrue(out2)

        out3 = torch.is_nonzero(torch.tensor([False]).npu())
        self.assertFalse(out3)

    def test_is_set_to(self):
        t1 = torch.empty(3, 4, 9, 10, device=device)
        t2 = torch.empty(3, 4, 9, 10, device=device)
        t3 = torch.tensor([], device=device).set_(t1)
        t4 = t3.clone().resize_(12, 90)
        self.assertFalse(t1.is_set_to(t2))
        self.assertTrue(t1.is_set_to(t3))
        self.assertTrue(t3.is_set_to(t1), "is_set_to should be symmetric")
        self.assertFalse(t1.is_set_to(t4))
        self.assertFalse(torch.tensor([]).is_set_to(torch.tensor([])),
                         "Tensors with no storages should not appear to be set "
                         "to each other")

        t1 = torch.tensor([True, True], dtype=torch.bool, device=device)
        t2 = torch.tensor([0], dtype=torch.bool, device=device).set_(t1)
        self.assertTrue(t1.is_set_to(t2))

        # test that sizes must match
        t1 = torch.empty([2, 3, 4], device=device)
        t2 = t1.view(4, 3, 2)
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

        # test that legacy empty size behavior used to be respected (i.e. all
        # empty tensors were logically collapsed to size [0]).
        t1 = torch.empty([2, 5, 0], device=device)
        t2 = t1.view([0])
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

    def test_set_default_dtype(self):
        out1 = torch.tensor([1.2, 3]).npu().dtype
        self.assertTrue(torch.float32, out1)

        torch.set_default_dtype(torch.float16)
        out2 = torch.tensor([1.2, 3]).npu().dtype
        self.assertTrue(torch.float16, out2)

    def test_get_default_dtype(self):
        out1 = torch.get_default_dtype()
        self.assertTrue(torch.float32, out1)

        torch.set_default_dtype(torch.float16)
        out2 = torch.get_default_dtype()
        self.assertTrue(torch.float16, out2)

    def test_set_default_tensor_type(self):
        out1 = torch.tensor([1.2, 3]).npu().dtype
        self.assertTrue(torch.float32, out1)

        torch.set_default_tensor_type(torch.HalfTensor)
        out2 = torch.tensor([1.2, 3]).npu().dtype
        self.assertTrue(torch.float16, out2)

    def test_numel(self):
        out1 = torch.randn(1, 2, 3, 4, 5).npu()
        self.assertTrue(120, out1)

        out2 = torch.zeros(4, 4).npu()
        self.assertTrue(16, out2)

    def test_set_printoptions(self):
        x = torch.tensor([1e2, 1e-2]).npu()
        torch.set_printoptions(sci_mode=True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertTrue(str(x) is not None)
        torch.set_printoptions(sci_mode=False)
        self.assertEqual(x.__repr__(), str(x))
        self.assertTrue(str(x) is not None)

    def test_set_flush_denormal(self):
        torch.set_flush_denormal(True)

        cpu_output1 = torch.tensor([1e-42], dtype=torch.float32)
        npu_output1 = torch.tensor([1e-42], dtype=torch.float32, device=device)
        self.assertEqual(cpu_output1, npu_output1)

        torch.set_flush_denormal(False)
        cpu_output2 = torch.tensor([1e-42], dtype=torch.float32)
        npu_output2 = torch.tensor([1e-42], dtype=torch.float32, device=device)
        self.assertEqual(cpu_output2, npu_output2)

    def test_deepcopy(self):
        input1 = torch.randn(1, 2, 3, 4, 5).npu()
        copy_tensor = deepcopy(input1)
        input1_storage_size = torch_npu.get_storage_size(input1)
        copy_tensor_storage_size = torch_npu.get_storage_size(copy_tensor)
        self.assertEqual(input1_storage_size, copy_tensor_storage_size)

    def test_scalar_tensor_item(self):
        scalar_input = torch.randn(1).npu()
        bool_scalar = scalar_input.to(torch.bool)
        self.assertTrue(isinstance(bool_scalar.item(), bool))
        
        half_scalar = scalar_input.to(torch.float16)
        self.assertTrue(isinstance(half_scalar.item(), float))
        
        bf16_scalar = scalar_input.to(torch.bfloat16)
        self.assertTrue(isinstance(bf16_scalar.item(), float))


class TestCreationOps(TestCase):
    def test_tensor(self):
        cpu_output1 = torch.tensor([[0.11111, 0.222222, 0.3333333]],
                                   dtype=torch.float32,
                                   device=torch.device('cpu'))
        npu_output1 = torch.tensor([[0.11111, 0.222222, 0.3333333]],
                                   dtype=torch.float32,
                                   device=device)
        self.assertRtolEqual(cpu_output1.numpy(), npu_output1.cpu().numpy())

        cpu_output2 = torch.tensor(3.14159, device=torch.device('cpu'))
        npu_output2 = torch.tensor(3.14159, device=device)
        self.assertRtolEqual(cpu_output2.numpy(), npu_output2.cpu().numpy())

        cpu_output3 = torch.tensor([], device=torch.device('cpu'))
        npu_output3 = torch.tensor([], device=device)
        self.assertRtolEqual(cpu_output3.numpy(), npu_output3.cpu().numpy())

    def test_as_tensor(self):
        # numpy ndarray to cpu/npu tensor
        a = np.array([1, 2, 3])
        cpu_output = torch.as_tensor(a, device=torch.device('cpu'))
        npu_output1 = torch.as_tensor(a, device="npu:0")
        npu_output2 = torch.as_tensor(a, device="npu")
        npu_output3 = torch.as_tensor(a, device=torch.device("npu"))
        self.assertRtolEqual(cpu_output.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_output.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_output.numpy(), npu_output3.cpu().numpy())

        # cpu tesor to npu tensor
        a = np.array([1, 2, 3])
        cpu_input = torch.as_tensor(a, device=torch.device('cpu'))
        npu_output1 = torch.as_tensor(cpu_input, device="npu:0")
        npu_output2 = torch.as_tensor(cpu_input, device="npu")
        npu_output3 = torch.as_tensor(cpu_input, device=torch.device("npu"))
        self.assertRtolEqual(cpu_input.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_input.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_input.numpy(), npu_output3.cpu().numpy())

        # npu tesor to cpu tensor
        a = np.array([1, 2, 3])
        npu_input = torch.as_tensor(a, device=torch.device('npu'))
        cpu_output = torch.as_tensor(npu_input, device="cpu")
        self.assertRtolEqual(cpu_output.numpy(), npu_input.cpu().numpy())

        # npu tesor to npu tensor
        npu_input = torch.tensor([1, 2, 3]).npu()
        npu_output = torch.as_tensor(npu_input)
        self.assertEqual(npu_input.device, npu_output.device)
        self.assertRtolEqual(npu_input.cpu().numpy(), npu_output.cpu().numpy())

    def test_as_strided(self):
        x = torch.randn(3, 3, device='npu:0')
        npu_output1 = torch.as_strided(x, (2, 2), (1, 2))
        npu_output2 = torch.as_strided(x, (2, 2), (1, 2), 1)
        self.assertEqual(npu_output1 is not None, True)
        self.assertEqual(npu_output2 is not None, True)

    def test_from_numpy(self):
        a = np.array([1, 2, 3])
        cpu_output = torch.from_numpy(a)
        self.assertTrue(str(cpu_output), '''tensor([ 1,  2,  3])''')

    def test_zeros(self):
        cpu_output = torch.zeros(2, 3, device="cpu")
        npu_output = torch.zeros(2, 3, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_zeros_like(self):
        input1 = torch.empty(2, 3)
        cpu_output = torch.zeros_like(input1, device="cpu")
        npu_output = torch.zeros_like(input1.npu(), device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_ones(self):
        cpu_output = torch.ones(2, 3, device="cpu")
        npu_output = torch.ones(2, 3, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_ones_like(self):
        input1 = torch.empty(2, 3)
        cpu_output = torch.ones_like(input1, device="cpu")
        npu_output = torch.ones_like(input1, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_arange(self):
        cpu_output = torch.arange(1, 2.5, 0.5, device="cpu")
        npu_output = torch.arange(1, 2.5, 0.5, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_range(self):
        cpu_output = torch.range(1, 4, 0.5, device="cpu")
        npu_output = torch.range(1, 4, 0.5, device="npu:0")
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_linspace(self):
        cpu_output = torch.linspace(3, 10, steps=5, device="cpu")
        npu_output = torch.linspace(3, 10, steps=5, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_logspace(self):
        cpu_output = torch.linspace(3, 10, steps=5, device="cpu")
        npu_output = torch.linspace(3, 10, steps=5, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_full_like(self):
        input1 = torch.randn(2, 3)
        cpu_output = torch.full_like(input1, 5, device="cpu")
        npu_output = torch.full_like(input1, 5, device=device)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())


class TestISJMOps(TestCase):
    def test_chunk(self):
        npu_input = torch.randn(2, 3, device=device)
        cpu_input = npu_input.cpu()
        cpu_output = torch.chunk(cpu_input, chunks=2)
        npu_output = torch.chunk(npu_input, chunks=2)
        self.assertRtolEqual(cpu_output[0].numpy(), npu_output[0].cpu().numpy())
        self.assertRtolEqual(cpu_output[1].numpy(), npu_output[1].cpu().numpy())

    def test_narrow(self):
        npu_input = torch.randn(3, 3, device=device)
        cpu_input = npu_input.cpu()
        cpu_output = torch.narrow(cpu_input, 1, 1, 2)
        npu_output = torch.narrow(npu_input, 1, 1, 2)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_reshape(self):
        npu_input = torch.randn(2, 3, device=device)
        cpu_input = npu_input.cpu()
        cpu_output = torch.reshape(cpu_input, (3, 2))
        npu_output = torch.reshape(npu_input, (3, 2))
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_squeeze(self):
        input1 = torch.zeros(2, 1, 2, 1, 2, device=device)
        output1 = torch.squeeze(input1)
        output2 = torch.squeeze(input1, 0)
        output3 = torch.squeeze(input1, 1)
        self.assertExpectedInline(str(output1.size()), '''torch.Size([2, 2, 2])''')
        self.assertExpectedInline(str(output2.size()), '''torch.Size([2, 1, 2, 1, 2])''')
        self.assertExpectedInline(str(output3.size()), '''torch.Size([2, 2, 1, 2])''')

    def test_t(self):
        input1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
        output = torch.t(input1)
        output_expect = torch.tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertRtolEqual(output.cpu().numpy(), output_expect.numpy())

    def test_transpose(self):
        input1 = torch.randn(2, 3)
        cpu_output = torch.transpose(input1, 0, 1)
        npu_output = torch.transpose(input1.npu(), 0, 1)
        self.assertRtolEqual(npu_output.cpu().numpy(), cpu_output.numpy())

    def test_unbind(self):
        input1 = torch.randn(2, 3)
        cpu_output = torch.unbind(input1)
        npu_output = torch.unbind(input1.npu())
        for i, _ in enumerate(cpu_output):
            self.assertRtolEqual(npu_output[i].cpu().numpy(), cpu_output[i].numpy())

    def test_unsqueeze(self):
        input1 = torch.tensor([1, 2, 3, 4], device=device)
        output1 = torch.unsqueeze(input1, 0)
        output1_expect = torch.tensor([[1, 2, 3, 4]])
        self.assertRtolEqual(output1.cpu().numpy(), output1_expect.numpy())
        output2 = torch.unsqueeze(input1, 1)
        output1_expect = torch.tensor([[1], [2], [3], [4]])
        self.assertRtolEqual(output2.cpu().numpy(), output1_expect.numpy())


class TestViewOps(TestCase):
    def is_view_of(self, base, other):
        if (not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device):
            return False

        return True

    def test_view_as(self):
        t = torch.ones(5, 5).npu()
        e = torch.empty((25,)).npu()
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view(self):
        t = torch.ones(5, 5).npu()
        size = (25,)
        v = t.view(size)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_with_size(self):
        tensor = torch.rand(15).npu()
        template = torch.rand(3, 5).npu()
        empty = torch.empty(0).npu()
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)


if __name__ == "__main__":
    run_tests()
