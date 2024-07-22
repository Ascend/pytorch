import torch
import torch_npu
from torch_npu.utils import npu_combine_tensors, get_part_combined_tensor, is_combined_tensor_valid

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestCombineTensors(TestCase):

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_change_data_ptr(self, dtype, device="npu"):
        x = torch.randn((2, 2, 2, 2), device=device, dtype=dtype)
        y = torch.randn((4, 4), device=device, dtype=dtype)
        z = torch.randn((3, 3, 3), device=device, dtype=dtype)
        x_clone = x.clone()
        y_clone = y.clone()
        z_clone = z.clone()

        list_of_tensor = [x, y, z]
        total_numel = 0
        for tensor in list_of_tensor:
            total_numel += torch_npu.get_storage_size(tensor)
        combined_tensor = torch.zeros(total_numel, dtype=dtype).npu()
        idx = 0
        for tensor in list_of_tensor:
            tmp = tensor.clone()
            torch_npu.npu_change_data_ptr(tensor, combined_tensor, idx)
            tensor.copy_(tmp)
            idx += torch_npu.get_storage_size(tensor)

        self.assertEqual(x, x_clone)
        self.assertEqual(y, y_clone)
        self.assertEqual(z, z_clone)

        x_new = torch.zeros((2, 2, 2, 2), device=device, dtype=dtype)
        y_new = torch.zeros((4, 4), device=device, dtype=dtype)
        z_new = torch.zeros((3, 3, 3), device=device, dtype=dtype)
        list_of_tensor_new = [x_new, y_new, z_new]
        idx = 0
        for tensor_new, tensor in zip(list_of_tensor_new, list_of_tensor):
            torch_npu.npu_change_data_ptr(tensor_new, combined_tensor, idx)
            self.assertEqual(tensor_new, tensor)
            idx += torch_npu.get_storage_size(tensor)

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_storage_resize(self, dtype, device="npu"):
        x = torch.randn((2, 2, 2, 2), device=device, dtype=dtype)
        y = torch.randn((4, 4), device=device, dtype=dtype)
        z = torch.randn((3, 3, 3), device=device, dtype=dtype)
        x_clone = x.view(-1)[:4].clone()
        y_clone = y.view(-1)[:4].clone()
        z_clone = z.view(-1)[:4].clone()

        x.storage().resize_(4)
        y.storage().resize_(4)
        z.storage().resize_(4)

        list_of_tensor = [x, y, z]
        total_numel = 12
        combined_tensor = torch.zeros(total_numel, dtype=dtype).npu()
        idx = 0
        for tensor in list_of_tensor:
            tmp = tensor.clone()
            torch_npu.npu_change_data_ptr(tensor, combined_tensor, idx)
            tensor.copy_(tmp)
            idx += torch_npu.get_storage_size(tensor)

        self.assertEqual(x.storage(), x_clone.storage())
        self.assertEqual(y.storage(), y_clone.storage())
        self.assertEqual(z.storage(), z_clone.storage())

        x_new = torch.zeros((4), device=device, dtype=dtype)
        y_new = torch.zeros((4), device=device, dtype=dtype)
        z_new = torch.zeros((4), device=device, dtype=dtype)
        list_of_tensor_new = [x_new, y_new, z_new]
        idx = 0
        for tensor_new, tensor in zip(list_of_tensor_new, list_of_tensor):
            torch_npu.npu_change_data_ptr(tensor_new, combined_tensor, idx)
            self.assertEqual(tensor_new.storage(), tensor.storage())
            idx += torch_npu.get_storage_size(tensor)

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_untyped_storage_resize(self, dtype, device="npu"):
        a = torch.randn((2, 2, 2, 2, 2), device=device, dtype=dtype)
        a.untyped_storage().resize_(0)
        a.untyped_storage().resize_(128)

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_combine_tensors(self, dtype, device="npu"):
        x = torch.zeros((2, 2, 2, 2), device=device, dtype=dtype)
        y = torch.zeros((4, 4), device=device, dtype=dtype)
        z = torch.zeros((3, 3, 3), device=device, dtype=dtype)

        lst = [x, y, z]
        combine_tensor = npu_combine_tensors(lst)

        x_storage_size = torch_npu.get_storage_size(x)
        y_storage_size = torch_npu.get_storage_size(y)
        z_storage_size = torch_npu.get_storage_size(z)
        combine_tensor_storage_size = torch_npu.get_storage_size(combine_tensor)

        # test if combine_tensor is contiguous, and x,y,z are moved into the combine_tensor.
        self.assertEqual(True, combine_tensor.is_contiguous())
        self.assertEqual(combine_tensor.data_ptr(), x.data_ptr())
        self.assertEqual(x.data_ptr() + x_storage_size * x.element_size(), y.data_ptr())
        self.assertEqual(y.data_ptr() + y_storage_size * y.element_size(), z.data_ptr())
        self.assertEqual(combine_tensor_storage_size, x_storage_size + y_storage_size + z_storage_size)

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_combine_tensors_large(self, dtype, device="npu"):
        x = torch.zeros((200, 20, 200, 20), device=device, dtype=dtype)
        y = torch.zeros((4000, 4000), device=device, dtype=dtype)
        z = torch.zeros((300, 300, 300), device=device, dtype=dtype)

        lst = [x, y, z]
        combine_tensor = npu_combine_tensors(lst)

        x_storage_size = torch_npu.get_storage_size(x)
        y_storage_size = torch_npu.get_storage_size(y)
        z_storage_size = torch_npu.get_storage_size(z)
        combine_tensor_storage_size = torch_npu.get_storage_size(combine_tensor)

        # test if combine_tensor is contiguous, and x,y,z are moved into the combine_tensor.
        self.assertEqual(True, combine_tensor.is_contiguous())
        self.assertEqual(combine_tensor.data_ptr(), x.data_ptr())
        self.assertEqual(x.data_ptr() + x_storage_size * x.element_size(), y.data_ptr())
        self.assertEqual(y.data_ptr() + y_storage_size * y.element_size(), z.data_ptr())
        self.assertEqual(combine_tensor_storage_size, x_storage_size + y_storage_size + z_storage_size)

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_computation(self, dtype, device="npu"):
        x = torch.zeros((2, 2, 2, 2), device=device, dtype=dtype)
        y = torch.zeros((4, 4), device=device, dtype=dtype)
        z = torch.zeros((3, 3, 3), device=device, dtype=dtype)

        lst = [x, y, z]
        combine_tensor = npu_combine_tensors(lst)

        combine_tensor += 2

        self.assertEqual(32, x.sum())
        self.assertEqual(32, y.sum())
        self.assertEqual(54, z.sum())

        for tensor in lst:
            tensor.mul_(2)

        self.assertEqual(236, combine_tensor.sum().long().item())
        self.assertEqual(combine_tensor.sum().long().item(), (x.sum() + y.sum() + z.sum()).long().item())

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_get_part_combined_tensor(self, dtype, device="npu"):
        x = torch.randn((2, 2, 2, 2), device=device, dtype=dtype)
        y = torch.randn((4, 4), device=device, dtype=dtype)
        z = torch.randn((3, 3, 3), device=device, dtype=dtype)

        lst = [x, y, z]
        combine_tensor = npu_combine_tensors(lst)

        x_storage_size = torch_npu.get_storage_size(x)
        y_storage_size = torch_npu.get_storage_size(y)
        z_storage_size = torch_npu.get_storage_size(z)
        part_tensor_x = get_part_combined_tensor(combine_tensor, 0, x_storage_size)
        part_tensor_y = get_part_combined_tensor(combine_tensor, x_storage_size, y_storage_size)
        part_tensor_z = get_part_combined_tensor(
            combine_tensor, x_storage_size + y_storage_size, z_storage_size)

        self.assertEqual(part_tensor_x.reshape_as(x), x)
        self.assertEqual(part_tensor_y.reshape_as(y), y)
        self.assertEqual(part_tensor_z.reshape_as(z), z)

    @Dtypes(torch.half, torch.float, torch.bfloat16)
    def test_is_combined_tensor_valid(self, dtype, device="npu"):
        x = torch.randn((2, 2, 2, 2), device=device, dtype=dtype)
        y = torch.randn((4, 4), device=device, dtype=dtype)
        z = torch.randn((3, 3, 3), device=device, dtype=dtype)

        lst = [x, y, z]
        combine_tensor = npu_combine_tensors(lst)
        self.assertEqual(True, is_combined_tensor_valid(combine_tensor, lst))
        self.assertEqual(False, is_combined_tensor_valid(combine_tensor, lst + [None]))
        self.assertEqual(False, is_combined_tensor_valid(combine_tensor, [x.clone(), y.clone(), z.clone()]))

        lst = []
        combine_tensor = npu_combine_tensors(lst)
        self.assertEqual(True, is_combined_tensor_valid(combine_tensor, lst))


if __name__ == '__main__':
    run_tests()
