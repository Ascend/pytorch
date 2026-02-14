
import unittest
import copy
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestStorage(TestCase):

    def test_storage_method(self):
        # The commented out part are unsupported data types by operators.
        storage_types = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.half,
            torch.float32,
            torch.float64,
            torch.bool,
            torch.uint8,
            torch.bfloat16,
            # torch.cdouble,
            # torch.cfloat,
            # torch.qint8,
            # torch.qint32,
            # torch.quint8,
            # torch.quint4x2,
            # torch.quint2x4,
        ]
        for dtype in storage_types:

            def _test_cpu(cpu_storage, npu_storage):
                npu_res = npu_storage.cpu()
                self.assertEqual(cpu_storage, npu_res)

            def _test_npu(cpu_storage, npu_storage):
                cpu_res = cpu_storage.npu().cpu()
                npu_res = npu_storage.cpu()
                self.assertEqual(cpu_res, npu_res)

            def _test_clone(cpu_storage, npu_storage):
                cpu_res = cpu_storage.clone()
                npu_res = npu_storage.clone().cpu()
                self.assertEqual(cpu_res, npu_res)

            def _test_copy_(cpu_storage, npu_storage):
                cpu_res = torch.ones([3, 1, 2, 2]).to(dtype).storage()
                npu_res = torch.zeros([3, 1, 2, 2]).npu().to(dtype).storage()

                cpu_res.copy_(cpu_storage)
                npu_res.copy_(npu_storage)
                self.assertEqual(cpu_res, npu_res.cpu())

            def _test_untyped(cpu_storage, npu_storage):
                cpu_res = cpu_storage.untyped()
                npu_res = npu_storage.untyped()
                if dtype == torch.float64:
                    self.assertEqual(cpu_storage.float().untyped(), npu_res)
                else:
                    self.assertEqual(cpu_res, npu_res.cpu())

            def _test_data_ptr(cpu_storage, npu_storage):
                cpu_res = cpu_storage.data_ptr()
                npu_res = npu_storage.data_ptr()
                self.assertNotEqual(npu_res, None)
                self.assertNotEqual(cpu_res, npu_res)

            def _test_element_size(cpu_storage, npu_storage):
                cpu_res = cpu_storage.element_size()
                npu_res = npu_storage.element_size()
                if dtype == torch.float64:
                    self.assertEqual(cpu_res, npu_res * 2)
                else:
                    self.assertEqual(cpu_res, npu_res)

            def _test_fill_(cpu_storage, npu_storage):
                cpu_storage.fill_(1)
                npu_storage.fill_(1)
                npu_res = npu_storage.cpu()
                self.assertEqual(cpu_storage, npu_res)

            def _test_get_device(cpu_storage, npu_storage):
                npu_res = npu_storage.get_device()
                self.assertEqual(npu_res, 0)

            def _test_is_pinned(cpu_storage, npu_storage):
                cpu_res = cpu_storage.is_pinned()
                npu_res = npu_storage.is_pinned()
                self.assertEqual(cpu_res, False)
                self.assertEqual(npu_res, False)
                cpu_res = cpu_storage.is_pinned("npu")
                npu_res = npu_storage.is_pinned("npu")
                self.assertEqual(cpu_res, False)
                self.assertEqual(npu_res, False)

            def _test_pin_memory(cpu_storage, npu_storage):
                ori_ptr = cpu_storage.data_ptr()
                cpu_pin_storage = cpu_storage.pin_memory("npu")
                self.assertEqual(cpu_storage.data_ptr(), ori_ptr)
                self.assertNotEqual(cpu_pin_storage.data_ptr(), ori_ptr)
                self.assertEqual(cpu_storage.is_pinned("npu"), False)
                self.assertEqual(cpu_pin_storage.is_pinned("npu"), True)

            def _test_nbytes(cpu_storage, npu_storage):
                cpu_res = cpu_storage.nbytes()
                npu_res = npu_storage.nbytes()
                if dtype == torch.float64:
                    self.assertEqual(cpu_res, npu_res * 2)
                else:
                    self.assertEqual(cpu_res, npu_res)

            def _test_pickle_storage_type(cpu_storage, npu_storage):
                cpu_res = cpu_storage.pickle_storage_type()
                npu_res = npu_storage.pickle_storage_type()
                if dtype == torch.float64:
                    self.assertEqual(npu_res, "FloatStorage")
                else:
                    self.assertEqual(cpu_res, npu_res)

            def _test_size(cpu_storage, npu_storage):
                cpu_res = cpu_storage.size()
                npu_res = npu_storage.size()
                self.assertEqual(cpu_res, npu_res)

            def _test_tolist(cpu_storage, npu_storage):
                cpu_res = cpu_storage.tolist()
                npu_res = npu_storage.tolist()
                self.assertEqual(cpu_res, npu_res)

            def _test_resize_(cpu_storage, npu_storage):
                cpu_ori_ptr = cpu_storage.data_ptr()
                npu_ori_ptr = npu_storage.data_ptr()
                cpu_storage.resize_(24)
                npu_storage.resize_(24)
                self.assertEqual(cpu_storage.size(), npu_storage.size())
                self.assertEqual(cpu_storage.tolist()[:12], npu_storage.tolist()[:12])
                self.assertNotEqual(cpu_storage.data_ptr(), cpu_ori_ptr)
                self.assertNotEqual(npu_storage.data_ptr(), npu_ori_ptr)

                cpu_ori_ptr = cpu_storage.data_ptr()
                npu_ori_ptr = npu_storage.data_ptr()
                cpu_storage.resize_(8)
                npu_storage.resize_(8)
                self.assertEqual(cpu_storage.size(), npu_storage.size())
                self.assertEqual(cpu_storage.tolist(), npu_storage.tolist())
                self.assertNotEqual(cpu_storage.data_ptr(), cpu_ori_ptr)
                self.assertNotEqual(npu_storage.data_ptr(), npu_ori_ptr)

                cpu_storage.resize_(0)
                npu_storage.resize_(0)
                self.assertEqual(cpu_storage.size(), npu_storage.size())
                cpu_storage.resize_(16)
                npu_storage.resize_(16)
                cpu_storage.fill_(4)
                npu_storage.fill_(4)

            def _test_is_shared(cpu_storage, npu_storage):
                cpu_res = cpu_storage.is_shared()
                npu_res = npu_storage.is_shared()
                self.assertEqual(cpu_res, False)
                self.assertEqual(npu_res, False)

            def _test_share_memory_(cpu_storage, npu_storage):
                npu_ori_ptr = npu_storage.data_ptr()
                cpu_res = cpu_storage.share_memory_()
                npu_res = npu_storage.share_memory_()
                self.assertEqual(npu_storage.data_ptr(), npu_ori_ptr)
                self.assertEqual(cpu_res.is_shared(), True)
                self.assertEqual(npu_res.is_shared(), False)

            def _test_dtype(cpu_storage, npu_storage):
                cpu_res = cpu_storage.dtype
                npu_res = npu_storage.dtype
                if cpu_res == torch.float64:
                    self.assertEqual(npu_res, torch.float32)
                else:
                    self.assertEqual(npu_res, cpu_res)

            def _test_device(cpu_storage, npu_storage):
                cpu_res = cpu_storage.device
                npu_res = npu_storage.device
                self.assertEqual(cpu_res.type, "cpu")
                self.assertEqual(npu_res.type, "npu")

            def _test_datatype_cast(cpu_storage, npu_storage):
                dtypes = [
                    "bool",
                    "double",
                    "float",
                    "half",
                    "long",
                    "int",
                    "short",
                    "byte",
                    "char",
                    # "bfloat16",
                ]
                for dt in dtypes:
                    cpu_res = eval("cpu_storage" + "." + dt + "()")
                    npu_res = eval("npu_storage" + "." + dt + "()")
                    self.assertEqual(cpu_res.size(), npu_res.size())
                    self.assertEqual(cpu_res, npu_res.cpu())
                    self.assertEqual(cpu_res.tolist(), npu_res.cpu().tolist())
            
            @SupportedDevices(['Ascend910B'])
            def _test_datatype_cast_complex(cpu_storage, npu_storage):
                dtypes = [
                    "complex_double",
                    "complex_float",
                ]
                for dt in dtypes:
                    cpu_res = eval("cpu_storage" + "." + dt + "()")
                    npu_res = eval("npu_storage" + "." + dt + "()")
                    self.assertEqual(cpu_res.size(), npu_res.size())

            def _test_from_buffer(cpu_storage, npu_storage):
                cpu_list = [2, 3, 3, 2, 5]
                cpu_buffer = bytearray(cpu_list)
                cpu_res = torch.ByteStorage.from_buffer(cpu_buffer)
                self.assertEqual(cpu_res.tolist(), cpu_list)

            ''' test TypedStorage, FloatStorage and so on '''
            cpu_tensor = torch.randn([3, 1, 2, 2])
            npu_tensor = cpu_tensor.npu()
            cpu_storage = cpu_tensor.to(dtype).storage()
            npu_storage = npu_tensor.to(dtype).storage()

            _test_fill_(cpu_storage, npu_storage)
            _test_cpu(cpu_storage, npu_storage)
            _test_npu(cpu_storage, npu_storage)
            _test_clone(cpu_storage, npu_storage)
            _test_copy_(cpu_storage, npu_storage)
            _test_untyped(cpu_storage, npu_storage)
            _test_data_ptr(cpu_storage, npu_storage)
            _test_element_size(cpu_storage, npu_storage)
            _test_fill_(cpu_storage, npu_storage)
            _test_get_device(cpu_storage, npu_storage)
            _test_is_pinned(cpu_storage, npu_storage)
            _test_pin_memory(cpu_storage, npu_storage)
            _test_nbytes(cpu_storage, npu_storage)
            _test_pickle_storage_type(cpu_storage, npu_storage)
            _test_size(cpu_storage, npu_storage)
            _test_tolist(cpu_storage, npu_storage)
            _test_resize_(cpu_storage, npu_storage)
            _test_is_shared(cpu_storage, npu_storage)
            _test_share_memory_(cpu_storage, npu_storage)
            _test_dtype(cpu_storage, npu_storage)
            _test_device(cpu_storage, npu_storage)
            _test_datatype_cast(cpu_storage, npu_storage)
            _test_datatype_cast_complex(cpu_storage, npu_storage)
            _test_from_buffer(cpu_storage, npu_storage)

            ''' test untyped storage only on a certain data type'''
            if dtype == torch.int8:
                def _test_mps(cpu_storage, npu_storage):
                    npu_res = npu_storage.mps()
                    cpu_res = cpu_storage.mps()
                    self.assertEqual(cpu_res, npu_res)

                def _test_new(cpu_storage, npu_storage):
                    npu_res = npu_storage.new()
                    cpu_res = cpu_storage.new()
                    self.assertEqual(cpu_res.size(), npu_res.size())
                    self.assertEqual(cpu_res.type(), npu_res.type())
                    self.assertEqual(cpu_res.device.type, cpu_storage.device.type)
                    self.assertEqual(npu_res.device.type, npu_storage.device.type)

                def _test_type(cpu_storage, npu_storage):
                    npu_res = npu_storage.type()
                    self.assertEqual(npu_res, "torch.storage.UntypedStorage")

                def _test_copy_(cpu_storage, npu_storage):
                    cpu_res = torch.ones([3, 1, 2, 2]).untyped_storage()
                    npu_res = torch.zeros([3, 1, 2, 2]).npu().untyped_storage()

                    cpu_res.copy_(cpu_storage)
                    npu_res.copy_(npu_storage)
                    self.assertEqual(cpu_res, npu_res.cpu())

                cpu_tensor = torch.randn([3, 1, 2, 2])
                npu_tensor = cpu_tensor.npu()
                cpu_storage = cpu_tensor.untyped_storage()
                npu_storage = npu_tensor.untyped_storage()

                ''' test typed storage '''
                _test_new(cpu_storage, npu_storage)
                _test_cpu(cpu_storage, npu_storage)
                _test_npu(cpu_storage, npu_storage)
                _test_clone(cpu_storage, npu_storage)
                _test_copy_(cpu_storage, npu_storage)
                _test_untyped(cpu_storage, npu_storage)
                _test_data_ptr(cpu_storage, npu_storage)
                _test_element_size(cpu_storage, npu_storage)
                _test_fill_(cpu_storage, npu_storage)
                _test_get_device(cpu_storage, npu_storage)
                _test_is_pinned(cpu_storage, npu_storage)
                _test_pin_memory(cpu_storage, npu_storage)
                _test_nbytes(cpu_storage, npu_storage)
                _test_size(cpu_storage, npu_storage)
                _test_tolist(cpu_storage, npu_storage)
                _test_type(cpu_storage, npu_storage)
                _test_resize_(cpu_storage, npu_storage)
                _test_is_shared(cpu_storage, npu_storage)
                _test_share_memory_(cpu_storage, npu_storage)
                _test_device(cpu_storage, npu_storage)
                _test_datatype_cast(cpu_storage, npu_storage)
                _test_datatype_cast_complex(cpu_storage, npu_storage)
                _test_from_buffer(cpu_storage, npu_storage)
                with self.assertRaisesRegex(RuntimeError, "Storage device not recognized: mps"):
                    _test_mps(cpu_storage, npu_storage)

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        supported_dtypes = ["float", "half", "long", "short", "int", "bool", "char", "byte"]

        for dtype in supported_dtypes:
            self.assertIsInstance(getattr(x.npu(), dtype)(), getattr(torch.npu, dtype.title() + "Tensor"))
            self.assertIsInstance(getattr(x.float().cpu(), dtype)(), getattr(torch, dtype.title() + "Tensor"))

        y = x.storage()
        for dtype in supported_dtypes:
            self.assertIsInstance(getattr(y.npu(), dtype)(), getattr(torch.npu, dtype.title() + "Storage"))
            self.assertIsInstance(getattr(y.float().cpu(), dtype)(), getattr(torch, dtype.title() + "Storage"))
    
    @unittest.skip("Temporarily disabled")
    def test_deepcopy(self):
        x = torch.tensor([1])
        y = copy.deepcopy(x)

        x = torch.tensor([1]).npu()
        y = copy.deepcopy(x)
        self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())

        x = torch.rand(3, 3).npu()
        x = torch_npu.npu_format_cast(x, 29)
        y = copy.deepcopy(x)
        self.assertEqual(torch_npu.get_npu_format(y), 29)
        self.assertEqual(x, y)


if __name__ == '__main__':
    run_tests()
