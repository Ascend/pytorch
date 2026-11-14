# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for Tensor subclass APIs on NPU:
1. Community tests lack complete argument coverage for storage-backed subclasses.
2. This file validates torch.Tensor._make_subclass (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._python_dispatch import TorchDispatchMode

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class StorageTensor(torch.Tensor):
    pass


class ChildTensor(StorageTensor):
    pass


class TestMakeSubclass(TestCase):
    def test_storage_and_metadata(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16,
                      torch.int32, torch.int64, torch.bool):
            base = torch.ones((4, 6), dtype=dtype, device=device_type)
            for data in (base, base[1:, 1::2], base[:0], base[0, 0]):
                for cls in (torch.Tensor, StorageTensor, ChildTensor):
                    with self.subTest(dtype=dtype, shape=data.shape, cls=cls):
                        result = torch.Tensor._make_subclass(cls, data)
                        self.assertIs(type(result), cls)
                        self.assertEqual(result.device, data.device)
                        self.assertEqual(result.dtype, data.dtype)
                        self.assertEqual(result.size(), data.size())
                        self.assertEqual(result.stride(), data.stride())
                        self.assertEqual(result.storage_offset(), data.storage_offset())
                        self.assertEqual(result.untyped_storage().data_ptr(),
                                         data.untyped_storage().data_ptr())
                        self.assertEqual(result.as_subclass(torch.Tensor), data)
                        self.assertFalse(result.requires_grad)
                        self.assertTrue(result.is_leaf)

    def test_aliasing_and_subclass_input(self):
        base = torch.arange(12, dtype=torch.float32, device=device_type).reshape(3, 4)
        first = torch.Tensor._make_subclass(StorageTensor, base[:, 1:])
        second = torch.Tensor._make_subclass(ChildTensor, first)
        second.add_(2)
        self.assertEqual(first.as_subclass(torch.Tensor), base[:, 1:])
        self.assertEqual(second.as_subclass(torch.Tensor), base[:, 1:])
        self.assertEqual(base[:, 1:],
                         torch.tensor([[3, 4, 5], [7, 8, 9], [11, 12, 13]],
                                      dtype=torch.float32, device=device_type))

    def test_requires_grad_and_autograd(self):
        source = torch.ones(4, device=device_type, requires_grad=True)
        data = source * 3
        for require_grad in (False, True):
            for keyword in (False, True):
                with self.subTest(require_grad=require_grad, keyword=keyword):
                    if keyword:
                        result = torch.Tensor._make_subclass(
                            cls=StorageTensor, data=data, require_grad=require_grad)
                    else:
                        result = torch.Tensor._make_subclass(StorageTensor, data, require_grad)
                    self.assertEqual(result.requires_grad, require_grad)
                    self.assertTrue(result.is_leaf)
                    self.assertIsNone(result.grad_fn)
                    if require_grad:
                        (result * 2).sum().backward()
                        self.assertEqual(result.grad, torch.full_like(source, 2))
                        self.assertEqual(result.grad.device, source.device)
                        self.assertIsNone(source.grad)
        result = torch.Tensor._make_subclass(StorageTensor, data)
        self.assertFalse(result.requires_grad)
        self.assertTrue(data.requires_grad)
        for dtype in (torch.int64, torch.bool):
            with self.assertRaisesRegex(RuntimeError, "floating point|complex"):
                torch.Tensor._make_subclass(
                    StorageTensor, torch.ones(2, dtype=dtype, device=device_type), True)

    def test_backend_keys(self):
        data = torch.ones(3, device=device_type)
        for backend in (None, device_type, str(data.device), data.device, data.device.index):
            with self.subTest(backend=backend):
                result = torch.Tensor._make_subclass(
                    StorageTensor, data, device_for_backend_keys=backend)
                self.assertEqual(result.device, data.device)
                self.assertEqual(torch._C._dispatch_keys(result).raw_repr(),
                                 torch._C._dispatch_keys(data).raw_repr())
                self.assertEqual((result + 2).as_subclass(torch.Tensor), data + 2)
                self.assertEqual(result.data_ptr(), data.data_ptr())

    def test_backend_override_preserves_storage(self):
        data = torch.ones(3, device=device_type)
        for backend, key in (("cpu", torch._C.DispatchKey.CPU),
                             ("cuda", torch._C.DispatchKey.CUDA),
                             ("meta", torch._C.DispatchKey.Meta)):
            with self.subTest(backend=backend):
                result = torch.Tensor._make_subclass(
                    StorageTensor, data, device_for_backend_keys=backend)
                self.assertTrue(torch._C._dispatch_keys(result).has(key))
                self.assertEqual(result.device, data.device)
                self.assertEqual(result.untyped_storage().device, data.device)
                self.assertEqual(result.data_ptr(), data.data_ptr())
                # Only inspect metadata: overriding keys does not move NPU storage.

    def test_metadata_dispatch_options(self):
        data = torch.ones((3, 2), device=device_type)
        calls = []

        class MetadataTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                calls.append(func)
                responses = {
                    torch.ops.aten.dim.default: 2,
                    torch.ops.aten.sym_size.default: (3, 2),
                    torch.ops.aten.sym_stride.default: (2, 1),
                    torch.ops.prim.device.default: data.device,
                    torch.ops.prim.layout.default: data.layout,
                }
                return responses.get(func, NotImplemented)

        for policy in (None, "strides", "sizes"):
            for dispatch_device in (False, True):
                for dispatch_layout in (False, True):
                    with self.subTest(policy=policy, device=dispatch_device, layout=dispatch_layout):
                        result = torch.Tensor._make_subclass(
                            MetadataTensor, data, dispatch_sizes_strides_policy=policy,
                            dispatch_device=dispatch_device, dispatch_layout=dispatch_layout)
                        calls.clear()
                        self.assertEqual(result.size(), data.size())
                        self.assertEqual(result.stride(), data.stride())
                        self.assertEqual(result.device, data.device)
                        self.assertEqual(result.layout, data.layout)
                        self.assertEqual(torch.ops.aten.sym_size.default in calls, policy == "sizes")
                        self.assertEqual(torch.ops.aten.sym_stride.default in calls, policy is not None)
                        self.assertEqual(torch.ops.prim.device.default in calls, dispatch_device)
                        self.assertEqual(torch.ops.prim.layout.default in calls, dispatch_layout)

    def test_dispatch_defaults_and_missing_handlers(self):
        class UnhandledTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        data = torch.ones((3, 2), device=device_type)
        result = torch.Tensor._make_subclass(UnhandledTensor, data)
        self.assertEqual(result.size(), data.size())
        self.assertEqual(result.stride(), data.stride())
        self.assertEqual(result.device, data.device)
        self.assertEqual(result.layout, data.layout)
        cases = (
            ({"dispatch_sizes_strides_policy": "sizes"}, lambda x: x.size()),
            ({"dispatch_sizes_strides_policy": "strides"}, lambda x: x.stride()),
            ({"dispatch_device": True}, lambda x: x.device),
            ({"dispatch_layout": True}, lambda x: x.layout),
        )
        for options, query in cases:
            with self.subTest(options=options):
                result = torch.Tensor._make_subclass(UnhandledTensor, data, **options)
                with self.assertRaisesRegex(TypeError, "Multiple dispatch failed"):
                    query(result)

    def test_modes_are_stashed(self):
        calls = []

        class RecordingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                calls.append(func)
                return func(*args, **(kwargs or {}))

        data = torch.ones(2, device=device_type)
        with RecordingMode():
            result = torch.Tensor._make_subclass(StorageTensor, data)
            self.assertEqual(calls, [])
            value = result + 1
        self.assertTrue(calls)
        self.assertEqual(value.as_subclass(torch.Tensor), data + 1)

    def test_inference_and_no_grad(self):
        with torch.inference_mode():
            data = torch.ones(2, device=device_type)
            result = torch.Tensor._make_subclass(StorageTensor, data)
            self.assertTrue(result.is_inference())
            self.assertEqual(result.device, data.device)
            self.assertEqual(result.data_ptr(), data.data_ptr())
        with torch.no_grad():
            data = torch.ones(2, device=device_type)
            result = torch.Tensor._make_subclass(StorageTensor, data, True)
            self.assertTrue(result.requires_grad)
            self.assertTrue(result.is_leaf)

    def test_invalid_arguments(self):
        data = torch.ones(2, device=device_type)
        for cls in (None, 1, "Tensor"):
            with self.subTest(cls=cls), self.assertRaises(TypeError):
                torch.Tensor._make_subclass(cls, data)
        with self.assertRaisesRegex((TypeError, RuntimeError), "Tensor|subclass"):
            torch.Tensor._make_subclass(object, data)
        for invalid_data in (None, 1, [1, 2], (1, 2)):
            with self.subTest(data=invalid_data), self.assertRaises(TypeError):
                torch.Tensor._make_subclass(StorageTensor, invalid_data)
        for name in ("require_grad", "dispatch_device", "dispatch_layout",
                     "dispatch_sizes_strides_policy", "device_for_backend_keys"):
            with self.subTest(argument=name), self.assertRaises(TypeError):
                torch.Tensor._make_subclass(StorageTensor, data, **{name: []})
        for name in ("require_grad", "dispatch_device", "dispatch_layout"):
            for value in (None, 0, "True"):
                with self.subTest(argument=name, value=value), self.assertRaises(TypeError):
                    torch.Tensor._make_subclass(StorageTensor, data, **{name: value})
        for policy in ("", "invalid"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "sizes_strides_policy"):
                torch.Tensor._make_subclass(StorageTensor, data, dispatch_sizes_strides_policy=policy)
        with self.assertRaises((RuntimeError, ValueError)):
            torch.Tensor._make_subclass(StorageTensor, data, device_for_backend_keys="invalid")
        with self.assertRaises(TypeError):
            torch.Tensor._make_subclass(StorageTensor)
        with self.assertRaises(TypeError):
            torch.Tensor._make_subclass(StorageTensor, data, False, None)
        with self.assertRaises(TypeError):
            torch.Tensor._make_subclass(StorageTensor, data, unexpected=True)


if __name__ == "__main__":
    run_tests()
