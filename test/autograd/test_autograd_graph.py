# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
Add validation cases for torch.autograd.graph APIs on NPU:
1. Upstream direct cases omit iterable inputs, aliases, backward detection and invalid inputs.
2. This file validates torch.autograd.graph.increment_version (extendable).
"""
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestAutogradGraph(TestCase):
    def test_tensor_shapes_and_dtypes(self):
        for shape in ((), (0,), (2, 3)):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8,
                          torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                with self.subTest(shape=shape, dtype=dtype):
                    tensor = torch.zeros(shape, dtype=dtype).to(device_type)
                    self.assertEqual(tensor.device.type, "npu")
                    before = tensor.cpu().clone()
                    version = tensor._version
                    self.assertIsNone(torch.autograd.graph.increment_version(tensor=tensor))
                    self.assertEqual(tensor._version, version + 1)
                    self.assertTrue(torch.equal(tensor.cpu(), before))

    def test_iterable_inputs(self):
        for name, kind in (("list", list), ("tuple", tuple), ("iterator", iter),
                           ("generator", lambda values: (value for value in values))):
            with self.subTest(kind=name):
                first, second = torch.ones(2).to(device_type), torch.zeros(2).to(device_type)
                versions = first._version, second._version
                self.assertIsNone(torch.autograd.graph.increment_version(kind([first, second])))
                self.assertEqual((first._version, second._version), tuple(v + 1 for v in versions))
                self.assertIsNone(torch.autograd.graph.increment_version(kind([])))

    def test_repeated_and_duplicate_inputs(self):
        tensor = torch.zeros(3).to(device_type)
        version = tensor._version
        torch.autograd.graph.increment_version([tensor, tensor])
        self.assertEqual(tensor._version, version + 2)
        for _ in range(3):
            torch.autograd.graph.increment_version(tensor)
        self.assertEqual(tensor._version, version + 5)

    def test_view_detach_noncontiguous(self):
        base = torch.arange(6, dtype=torch.float32).to(device_type).reshape(2, 3)
        view, detached = base.t(), base.detach()
        self.assertFalse(view.is_contiguous())
        version = base._version
        torch.autograd.graph.increment_version(view)
        self.assertEqual((base._version, view._version, detached._version), (version + 1,) * 3)
        torch.autograd.graph.increment_version(detached)
        self.assertEqual((base._version, view._version, detached._version), (version + 2,) * 3)

    def test_inference_and_normal_mixture(self):
        with torch.inference_mode():
            inference = torch.ones(3).to(device_type)
            self.assertIsNone(torch.autograd.graph.increment_version(inference))
        normal = torch.ones(3).to(device_type)
        version = normal._version
        self.assertIsNone(torch.autograd.graph.increment_version([inference, normal]))
        self.assertEqual(normal._version, version + 1)
        with self.assertRaisesRegex(RuntimeError, "Inference tensors do not track version counter"):
            _ = inference._version

    def test_backward_change_detection(self):
        source = torch.ones(3).to(device_type).requires_grad_()
        output = (source * source).sum()
        torch.autograd.graph.increment_version(source)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            output.backward()

    def test_backward_without_increment(self):
        source = torch.ones(3).to(device_type).requires_grad_()
        (source * source).sum().backward()
        self.assertTrue(torch.equal(source.grad.cpu(), torch.full((3,), 2.0)))

    def test_mixed_devices(self):
        # Mixed-device iterable is an API input; only the reference tensor stays on CPU.
        cpu, npu = torch.ones(3), torch.ones(3).to(device_type)
        versions = cpu._version, npu._version
        torch.autograd.graph.increment_version([cpu, npu])
        self.assertEqual((cpu._version, npu._version), tuple(v + 1 for v in versions))

    def test_invalid_inputs_match_cpu(self):
        # CPU calls establish upstream exception types, rather than assuming a backend-specific type.
        factories = (lambda t: None, lambda t: 1, lambda t: [None],
                     lambda t: [1], lambda t: [t, None], lambda t: {"tensor": t})
        for index, factory in enumerate(factories):
            with self.subTest(input=index):
                with self.assertRaises(Exception) as cpu_error:
                    torch.autograd.graph.increment_version(factory(torch.ones(1)))
                with self.assertRaises(type(cpu_error.exception)):
                    torch.autograd.graph.increment_version(factory(torch.ones(1).to(device_type)))
        with self.assertRaises(TypeError):
            torch.autograd.graph.increment_version()
        with self.assertRaises(TypeError):
            torch.autograd.graph.increment_version(torch.ones(1).to(device_type), unexpected=True)

    def test_parameter_input(self):
        parameter = torch.nn.Parameter(torch.zeros(3).to(device_type))
        version = parameter._version
        self.assertIsNone(torch.autograd.graph.increment_version(parameter))
        self.assertEqual(parameter._version, version + 1)
        self.assertTrue(parameter.requires_grad)

    def test_clone_has_independent_counter(self):
        base = torch.zeros(3).to(device_type)
        clone = base.clone()
        base_version, clone_version = base._version, clone._version
        torch.autograd.graph.increment_version(clone)
        self.assertEqual(base._version, base_version)
        self.assertEqual(clone._version, clone_version + 1)


if __name__ == "__main__":
    run_tests()
