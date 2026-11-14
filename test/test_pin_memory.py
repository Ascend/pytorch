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

"""Validate data-loading utility APIs on NPU (extendable).

Directly cover torch.utils.data._utils.pin_memory.pin_memory branches that
DataLoader integration tests do not fully exercise. Pinned tensors remain on
CPU by contract; the transfer test separately verifies their NPU consumption.
"""

from collections import OrderedDict, defaultdict, deque, namedtuple
from collections.abc import Mapping, Sequence

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data._utils.pin_memory import pin_memory


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

# Fixed expectations for the v2.11.0 source contract.
TUPLE_RESULT_TYPE = tuple
CUSTOM_METHOD_FIRST = True
FORWARDS_DEVICE = False


class FrozenMapping(Mapping):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class NonReconstructibleMapping(FrozenMapping):
    def __init__(self, values, marker):
        super().__init__(values)
        self.marker = marker


class NonCopyableMapping(dict):
    def __copy__(self):
        raise TypeError("copy unsupported")


class NonUpdatableMapping(dict):
    def update(self, *args, **kwargs):
        raise TypeError("update unsupported")


class TaggedMapping(dict):
    pass


class FrozenSequence(Sequence):
    def __init__(self, values):
        self.values = tuple(values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class NonReconstructibleSequence(FrozenSequence):
    def __init__(self, values, marker):
        super().__init__(values)
        self.marker = marker


class NonCopyableSequence(list):
    def __copy__(self):
        raise TypeError("copy unsupported")


class NonAssignableSequence(list):
    def __setitem__(self, index, value):
        raise TypeError("assignment unsupported")


class TaggedSequence(list):
    pass


class CustomBatch:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def pin_memory(self):
        self.calls += 1
        return self.result


class CustomMapping(dict):
    def pin_memory(self):
        return "mapping custom result"


class CustomSequence(list):
    def pin_memory(self):
        return "sequence custom result"


class RaisingBatch:
    def pin_memory(self):
        raise RuntimeError("custom pin failure")


class TestPinMemory(TestCase):
    def _assert_pinned_tensor(self, actual, source):
        self.assertEqual(device_type, "npu")
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, source.dtype)
        self.assertEqual(actual.shape, source.shape)
        self.assertEqual(actual.stride(), source.stride())
        self.assertEqual(actual, source)
        # Empty tensors have no data region whose pinning can be asserted.
        if source.numel():
            self.assertTrue(actual.is_pinned())

    def test_tensor_types_shapes_and_strides(self):
        dtypes = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                  torch.int64, torch.float16, torch.bfloat16, torch.float32,
                  torch.float64, torch.complex64, torch.complex128)
        for dtype in dtypes:
            base = torch.arange(12).to(dtype).reshape(3, 4)
            for source in (base, base.t(), base[:, ::2], base[0, 0], base[:0]):
                with self.subTest(dtype=dtype, shape=source.shape, stride=source.stride()):
                    actual = pin_memory(source)
                    self._assert_pinned_tensor(actual, source)
                    if source.numel():
                        self.assertFalse(source.is_pinned())
                        self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    def test_optional_device(self):
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self._assert_pinned_tensor(pin_memory(source), source)
        for device in (None, "npu", torch.device("npu")):
            with self.subTest(device=device):
                self._assert_pinned_tensor(pin_memory(source, device), source)
                self._assert_pinned_tensor(pin_memory(data=source, device=device), source)

    def test_invalid_device_argument(self):
        source = torch.arange(3, dtype=torch.float32)
        for device, error in ((object(), TypeError), ("cpu", RuntimeError), ("", RuntimeError)):
            with self.subTest(device_type=type(device).__name__):
                if FORWARDS_DEVICE:
                    with self.assertRaises(error):
                        pin_memory(source, device=device)
                else:
                    self._assert_pinned_tensor(pin_memory(source, device=device), source)

    def test_pinned_input_reuses_storage(self):
        source = torch.arange(8, dtype=torch.float32).pin_memory()
        actual = pin_memory(source)
        self._assert_pinned_tensor(actual, source)
        self.assertEqual(actual.data_ptr(), source.data_ptr())

    def test_transfer_to_npu(self):
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        pinned = pin_memory(source)
        self._assert_pinned_tensor(pinned, source)
        actual = pinned.to(device_type, non_blocking=True)
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.cpu(), source)

    def test_npu_input_rejected(self):
        self.assertEqual(device_type, "npu")
        source = torch.arange(4, dtype=torch.float32).to(device_type)
        with self.assertRaises(RuntimeError):
            pin_memory(source)

    def test_sparse_input_rejected(self):
        self.assertEqual(device_type, "npu")
        source = torch.sparse_coo_tensor([[0, 1]], [1.0, 2.0], (3,))
        with self.assertRaises(RuntimeError):
            pin_memory(source)

    def test_nested_containers_and_input_unchanged(self):
        point_type = namedtuple("Point", "tensor label")
        tensor = torch.arange(4, dtype=torch.float32)
        source = {"nested": [point_type(tensor, "label"), (tensor, b"bytes")],
                  "none": None, "number": 7}
        actual = pin_memory(source)
        self.assertIsNot(actual, source)
        self.assertIsNot(actual["nested"], source["nested"])
        self.assertIsInstance(actual["nested"][0], point_type)
        self.assertEqual(actual["nested"][0].label, "label")
        self.assertIsInstance(actual["nested"][1], TUPLE_RESULT_TYPE)
        self.assertEqual(actual["nested"][1][1], b"bytes")
        self.assertIsNone(actual["none"])
        self.assertEqual(actual["number"], 7)
        self._assert_pinned_tensor(actual["nested"][0].tensor, tensor)
        self._assert_pinned_tensor(actual["nested"][1][0], tensor)
        self.assertIs(source["nested"][0].tensor, tensor)
        self.assertIsInstance(source["nested"][1], tuple)
        self.assertFalse(tensor.is_pinned())

    def test_mapping_types_and_metadata(self):
        tensor = torch.arange(3, dtype=torch.float32)
        tagged = TaggedMapping(value=tensor)
        tagged.label = "preserve me"
        sources = (dict(value=tensor), OrderedDict(value=tensor),
                   defaultdict(list, value=tensor), tagged, FrozenMapping({"value": tensor}))
        for source in sources:
            with self.subTest(mapping=type(source).__name__):
                actual = pin_memory(source)
                self.assertIsInstance(actual, type(source))
                self.assertIsNot(actual, source)
                self.assertEqual(list(actual), ["value"])
                self._assert_pinned_tensor(actual["value"], tensor)
                self.assertIs(source["value"], tensor)
                if isinstance(source, defaultdict):
                    self.assertIs(actual.default_factory, list)
                if isinstance(source, TaggedMapping):
                    self.assertEqual(actual.label, "preserve me")

    def test_mapping_fallback_pins_values(self):
        tensor = torch.arange(3, dtype=torch.float32)
        for source in (NonCopyableMapping(value=tensor), NonUpdatableMapping(value=tensor),
                       NonReconstructibleMapping({"value": tensor}, "marker")):
            with self.subTest(mapping=type(source).__name__):
                actual = pin_memory(source)
                self.assertIs(type(actual), dict)
                self._assert_pinned_tensor(actual["value"], tensor)
                self.assertIs(source["value"], tensor)

    def test_sequence_types_and_metadata(self):
        tensor = torch.arange(3, dtype=torch.float32)
        tagged = TaggedSequence([tensor, "text"])
        tagged.label = "preserve me"
        sources = ([tensor, "text"], tagged, deque([tensor, "text"], maxlen=3),
                   FrozenSequence([tensor, "text"]))
        for source in sources:
            with self.subTest(sequence=type(source).__name__):
                actual = pin_memory(source)
                self.assertIsInstance(actual, type(source))
                self.assertIsNot(actual, source)
                self._assert_pinned_tensor(actual[0], tensor)
                self.assertEqual(actual[1], "text")
                self.assertIs(source[0], tensor)
                if isinstance(source, deque):
                    self.assertEqual(actual.maxlen, 3)
                if isinstance(source, TaggedSequence):
                    self.assertEqual(actual.label, "preserve me")

    def test_sequence_fallback_pins_values(self):
        tensor = torch.arange(3, dtype=torch.float32)
        for source in (NonCopyableSequence([tensor]), NonAssignableSequence([tensor]),
                       NonReconstructibleSequence([tensor], "marker")):
            with self.subTest(sequence=type(source).__name__):
                actual = pin_memory(source)
                self.assertIs(type(actual), list)
                self._assert_pinned_tensor(actual[0], tensor)
                self.assertIs(source[0], tensor)

    def test_passthrough_objects(self):
        values = (None, True, 1, 1.5, 2j, "text", b"bytes", object(),
                  {1, 2}, frozenset({1}), (value for value in range(2)))
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                self.assertIs(pin_memory(value), value)
                self.assertIs(pin_memory(value, device="npu"), value)

    def test_empty_containers_and_tuple_contract(self):
        point_type = namedtuple("Point", "x y")
        for source in ({}, [], (), OrderedDict(), FrozenMapping({}), FrozenSequence([])):
            actual = pin_memory(source)
            expected_type = TUPLE_RESULT_TYPE if isinstance(source, tuple) else type(source)
            self.assertIsInstance(actual, expected_type)
            self.assertEqual(len(actual), 0)
        actual = pin_memory((1, "text"))
        self.assertIs(type(actual), TUPLE_RESULT_TYPE)
        self.assertEqual(list(actual), [1, "text"])
        point = pin_memory(point_type(1, "text"))
        self.assertIsInstance(point, point_type)
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, "text")

    def test_mapping_fallback_without_tensors(self):
        for source in (NonCopyableMapping(value="text"), NonUpdatableMapping(value="text"),
                       NonReconstructibleMapping({"value": "text"}, "marker")):
            actual = pin_memory(source)
            self.assertIs(type(actual), dict)
            self.assertEqual(actual, {"value": "text"})

    def test_sequence_fallback_without_tensors(self):
        sources = (NonCopyableSequence([1, "text"]), NonAssignableSequence([1, "text"]),
                   NonReconstructibleSequence([1, "text"], "marker"), range(3))
        for source in sources:
            actual = pin_memory(source)
            self.assertIs(type(actual), list)
            self.assertEqual(actual, list(source))

    def test_custom_method_called_without_device(self):
        expected = object()
        for device in (None, "npu", object()):
            source = CustomBatch(expected)
            self.assertIs(pin_memory(source, device=device), expected)
            self.assertEqual(source.calls, 1)

    def test_custom_method_container_precedence(self):
        for source, expected in ((CustomMapping(value="text"), "mapping custom result"),
                                 (CustomSequence([1, "text"]), "sequence custom result")):
            actual = pin_memory(source)
            if CUSTOM_METHOD_FIRST:
                self.assertEqual(actual, expected)
            else:
                self.assertIsInstance(actual, type(source))
                self.assertIsNot(actual, source)
                self.assertEqual(actual, source)

    def test_custom_method_exception_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "custom pin failure"):
            pin_memory(RaisingBatch())
        non_callable = type("InvalidBatch", (), {"pin_memory": 1})()
        with self.assertRaises(TypeError):
            pin_memory(non_callable)

    def test_invalid_call_signature(self):
        with self.assertRaises(TypeError):
            pin_memory()
        with self.assertRaises(TypeError):
            pin_memory(None, None, None)
        with self.assertRaises(TypeError):
            pin_memory(None, unexpected=True)


if __name__ == "__main__":
    run_tests()
