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

"""Validate distributed_c10d APIs without direct PyTorch community coverage."""

import pickle

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import TestCase, run_tests


TEST_NPU = torch.npu.is_available()


class TestDistributedC10dAPIs(TestCase):

    def setUp(self):
        super().setUp()
        if not TEST_NPU:
            self.skipTest("NPU unavailable")

        torch.npu.set_device(0)
        store = dist.HashStore()
        dist.init_process_group("hccl", store=store, rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)

    @staticmethod
    def _tensor_to_object_inputs():
        payload = {
            "api": "torch.distributed.distributed_c10d._tensor_to_object",
            "values": [1, None, "npu"],
        }
        tensor, tensor_size = c10d._object_to_tensor(
            payload, torch.device("npu"), None
        )
        return payload, tensor, tensor_size

    def _assert_tensor_to_object_error(
        self,
        cpu_tensor,
        npu_tensor,
        cpu_tensor_size,
        npu_tensor_size,
        exception_type,
        error_regex,
    ):
        messages = []
        for tensor, tensor_size in (
            (cpu_tensor, cpu_tensor_size),
            (npu_tensor, npu_tensor_size),
        ):
            with self.assertRaisesRegex(exception_type, error_regex) as error:
                c10d._tensor_to_object(tensor, tensor_size, None)
            messages.append(str(error.exception))
        self.assertEqual(messages[0], messages[1])

    def test_tensor_to_object_input_types(self):
        payload, npu_tensor, npu_tensor_size = self._tensor_to_object_inputs()
        cpu_tensor = npu_tensor.cpu()
        cpu_tensor_size = npu_tensor_size.cpu()
        tensor_size = int(npu_tensor_size.item())

        self.assertEqual(npu_tensor.device.type, "npu")
        self.assertEqual(npu_tensor.dtype, torch.uint8)
        self.assertEqual(npu_tensor_size.device.type, "npu")
        self.assertEqual(npu_tensor_size.dtype, torch.int64)
        self.assertEqual(npu_tensor_size.shape, torch.Size([1]))
        valid_inputs = (
            (npu_tensor, npu_tensor_size, None),
            (npu_tensor, cpu_tensor_size, None),
            (npu_tensor, npu_tensor_size[0], None),
            (npu_tensor, tensor_size, None),
            (cpu_tensor, tensor_size, None),
            (npu_tensor, npu_tensor_size, dist.group.WORLD),
        )
        for tensor, size, group in valid_inputs:
            with self.subTest(
                tensor_device=tensor.device.type,
                tensor_size_type=type(size).__name__,
                group_is_none=group is None,
            ):
                self.assertEqual(
                    c10d._tensor_to_object(tensor, size, group), payload
                )

    def test_tensor_to_object_tensor_size_boundaries(self):
        payload, npu_tensor, npu_tensor_size = self._tensor_to_object_inputs()
        cpu_tensor = npu_tensor.cpu()
        tensor_size = int(npu_tensor_size.item())

        for tensor in (cpu_tensor, npu_tensor):
            with self.subTest(tensor_device=tensor.device.type, boundary="exact"):
                self.assertEqual(
                    c10d._tensor_to_object(tensor, tensor_size, None), payload
                )
            with self.subTest(tensor_device=tensor.device.type, boundary="oversized"):
                self.assertEqual(
                    c10d._tensor_to_object(tensor, tensor_size + 1, None), payload
                )

        self._assert_tensor_to_object_error(
            cpu_tensor, npu_tensor, 0, 0, EOFError, "Ran out of input"
        )
        for boundary in (tensor_size - 1, -1):
            self._assert_tensor_to_object_error(
                cpu_tensor,
                npu_tensor,
                boundary,
                boundary,
                pickle.UnpicklingError,
                "pickle data was truncated",
            )

    def test_tensor_to_object_invalid_inputs(self):
        _, npu_tensor, npu_tensor_size = self._tensor_to_object_inputs()
        cpu_tensor = npu_tensor.cpu()
        tensor_size = int(npu_tensor_size.item())

        self._assert_tensor_to_object_error(
            cpu_tensor,
            npu_tensor,
            float(tensor_size),
            float(tensor_size),
            TypeError,
            "slice indices must be integers",
        )
        self._assert_tensor_to_object_error(
            cpu_tensor,
            npu_tensor,
            torch.tensor([tensor_size, tensor_size]),
            torch.tensor([tensor_size, tensor_size], device="npu"),
            TypeError,
            "only integer tensors of a single element can be converted to an index",
        )
        self._assert_tensor_to_object_error(
            cpu_tensor.float(),
            npu_tensor.float(),
            tensor_size,
            tensor_size,
            pickle.UnpicklingError,
            "invalid load key",
        )
        with self.assertRaisesRegex(AttributeError, "has no attribute 'cpu'"):
            c10d._tensor_to_object(npu_tensor.tolist(), tensor_size, None)

if __name__ == "__main__":
    run_tests()
