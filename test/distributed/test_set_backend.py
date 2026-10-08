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
Validation cases for torch.distributed._symmetric_memory.set_backend and
get_backend (symmetric memory backend configuration):
1. set_backend configures the global symmetric memory allocation backend.
   The backends a build exposes for allocation are build-dependent: CUDA
   builds register "NVSHMEM"/"CUDA"/"NCCL", while the torch_npu build
   exposes no allocation backend, so set_backend rejects every name
   (documented or not) with RuntimeError and rejects non-str arguments
   with TypeError.
2. get_backend reports the backend registered for a device: "NPUSHMEM"
   for the npu device (registered by torch_npu), accepted as torch.device
   or str; None for devices without a registered backend; TypeError for
   arguments torch.device() cannot parse.
3. torch_npu is a required dependency of this repository's CI, so it is
   imported unconditionally.
"""
import torch
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401  # Registers the "npu" device and the NPUSHMEM backend.


class TestSetBackend(TestCase):
    # Validate torch.distributed._symmetric_memory.set_backend / get_backend.

    def test_set_backend_invalid_names_raise(self):
        # The torch_npu build exposes no symmetric memory allocation
        # backend, so every backend name is rejected.
        for name in ["NVSHMEM", "CUDA", "NCCL", "NPUSHMEM", "invalid"]:
            with self.assertRaises(RuntimeError):
                symm_mem.set_backend(name)

    def test_set_backend_invalid_type_raises(self):
        # The backend name must be a str; other types are rejected.
        for value in (123, None):
            with self.assertRaises(TypeError):
                symm_mem.set_backend(value)

    def test_get_backend_npu(self):
        # torch_npu registers the NPUSHMEM symmetric memory backend for
        # the npu device; both torch.device and str forms are accepted.
        self.assertEqual(symm_mem.get_backend(torch.device("npu:0")), "NPUSHMEM")
        self.assertEqual(symm_mem.get_backend("npu:0"), "NPUSHMEM")

    def test_get_backend_unregistered_devices(self):
        # Devices without a registered symmetric memory backend report None.
        for device in ("cpu", "cpu:0", "meta", "cuda:0"):
            self.assertIsNone(symm_mem.get_backend(torch.device(device)))

    def test_get_backend_invalid_type_raises(self):
        # Arguments torch.device() cannot parse are rejected.
        for value in (None, 3.5):
            with self.assertRaises(TypeError):
                symm_mem.get_backend(value)


if __name__ == "__main__":
    run_tests()
