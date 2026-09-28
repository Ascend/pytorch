# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
Validate torch.Tensor.__setstate__ API completion on NPU.

This file covers torch.Tensor.__setstate__ directly because the upstream
coverage is only indirect through serialization/JIT-style __setstate__ paths.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTensorSetstateApiCompletion(TestCase):
    def test_tensor_setstate(self):
        tensor = torch.ones(2, device=device_type)
        self.assertFalse(tensor.requires_grad)

        tensor.__setstate__((True, None, None))
        self.assertTrue(tensor.requires_grad)

        with self.assertRaisesRegex(
            RuntimeError, "__setstate__ can be only called on leaf Tensors"
        ):
            (tensor + 1).__setstate__((False, None, None))


if __name__ == "__main__":
    run_tests()
