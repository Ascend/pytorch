# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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

from enum import IntEnum, unique
import torch
import torch.distributed as c10d
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, load_tests, run_tests

if not c10d.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)

@unique
class Format(IntEnum):
    NCHW = 0
    ND = 2
    NC1HWC0 = 3
    NZ = 29

# bucket分配原则：
# 1.bucket计算大小以byte为单位，fp32占4个byte，fp16占2个byte
# 2.npu计算bucket需要以时间占用物理内存计算，例如100个元素的fp32,nz格式的数据占用1792个byte,而不是400个byte
# 3.同一种数据类型(fp32, fp16)放在一个bucket
# 4.当一个数据只能部分放在一个bucket时，则改数据仍然放在该bucket中。
# 5.当有多个limit时,如[40, 80]，第一个limit(40)使用一次，以后均使用最后一个(80)。
class ComputeBucketAssignmentTest(TestCase):
    def test_single_limit_single_dtype(self):
        tensors = [
            torch.empty([100], dtype=torch.float).npu().npu_format_cast(Format.NZ),
            torch.empty([200], dtype=torch.float).npu(),
            torch.empty([100], dtype=torch.float).npu(),
            torch.empty([50], dtype=torch.float).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [1792 * 4 + 1])
        self.assertEqual([[0, 1], [2, 3]], result)

    def test_single_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    def test_multi_limit_single_dtype(self):
        tensors = [
            torch.empty([10], dtype=torch.float).npu(),
            torch.empty([10], dtype=torch.float).npu(),
            torch.empty([10], dtype=torch.float).npu(),
            torch.empty([10], dtype=torch.float).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [40, 80])
        self.assertEqual([[0], [1, 2], [3]], result)

    def test_multi_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),

            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),

            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5], [6, 8], [7, 9]], result)

if __name__ == '__main__':
    assert not torch.npu._initialized, "test_distributed must not have initialized NPU context on main process"

    run_tests()
