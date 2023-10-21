# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestDataParallelLayers(TestCase):
    def test_parallel_DistributedDataParallel(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
                self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
                self.dense2 = torch.nn.Linear(128, 10)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv(x)), 2)
                x = x.view(x.size(0), -1)
                x = F.relu(self.dense1(x))
                x = self.dense2(x)
                return x

        model = Net()
        import os
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29688"

        LOCAL_RANK = int(os.getenv('LOCAL_RANK', 0))
        RANK = int(os.getenv('RANK', 0))
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        torch.distributed.init_process_group(backend="hccl", rank=RANK, world_size=WORLD_SIZE)
        model = model.npu()
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False)

        self.assertEqual(net is not None, True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()
