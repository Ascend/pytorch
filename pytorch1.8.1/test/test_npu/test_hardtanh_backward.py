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
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestHardtanhBackward(TestCase):

    def cpu_op_exec(self, input_x, min_val, max_val):
        input_x.requires_grad_(True)
        m = torch.nn.Hardtanh(min_val, max_val)
        output = m(input_x)
        w = torch.ones_like(output)
        output.backward(w)
        out = input_x.grad
        return out

    def npu_op_exec(self, input_x, min_val, max_val):
        input_x.requires_grad_(True)
        m = torch.nn.Hardtanh(min_val, max_val)
        output = m(input_x)
        w = torch.ones_like(output)
        w = w.to("npu")
        output.backward(w)
        out = input_x.grad.to('cpu')
        return out

    def test_hardtanh_backwardfloat32(self, device):
        shape_format = [
            [[np.float32, 0, (10, 10)], -1, 1], [[np.float32, 0, (5, 6, 7)], -1, 1], 
            [[np.float32, -1, (6, 6, 6)], -1, 3], [[np.float32, 3, (8, 6, 4)], -2, 2], 
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())

    def test_hardtanh_backwardfloat16(self, device):
        shape_format = [
            [[np.float16, 0, (10, 10, 10)], -1, 1], [[np.float16, 0, (7, 7, 7)], -1, 1], 
            [[np.float16, -1, (6, 6, 6, 6)], -3, 1], [[np.float16, 3, (10, 10, 10, 10)], -1, 3],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            cpu_output = cpu_output.to(torch.float16)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())

instantiate_device_type_tests(TestHardtanhBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()