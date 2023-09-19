import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import graph_mode
from torch_npu.testing.common_utils import create_common_tensor


class TestAsynchronousPrint(TestCase):
    
    
    def cpu_op_exec(self, input1, input2):
        output = torch.sub(input1, input2)
        output = torch.add(output, input2)
        output = output.numpy()
        return output
    
    def npu_op_exec(self, input1, input2):   
        output = torch.sub(input1, input2)
        print("npu_output_sub: {}".format(output))
        output = torch.add(output, input2)
        print("npu_output_add: {}".format(output))
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    @graph_mode
    def test_asynchronous_print(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [4,8]]  for i in format_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
                
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            for _ in range(10):
                npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            print("npu_output: {}".format(npu_output))
            print("cpu_output: {}".format(cpu_output))
            
            self.assertRtolEqual(cpu_output, npu_output)
    
if __name__ == "__main__":
    run_tests()

