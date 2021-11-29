import copy
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import RunFuncInGraphMode


class TestCastNpuDtype(TestCase):
    def npu_op_exec(self, input1, scalartype):
        output = torch.npu_dtype_cast(input1, scalartype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1, input2):
        output = input1.npu_dtype_cast_(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    @RunFuncInGraphMode 
    def test_npu_dtype_cast_inplace(self, device):
        shape_format = [
            [[np.float32, -1, (4, 3)]],
            [[np.float32, -1, (4, 3, 1)]],
            [[np.int32,   -1, (2, 3)]],
            [[np.int32,   -1, (4, 3, 1)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input1 = cpu_input1.byte()
            npu_dtype_tensor = cpu_input1.to("npu")
            npu_output = self.npu_op_exec_inplace(npu_input1, npu_dtype_tensor)            
            self.assertEqual(cpu_input1, npu_output)

    @RunFuncInGraphMode
    def test_npu_dtype_cast(self, device):
        shape_format = [
            [[np.float32, -1, (4, 3)]],
            [[np.float32, -1, (4, 3, 1)]],
            [[np.int32,   -1, (2, 3)]],
            [[np.int32,   -1, (4, 3, 1)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input1 = cpu_input1.byte()
            npu_output = self.npu_op_exec(npu_input1, cpu_input1.dtype)            
            self.assertEqual(cpu_input1, npu_output)
    
    @RunFuncInGraphMode
    def test_npu_dtype_cast_noneed(self, device):
        shape_format = [
            [[np.float32, -1, (4, 3)]],
            [[np.float32, -1, (4, 3, 1)]],
            [[np.int32,   -1, (2, 3)]],
            [[np.int32,   -1, (4, 3, 1)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_output = self.npu_op_exec(npu_input1, npu_input1.dtype)            
            self.assertEqual(cpu_input1, npu_output)


instantiate_device_type_tests(TestCastNpuDtype, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
