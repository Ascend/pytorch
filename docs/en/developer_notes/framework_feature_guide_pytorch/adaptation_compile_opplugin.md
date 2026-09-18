# Compilation Verification

1. After the operator adaptation is complete, compile the TorchNPU package. You are advised to compile and install it in a container scenario. For details, see the "Method 1 (Recommended): Container Scenario" section in the "[Method 2: Source Code Installation](../installation_guide/compilation_installation_using_source_code.md)" chapter of the *Software Installation* guide.
2. After TorchNPU is installed, perform a unit test (UT) on the newly added operator interfaces. UT ensures that the operator implementation meets expectations through functional correctness verification and boundary condition coverage, reducing joint debugging costs. It also serves as a quality baseline for long-term maintenance, ensuring the stability of the operator adaptation throughout its lifecycle. The test directory for custom operator adaptation is `test/test_custom_ops`.
  Taking `npu_reshape` as an example, implement the following test cases:

    ```python
    import torch
    import numpy as np
    import torch_npu

    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import create_common_tensor


    class TestNpuReshape(TestCase):
        def test_npu_reshape(self):
            def cpu_op_exec(input1, shape):
                output = torch.reshape(input1, shape)
                output = output.numpy()
                return output

            def npu_op_exec(input1, shape):
                output = torch_npu.npu_reshape(input1, shape)
                output = output.to("cpu")
                output = output.numpy()
                return output

            shape_format = [
                [[np.float32, 0, (8, 8)], [4, 16]],
                [[np.float16, 0, (8, 8)], [4, 16]],
                [[np.float32, 0, (2, 4, 8)], [2, 32]],
                [[np.float16, 0, (2, 4, 4, 2)], [64, 1]],
            ]

            for item in shape_format:
                cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
                cpu_output = cpu_op_exec(cpu_input, item[1])
                npu_output = npu_op_exec(npu_input, item[1])
                self.assertRtolEqual(cpu_output, npu_output)

        def test_npu_reshape_boundary(self):
            def cpu_op_exec(input1, shape):
                output = torch.reshape(input1, shape)
                output = output.numpy()
                return output

            def npu_op_exec(input1, shape):
                output = torch_npu.npu_reshape(input1, shape)
                output = output.to("cpu")
                output = output.numpy()
                return output

            shape_format = [
                [[np.float32, 0, (0, 8)], [0, 8]],
                [[np.float16, 0, (4, 0)], [0, 4]],
            ]

            for item in shape_format:
                cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
                cpu_output = cpu_op_exec(cpu_input, item[1])
                npu_output = npu_op_exec(npu_input, item[1])
                self.assertRtolEqual(cpu_output, npu_output)


    if __name__ == "__main__":
        run_tests()
    ```
