# Compilation Verification

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:48:17.751Z pushedAt=2026-06-15T12:00:44.055Z -->

1. After the operator adaptation is complete, the torch_npu package needs to be compiled. It is recommended to use a container scenario for compilation and installation. For detailed operations, refer to the "Method 1 (Recommended): Container Scenario" section in the [Method 2: Installation from Source Code](../installation_guide/compilation_installation_using_source_code.md) chapter of the *AscendExtension for PyTorch Software Installation Guide*.
2. After torch_npu is installed, unit test (UT) of the newly added operator interfaces is performed. UT ensures that the operator implementation meets expectations through functional correctness verification and boundary condition coverage, reducing joint debugging costs. It also serves as a quality baseline for long-term maintenance, ensuring the stability of the operator adaptation throughout its lifecycle. The test directory for custom operator adaptation is `test/test_custom_ops`.
  Taking npu_reshape as an example, the following test cases need to be implemented:

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
