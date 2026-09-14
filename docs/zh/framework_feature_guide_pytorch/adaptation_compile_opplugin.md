# 编译验证

1. 算子适配完成后，需编译TorchNPU包，推荐使用容器场景进行编译安装，具体操作可参考《软件安装》中的“[方式二：源码安装](../installation_guide/compilation_installation_using_source_code.md)”章节的“方式一（推荐）：容器场景”。
2. TorchNPU安装完成后进行新增算子接口的测试验证。测试验证（UT）通过功能正确性验证、边界条件覆盖等，确保算子实现预期，降低联调成本，同时作为长期维护的质量基线，保障算子适配全生命周期的稳定性，自定义算子适配test目录为test/test\_custom\_ops。
  以npu_linear为例，需要实现以下用例：

    ```python
    import torch
    import numpy as np
    import torch_npu

    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import create_common_tensor


    class TestLinear(TestCase):
        def test_linear(self):
            def cpu_op_exec(input1, weight, bias):
                output = torch.nn.functional.linear(input1, weight, bias)
                output = output.numpy()
                return output

            def npu_op_exec(input1, weight, bias):
                output = torch_npu.npu_linear(input1, weight, bias)
                output = output.to("cpu")
                output = output.numpy()
                return output

            shape_format = [
                [[np.float32, 0, (5, 3)], [np.float32, 0, (4, 3)], [np.float32, 0, (4,)]],
                [[np.float16, 0, (5, 3)], [np.float16, 0, (4, 3)], [np.float16, 0, (4,)]],
            ]

            for item in shape_format:
                cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
                cpu_weight, npu_weight = create_common_tensor(item[1], 0, 100)
                cpu_bias, npu_bias = create_common_tensor(item[2], 0, 100)
                cpu_output = cpu_op_exec(cpu_input, cpu_weight, cpu_bias)
                npu_output = npu_op_exec(npu_input, npu_weight, npu_bias)

                self.assertRtolEqual(cpu_output, npu_output)


    if __name__ == "__main__":
        run_tests()
    ```
