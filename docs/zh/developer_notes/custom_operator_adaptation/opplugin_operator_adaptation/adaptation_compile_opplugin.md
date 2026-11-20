# 编译验证

本文档介绍自定义算子适配完成后的编译验证流程，包括TorchNPU包的编译安装与新增算子接口的UT测试验证，适用于进行自定义算子适配的开发者。

1. 算子适配完成后，需编译TorchNPU包，推荐使用容器场景进行编译安装，具体操作可参考[源码安装](../../../installation_guide/building_from_source.md)章节的“方式一（推荐）：容器场景”。
2. TorchNPU 安装完成后，在运行 UT 的同一 Python 环境中安装测试依赖。以下命令从任意目录执行，请替换源码路径并选择与 PyTorch 版本匹配的测试依赖文件：

    ```bash
    python -m pip install -r /path/to/pytorch/test/requirements.txt
    ```

3. 进行新增算子接口的测试验证。单元测试（UT）通过功能正确性验证、边界条件覆盖等，确保算子实现符合预期，降低联调成本，同时作为长期维护的质量基线，保障算子适配全生命周期的稳定性。自定义算子适配的test目录为test/test\_custom\_ops。
  以npu_reshape为例，需要实现以下用例：

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

4. 将示例保存到 OpPlugin 源码目录的 `test/test_custom_ops/test_npu_reshape.py`，从该仓根目录执行：

    ```bash
    python test/test_custom_ops/test_npu_reshape.py -v
    ```

    两个用例均通过时，输出包含 `OK`。
