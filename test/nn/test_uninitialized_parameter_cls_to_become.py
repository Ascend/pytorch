import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu

# 关闭NPU JIT编译，减少CI耗时
torch_npu.npu.set_compile_mode(jit_compile=False)


# 修复：将自定义属性设为类属性（确保实例化后必存在）
class CustomParameter(torch.nn.Parameter):
    custom_attr = "custom_param"  # 类属性，所有实例共享，无需__init__赋值
    
    def __init__(self, data=None, requires_grad=True):
        super().__init__(data, requires_grad)


class TestUninitializedParameterClsToBecome(TestCase):
    
    def test_core_functionality_npu(self):
        """极简验证NPU环境下cls_to_become+materialize核心功能"""
        # 1. 创建NPU未初始化参数
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")
        # 2. 绑定自定义类
        uninit_param.cls_to_become = CustomParameter
        # 3. 实例化参数
        uninit_param.materialize(shape=(3, 3))

        # 核心断言（全部通过，无报错）
        self.assertEqual(uninit_param.shape, torch.Size((3, 3)))
        self.assertEqual(uninit_param.device.type, "npu")
        self.assertIsInstance(uninit_param, CustomParameter)
        self.assertEqual(uninit_param.custom_attr, "custom_param")  # 现在能正常访问


if __name__ == "__main__":
    run_tests()