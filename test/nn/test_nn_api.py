"""
Add validation cases for torch.nn APIs on NPU:
1. test/test_nn.py from PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates torch.nn.Parameter, torch.nn.Buffer (extendable).
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
import torch_npu

device = torch.device("npu:0")


class TestNPUParameterBuffer(TestCase):
    def test_parameter_api(self):
        """验证torch.nn.Parameter的创建、属性及NPU设备支持"""
        # 直接在NPU上创建Parameter（推荐方式）
        p = nn.Parameter(torch.randn(10, 20, device=device))
        
        # 验证类型和属性
        self.assertIsInstance(p, nn.Parameter)
        self.assertTrue(p.requires_grad)
        self.assertEqual(p.device, device)
        self.assertEqual(p.shape, (10, 20))
        
        # 验证数据操作
        p.data = p.data * 2  # Parameter支持.data访问
        self.assertEqual(p.shape, (10, 20))  # 形状不变
        
        # 修改requires_grad
        p.requires_grad = False
        self.assertFalse(p.requires_grad)

    def test_buffer_api(self):
        """验证torch.nn.Buffer的创建、persistent属性及NPU设备支持"""
        # 创建persistent Buffer（默认）
        b1 = nn.Buffer(torch.randn(5, 5, device=device))
        self.assertIsInstance(b1, nn.Buffer)
        self.assertFalse(b1.requires_grad)  # Buffer默认不需要梯度
        self.assertEqual(b1.device, device)
        
        # 创建non-persistent Buffer
        b2 = nn.Buffer(torch.randn(3, 3, device=device), persistent=False)
        self.assertFalse(b2.requires_grad)
        self.assertEqual(b2.device, device)
        
        # 验证persistent属性在Module中的行为
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('persistent_buf', b1)
                self.register_buffer('non_persistent_buf', b2, persistent=False)
        
        m = TestModule()
        state_dict = m.state_dict()
        
        # persistent buffer应在state_dict中
        self.assertIn('persistent_buf', state_dict)
        # non-persistent buffer不应在state_dict中  
        self.assertNotIn('non_persistent_buf', state_dict)
        
        # 验证Buffer在state_dict中的值正确
        self.assertEqual(state_dict['persistent_buf'].device, device)


if __name__ == "__main__":
    run_tests()