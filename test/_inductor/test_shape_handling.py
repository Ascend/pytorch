import torch
from torch.nn.parallel import scatter_gather
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_REMOTE_GPU,
    IS_SANDCASTLE,
    NoTest,
    TEST_PRIVATEUSE1,
    TestCase,
    instantiate_parametrized_tests,
    run_tests,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
)
import torch_npu
import torch_npu._inductor
import torch_npu.testing
from torch_npu.testing.common_utils import get_cycles_per_ms


class TestShapeHandling(TestCase):
    def test_init_no_input(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        self.assertNotEqual(shape_handling, None)
 
    def test_init_with_sizes(self):
        """测试使用sizes列表初始化"""
        sizes = [16, 32, 64]
 
        shape_handling = torch_npu._inductor.NPUShapeHandling(sizes=sizes)
        self.assertNotEqual(shape_handling, None)
 
    def test_init_with_policy(self):
        min_size = 2
        max_size = 8
        policy = "TIMES"
        shape_handling = torch_npu._inductor.NPUShapeHandling(min_size, max_size, policy)
        self.assertNotEqual(shape_handling, None)
 
    def test_transform_no_operation(self):
        """测试无需变换的情况"""
        shape_handling = torch_npu._inductor.NPUShapeHandling(sizes=[32], dim=0)
        input_tensor = torch.randn(32, 128)
        outputs = shape_handling.transform([input_tensor])
        # 验证输出与输入一致
        self.assertEqual(outputs[0][0].shape, input_tensor.shape)
 
    def test_transform_padding(self):
        """测试填充操作"""
        # 使用min_size和max_size初始化，生成2^n序列：16,32,64
        shape_handling = torch_npu._inductor.NPUShapeHandling(min_size=16, max_size=64, policy="TIMES", dim=0)
        input_tensor = torch.randn(48, 128)  # 48不在序列中，会填充到64
        outputs = shape_handling.transform([input_tensor])
        # 验证填充到最近的2^n尺寸64
        self.assertEqual(outputs[0][0].shape, (64, 128))
    
    def test_transform_padding_with_index(self):
        """测试填充操作"""
        # 使用min_size和max_size初始化，生成2^n序列：16,32,64
        indexs = [0]
        shape_handling = torch_npu._inductor.NPUShapeHandling(min_size=16, max_size=64, policy="TIMES", indexs=indexs)
        input_tensor1 = torch.randn(48, 128)  # 48不在序列中，会填充到64
        input_tensor2 = torch.randn(48, 128)  # 48不在序列中，会填充到64
        outputs = shape_handling.transform([input_tensor1, input_tensor2])
        # 验证填充到最近的2^n尺寸64
        self.assertEqual(outputs[0][0].shape, (64, 128))
        self.assertEqual(outputs[0][1].shape, (48, 128))
 
    def test_transform_split(self):
        """测试分割操作"""
        # 设置max_size为128，那么超过128的会被分割
        shape_handling = torch_npu._inductor.NPUShapeHandling(min_size=64, max_size=128, dim=0)
        input_tensor = torch.randn(200, 128)  # 超过max_size
        outputs = shape_handling.transform([input_tensor])
        # 验证分割结果：分割为两个组，第一段128，第二段72，再填充为128
        self.assertEqual(len(outputs), 2)  # 分割为两个组
        self.assertEqual(outputs[0][0].shape, (128, 128))  # 第一段
        self.assertEqual(outputs[1][0].shape, (128, 128))  # 第二段
 
    def test_recover_padding(self):
        """测试恢复填充的张量"""
        shape_handling = torch_npu._inductor.NPUShapeHandling(min_size=16, max_size=64, dim=0)
        orig_tensor = torch.randn(48, 128)
        padded_group = shape_handling.transform([orig_tensor])
        # 执行恢复
        recovered = shape_handling.recover(padded_group)
        # 验证尺寸还原
        self.assertEqual(recovered[0].shape, orig_tensor.shape)
        # 验证数据一致性
        self.assertTrue(torch.allclose(recovered[0], orig_tensor))
 
    def test_recover_split(self):
        """测试恢复分割的张量组"""
        shape_handling = torch_npu._inductor.NPUShapeHandling(min_size=64, max_size=128, dim=0)
        orig_tensor = torch.randn(200, 128)
        split_groups = shape_handling.transform([orig_tensor])
        # 执行恢复
        recovered = shape_handling.recover(split_groups)
        # 验证拼接还原
        self.assertEqual(recovered[0].shape, orig_tensor.shape)
        self.assertTrue(torch.allclose(recovered[0], orig_tensor))
 
    def test_invalid_dim(self):
        """测试无效维度异常"""
        shape_handling = torch_npu._inductor.NPUShapeHandling(dim=3)
        input_tensor = torch.randn(32, 128)
        # 验证越界维度触发异常
        with self.assertRaises(RuntimeError):
            shape_handling.transform([input_tensor])  # 有效维度应为0或1
 
instantiate_parametrized_tests(TestShapeHandling)
 
if __name__ == '__main__':
    run_tests()