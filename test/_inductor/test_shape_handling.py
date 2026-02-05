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
from torch_npu._inductor.shape_handling import unified_copy

torch._dynamo.config.cache_size_limit = 128

if not torch.npu.is_available():
    raise unittest.SkipTest("NPU is not available")

device = "npu"


def model_fn(A, B):  
    return A + B

shape_options = {
    "enable_shape_handling": True,
    "shape_handling_configs": [
        {
            "type": "BATCHSIZE", # 处理的维度类型(Required)
            "dimensions": 0,      # 该维度在tensor中的下标(BATCHSIZE默认为0)(Optional)
            # "indices": [0, 1],   # 需要处理的tensor下标(默认为所有tensor)(Optional)
            "value": 0.0,        # padding时填充的值, 默认为0.0(Optional)
            "gears": [],         # 自定义档位信息(Optional)
            "min_size": 1,       # 该维度的最小大小(档位), 默认为1(Optional)           
            "max_size": 1024,    # 该维度的最大大小(档位), 默认为1024(Optional)
            "policy": "TIMES",   # 依据min_size, max_size自动生成gears的策略, 默认为TIMES, 表示生成范围内2的整数幂档位(Optional) 
        },
        {
            "type": "SEQLEN",          # 处理的维度类型(Required)
            "dimensions": [1, 1],      # 该维度在tensor中的下标, 与indices一一对应, 也可以接收dimension表示所有tensor使用相同的dimension index, 优先接收dimension(SEQLEN默认为1)(Optional)
            # "indices": [0, 1],         # 需要处理的tensor下标(默认为所有tensor)(Optional)
            "value": 0.0,              # padding时填充的值, 默认为0.0(Optional)
            "gears": [],               # 自定义档位信息(Optional)
            "min_size": 1,             # 该维度的最小大小(档位), 默认为1(Optional)           
            "max_size": 1024,          # 该维度的最大大小(档位), 默认为1024(Optional)
            "policy": "TIMES",         # 依据min_size, max_size自动生成gears的策略, 默认为TIMES, 表示生成范围内2的整数幂档位(Optional) 
        }
    ]
}


class TestShapeHandling(TestCase):
    def test_init_no_input(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        self.assertNotEqual(shape_handling, None)
    
    def test_init_with_empty_conifg(self):
        configs = []
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertNotEqual(shape_handling, None)
    
    def test_init_with_gears(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "gears": [16, 32, 64]
            },
            {
                "type": "SEQLEN",
                "gears": [16, 32, 64]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertNotEqual(shape_handling, None)
 
    def test_init_with_policy(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 2,
                "max_size": 8,
                "policy": "TIMES"
            },
            {
                "type": "SEQLEN",
                "min_size": 2,
                "max_size": 8,
                "policy": "TIMES"
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertNotEqual(shape_handling, None)
 
    def test_transform_no_operation(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "gears": [32],
                "dimensions": 0
            },
            {
                "type": "SEQLEN",
                "gears": [128],
                "dimensions": [1]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        input_tensor = torch.randn(32, 128)
        outputs = shape_handling.transform([input_tensor])
        self.assertEqual(outputs[0][0].shape, input_tensor.shape)
 
    def test_transform_padding(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": 0
            },
            {
                "type": "SEQLEN",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": [1]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        input_tensor = torch.randn(48, 96)
        outputs = shape_handling.transform([input_tensor])
        self.assertEqual(outputs[0][0].shape, (64, 128))
    
    def test_transform_padding_with_indices(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": 0,
                "indices": [0]
            },
            {
                "type": "SEQLEN",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": [1],
                "indices": [0]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        input_tensor1 = torch.randn(48, 96)
        input_tensor2 = torch.randn(48, 96) 
        outputs = shape_handling.transform([input_tensor1, input_tensor2])
        self.assertEqual(outputs[0][0].shape, (64, 128))
        self.assertEqual(outputs[0][1].shape, (48, 96))
 
    def test_transform_split(self):
        """测试分割操作"""
        # 设置max_size为128，那么超过128的会被分割
        configs = [
            {
                "type": "BATCHSIZE",
                "gears": [64, 128],
                "dimensions": 0
            },
            {
                "type": "SEQLEN",
                "gears": [64, 128],
                "dimensions": [1]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        input_tensor = torch.randn(200, 96)  # 超过max_size
        outputs = shape_handling.transform([input_tensor])
        # 验证分割结果：分割为两个组，第一段128，第二段72，再填充为128
        self.assertEqual(len(outputs), 2)  # 分割为两个组
        self.assertEqual(outputs[0][0].shape, (128, 128))  # 第一段
        self.assertEqual(outputs[1][0].shape, (128, 128))  # 第二段
 
    def test_recover_padding(self):
        """测试恢复填充的张量"""
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": 0,
                "indices": [0]
            },
            {
                "type": "SEQLEN",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": [1],
                "indices": [0]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
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
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 16,
                "max_size": 128,
                "policy": "TIMES",
                "dimensions": 0,
                "indices": [0]
            },
            {
                "type": "SEQLEN",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": [1],
                "indices": [0]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        orig_tensor = torch.randn(200, 128)
        split_groups = shape_handling.transform([orig_tensor])
        # 执行恢复
        recovered = shape_handling.recover(split_groups)
        # 验证拼接还原
        self.assertEqual(recovered[0].shape, orig_tensor.shape)
        self.assertTrue(torch.allclose(recovered[0], orig_tensor))
 
    def test_invalid_dim(self):
        """测试无效维度异常"""
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 16,
                "max_size": 128,
                "policy": "TIMES",
                "dimensions": 3,
                "indices": [0]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        input_tensor = torch.randn(32, 128)
        # 验证越界维度触发异常
        with self.assertRaises(RuntimeError):
            shape_handling.transform([input_tensor])  # 有效维度应为0或1

    def test_register_custom_strategy(self):
        """测试注册自定义策略"""
        # 创建自定义策略类
        class CustomBsShapeOp(torch_npu._C._BSShapeOpStrategy):
            def __init__(self):
                super().__init__()
                self.transform_called = False
                self.recover_called = False

            def Transform(self, inputs, outputs):
                self.transform_called = True
                # 简单实现：直接复制输入到输出
                if inputs:
                    outputs.append([tensor.clone() for tensor in inputs])

            def Recover(self, inputs, outputs):
                self.recover_called = True
                # 简单实现：直接复制第一个组的第一个张量
                if inputs and inputs[0]:
                    outputs.append(inputs[0][0].clone())
        
        class CustomSeqShapeOp(torch_npu._C._SeqShapeOpStrategy):
            def __init__(self):
                super().__init__()
                self.transform_called = False
                self.recover_called = False

            def Transform(self, inputs, outputs):
                self.transform_called = True
                # 简单实现：直接复制输入到输出
                if inputs:
                    outputs.append([tensor.clone() for tensor in inputs])
        
        configs = [
            {
                "type": "BATCHSIZE",
                "min_size": 16,
                "max_size": 128,
                "policy": "TIMES",
                "dimensions": 0,
                "indices": [0]
            },
            {
                "type": "SEQLEN",
                "min_size": 16,
                "max_size": 256,
                "policy": "TIMES",
                "dimensions": [1],
                "indices": [0]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        custom_bs_strategy = CustomBsShapeOp()
        custom_seq_strategy = CustomSeqShapeOp()

        # 注册自定义策略
        shape_handling.register_batch_size_strategy(custom_bs_strategy)
        shape_handling.register_sequence_strategy(custom_seq_strategy)

        # 验证自定义策略被调用
        input_tensor = torch.randn(32, 128)
        shape_handling.transform([input_tensor])

        self.assertTrue(custom_bs_strategy.transform_called)
        self.assertTrue(custom_seq_strategy.transform_called)


    
class TestDynamicShapeCompile(TestCase):
    def test_npu_dynamic_shape_reuse_with_no_bucket(self):
    
        compiled_fn = torch.compile(
            model_fn, 
            backend='inductor', 
            dynamic=False,
        )

        # 运行不同形状，验证是否只触发 4 次编译
        test_shapes = [(3, 20), (4, 20), (5, 20), (6, 20)]
        

        if hasattr(torch._inductor.metrics.generated_kernel_count, 'reset'):
            torch._inductor.metrics.generated_kernel_count.reset()
        else:
            torch._inductor.metrics.generated_kernel_count = 0

        for shape in test_shapes:
            A = torch.randn(shape, device=device)
            B = torch.randn(shape, device=device)
            out = compiled_fn(A, B)
            self.assertTrue(torch.allclose(out, A + B))


        compile_count = torch._inductor.metrics.generated_kernel_count

        # 获取 Inductor 编译的总次数（生成的 kernel 数量相关）
        print(f"\n[结果] Inductor 编译次数 (generated_kernel_count): {compile_count}")

        # 如果动态形状分档不生效，编译次数应为 4
        self.assertEqual(compile_count, 4)

    def test_npu_dynamic_shape_reuse_with_bucket(self):
    
        compiled_fn = torch.compile(
            model_fn, 
            backend='inductor', 
            dynamic=False,
            options=shape_options
        )

        # 运行不同形状，验证是否只触发 2 次编译
        test_shapes = [(3, 32), (4, 32), (5, 32), (6, 32)]
        if hasattr(torch._inductor.metrics.generated_kernel_count, 'reset'):
            torch._inductor.metrics.generated_kernel_count.reset()
        else:
            torch._inductor.metrics.generated_kernel_count = 0

        for shape in test_shapes:
            A = torch.randn(shape, device=device)
            B = torch.randn(shape, device=device)
            out = compiled_fn(A, B)
            self.assertTrue(torch.allclose(out, A + B))
        

        compile_count = torch._inductor.metrics.generated_kernel_count
  
        # 获取 Inductor 编译的总次数（生成的 kernel 数量相关）
        print(f"\n[结果] Inductor 编译次数 (generated_kernel_count): {compile_count}")

        # 如果动态形状分档生效，编译次数应为 2
        self.assertEqual(compile_count, 2)

    def test_npu_dynamic_shape_reuse_with_symbolic_shape(self):
    
        compiled_fn = torch.compile(
            model_fn, 
            backend='inductor', 
            dynamic=True,
        )

        # 运行不同形状，验证是否只触发 1 次编译
        test_shapes = [(3, 32), (4, 32), (5, 32), (6, 32)]
        
        if hasattr(torch._inductor.metrics.generated_kernel_count, 'reset'):
            torch._inductor.metrics.generated_kernel_count.reset()
        else:
            torch._inductor.metrics.generated_kernel_count = 0

        for shape in test_shapes:
            A = torch.randn(shape, device=device)
            B = torch.randn(shape, device=device)
            out = compiled_fn(A, B)
            self.assertTrue(torch.allclose(out, A + B))

        compile_count = torch._inductor.metrics.generated_kernel_count



        # 获取 Inductor 编译的总次数（生成的 kernel 数量相关）
        print(f"\n[结果] Inductor 编译次数 (generated_kernel_count): {compile_count}")

        # 如果动态形状符号化生效，编译次数应为 1
        self.assertEqual(compile_count, 1)
    
    def test_npu_dynamic_shape_reuse_with_symbolic_shape_and_bucket(self):
    
        compiled_fn = torch.compile(
            model_fn, 
            backend='inductor', 
            dynamic=True,
            options=shape_options
        )

        # 运行不同形状，验证是否只触发 1 次编译
        test_shapes = [(3, 32), (4, 32), (5, 32), (6, 32)]
        
        if hasattr(torch._inductor.metrics.generated_kernel_count, 'reset'):
            torch._inductor.metrics.generated_kernel_count.reset()
        else:
            torch._inductor.metrics.generated_kernel_count = 0

        for shape in test_shapes:
            A = torch.randn(shape, device=device)
            B = torch.randn(shape, device=device)
            out = compiled_fn(A, B)
            self.assertTrue(torch.allclose(out, A + B))

        compile_count = torch._inductor.metrics.generated_kernel_count

    
        # 获取 Inductor 编译的总次数（生成的 kernel 数量相关）
        print(f"\n[结果] Inductor 编译次数 (generated_kernel_count): {compile_count}")

        # 如果动态形状符号化生效，编译次数应为 1
        self.assertEqual(compile_count, 1)
    
    def test_npu_invalid_shape_options_key_error(self):

        # 构造一个非法的配置：最小尺寸大于最大尺寸
        invalid_shape_options = {
            "enable_shape_handling": True,
            "shape_handling_configs": [
                {
                    "type": "BATCHSIZE",
                    "min_size": 1024,   # 错误：min > max
                    "max_size": 1,
                    "policy": "TIMES",
                }
            ]
        }

        # 尝试编译，预期抛出 ValueError 或相关配置错误
        # 注意：有些错误可能在编译阶段 (compile) 触发，有些可能在首次运行 (call) 时触发
        try:
            compiled_fn = torch.compile(
                model_fn, 
                backend='inductor', 
                dynamic=False,
                options=invalid_shape_options
            )
            
            # 准备输入数据触发编译
            A = torch.randn((2, 32), device=device)
            B = torch.randn((2, 32), device=device)
            
            with self.assertRaises((ValueError, RuntimeError, TypeError)) as cm:
                compiled_fn(A, B)
            
            print(f"\n[成功捕获异常]: {cm.exception}")
            
        except Exception as e:
            # 如果在 torch.compile 阶段就直接崩溃，也算捕获成功
            print(f"\n[编译阶段直接报错]: {e}")
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))

    def test_npu_malformed_option_type(self):
        
        # 构造格式完全错误的 options
        malformed_options = {
            "shape_handling_configs": "this should be a list, not a string"
        }

        with self.assertRaises(Exception):
            compiled_fn = torch.compile(model_fn, options=malformed_options)
            compiled_fn(torch.randn(2, device="npu"), torch.randn(2, device="npu"))

    def test_npu_shape_options_with_handling_disabled(self):
        # enable_shape_handling 设置为False, 同时保留主流的配置选项，测试是否会存在选项异常报错
    

        shape_options_off = {
            "enable_shape_handling": False,
            "shape_handling_configs": [
                {
                    "type": "BATCHSIZE",
                    "min_size": 1,
                    "max_size": 1024,
                    "policy": "TIMES",
                }
            ]
        }

        # 尝试编译，预期可正确处理此类配置，则通过测试
        try:
            compiled_fn = torch.compile(
                model_fn, 
                backend='inductor', 
                dynamic=False,
                options=shape_options_off
            )
            
            # 准备输入数据触发编译
            A = torch.randn((2, 32), device=device)
            B = torch.randn((2, 32), device=device)
            compiled_fn(A, B)

        except Exception as e:
            self.fail(f"torch.compile raised {type(e).__name__} unexpectedly: {e}")


class TestUnifiedCopy(TestCase):
    def test_none_and_simple_types(self):
        """Test unified_copy with None, list, dict, and tuple."""
        # None
        self.assertIsNone(unified_copy(None))

        # list
        lst = [1, 2, 3]
        copied_lst = unified_copy(lst)
        self.assertEqual(lst, copied_lst)
        self.assertIsNot(lst, copied_lst)

        # dict
        d = {"a": 1, "b": [2, 3]}
        copied_d = unified_copy(d)
        self.assertEqual(d, copied_d)
        self.assertIsNot(d, copied_d)

        # tuple
        t = (1, [2, 3])
        copied_t = unified_copy(t)
        self.assertEqual(t, copied_t)
        self.assertIsInstance(copied_t, tuple)

    def test_nested_structure_with_tensor(self):
        """Test nested structure containing NPU tensor."""
        original = {
            "data": [torch.tensor([1.0, 2.0]), 42],
            "meta": {"shape": (2,)}
        }
        copied = unified_copy(original)

        # 结构和值一致
        self.assertEqual(original["meta"], copied["meta"])
        self.assertTrue(torch.equal(original["data"][0], copied["data"][0]))
        self.assertEqual(original["data"][1], copied["data"][1])

        # 验证独立副本
        original["data"][0][0] = 888.0
        self.assertNotEqual(copied["data"][0][0].item(), 888.0)


instantiate_parametrized_tests(TestShapeHandling)
instantiate_parametrized_tests(TestUnifiedCopy)
instantiate_parametrized_tests(TestDynamicShapeCompile)
 
if __name__ == '__main__':
    run_tests()