import unittest
from unittest import mock
import threading
import time

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
import torch_npu._inductor.shape_handling as shape_handling_module

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
        """Constructing NPUShapeHandling with no arguments should succeed (delay_init path)."""
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        # object created successfully
        self.assertNotEqual(shape_handling, None)

    def test_init_with_empty_conifg(self):
        """Empty config list should not crash — falls through to delay_init."""
        configs = []
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertNotEqual(shape_handling, None)

    def test_init_with_gears(self):
        """Explicit gears for two dimension types should be accepted."""
        configs = [
            {"type": "BATCHSIZE", "gears": [16, 32, 64]},
            {"type": "SEQLEN", "gears": [16, 32, 64]},
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertNotEqual(shape_handling, None)

    def test_init_with_policy(self):
        """TIMES policy with min/max should auto-generate gears."""
        configs = [
            {"type": "BATCHSIZE", "min_size": 2, "max_size": 8, "policy": "TIMES"},
            {"type": "SEQLEN", "min_size": 2, "max_size": 8, "policy": "TIMES"},
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertNotEqual(shape_handling, None)

    def test_normalize_configs_edge_cases(self):
        """_normalize_configs handles empty input, dedup, float→int, TIMES expansion."""
        from torch_npu._inductor.adaptive_gears import AdaptiveGearRuntime
        from unittest.mock import MagicMock

        runtime = MagicMock(spec=AdaptiveGearRuntime)
        runtime._expand_policy_gears = lambda config: AdaptiveGearRuntime._expand_policy_gears(runtime, config)

        # Empty / None input → returns empty list without crashing
        self.assertEqual(AdaptiveGearRuntime._normalize_configs(runtime, []), [])
        self.assertEqual(AdaptiveGearRuntime._normalize_configs(runtime, None), [])

        # Duplicate gears deduped and sorted ascending
        config = {"type": "BATCHSIZE", "gears": [64, 16, 32, 16]}
        result = AdaptiveGearRuntime._normalize_configs(runtime, [config])
        self.assertEqual(result[0]["gears"], [16, 32, 64])

        # Float gears (e.g. from JSON) cast to int
        config = {"type": "BATCHSIZE", "gears": [16.0, 32.0]}
        result = AdaptiveGearRuntime._normalize_configs(runtime, [config])
        self.assertEqual(result[0]["gears"], [16, 32])
        # each value is a plain int
        self.assertIsInstance(result[0]["gears"][0], int)

        # TIMES: min == max → single gear (no expansion needed)
        config = {"type": "BATCHSIZE", "min_size": 4, "max_size": 4}
        result = AdaptiveGearRuntime._normalize_configs(runtime, [config])
        self.assertEqual(result[0]["gears"], [4])

        # TIMES: non-power-of-2 min_size → min preserved, then powers of 2, max appended
        config = {"type": "BATCHSIZE", "min_size": 3, "max_size": 100}
        result = AdaptiveGearRuntime._normalize_configs(runtime, [config])
        # min_size is the anchor
        self.assertEqual(result[0]["gears"][0], 3)
        # max_size is the cap
        self.assertEqual(result[0]["gears"][-1], 100)
        for g in result[0]["gears"][1:-1]:
            # power-of-2 check
            self.assertEqual(g & (g - 1), 0, f"{g} is not a power of 2")

        # TIMES: power-of-2 min_size → next power of 2, then double
        config = {"type": "BATCHSIZE", "min_size": 8, "max_size": 128}
        result = AdaptiveGearRuntime._normalize_configs(runtime, [config])
        self.assertEqual(result[0]["gears"], [8, 16, 32, 64, 128])

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
        # 分割为两个组
        self.assertEqual(len(outputs), 2)
        # 第一段
        self.assertEqual(outputs[0][0].shape, (128, 128))
        # 第二段
        self.assertEqual(outputs[1][0].shape, (128, 128))
 
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


    
class TestShapeHandlingBranchCoverage(TestCase):
    def test_validate_configs_error_branches(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": 1}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "UNKNOWN"}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE", "dimensions": "0"}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE", "dimensions": [0, "1"]}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE", "min_size": 1.5}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE", "value": "bad"}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE", "policy": 1}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE", "policy": "BAD"}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs([{"type": "BATCHSIZE"}, {"type": "BATCHSIZE"}])

        with self.assertRaises(ValueError):
            shape_handling._validate_configs(
                [{"type": "BATCHSIZE"}, {"type": "SEQLEN"}, {"type": "BATCHSIZE"}]
            )

    def test_validate_adaptive_configs(self):
        """Test adaptive config validation: unknown keys, range errors, type compatibility."""
        shape_handling = torch_npu._inductor.NPUShapeHandling()

        # Unknown key
        with self.assertRaises(ValueError):
            shape_handling._validate_adaptive_configs({"unknown_key": 1})

        # Ratio out of range
        with self.assertRaises(ValueError):
            shape_handling._validate_adaptive_configs({"weight_hit": 1.5})
        with self.assertRaises(ValueError):
            shape_handling._validate_adaptive_configs({"pad_add_threshold": -0.1})

        # Seconds must be >= 0
        with self.assertRaises(ValueError):
            shape_handling._validate_adaptive_configs({"window_seconds": -1})
        # Zero is allowed (business layer guards against division-by-zero)
        shape_handling._validate_adaptive_configs({
            "window_seconds": 0,
            "recent_use_protect_seconds": 0,
            "update_interval_seconds": 0,
        })

        # Int must be >= 1
        with self.assertRaises(ValueError):
            shape_handling._validate_adaptive_configs({"min_samples_per_gear": 0})

        # Int accepted for float fields (ratio, seconds)
        shape_handling._validate_adaptive_configs({
            "weight_hit": 1,          # int for ratio field
            "window_seconds": 300,    # int for seconds field
        })

    def test_construct_indices_branches(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        tensors = [torch.randn(2, 3), torch.randn(2)]

        bs_indices = shape_handling._construct_indices(tensors, [], "BATCHSIZE")
        self.assertEqual(bs_indices, [0, 1])

        seq_indices_default = shape_handling._construct_indices(tensors, [], "SEQLEN")
        self.assertEqual(seq_indices_default, [0])

        seq_indices_dim0 = shape_handling._construct_indices(tensors, [0], "SEQLEN")
        self.assertEqual(seq_indices_dim0, [0, 1])

    def test_delay_initialize_builds_missing_indices(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "dimensions": [0],
                "indices": [],
                "gears": [8]
            },
            {
                "type": "SEQLEN",
                "dimensions": [1],
                "indices": [0],
                "gears": [8]
            }
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        self.assertTrue(shape_handling.delay_init)

        shape_handling.delay_initialize([torch.randn(4, 5)])
        self.assertEqual(shape_handling.configs[0]["indices"], [0])
        self.assertEqual(shape_handling.configs[1]["indices"], [0])
        self.assertFalse(shape_handling.delay_init)

    def test_flatten_and_unflatten_with_and_without_tensor(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()

        tensors, indices, leaves, spec = shape_handling.flatten_to_tensors({"a": 1, "b": "x"})
        self.assertEqual(list(tensors), [])
        self.assertEqual(list(indices), [])
        self.assertEqual(len(leaves), 2)
        self.assertIsNotNone(spec)

        structure = ((torch.tensor([1.0]), "k"), {"v": torch.tensor([2.0])})
        tensors, indices, leaves, spec = shape_handling.flatten_to_tensors(structure)
        rebuilt = shape_handling.unflatten_from_tensors(
            [torch.tensor([10.0]), torch.tensor([20.0])], indices, leaves, spec
        )
        self.assertEqual(rebuilt[0][1], "k")
        self.assertTrue(torch.equal(rebuilt[0][0], torch.tensor([10.0])))
        self.assertTrue(torch.equal(rebuilt[1]["v"], torch.tensor([20.0])))

    def test_transform_hook_default_and_custom_paths(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        input_a = torch.tensor([1.0])
        input_b = torch.tensor([2.0])
        transform_res = [
            [torch.tensor([3.0]), torch.tensor([4.0])],
            [torch.tensor([5.0]), torch.tensor([6.0])]
        ]

        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling, "transform", return_value=transform_res
        ):
            args_list, kwargs_list = shape_handling.transform_hook(input_a, extra=input_b)
            self.assertEqual(len(args_list), 2)
            self.assertEqual(len(kwargs_list), 2)
            self.assertTrue(torch.equal(args_list[0][0], torch.tensor([3.0])))
            self.assertTrue(torch.equal(kwargs_list[1]["extra"], torch.tensor([6.0])))

        recorded = {}

        def pre_fn(*args, **kwargs):
            recorded["pre"] = (args, kwargs)
            return [args[0]]

        def post_fn(outputs):
            recorded["post"] = outputs
            return [["ok"]], [{"done": True}]

        shape_handling_custom = torch_npu._inductor.NPUShapeHandling(
            transform_pre_fn=pre_fn,
            transform_post_fn=post_fn,
        )
        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling, "transform", return_value=[[torch.tensor([9.0])]]
        ):
            out_args, out_kwargs = shape_handling_custom.transform_hook(torch.tensor([7.0]))
            self.assertIn("pre", recorded)
            self.assertIn("post", recorded)
            self.assertEqual(out_args, [["ok"]])
            self.assertEqual(out_kwargs, [{"done": True}])

        shape_handling_none = torch_npu._inductor.NPUShapeHandling(transform_post_fn=lambda _: None)
        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling, "transform", return_value=[[torch.tensor([1.0])]]
        ):
            self.assertIsNone(shape_handling_none.transform_hook(torch.tensor([1.0])))

    def test_recover_hook_default_and_custom_paths(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        groups = [
            ((torch.tensor([1.0]), "x"), {"y": torch.tensor([2.0])})
        ]

        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling,
            "recover",
            side_effect=lambda tensor_groups: list(tensor_groups[0]),
        ):
            outputs = shape_handling.recover_hook(groups)
            self.assertEqual(outputs[0][1], "x")
            self.assertTrue(torch.equal(outputs[0][0], torch.tensor([1.0])))
            self.assertTrue(torch.equal(outputs[1]["y"], torch.tensor([2.0])))

        custom = torch_npu._inductor.NPUShapeHandling(
            recover_pre_fn=lambda groups: [[torch.tensor([5.0])]],
            recover_post_fn=lambda recover_res: {"result": recover_res},
        )
        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling, "recover", return_value=[torch.tensor([8.0])]
        ):
            outputs = custom.recover_hook(groups)
            self.assertEqual(list(outputs.keys()), ["result"])
            self.assertTrue(torch.equal(outputs["result"][0], torch.tensor([8.0])))

    def test_get_shape_safe_and_patch_shape_handling(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling()
        shape_info = shape_handling.get_shape_safe((torch.randn(2, 3), [torch.randn(1)]))
        self.assertEqual(shape_info[0], [2, 3])
        self.assertEqual(shape_info[1][0], [1])
        self.assertIs(shape_handling.get_shape_safe(1), int)

        if hasattr(shape_handling_module.patch_shape_handling, "_is_patched"):
            delattr(shape_handling_module.patch_shape_handling, "_is_patched")
        called = []
        with mock.patch.object(shape_handling_module, "patch_dynamo_context", side_effect=lambda: called.append(1)):
            shape_handling_module.patch_shape_handling()
            shape_handling_module.patch_shape_handling()
        self.assertEqual(len(called), 1)
        if hasattr(shape_handling_module.patch_shape_handling, "_is_patched"):
            delattr(shape_handling_module.patch_shape_handling, "_is_patched")

    def test_transform_metadata_collection(self):
        configs = [
            {
                "type": "BATCHSIZE",
                "gears": [64],
                "dimensions": 0,
                "indices": [0],
            },
            {
                "type": "SEQLEN",
                "gears": [128],
                "dimensions": [1],
                "indices": [0],
            },
        ]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)
        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling,
            "transform",
            return_value=[[torch.randn(64, 128)], [torch.randn(64, 128)]],
        ):
            _, metadata = shape_handling._transform_with_metadata(torch.randn(96, 96))
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata[0]["raw_gear_values"], [[96], [96]])
        self.assertEqual(metadata[0]["mapped_gear_values"], [[64], [128]])
        self.assertGreater(metadata[0]["pad_ratios"][1][0], 0.0)
        self.assertGreater(metadata[0]["split_ratios"][0][0], 0.0)

    def test_transform_metadata_collection_degrades_on_failure(self):
        """Metadata collection failure should not affect transform output."""
        configs = [{"type": "BATCHSIZE", "gears": [32], "dimensions": 0, "indices": [0]}]
        shape_handling = torch_npu._inductor.NPUShapeHandling(configs)

        with mock.patch.object(
            torch_npu._inductor.NPUShapeHandling, "transform",
            return_value=[[torch.randn(32, 8)]],
        ):
            with mock.patch(
                "torch_npu._inductor.adaptive_gears.collect_transform_metadata",
                side_effect=RuntimeError("boom"),
            ):
                outputs, metadata = shape_handling._transform_with_metadata(torch.randn(16, 8))

        # Transform output is intact
        self.assertEqual(len(outputs[0]), 1)
        # Metadata gracefully degraded to empty
        self.assertEqual(metadata, [])

class TestAsyncWorkerAndConcurrency(TestCase):
    """测试异步Worker和并发控制机制"""

    def test_async_worker_creation_and_execution(self):
        """测试后台Worker线程异步执行 run_once"""
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "update_interval_seconds": 0.1,
                "min_samples_per_gear": 1,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager
        # adaptive manager created
        self.assertIsNotNone(manager)
        # worker attached
        self.assertIsNotNone(manager.worker)
        # daemon thread running
        self.assertTrue(manager._worker_thread.is_alive())

        run_event = threading.Event()
        run_thread_id = None

        def mock_run_once(ts):
            nonlocal run_thread_id
            run_thread_id = threading.current_thread().ident
            run_event.set()

        with mock.patch.object(manager.worker, "run_once", side_effect=mock_run_once):
            # worker called run_once
            self.assertTrue(run_event.wait(timeout=5.0))

        # thread recorded id
        self.assertIsNotNone(run_thread_id)
        # ran on background thread
        self.assertNotEqual(run_thread_id, threading.current_thread().ident)
        manager.shutdown()

    def test_update_loop_survives_exception(self):
        """Worker thread should continue after run_once raises."""
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "update_interval_seconds": 0.1,
                "min_samples_per_gear": 1,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager

        call_count = 0
        run_event = threading.Event()

        def mock_run_once(ts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated failure")
            run_event.set()

        with mock.patch.object(manager.worker, "run_once", side_effect=mock_run_once):
            # 2nd call succeeded
            self.assertTrue(run_event.wait(timeout=5.0))

        # 1st raised, 2nd invoked → loop survived
        self.assertGreaterEqual(call_count, 2)
        manager.shutdown()

    def test_snapshot_isolation_after_update(self):
        """Snapshot taken before an update must still reflect old gear set (clone-on-read)."""
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "min_samples_per_gear": 1,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager

        snapshot1 = manager.get_snapshot()
        original_gears = snapshot1.active_gears["BATCHSIZE"].copy()

        manager.record_event([[20]], [[32]], [[0.375]], [[0.0]], 100.0)
        manager.worker.run_once(150.0)

        # old snapshot unchanged
        self.assertEqual(snapshot1.active_gears["BATCHSIZE"], original_gears)
        manager.shutdown()

    def test_high_concurrent_gear_update_scenarios(self):
        """测试高并发场景下的gear更新"""
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32, 64], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "min_samples_per_gear": 1,
                "add_min_samples": 1,
                "min_gear_count_per_type": 2,
                "recent_use_protect_seconds": 0.0,
                "replace_loss_threshold": 0.20,
            },
        )
        manager = shape_handling.adaptive_manager

        def run_update(timestamp):
            manager.record_event([[16], [64]], [[32], [64]], [[0.5], [0.0]], [[0.0], [0.0]], 100.0 + timestamp)
            manager.worker.run_once(200.0 + timestamp)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_update, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        latest_snapshot = manager.get_snapshot()
        self.assertIsNotNone(latest_snapshot)
        self.assertEqual(sorted(latest_snapshot.active_gears["BATCHSIZE"]), [32, 64])
        manager.shutdown()


class TestAdaptiveShapeHandling(TestCase):
    def test_npu_shape_handling_creates_snapshot_handler(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [64, 128], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={"recent_use_protect_seconds": 1.0},
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        self.assertEqual(snapshot.active_gears["BATCHSIZE"], [16, 32])
        self.assertEqual(snapshot.active_gears["SEQLEN"], [64, 128])
        self.assertIsNotNone(snapshot.shape_handling)

    def test_npu_shape_handling_records_event_and_builds_stats(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [64, 128], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={"window_seconds": 300.0},
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event(
            raw_gear_values=[[20], [96]],
            mapped_gear_values=[[32], [128]],
            pad_ratios=[[0.375], [0.25]],
            split_ratios=[[0.0], [0.0]],
            event_ts=100.0,
        )

        stats = manager.build_stats_snapshot(100.0)
        self.assertIn("BATCHSIZE:32", stats)
        self.assertIn("SEQLEN:128", stats)
        self.assertEqual(stats["BATCHSIZE:32"]["sample_count"], 1)
        self.assertEqual(stats["BATCHSIZE:32"]["pad_sample_count"], 1)
        self.assertEqual(stats["BATCHSIZE:32"]["split_sample_count"], 0)
        self.assertAlmostEqual(stats["BATCHSIZE:32"]["avg_pad_ratio"], 0.375)
        self.assertAlmostEqual(stats["SEQLEN:128"]["avg_pad_ratio"], 0.25)

    def test_npu_shape_handling_records_cleanup_keys_per_gear(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [64, 128], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={"window_seconds": 300.0},
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        fake_key = 42  # opaque integer key from pool.register()
        manager.record_event(
            raw_gear_values=[[20], [96]],
            mapped_gear_values=[[32], [128]],
            pad_ratios=[[0.375], [0.25]],
            split_ratios=[[0.0], [0.0]],
            event_ts=100.0,
            cleanup_key=fake_key,
        )

        self.assertEqual(manager._states["BATCHSIZE:32"].cleanup_keys, {fake_key})
        self.assertEqual(manager._states["SEQLEN:128"].cleanup_keys, {fake_key})
        self.assertIsInstance(next(iter(manager._states["BATCHSIZE:32"].cleanup_keys)), int)

    def test_npu_shape_handling_commit_update_uses_recorded_cleanup_keys(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={"window_seconds": 300.0},
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        fake_key = 42  # opaque integer key
        manager.record_event(
            raw_gear_values=[[20]],
            mapped_gear_values=[[32]],
            pad_ratios=[[0.375]],
            split_ratios=[[0.0]],
            event_ts=100.0,
            cleanup_key=fake_key,
        )

        _, removed_keys = manager.commit_update(
            configs=[{"type": "BATCHSIZE", "gears": [16], "dimensions": 0, "indices": [0], "policy": "CUSTOM"}],
            removed_gears=["BATCHSIZE:32"],
            now_ts=101.0,
        )

        self.assertEqual(removed_keys, [fake_key])
        self.assertIsInstance(removed_keys[0], int)
        self.assertNotIn("BATCHSIZE:32", manager._states)

    def test_npu_shape_handling_builds_separate_pad_and_split_sample_counts(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32, 64], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={"window_seconds": 300.0},
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[20]], [[32]], [[0.375]], [[0.0]], 100.0)
        manager.record_event([[80]], [[64]], [[0.0]], [[0.20]], 101.0)

        stats = manager.build_stats_snapshot(101.0)
        self.assertEqual(stats["BATCHSIZE:32"]["pad_sample_count"], 1)
        self.assertEqual(stats["BATCHSIZE:32"]["split_sample_count"], 0)
        self.assertEqual(stats["BATCHSIZE:64"]["pad_sample_count"], 0)
        self.assertEqual(stats["BATCHSIZE:64"]["split_sample_count"], 1)

    def test_npu_shape_handling_builds_two_dim_stats_per_dimension(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [64, 128], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={"window_seconds": 300.0},
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[20], [96]], [[32], [128]], [[0.375], [0.25]], [[0.0], [0.0]], 100.0)
        manager.record_event([[24], [112]], [[32], [128]], [[0.25], [0.125]], [[0.0], [0.0]], 101.0)

        stats = manager.build_stats_snapshot(101.0)
        self.assertEqual(stats["BATCHSIZE:32"]["sample_count"], 2)
        self.assertEqual(stats["BATCHSIZE:32"]["raw_samples"], [20, 24])
        self.assertEqual(stats["SEQLEN:128"]["sample_count"], 2)
        self.assertEqual(stats["SEQLEN:128"]["raw_samples"], [96, 112])

    def test_npu_shape_handling_recent_use_protect_skips_recent_hit_gear(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [8, 16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 60.0,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[20]], [[16]], [[0.375]], [[0.0]], 100.0)

        stats = manager.build_stats_snapshot(120.0)
        breakdowns = manager.scorer.build_score_breakdown(manager.get_snapshot(), stats)
        candidates = manager.worker.build_eviction_candidates(
            breakdowns,
            manager.get_snapshot(),
            stats,
            120.0,
        )
        self.assertEqual(candidates, {"BATCHSIZE": "BATCHSIZE:8"})

    def test_npu_shape_handling_eviction_protection_skips_gear(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [8, 16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 300.0,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[20]], [[32]], [[0.375]], [[0.0]], 0.0)

        manager.protect_gear_from_eviction("BATCHSIZE:16", 500.0)

        stats = manager.build_stats_snapshot(500.0)
        breakdowns = manager.scorer.build_score_breakdown(manager.get_snapshot(), stats)
        candidates = manager.worker.build_eviction_candidates(
            breakdowns,
            manager.get_snapshot(),
            stats,
            500.0,
        )
        self.assertEqual(candidates, {"BATCHSIZE": "BATCHSIZE:8"})

    def test_npu_shape_handling_zero_usage_gears_are_eviction_candidates(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 0.0,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[30]], [[32]], [[0.0625]], [[0.0]], 100.0)

        stats = manager.build_stats_snapshot(120.0)
        breakdowns = manager.scorer.build_score_breakdown(manager.get_snapshot(), stats)
        candidates = manager.worker.build_eviction_candidates(
            breakdowns,
            manager.get_snapshot(),
            stats,
            120.0,
        )
        # only unused gear is candidate
        self.assertEqual(candidates, {"BATCHSIZE": "BATCHSIZE:16"})

    def test_npu_shape_handling_update_adds_new_gear(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [64], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "min_samples_per_gear": 2,
                "add_min_samples": 2,
                "pad_add_threshold": 0.20,
                "recent_use_protect_seconds": 0.0,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[20], [96]], [[32], [64]], [[0.375], [0.50]], [[0.0], [0.0]], 100.0)
        manager.record_event([[20], [96]], [[32], [64]], [[0.375], [0.50]], [[0.0], [0.0]], 101.0)

        manager.worker.run_once(150.0)
        latest_snapshot = manager.get_snapshot()
        # pad-driven gear added
        self.assertIn(96, latest_snapshot.active_gears["SEQLEN"])

    def test_npu_shape_handling_new_zero_usage_gear_respects_recent_create_protection(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32, 64], "min_size": 1, "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "add_min_samples": 2,
                "pad_add_threshold": 0.20,
                "recent_use_protect_seconds": 60.0,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[20]], [[32]], [[0.375]], [[0.0]], 10.0)
        manager.record_event([[20]], [[32]], [[0.375]], [[0.0]], 11.0)

        manager.worker.run_once(100.0)

        latest_snapshot = manager.get_snapshot()
        # gear 20 was added
        self.assertIn(20, latest_snapshot.active_gears["BATCHSIZE"])

        stats = manager.build_stats_snapshot(120.0)
        breakdowns = manager.scorer.build_score_breakdown(latest_snapshot, stats)
        candidates = manager.worker.build_eviction_candidates(
            breakdowns, latest_snapshot, stats, 120.0,
        )
        # new gear has zero hits
        self.assertEqual(stats["BATCHSIZE:20"]["sample_count"], 0)
        # creation timestamp preserved
        self.assertEqual(stats["BATCHSIZE:20"]["created_ts"], 100.0)
        # gear 32 evictable, gear 20 protected
        self.assertEqual(candidates, {"BATCHSIZE": "BATCHSIZE:32"})

    def test_npu_shape_handling_update_adds_split_driven_gear(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [64], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "add_min_samples": 2,
                "pad_add_threshold": 0.90,
                "split_add_threshold": 0.10,
                "recent_use_protect_seconds": 0.0,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[96]], [[64]], [[0.0]], [[0.34]], 100.0)
        manager.record_event([[96]], [[64]], [[0.0]], [[0.34]], 101.0)

        manager.worker.run_once(150.0)
        latest_snapshot = manager.get_snapshot()
        # split-driven gear added at median
        self.assertIn(96, latest_snapshot.active_gears["BATCHSIZE"])

    def test_npu_shape_handling_two_dim_addition_candidates_are_built_per_dimension(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [8], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [32], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "add_min_samples": 2,
                "split_add_threshold": 0.10,
                "recent_use_protect_seconds": 0.0,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[16], [192]], [[8], [32]], [[0.0], [0.0]], [[0.50], [0.83]], 100.0)
        manager.record_event([[16], [192]], [[8], [32]], [[0.0], [0.0]], [[0.50], [0.83]], 101.0)
        manager.record_event([[48], [64]], [[8], [32]], [[0.0], [0.0]], [[0.83], [0.50]], 102.0)
        manager.record_event([[48], [64]], [[8], [32]], [[0.0], [0.0]], [[0.83], [0.50]], 103.0)

        stats = manager.build_stats_snapshot(103.0)
        candidates = manager.worker.build_addition_candidates(
            manager.get_snapshot(),
            stats,
        )
        result = {shape_type: value for _, _, shape_type, value in candidates}
        self.assertEqual(result, {"BATCHSIZE": 32, "SEQLEN": 128})

    def test_npu_shape_handling_update_evicts_low_value_gear(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [8, 16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [64], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "min_samples_per_gear": 1,
                "min_gear_count_per_type": 2,
                "recent_use_protect_seconds": 0.0,
                "replace_loss_threshold": 1.0,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[8], [64]], [[8], [64]], [[0.0], [0.0]], [[0.0], [0.0]], 10.0)
        manager.record_event([[8], [64]], [[8], [64]], [[0.0], [0.0]], [[0.0], [0.0]], 11.0)
        manager.record_event([[10], [64]], [[16], [64]], [[0.375], [0.0]], [[0.0], [0.0]], 12.0)
        manager.record_event([[32], [64]], [[32], [64]], [[0.0], [0.0]], [[0.0], [0.0]], 13.0)

        manager.worker.run_once(20.0)
        latest_snapshot = manager.get_snapshot()
        self.assertNotIn(16, latest_snapshot.active_gears["BATCHSIZE"])
        self.assertIn(8, latest_snapshot.active_gears["BATCHSIZE"])
        self.assertIn(32, latest_snapshot.active_gears["BATCHSIZE"])

    def test_npu_shape_handling_two_dim_evicts_per_dimension(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [8, 16, 32], "dimensions": 0, "indices": [0]},
                {"type": "SEQLEN", "gears": [32, 64, 128], "dimensions": [1], "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "min_samples_per_gear": 1,
                "min_gear_count_per_type": 2,
                "add_min_samples": 10,
                "recent_use_protect_seconds": 0.0,
                "replace_loss_threshold": 1.0,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[8], [64]], [[8], [64]], [[0.0], [0.0]], [[0.0], [0.0]], 18.0)
        manager.record_event([[10], [96]], [[16], [128]], [[0.375], [0.25]], [[0.0], [0.0]], 19.0)
        manager.record_event([[32], [64]], [[32], [64]], [[0.0], [0.0]], [[0.0], [0.0]], 19.0)

        manager.worker.run_once(20.0)
        latest_snapshot = manager.get_snapshot()
        self.assertEqual(latest_snapshot.active_gears["BATCHSIZE"], [8, 32])
        self.assertEqual(latest_snapshot.active_gears["SEQLEN"], [64, 128])

    def test_npu_shape_handling_resource_pressure_triggers_update(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 300.0,
                "device_memory_usage_threshold_ratio": 0.80,
            },
        )
        manager = shape_handling.adaptive_manager
        # Memory usage ratio = 1.0 - 20/100 = 0.80 >= threshold → high pressure
        with mock.patch("torch_npu._inductor.adaptive_gears.torch.npu.mem_get_info", return_value=(20, 100)):
            budget = manager.build_resource_budget()
            self.assertTrue(budget["device_memory_usage_high"])
        # Memory usage ratio = 1.0 - 70/100 = 0.30 < threshold → no pressure
        with mock.patch("torch_npu._inductor.adaptive_gears.torch.npu.mem_get_info", return_value=(70, 100)):
            budget = manager.build_resource_budget()
            self.assertFalse(budget["device_memory_usage_high"])

    def test_npu_shape_handling_resource_budget_uses_device_ratio(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "device_memory_usage_threshold_ratio": 0.60,
            },
        )
        manager = shape_handling.adaptive_manager
        with mock.patch("torch_npu._inductor.adaptive_gears.torch.npu.mem_get_info", return_value=(40, 100)):
            budget = manager.build_resource_budget()

        self.assertAlmostEqual(budget["device_memory_usage_ratio"], 0.60)
        self.assertEqual(budget["device_memory_usage_threshold_ratio"], 0.60)
        self.assertTrue(budget["device_memory_usage_high"])

    def test_npu_shape_handling_device_pressure_blocks_direct_add(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [64], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 1.0,
                "add_min_samples": 2,
                "pad_add_threshold": 0.20,
                "device_memory_usage_threshold_ratio": 0.50,
                "min_gear_count_per_type": 1,
                "recent_use_protect_seconds": 0.0,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[40]], [[64]], [[0.375]], [[0.0]], 100.0)
        manager.record_event([[40]], [[64]], [[0.375]], [[0.0]], 101.0)

        with mock.patch("torch_npu._inductor.adaptive_gears.torch.npu.mem_get_info", return_value=(40, 100)):
            manager.worker.run_once(150.0)
        latest_snapshot = manager.get_snapshot()
        self.assertEqual(latest_snapshot.active_gears["BATCHSIZE"], [64])

    def test_npu_shape_handling_max_gear_guard_blocks_unsafe_delete(self):
        shape_handling = torch_npu._inductor.NPUShapeHandling(
            configs=[
                {"type": "BATCHSIZE", "gears": [32, 64], "dimensions": 0, "indices": [0]},
            ],
            adaptive_configs={
                "recent_use_protect_seconds": 0.0,
                "replace_loss_threshold": 1.0,
                "min_gear_count_per_type": 1,
            },
        )
        manager = shape_handling.adaptive_manager
        snapshot = manager.get_snapshot()
        manager.record_event([[40]], [[64]], [[0.375]], [[0.0]], 100.0)

        stats = manager.build_stats_snapshot(100.0)
        breakdowns = manager.scorer.build_score_breakdown(manager.get_snapshot(), stats)
        candidates = manager.worker.build_eviction_candidates(
            breakdowns,
            manager.get_snapshot(),
            stats,
            100.0,
        )
        self.assertNotIn("BATCHSIZE:64", candidates.values())

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

    def test_npu_shape_handling_whit_mutil_compile(self):
        # 多次编译同一个函数，验证 shape handling 是否正常工作
        # 1. 首次编译触发 shape handling
        # 2. 后续编译复用已有的 shape handling 逻辑
        try:
            compiled_fn = torch.compile(
                model_fn, 
                backend='inductor', 
                dynamic=False, 
                options=shape_options
            )
            # 第一次spilt
            compiled_fn(torch.randn((1025, 32), device=device), torch.randn((1025, 32), device=device))

            # 第二次spilt
            compiled_fn(torch.randn((2048, 32), device=device), torch.randn((2048, 32), device=device))

            # 第三次spilt
            compiled_fn(torch.randn((10240, 32), device=device), torch.randn((10240, 32), device=device))
        except Exception as e:
            self.fail(f"torch.compile raised {type(e).__name__} unexpectedly: ")


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


    def test_deepcopy_failure_returns_original(self):
        class NonCopyable:
            def __deepcopy__(self, memo):
                raise TypeError("deepcopy not supported")

        obj = NonCopyable()
        copied = unified_copy(obj)
        self.assertIs(copied, obj)


# ---------------------------------------------------------------------------
# ST: Adaptive Gear end-to-end integration tests
# ---------------------------------------------------------------------------

_ST_ADAPTIVE_CONFIGS = {
    "window_seconds": 300.0,
    "recent_use_protect_seconds": 0,
    "pad_add_threshold": 0.20,
    "split_add_threshold": 0.20,
    "add_min_samples": 2,
    "min_samples_per_gear": 1,
    "min_gear_count_per_type": 2,
    "replace_loss_threshold": 1.0,
    "update_interval_seconds": 9999.0,
}

_ST_DAEMON_ADAPTIVE_CONFIGS = dict(_ST_ADAPTIVE_CONFIGS)
_ST_DAEMON_ADAPTIVE_CONFIGS["update_interval_seconds"] = 0.1


def _make_adaptive_options(shape_configs, adaptive_configs=None):
    opts = {
        "enable_shape_handling": True,
        "shape_handling_configs": shape_configs,
    }
    if adaptive_configs is not None:
        opts["shape_handling_dict"] = {"adaptive_gears": adaptive_configs}
    return opts


def _run_model(compiled_fn, shape):
    A = torch.randn(shape, device=device)
    B = torch.randn(shape, device=device)
    out = compiled_fn(A, B)
    return out, A, B


class TestAdaptiveGearsCompileST(TestCase):

    def setUp(self):
        torch._dynamo.reset()
        self._captured_managers = []
        self._original_sh_init = shape_handling_module.NPUShapeHandling.__init__

        test_self = self

        def capturing_init(sh_self, *args, **kwargs):
            test_self._original_sh_init(sh_self, *args, **kwargs)
            if sh_self.adaptive_manager is not None:
                test_self._captured_managers.append(sh_self.adaptive_manager)

        shape_handling_module.NPUShapeHandling.__init__ = capturing_init

    def tearDown(self):
        shape_handling_module.NPUShapeHandling.__init__ = self._original_sh_init
        for mgr in self._captured_managers:
            try:
                mgr.shutdown()
            except Exception:
                pass
        self._captured_managers.clear()
        torch._dynamo.reset()

    def _get_manager(self):
        self.assertTrue(
            len(self._captured_managers) > 0,
            "No AdaptiveGearRuntime was captured — torch.compile did not create one",
        )
        return self._captured_managers[-1]

    def _stop_daemon(self, manager):
        manager._shutdown_event.set()
        manager._worker_thread.join(timeout=2.0)

    def test_adaptive_gears_basic_compile_and_compute(self):
        shape_configs = [
            {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0, 1]},
        ]
        options = _make_adaptive_options(shape_configs, _ST_ADAPTIVE_CONFIGS)

        compiled_fn = torch.compile(
            model_fn,
            backend="inductor",
            dynamic=False,
            options=options,
        )

        for shape in [(8, 32), (24, 32), (10, 32)]:
            out, A, B = _run_model(compiled_fn, shape)
            self.assertTrue(
                torch.allclose(out, A + B),
                f"Output mismatch for shape {shape}",
            )

        manager = self._get_manager()

        self.assertTrue(
            manager._worker_thread.is_alive(),
            "Daemon worker thread should be alive",
        )

    def test_adaptive_gears_event_recording_and_stats(self):
        shape_configs = [
            {"type": "BATCHSIZE", "gears": [16, 32], "dimensions": 0, "indices": [0, 1]},
        ]
        options = _make_adaptive_options(shape_configs, _ST_ADAPTIVE_CONFIGS)

        compiled_fn = torch.compile(
            model_fn,
            backend="inductor",
            dynamic=False,
            options=options,
        )
        manager = self._get_manager()

        for _ in range(3):
            out, A, B = _run_model(compiled_fn, (24, 32))
            self.assertTrue(torch.allclose(out, A + B))

        stats = manager.build_stats_snapshot(time.time())

        self.assertIn("BATCHSIZE:32", stats)
        self.assertEqual(
            stats["BATCHSIZE:32"]["sample_count"], 6,
            "Exactly 6 samples should be recorded for gear 32 (3 calls × 2 tensors)",
        )
        self.assertEqual(
            stats["BATCHSIZE:32"]["pad_sample_count"], 6,
            "All 6 events should be classified as padding",
        )
        self.assertAlmostEqual(
            stats["BATCHSIZE:32"]["avg_pad_ratio"], 0.25, places=2,
            msg="avg_pad_ratio should be (32-24)/32 = 0.25",
        )
        self.assertEqual(
            stats["BATCHSIZE:32"]["raw_samples"], [24, 24, 24, 24, 24, 24],
            "raw_samples should have 6 entries (3 calls × 2 tensors)",
        )

    def test_adaptive_gears_gear_eviction_synchronous(self):
        shape_configs = [
            {"type": "BATCHSIZE", "gears": [16, 32, 64], "dimensions": 0, "indices": [0, 1]},
        ]
        options = _make_adaptive_options(shape_configs, _ST_ADAPTIVE_CONFIGS)

        compiled_fn = torch.compile(
            model_fn,
            backend="inductor",
            dynamic=False,
            options=options,
        )
        manager = self._get_manager()
        self._stop_daemon(manager)

        for _ in range(5):
            out, A, B = _run_model(compiled_fn, (24, 32))
            self.assertTrue(torch.allclose(out, A + B))

        manager.worker.run_once(time.time())

        snapshot_after = manager.get_snapshot()
        gears_after = sorted(snapshot_after.active_gears["BATCHSIZE"])

        # Gear 16 evicted (zero hits) + gear 24 added (pad_ratio=0.25 > threshold 0.20)
        self.assertEqual(
            gears_after, [24, 32, 64],
            f"Expected [24, 32, 64] after eviction+addition, got {gears_after}",
        )

        out, A, B = _run_model(compiled_fn, (8, 32))
        self.assertTrue(
            torch.allclose(out, A + B),
            "Computation should be correct after gear eviction",
        )

    def test_adaptive_gears_gear_addition_synchronous(self):
        shape_configs = [
            {"type": "BATCHSIZE", "gears": [64], "dimensions": 0, "indices": [0, 1]},
        ]
        add_configs = dict(_ST_ADAPTIVE_CONFIGS)
        add_configs["min_gear_count_per_type"] = 1
        options = _make_adaptive_options(shape_configs, add_configs)

        compiled_fn = torch.compile(
            model_fn,
            backend="inductor",
            dynamic=False,
            options=options,
        )
        manager = self._get_manager()
        self._stop_daemon(manager)

        for _ in range(3):
            out, A, B = _run_model(compiled_fn, (20, 32))
            self.assertTrue(torch.allclose(out, A + B))

        manager.worker.run_once(time.time())

        snapshot_after = manager.get_snapshot()
        gears_after = snapshot_after.active_gears["BATCHSIZE"]

        self.assertEqual(
            len(gears_after), 2,
            f"A new gear should be added, got {gears_after}",
        )
        self.assertIn(
            20, gears_after,
            f"New gear 20 (median of raw samples) should be added, got {gears_after}",
        )

        out, A, B = _run_model(compiled_fn, (20, 32))
        self.assertTrue(
            torch.allclose(out, A + B),
            "Computation should be correct after gear addition",
        )

    def test_adaptive_gears_graph_cleanup_after_eviction(self):
        from torch_npu.npu._graph_resource_pool import GraphResourcePool

        shape_configs = [
            {"type": "BATCHSIZE", "gears": [16, 32, 64], "dimensions": 0, "indices": [0, 1]},
        ]
        options = _make_adaptive_options(shape_configs, _ST_ADAPTIVE_CONFIGS)
        options["triton.cudagraphs"] = True
        options["triton.cudagraph_trees"] = True

        compiled_fn = torch.compile(
            model_fn,
            backend="inductor",
            dynamic=False,
            options=options,
        )
        manager = self._get_manager()
        self._stop_daemon(manager)

        for shape in [(8, 32), (24, 32), (48, 32)]:
            for _ in range(2):
                out, A, B = _run_model(compiled_fn, shape)
                self.assertTrue(torch.allclose(out, A + B))

        device_index = torch.npu.current_device()
        pool = GraphResourcePool.get_pool(device_index)

        self.assertGreater(
            pool.entry_count, 0,
            f"Pool should have entries after compilation, got {pool.entry_count}",
        )

        manager.worker.run_once(time.time())

        snapshot_after = manager.get_snapshot()
        gears_after = sorted(snapshot_after.active_gears["BATCHSIZE"])

        self.assertEqual(
            gears_after, [8, 24, 32, 48, 64],
            f"Expected [8, 24, 32, 48, 64] after eviction+additions, got {gears_after}",
        )

        out, A, B = _run_model(compiled_fn, (24, 32))
        self.assertTrue(
            torch.allclose(out, A + B),
            "Computation should be correct after graph cleanup",
        )

    def test_adaptive_gears_daemon_thread_triggers_update(self):
        shape_configs = [
            {"type": "BATCHSIZE", "gears": [16, 32, 64], "dimensions": 0, "indices": [0, 1]},
        ]
        options = _make_adaptive_options(shape_configs, _ST_DAEMON_ADAPTIVE_CONFIGS)

        compiled_fn = torch.compile(
            model_fn,
            backend="inductor",
            dynamic=False,
            options=options,
        )
        manager = self._get_manager()
        original_gears = sorted(manager.get_snapshot().active_gears["BATCHSIZE"])

        for _ in range(5):
            out, A, B = _run_model(compiled_fn, (24, 32))
            self.assertTrue(torch.allclose(out, A + B))

        self.assertTrue(
            manager._worker_thread.is_alive(),
            "Daemon thread should be alive",
        )

        deadline = time.time() + 5.0
        current_gears = original_gears
        while time.time() < deadline:
            current_gears = sorted(manager.get_snapshot().active_gears["BATCHSIZE"])
            if current_gears != original_gears:
                break
            time.sleep(0.1)

        self.assertNotEqual(
            current_gears, original_gears,
            f"Daemon thread should trigger gear update within 5s. "
            f"Original: {original_gears}, Current: {current_gears}",
        )

        out, A, B = _run_model(compiled_fn, (24, 32))
        self.assertTrue(
            torch.allclose(out, A + B),
            "Computation should be correct after daemon-triggered update",
        )


instantiate_parametrized_tests(TestShapeHandling)
instantiate_parametrized_tests(TestShapeHandlingBranchCoverage)
instantiate_parametrized_tests(TestAsyncWorkerAndConcurrency)
instantiate_parametrized_tests(TestAdaptiveShapeHandling)
instantiate_parametrized_tests(TestUnifiedCopy)
instantiate_parametrized_tests(TestDynamicShapeCompile)
instantiate_parametrized_tests(TestAdaptiveGearsCompileST)
 
if __name__ == '__main__':
    run_tests()