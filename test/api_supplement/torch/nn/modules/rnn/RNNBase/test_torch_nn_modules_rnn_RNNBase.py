# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.rnn.RNNBase 接口功能正确性
API 名称：torch.nn.modules.rnn.RNNBase
API 签名：RNNBase(mode, input_size, hidden_size, num_layers=1, bias=True,
          batch_first=False, dropout=0.0, bidirectional=False, proj_size=0,
          device=None, dtype=None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 合法/非法 mode、bias=False 无偏置参数 | 已覆盖（test_rnn_base_modes_gate_size、
test_rnn_base_invalid_mode、test_rnn_base_bias_false_param_names） |
| 枚举选项 | mode=LSTM/GRU/RNN_TANH/RNN_RELU 全枚举，门控尺寸 4/3/1/1 | 已覆盖（test_rnn_base_modes_gate_size） |
| 参数类型 | dropout/bias/batch_first/input_size/hidden_size/num_layers 类型校验 | 已覆盖（test_rnn_base_dropout_range_validation、
test_rnn_base_type_validation、test_rnn_base_positive_value_validation） |
| 传参与不传参 | 默认构造与显式 num_layers/bidirectional/proj_size/dropout/device/dtype | 已覆盖（test_rnn_base_modes_gate_size、
test_rnn_base_param_names_bidirectional_multilayer、test_rnn_base_device_dtype_kwargs） |
| 等价类/边界值 | 尺寸为 0、proj_size 为负或 >= hidden_size、proj_size>0 增加 weight_hr | 已覆盖（test_rnn_base_positive_value_validation、
test_rnn_base_proj_size_validation、test_rnn_base_proj_size） |
| 正常传参场景 | 属性回填、参数命名、all_weights 同一性、setattr 与 to() 同步、
extra_repr、check_input/check_hidden_size 契约 | 已覆盖（test_rnn_base_param_names_bidirectional_multilayer、
test_rnn_base_all_weights_identity、test_rnn_base_setattr_updates_flat_weights、
test_rnn_base_to_updates_flat_weights、test_rnn_base_extra_repr、
test_rnn_base_check_input_and_hidden_size） |
| 异常传参场景 | 非法 mode、dropout 越界/布尔、类型错误、非正值、forward 未实现 | 已覆盖（test_rnn_base_invalid_mode、
test_rnn_base_dropout_range_validation、test_rnn_base_forward_not_implemented） |
| 混合设备类型 | CPU state_dict 加载到设备模块后逐参数对照（CPU Golden） | 已覆盖（test_rnn_base_state_dict_cross_device） |

未覆盖项及原因：
- permute_hidden 依赖已弃用的 apply_permutation（FutureWarning 路径），不在本类 API 范围内单独看护。
- RNNBase 自身不实现 forward（仅校验工具方法），无反向传播路径，反向维度不适用。
- dropout>0 时的真实丢弃行为在子类 forward 中生效，属于 RNN/LSTM/GRU 的用例范围，本文件只看护其参数校验与告警。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行对照；
     整数与结构性结果使用精确断言。
"""
import math
import os
import unittest
import warnings

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化；
# 被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

from torch.nn.modules.rnn import RNNBase


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRNNBase(TestCase):
    """Test cases for torch.nn.modules.rnn.RNNBase (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_rnn_base_modes_gate_size(self):
        """All four legal modes build modules with documented gate sizes."""
        gates_by_mode = {"LSTM": 4, "GRU": 3, "RNN_TANH": 1, "RNN_RELU": 1}
        for mode, gates in gates_by_mode.items():
            module = RNNBase(mode, 10, 20)
            self.assertEqual(module.mode, mode)
            self.assertEqual(module.input_size, 10)
            self.assertEqual(module.hidden_size, 20)
            self.assertEqual(module.num_layers, 1)
            self.assertEqual(module.bias, True)
            self.assertEqual(module.batch_first, False)
            self.assertEqual(module.dropout, 0.0)
            self.assertEqual(module.bidirectional, False)
            self.assertEqual(module.proj_size, 0)
            self.assertEqual(tuple(module.weight_ih_l0.shape), (gates * 20, 10))
            self.assertEqual(tuple(module.weight_hh_l0.shape), (gates * 20, 20))
            self.assertEqual(len(list(module.parameters())), 4)

    def test_rnn_base_param_names_bidirectional_multilayer(self):
        """num_layers=2 + bidirectional enumerates all 16 parameter names in order."""
        module = RNNBase("LSTM", 10, 20, num_layers=2, bidirectional=True)
        expected = []
        for layer in range(2):
            for suffix in ["", "_reverse"]:
                expected.extend(
                    [
                        f"weight_ih_l{layer}{suffix}",
                        f"weight_hh_l{layer}{suffix}",
                        f"bias_ih_l{layer}{suffix}",
                        f"bias_hh_l{layer}{suffix}",
                    ]
                )
        actual = [name for name, _ in module.named_parameters()]
        self.assertEqual(actual, expected)
        self.assertEqual(len(module._flat_weights), 16)

    def test_rnn_base_bias_false_param_names(self):
        """bias=False drops both bias tensors and keeps weights only."""
        module = RNNBase("GRU", 10, 20, bias=False)
        actual = [name for name, _ in module.named_parameters()]
        self.assertEqual(actual, ["weight_ih_l0", "weight_hh_l0"])

    def test_rnn_base_proj_size(self):
        """proj_size>0 adds weight_hr of shape (proj_size, hidden_size)."""
        module = RNNBase("LSTM", 10, 20, proj_size=5)
        self.assertEqual(tuple(module.weight_hr_l0.shape), (5, 20))
        names = [name for name, _ in module.named_parameters()]
        self.assertEqual(
            names,
            [
                "weight_ih_l0",
                "weight_hh_l0",
                "bias_ih_l0",
                "bias_hh_l0",
                "weight_hr_l0",
            ],
        )

    def test_rnn_base_all_weights_identity(self):
        """all_weights resolves to the very same Parameter objects (assertIs)."""
        module = RNNBase("LSTM", 10, 20, num_layers=2, bidirectional=True)
        all_weights = module.all_weights
        self.assertEqual(len(all_weights), 4)
        named = dict(module.named_parameters())
        values = list(named.values())
        for layer_weights in all_weights:
            self.assertEqual(len(layer_weights), 4)
            for weight in layer_weights:
                # Identity check only: Parameter == Parameter would compare tensors.
                self.assertTrue(
                    any(p is weight for p in values),
                    "all_weights entry is not among named_parameters objects",
                )

    def test_rnn_base_invalid_mode(self):
        """Unknown mode raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Unrecognized RNN mode"):
            RNNBase("BAD", 10, 20)

    def test_rnn_base_dropout_range_validation(self):
        """dropout outside [0, 1] or of bool type raises ValueError."""
        with self.assertRaisesRegex(ValueError, "dropout should be a number"):
            RNNBase("LSTM", 10, 20, dropout=1.5)
        with self.assertRaisesRegex(ValueError, "dropout should be a number"):
            RNNBase("LSTM", 10, 20, dropout=-0.5)
        with self.assertRaisesRegex(ValueError, "dropout should be a number"):
            RNNBase("LSTM", 10, 20, dropout=True)

    def test_rnn_base_type_validation(self):
        """Non-bool bias/batch_first and non-int sizes raise TypeError."""
        with self.assertRaisesRegex(TypeError, "bias should be of type bool"):
            RNNBase("LSTM", 10, 20, bias=1)
        with self.assertRaisesRegex(TypeError, "batch_first should be of type bool"):
            RNNBase("LSTM", 10, 20, batch_first="x")
        with self.assertRaisesRegex(TypeError, "input_size should be of type int"):
            RNNBase("LSTM", "10", 20)
        with self.assertRaisesRegex(TypeError, "hidden_size should be of type int"):
            RNNBase("LSTM", 10, "20")

    def test_rnn_base_positive_value_validation(self):
        """Non-positive input_size/hidden_size/num_layers raise ValueError."""
        with self.assertRaisesRegex(ValueError, "input_size must be greater than zero"):
            RNNBase("LSTM", 0, 20)
        with self.assertRaisesRegex(ValueError, "hidden_size must be greater than zero"):
            RNNBase("LSTM", 10, 0)
        with self.assertRaisesRegex(ValueError, "num_layers must be greater than zero"):
            RNNBase("LSTM", 10, 20, num_layers=0)

    def test_rnn_base_proj_size_validation(self):
        """proj_size must be zero or a positive int smaller than hidden_size."""
        with self.assertRaisesRegex(ValueError, "proj_size should be a positive integer or zero"):
            RNNBase("LSTM", 10, 20, proj_size=-1)
        with self.assertRaisesRegex(ValueError, "proj_size has to be smaller than hidden_size"):
            RNNBase("LSTM", 10, 20, proj_size=20)

    def test_rnn_base_dropout_num_layers_warning(self):
        """dropout>0 with num_layers==1 emits a UserWarning about num_layers."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RNNBase("RNN_TANH", 10, 20, dropout=0.5, num_layers=1)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("dropout" in m and "num_layers" in m for m in messages),
            f"expected dropout/num_layers warning, got: {messages}",
        )

    def test_rnn_base_forward_not_implemented(self):
        """RNNBase itself has no forward; calling the module raises NotImplementedError."""
        module = RNNBase("RNN_TANH", 10, 20)
        with self.assertRaises(NotImplementedError):
            module(torch.zeros(3, 4, 10))

    def test_rnn_base_setattr_updates_flat_weights(self):
        """Assigning a weight attribute keeps _flat_weights pointing at the new tensor."""
        module = RNNBase("RNN_TANH", 10, 20)
        replacement = torch.nn.Parameter(torch.zeros(3, 10))
        module.weight_ih_l0 = replacement
        self.assertIs(module._flat_weights[0], replacement)
        self.assertIs(module.weight_ih_l0, replacement)

    def test_rnn_base_to_updates_flat_weights(self):
        """Module.to(dtype) re-syncs _flat_weights with the converted parameters."""
        module = RNNBase("RNN_TANH", 10, 20)
        converted = module.to(torch.float64)
        self.assertEqual(converted._flat_weights[0].dtype, torch.float64)
        self.assertEqual(converted.weight_ih_l0.dtype, torch.float64)
        self.assertIs(converted._flat_weights[0], converted.weight_ih_l0)

    def test_rnn_base_extra_repr(self):
        """extra_repr lists non-default options and omits default ones."""
        module = RNNBase("LSTM", 10, 20)
        self.assertEqual(repr(module), "RNNBase(10, 20)")
        module = RNNBase(
            "LSTM", 10, 20, num_layers=2, dropout=0.5, bidirectional=True, proj_size=4
        )
        self.assertEqual(
            repr(module),
            "RNNBase(10, 20, proj_size=4, num_layers=2, dropout=0.5, bidirectional=True)",
        )

    def test_rnn_base_check_input_and_hidden_size(self):
        """check_input/get_expected_hidden_size/check_hidden_size enforce input contracts."""
        module = RNNBase("LSTM", 10, 20)
        module.check_input(torch.zeros(3, 4, 10), None)
        module.check_input(torch.zeros(4, 10), torch.tensor([4, 3, 1]))
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            module.check_input(torch.zeros(3, 10), None)
        with self.assertRaisesRegex(RuntimeError, "input_size"):
            module.check_input(torch.zeros(3, 4, 8), None)
        with self.assertRaisesRegex(ValueError, "dtype"):
            module.check_input(torch.zeros(3, 4, 10, dtype=torch.float64), None)
        self.assertEqual(
            module.get_expected_hidden_size(torch.zeros(3, 5, 10), None), (1, 5, 20)
        )
        self.assertEqual(
            module.get_expected_hidden_size(torch.zeros(2, 10), torch.tensor([4, 3, 1])),
            (1, 4, 20),
        )
        module.check_hidden_size(torch.zeros(1, 5, 20), (1, 5, 20))
        with self.assertRaisesRegex(RuntimeError, "Expected hidden size"):
            module.check_hidden_size(torch.zeros(2, 5, 20), (1, 5, 20))
        bi = RNNBase("RNN_TANH", 10, 20, bidirectional=True)
        self.assertEqual(
            bi.get_expected_hidden_size(torch.zeros(3, 5, 10), None), (2, 5, 20)
        )
        proj = RNNBase("LSTM", 10, 20, proj_size=6)
        self.assertEqual(
            proj.get_expected_hidden_size(torch.zeros(3, 5, 10), None), (1, 5, 6)
        )

    def test_rnn_base_device_dtype_kwargs(self):
        """device/dtype factory kwargs place initialized parameters on the device."""
        module = RNNBase("GRU", 6, 8, device=self.device_name)
        for name, param in module.named_parameters():
            self.assertEqual(param.device.type, self.device_name, f"param {name}")
            self.assertEqual(param.dtype, torch.float32, f"param {name}")
        stdv = 1.0 / math.sqrt(8)
        for name, param in module.named_parameters():
            self.assertTrue(
                bool((param.detach().abs() <= stdv + 1e-7).all()),
                f"param {name} outside uniform init range [-stdv, stdv]",
            )
        # dtype kwarg contract verified on CPU: float64 parameters.
        cpu_fp64 = RNNBase("GRU", 6, 8, dtype=torch.float64)
        self.assertEqual(cpu_fp64.weight_ih_l0.dtype, torch.float64)

    def test_rnn_base_state_dict_cross_device(self):
        """CPU state_dict loads into a device module and matches the CPU Golden."""
        source = RNNBase("LSTM", 10, 20)
        for name, param in source.named_parameters():
            flat = torch.arange(param.numel(), dtype=torch.float32) * 0.01 - 1.0
            with torch.no_grad():
                param.copy_(flat.reshape(param.shape))
        state_dict = {key: value.clone() for key, value in source.state_dict().items()}
        target = RNNBase("LSTM", 10, 20, device=self.device_name)
        target.load_state_dict(state_dict)
        for name, param in source.named_parameters():
            self.assertEqual(target.state_dict()[name].device.type, self.device_name)
            torch.testing.assert_close(
                target.state_dict()[name].cpu(), param.detach().cpu(), rtol=0.0, atol=0.0
            )

    def test_rnn_base_deterministic_fixed_seed(self):
        """Fixed seed on CPU yields bit-identical initialization across constructions."""
        torch.manual_seed(1234)
        first = RNNBase("LSTM", 6, 8)
        torch.manual_seed(1234)
        second = RNNBase("LSTM", 6, 8)
        for (name_a, param_a), (name_b, param_b) in zip(
            first.named_parameters(), second.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(param_a.detach(), param_b.detach(), rtol=0.0, atol=0.0)

    def test_rnn_base_memory_repeated_construction(self):
        """Repeated construction on device does not keep growing allocated memory."""
        if self.device_name == 'cpu':
            self.skipTest("requires a device allocator (NPU/CUDA)")
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        for _ in range(10):
            module = RNNBase("LSTM", 8, 16, device=self.device_name)
            self.assertEqual(module.weight_ih_l0.device.type, self.device_name)
            del module
        torch.cuda.empty_cache()
        grown = torch.cuda.memory_allocated()
        self.assertLessEqual(grown - base, 1024 * 1024)


if __name__ == "__main__":
    run_tests()
