# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.quantization.default_eval_fn 接口功能正确性
API 名称：torch.quantization.default_eval_fn
API 签名：default_eval_fn(model, calib_data)

覆盖维度表：
| 覆盖维度      | 说明                                   | 覆盖情况                                                |
|---------------|----------------------------------------|---------------------------------------------------------|
| 空/非空       | calib_data 为空 / 非空                 | 已覆盖：test_empty_calib_data、                         |
|               |                                        | test_list_of_pairs_invokes_model                        |
| 枚举选项      | calib_data 为 list / Dataset 两种形态  | 已覆盖：test_list_of_pairs_invokes_model、              |
|               |                                        | test_tensor_dataset_invokes_model                       |
| 参数类型      | model 为 nn.Module，calib_data         | 已覆盖：test_tensor_dataset_invokes_model               |
|               | 为可迭代对象                           |                                                         |
| 传参与不传参  | 该接口无可选参数                       | 不适用                                                  |
| 等价类/边界值 | 仅 data 传入 model（不含 target）；    | 已覆盖：test_model_receives_data_not_target             |
|               | 空迭代器                               |                                                         |
| 正常传参场景  | 逐样本前向，返回 None                  | 已覆盖：test_returns_none、                             |
|               |                                        | test_list_of_pairs_invokes_model                        |
| 异常传参场景  | 样本非二元组触发 ValueError            | 已覆盖：test_invalid_pair_length_raises                 |
| 混合设备类型  | 多 Tensor 异构设备输入                 | 未覆盖，见下                                            |

未覆盖项及原因：
- 传参与不传参：接口仅两个必填参数，无可选参数可覆盖。
- 混合设备类型：default_eval_fn 仅按数据迭代调用 model，是否异构设备由模型/数据决定；
  量化标定在 CPU 上执行，无 NPU/CUDA 对照路径，故不构造异构输入。

注：该 API 为纯 Python 驱动工具（框架类 A），不产生数值输出，按调用次数与传入对象身份断言，
     不做 CPU Golden 对照。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

from torch.quantization import default_eval_fn


class _Recorder(torch.nn.Module):
    """Records every input tensor it is called with."""

    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, x):
        self.received.append(x)
        return x


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDefaultEvalFnUt(TestCase):
    """Functional tests for torch.quantization.default_eval_fn."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_list_of_pairs_invokes_model(self):
        tensors = [torch.randn(1, 3), torch.randn(2, 3)]
        model = _Recorder()
        default_eval_fn(model, [(tensors[0], 0), (tensors[1], 1)])
        self.assertEqual(len(model.received), 2)

    def test_model_receives_data_not_target(self):
        data = torch.randn(1, 3)
        model = _Recorder()
        default_eval_fn(model, [(data, 42)])
        # The exact data tensor (not the target) must be forwarded into the model.
        self.assertIs(model.received[0], data)

    def test_tensor_dataset_invokes_model(self):
        images = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 0, 1])
        dataset = torch.utils.data.TensorDataset(images, labels)
        model = _Recorder()
        default_eval_fn(model, dataset)
        self.assertEqual(len(model.received), 4)

    def test_empty_calib_data(self):
        model = _Recorder()
        result = default_eval_fn(model, [])
        # No samples => the model is never invoked.
        self.assertEqual(model.received, [])
        self.assertIsNone(result)

    def test_returns_none(self):
        model = _Recorder()
        self.assertIsNone(default_eval_fn(model, [(torch.randn(1, 3), 0)]))

    def test_invalid_pair_length_raises(self):
        model = _Recorder()
        with self.assertRaises(ValueError):
            default_eval_fn(model, [(torch.randn(1, 3),)])


if __name__ == "__main__":
    run_tests()
