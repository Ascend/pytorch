# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Identity.forward 接口功能正确性
API 名称：torch.nn.Identity.forward
API 签名：forward(self, input: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                        | 覆盖情况                            |
|------------------|---------------------------------------------|-------------------------------------|
| 空/非空          | 输入可为任意 Tensor（多 dtype/shape）；恒等返回不改变输入         | 已覆盖（forward_identity_various_dtypes）             |
| 枚举选项         | 无枚举参数                                                       | 未覆盖（无该类参数）                                  |
| 参数类型         | input: Tensor；非 Tensor 输入按契约原样返回（identity 语义）     | 已覆盖（forward_non_tensor_passthrough）              |
| 传参与不传参     | 必须传 input；多余位置参数被签名拒绝                             | 已覆盖（forward_extra_positional_rejected）           |
| 等价类/边界值    | 标量/单元素/多维/零值边界 shape，返回同一对象且值不变            | 已覆盖（forward_identity_various_dtypes）             |
| 正常传参场景     | 返回输入对象本身，数值/dtype/shape/device 不变                   | 已覆盖（forward_returns_input_object 两例）           |
| 异常传参场景     | 两个位置参数触发 TypeError；失败后模块仍可用                     | 已覆盖（forward_extra_positional_rejected）           |
| 混合设备类型     | NPU 驻留输入经 forward 保持同一 NPU 对象；梯度以 CPU Golden 对照 | 已覆盖（forward_device_preserved/forward_backward_grad_flow） |

未覆盖项及原因：
- 本接口为恒等占位算子，不产生数值变换，浮点对照按「输出须等于输入」的原生契约以
  torch.testing.assert_close 校验，并辅以 assertIs 同一对象断言。
- 序列化与 Hook 维度：Identity 无状态（无参数/buffer），forward 语义不依赖序列化或
  hook 注册，未覆盖（无适用场景）。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
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

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestIdentityforward(TestCase):
    """Test cases for torch.nn.Identity.forward (NPU/CUDA portable).

    torch.nn.Identity.forward is ``def forward(self, input: Tensor) -> Tensor:
    return input`` (torch 2.15.0, torch/nn/modules/linear.py). Empirical
    contract (probed on torch 2.15.0+cpu + Ascend910B4):
      - the very same input object is returned (no copy), preserving values,
        dtype, shape and device;
      - non-tensor inputs are returned as-is (identity semantics, no type
        validation inside forward);
      - a second positional argument is rejected by the signature with
        TypeError, and the module stays usable afterwards;
      - gradients flow through unchanged (grad equals ones of the input).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_forward_returns_input_object(self):
        """forward returns the input tensor object itself, values untouched."""
        model = torch.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        x = torch.tensor([1.0, -2.0, 3.0])
        y = model(x)
        # Identity contract: the exact same object comes back (no copy).
        self.assertIs(y, x, "Identity.forward must return the input object itself")
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.dtype, torch.float32)
        torch.testing.assert_close(
            y, torch.tensor([1.0, -2.0, 3.0]),
            msg="forward output values must equal the input values")

    def test_forward_identity_various_dtypes(self):
        """Identity passes through scalar/int/bool/multi-dim inputs unchanged."""
        model = torch.nn.Identity()
        cases = [
            torch.tensor(3.5),                       # 0-dim scalar
            torch.tensor([[7]], dtype=torch.int64),  # int, 2-dim
            torch.tensor([True, False]),             # bool
            torch.zeros(2, 3, 4),                    # 3-dim boundary (all zeros)
        ]
        for x in cases:
            y = model(x)
            self.assertIs(y, x, "forward must return the same object for %s" % (tuple(x.shape),))
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(y.dtype, x.dtype)

    def test_forward_non_tensor_passthrough(self):
        """Non-tensor input is returned as-is: forward performs no validation."""
        model = torch.nn.Identity()
        sentinel = "plain string"
        y = model(sentinel)
        self.assertIs(y, sentinel,
                      "Identity.forward must return a non-tensor input unchanged")

    def test_forward_extra_positional_rejected(self):
        """A second positional argument exceeds the signature and raises TypeError."""
        model = torch.nn.Identity()
        x = torch.ones(2)
        with self.assertRaises(TypeError):
            model(x, x)
        # The failed call must leave the module usable.
        torch.testing.assert_close(
            model(x), torch.ones(2),
            msg="module must stay usable after the rejected extra-arg call")

    def test_forward_backward_grad_flow(self):
        """Gradient flows through Identity unchanged on CPU and the active device."""
        model = torch.nn.Identity()
        x_cpu = torch.ones(2, 3, requires_grad=True)
        model(x_cpu).sum().backward()
        self.assertIsNotNone(x_cpu.grad, "CPU forward must pass gradients through")
        torch.testing.assert_close(
            x_cpu.grad, torch.ones(2, 3),
            msg="Identity grad must be ones on CPU")

        x_dev = torch.ones(2, 3, device=self.device, requires_grad=True)
        model(x_dev).sum().backward()
        self.assertIsNotNone(x_dev.grad, "device forward must pass gradients through")
        torch.testing.assert_close(
            x_dev.grad.cpu(), torch.ones(2, 3),
            msg="Identity grad must be ones on %s" % self.device_name)

    def test_forward_device_preserved(self):
        """Device-resident input keeps its device and values through forward."""
        model = torch.nn.Identity()
        expected = torch.tensor([0.25, -1.5, 2.0])
        x = expected.clone().to(self.device)
        y = model(x)
        self.assertIs(y, x, "device input must be returned as the same object")
        self.assertEqual(y.device.type, self.device_name,
                         "forward must not move the tensor across devices")
        torch.testing.assert_close(
            y.cpu(), expected,
            msg="device-resident values must be unchanged by forward")

    def test_forward_training_eval_consistent(self):
        """forward output is identical in train() and eval() modes (no state)."""
        model = torch.nn.Identity()
        x = torch.randn(2, 2)
        model.train()
        y_train = model(x)
        model.eval()
        y_eval = model(x)
        self.assertIs(y_train, x)
        self.assertIs(y_eval, x)
        torch.testing.assert_close(
            y_train, y_eval,
            msg="train/eval modes must not change Identity.forward output")


if __name__ == "__main__":
    run_tests()
