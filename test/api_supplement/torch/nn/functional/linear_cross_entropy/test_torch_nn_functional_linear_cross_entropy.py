# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.functional.linear_cross_entropy 接口功能正确性
API 名称：torch.nn.functional.linear_cross_entropy
API 签名：linear_cross_entropy(input, linear_weight, target, *, linear_bias=None,
weight=None, reduction='mean', ignore_index=None, label_smoothing=0.0, options=None)
-> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | linear_bias/weight 缺省与显式、1D 非批量输入 | 已覆盖（_reference_with_bias 等） |
| 枚举选项 | reduction='none'/'mean'/'sum'、ignore_index 默认与显式 | 已覆盖（_reference_reductions/_reference_ignore_index） |
| 参数类型 | fp32 索引 target、概率 target、fp16 分块输入 | 已覆盖（_reference_index_target/_reference_prob_target/_chunked_fp16_acc_dtype） |
| 传参与不传参 | options=None 参考路径与 options=分块路径 | 已覆盖（_reference_* 与 _chunked_*） |
| 等价类/边界值 | 各字段组合、ignore_index 命中 | 已覆盖（_chunked_batch_chunk_size/_chunked_chunking_method/_ignore_index） |
| 正常传参场景 | 参考路径/分块路径前向、反向梯度、确定性重复执行 | 已覆盖（_reference_backward/_chunked_backward/_deterministic） |
| 异常传参场景 | input 维度、linear_weight 维度、in_features、bias 形状、概率 target+ignore_index | 已覆盖（_invalid_shapes_raise） |
| 混合设备类型 | input/linear_weight/target/linear_bias 同处当前设备执行 | 已覆盖（全部用例的设备内对照） |

未覆盖项及原因：
- 异构设备输入：原生语义要求各 Tensor 同设备（抛设备一致性错误），未构造跨设备断言以保证
  NPU/CUDA 可移植。
- K 维 loss（linear_weight 形状 (C,d1,...,in_features)）：分块路径按契约回退参考路径，功能
  行为由参考路径覆盖，本文件聚焦标准 (C, in_features) 形态。
- 分块路径 target 类索引越界：无效输入在设备侧行为不可移植，不构造。
- options 分块路径的二次反向/高阶 AD 契约已覆盖（_chunked_second_backward）；backward() 之外
  的 torch.compile/torch.func 组合不在本次功能范围。
- 内存维度专项：分块路径的省内存效果未做峰值内存度量，仅验证数值与梯度契约。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     CPU Golden 采用 docstring 的参考实现（F.linear + F.cross_entropy）独立于被测入口计算。
"""
import os
import unittest
import warnings

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the accelerator only when supplement tests actually run; regular CI
# collection skips the class below and never initializes a device.
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _reference_logits(input, linear_weight, linear_bias):
    # Docstring reference implementation: linear projection then reshape.
    in_features = input.shape[-1]
    num_classes = linear_weight.shape[0]
    weight = linear_weight.reshape(num_classes, in_features)
    bias = None if linear_bias is None else linear_bias.reshape(num_classes)
    return torch.nn.functional.linear(input, weight, bias)


def _reference_golden(
    input,
    linear_weight,
    target,
    linear_bias=None,
    weight=None,
    reduction="mean",
    ignore_index=None,
    label_smoothing=0.0,
):
    # Independent golden: F.cross_entropy over the docstring reference logits.
    logits = _reference_logits(input, linear_weight, linear_bias)
    resolved_ignore = -100 if ignore_index is None else ignore_index
    return torch.nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=resolved_ignore,
        label_smoothing=label_smoothing,
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLinearCrossEntropyFunctional(TestCase):
    """Test cases for torch.nn.functional.linear_cross_entropy (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _device_batch(self, num_batches=8, in_features=4, num_classes=10, seed=1):
        generator = torch.Generator().manual_seed(seed)
        input = torch.randn(
            num_batches, in_features, generator=generator
        ).to(self.device)
        weight = torch.randn(
            num_classes, in_features, generator=generator
        ).to(self.device)
        target = torch.randint(
            0, num_classes, (num_batches,), generator=generator
        ).to(self.device)
        return input, weight, target

    def test_linear_cross_entropy_reference_index_target_fp32(self):
        """options=None with int64 targets matches the independent CPU golden."""
        input, weight, target = self._device_batch()
        actual = torch.nn.functional.linear_cross_entropy(input, weight, target)
        cpu_input = input.cpu().requires_grad_(False)
        cpu_weight = weight.cpu()
        expected = torch.nn.functional.linear_cross_entropy(
            cpu_input, cpu_weight, target.cpu()
        )
        golden = _reference_golden(cpu_input, cpu_weight, target.cpu())
        torch.testing.assert_close(expected, golden)
        torch.testing.assert_close(actual.cpu(), expected)

    def test_linear_cross_entropy_reference_with_bias(self):
        """linear_bias (C,) is added before the softmax on the reference path."""
        input, weight, target = self._device_batch()
        bias = torch.randn(weight.shape[0], device=self.device)
        actual = torch.nn.functional.linear_cross_entropy(
            input, weight, target, linear_bias=bias
        )
        cpu_bias = bias.cpu()
        expected = torch.nn.functional.linear_cross_entropy(
            input.cpu(), weight.cpu(), target.cpu(), linear_bias=cpu_bias
        )
        torch.testing.assert_close(
            expected, _reference_golden(input.cpu(), weight.cpu(), target.cpu(), cpu_bias)
        )
        torch.testing.assert_close(actual.cpu(), expected)

    def test_linear_cross_entropy_reference_class_weight(self):
        """A (C,) class weight rescales per-class contributions."""
        input, weight, target = self._device_batch()
        class_weight = torch.rand(weight.shape[0], device=self.device) + 0.5
        actual = torch.nn.functional.linear_cross_entropy(
            input, weight, target, weight=class_weight
        )
        cpu_weight_class = class_weight.cpu()
        expected = torch.nn.functional.linear_cross_entropy(
            input.cpu(), weight.cpu(), target.cpu(), weight=cpu_weight_class
        )
        torch.testing.assert_close(
            actual.cpu(), expected
        )

    def test_linear_cross_entropy_reference_reductions(self):
        """reduction='none'/'mean'/'sum' on the reference path match CPU golden."""
        input, weight, target = self._device_batch()
        for reduction in ("none", "mean", "sum"):
            actual = torch.nn.functional.linear_cross_entropy(
                input, weight, target, reduction=reduction
            )
            expected = torch.nn.functional.linear_cross_entropy(
                input.cpu(), weight.cpu(), target.cpu(), reduction=reduction
            )
            torch.testing.assert_close(actual.cpu(), expected)
            if reduction == "none":
                self.assertEqual(actual.shape, target.shape)

    def test_linear_cross_entropy_reference_ignore_index(self):
        """ignore_index default -100 and explicit values exclude targets."""
        input, weight, _ = self._device_batch()
        target = torch.randint(1, weight.shape[0], (input.shape[0],), device=self.device)
        target[0] = 0
        target[3] = 0
        for ignore_index in (None, -100, 0):
            actual = torch.nn.functional.linear_cross_entropy(
                input, weight, target, ignore_index=ignore_index
            )
            expected = torch.nn.functional.linear_cross_entropy(
                input.cpu(), weight.cpu(), target.cpu(), ignore_index=ignore_index
            )
            golden = _reference_golden(
                input.cpu(), weight.cpu(), target.cpu(), ignore_index=ignore_index
            )
            torch.testing.assert_close(expected, golden)
            torch.testing.assert_close(actual.cpu(), expected)

    def test_linear_cross_entropy_reference_label_smoothing(self):
        """label_smoothing spreads probability mass over classes."""
        input, weight, target = self._device_batch()
        actual = torch.nn.functional.linear_cross_entropy(
            input, weight, target, label_smoothing=0.1
        )
        expected = torch.nn.functional.linear_cross_entropy(
            input.cpu(), weight.cpu(), target.cpu(), label_smoothing=0.1
        )
        golden = _reference_golden(
            input.cpu(), weight.cpu(), target.cpu(), label_smoothing=0.1
        )
        torch.testing.assert_close(expected, golden)
        torch.testing.assert_close(actual.cpu(), expected)

    def test_linear_cross_entropy_reference_prob_target(self):
        """Probability targets (N, C) match the reference golden."""
        input, weight, _ = self._device_batch()
        logits = _reference_logits(input, weight, None)
        target = torch.softmax(logits, dim=1).detach()
        actual = torch.nn.functional.linear_cross_entropy(input, weight, target)
        cpu_logits = _reference_logits(input.cpu(), weight.cpu(), None)
        cpu_target = torch.softmax(cpu_logits, dim=1)
        expected = torch.nn.functional.linear_cross_entropy(
            input.cpu(), weight.cpu(), cpu_target
        )
        torch.testing.assert_close(actual.cpu(), expected)

    def test_linear_cross_entropy_reference_1d_input(self):
        """Unbatched (in_features,) input with 0-dim target gives a scalar."""
        input, weight, _ = self._device_batch(num_batches=1)
        input = input.squeeze(0)
        target = torch.randint(
            0, weight.shape[0], (), device=self.device
        )
        actual = torch.nn.functional.linear_cross_entropy(input, weight, target)
        expected = torch.nn.functional.linear_cross_entropy(
            input.cpu(), weight.cpu(), target.cpu()
        )
        torch.testing.assert_close(actual.cpu(), expected)

    def test_linear_cross_entropy_reference_backward_fp32(self):
        """Input/weight/bias gradients match the CPU autograd golden."""
        input, weight, target = self._device_batch()
        bias = torch.randn(weight.shape[0], device=self.device)
        device_input = input.clone().requires_grad_(True)
        device_weight = weight.clone().requires_grad_(True)
        device_bias = bias.clone().requires_grad_(True)
        loss = torch.nn.functional.linear_cross_entropy(
            device_input, device_weight, target, linear_bias=device_bias
        )
        loss.backward()
        cpu_input = input.cpu().clone().requires_grad_(True)
        cpu_weight = weight.cpu().clone().requires_grad_(True)
        cpu_bias = bias.cpu().clone().requires_grad_(True)
        cpu_loss = torch.nn.functional.linear_cross_entropy(
            cpu_input, cpu_weight, target.cpu(), linear_bias=cpu_bias
        )
        cpu_loss.backward()
        for device_grad, cpu_grad in (
            (device_input.grad, cpu_input.grad),
            (device_weight.grad, cpu_weight.grad),
            (device_bias.grad, cpu_bias.grad),
        ):
            self.assertTrue(device_grad is not None)
            torch.testing.assert_close(device_grad.cpu(), cpu_grad)

    def test_linear_cross_entropy_chunked_matches_reference(self):
        """options chunked path matches both the device reference and CPU golden."""
        input, weight, target = self._device_batch(num_batches=16, seed=2)
        for reduction in ("mean", "sum", "none"):
            options = torch.nn.LinearCrossEntropyOptions()
            chunked = torch.nn.functional.linear_cross_entropy(
                input, weight, target, options=options, reduction=reduction
            )
            reference = torch.nn.functional.linear_cross_entropy(
                input, weight, target, reduction=reduction
            )
            cpu_input = input.cpu()
            cpu_weight = weight.cpu()
            golden = _reference_golden(
                cpu_input, cpu_weight, target.cpu(), reduction=reduction
            )
            torch.testing.assert_close(reference.cpu(), golden)
            torch.testing.assert_close(chunked.cpu(), golden)

    def test_linear_cross_entropy_chunked_backward_fp32(self):
        """Chunked-path gradients match the CPU reference autograd golden."""
        input, weight, target = self._device_batch(num_batches=16, seed=3)
        options = torch.nn.LinearCrossEntropyOptions()
        device_input = input.clone().requires_grad_(True)
        device_weight = weight.clone().requires_grad_(True)
        loss = torch.nn.functional.linear_cross_entropy(
            device_input, device_weight, target, options=options
        )
        loss.backward()
        cpu_input = input.cpu().clone().requires_grad_(True)
        cpu_weight = weight.cpu().clone().requires_grad_(True)
        cpu_loss = torch.nn.functional.linear_cross_entropy(
            cpu_input, cpu_weight, target.cpu()
        )
        cpu_loss.backward()
        torch.testing.assert_close(device_input.grad.cpu(), cpu_input.grad)
        torch.testing.assert_close(device_weight.grad.cpu(), cpu_weight.grad)

    def test_linear_cross_entropy_chunked_second_backward(self):
        """allow_retain_graph=False rejects a second backward; True allows it."""
        input, weight, target = self._device_batch(num_batches=16, seed=4)
        strict_input = input.clone().requires_grad_(True)
        strict_weight = weight.clone().requires_grad_(True)
        strict_loss = torch.nn.functional.linear_cross_entropy(
            strict_input,
            strict_weight,
            target,
            options=torch.nn.LinearCrossEntropyOptions(),
        )
        strict_loss.backward()
        with self.assertRaisesRegex(RuntimeError, "called twice"):
            strict_loss.backward()
        retain_input = input.clone().requires_grad_(True)
        retain_weight = weight.clone().requires_grad_(True)
        retain_loss = torch.nn.functional.linear_cross_entropy(
            retain_input,
            retain_weight,
            target,
            options=torch.nn.LinearCrossEntropyOptions(allow_retain_graph=True),
        )
        retain_loss.backward()
        first_input_grad = retain_input.grad.cpu().clone()
        retain_loss.backward()
        # The second backward accumulates the same pre-computed gradient again.
        torch.testing.assert_close(retain_input.grad.cpu(), first_input_grad * 2)

    def test_linear_cross_entropy_chunked_batch_chunk_size(self):
        """Explicit batch_chunk_size chunks the batch without changing results."""
        input, weight, target = self._device_batch(num_batches=16, seed=5)
        for chunk_size in (1, 3, 16):
            options = torch.nn.LinearCrossEntropyOptions(batch_chunk_size=chunk_size)
            chunked = torch.nn.functional.linear_cross_entropy(
                input, weight, target, options=options
            )
            reference = torch.nn.functional.linear_cross_entropy(input, weight, target)
            torch.testing.assert_close(chunked.cpu(), reference.cpu())

    def test_linear_cross_entropy_chunked_chunking_method(self):
        """chunking_method='aspect_ratio'[:N] resolves without changing results."""
        input, weight, target = self._device_batch(num_batches=16, seed=6)
        for method in ("auto", "aspect_ratio", "aspect_ratio:2", "aspect_ratio:4"):
            options = torch.nn.LinearCrossEntropyOptions(chunking_method=method)
            chunked = torch.nn.functional.linear_cross_entropy(
                input, weight, target, options=options
            )
            reference = torch.nn.functional.linear_cross_entropy(input, weight, target)
            torch.testing.assert_close(chunked.cpu(), reference.cpu())
        resolved = torch.nn.LinearCrossEntropyOptions()._adjust(
            num_batches=16,
            in_features=input.shape[-1],
            num_classes=weight.shape[0],
            dtype=input.dtype,
            device=input.device,
        )
        # Documented aspect_ratio formula: next_pow2(ceil(N / ceil(C / D))).
        # ceil(10/4)=3 chunks target ceil(16/3)=6 rows, next pow2 is 8.
        expected_chunk = 8
        self.assertEqual(resolved.batch_chunk_size, expected_chunk)
        self.assertEqual(resolved.chunking_method, "aspect_ratio")

    def test_linear_cross_entropy_chunked_class_weight(self):
        """Chunked path supports a (C,) class weight across reductions."""
        input, weight, target = self._device_batch(num_batches=16, seed=7)
        class_weight = torch.rand(weight.shape[0], device=self.device) + 0.5
        for reduction in ("mean", "sum", "none"):
            chunked = torch.nn.functional.linear_cross_entropy(
                input,
                weight,
                target,
                weight=class_weight,
                options=torch.nn.LinearCrossEntropyOptions(),
                reduction=reduction,
            )
            reference = torch.nn.functional.linear_cross_entropy(
                input, weight, target, weight=class_weight, reduction=reduction
            )
            torch.testing.assert_close(chunked.cpu(), reference.cpu())

    def test_linear_cross_entropy_chunked_ignore_index(self):
        """Chunked path honors ignore_index on class-index targets."""
        input, weight, _ = self._device_batch(num_batches=16, seed=8)
        target = torch.randint(1, weight.shape[0], (input.shape[0],), device=self.device)
        target[0] = 0
        target[3] = 0
        for ignore_index in (-100, 0):
            chunked = torch.nn.functional.linear_cross_entropy(
                input,
                weight,
                target,
                options=torch.nn.LinearCrossEntropyOptions(batch_chunk_size=2),
                ignore_index=ignore_index,
            )
            reference = torch.nn.functional.linear_cross_entropy(
                input, weight, target, ignore_index=ignore_index
            )
            torch.testing.assert_close(chunked.cpu(), reference.cpu())

    def test_linear_cross_entropy_chunked_prob_target(self):
        """Chunked path supports probability targets with matching dtype."""
        input, weight, _ = self._device_batch(num_batches=16, seed=9)
        logits = _reference_logits(input, weight, None)
        target = torch.softmax(logits, dim=1).detach()
        chunked = torch.nn.functional.linear_cross_entropy(
            input, weight, target, options=torch.nn.LinearCrossEntropyOptions()
        )
        reference = torch.nn.functional.linear_cross_entropy(input, weight, target)
        torch.testing.assert_close(chunked.cpu(), reference.cpu())

    def test_linear_cross_entropy_chunked_fp16_acc_dtype_fp32(self):
        """fp16 input with acc_dtype=float32 keeps fp32 accumulation accuracy."""
        generator = torch.Generator().manual_seed(10)
        input16 = torch.randn(16, 4, generator=generator).to(
            self.device, torch.float16
        )
        weight16 = torch.randn(10, 4, generator=generator).to(
            self.device, torch.float16
        )
        target = torch.randint(0, 10, (16,), generator=generator).to(self.device)
        options = torch.nn.LinearCrossEntropyOptions(acc_dtype=torch.float32)
        chunked = torch.nn.functional.linear_cross_entropy(
            input16, weight16, target, options=options
        )
        self.assertEqual(chunked.dtype, torch.float16)
        reference = torch.nn.functional.linear_cross_entropy(input16, weight16, target)
        torch.testing.assert_close(
            chunked.cpu(), reference.cpu(), rtol=2e-3, atol=2e-3
        )

    def test_linear_cross_entropy_chunked_label_smoothing_falls_back(self):
        """label_smoothing != 0 falls back to the reference path with a warning."""
        input, weight, target = self._device_batch(num_batches=16, seed=11)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = torch.nn.functional.linear_cross_entropy(
                input,
                weight,
                target,
                options=torch.nn.LinearCrossEntropyOptions(),
                label_smoothing=0.1,
            )
        matches = [str(w.message) for w in caught if "options" in str(w.message)]
        self.assertTrue(matches, "expected the options-ignored fallback warning")
        reference = torch.nn.functional.linear_cross_entropy(
            input, weight, target, label_smoothing=0.1
        )
        torch.testing.assert_close(actual.cpu(), reference.cpu())

    def test_linear_cross_entropy_deterministic_fixed_seed(self):
        """Both paths give identical results across repeated fixed-seed runs."""
        input, weight, target = self._device_batch(num_batches=16, seed=12)
        options = torch.nn.LinearCrossEntropyOptions(batch_chunk_size=4)
        first_chunked = torch.nn.functional.linear_cross_entropy(
            input, weight, target, options=options
        ).cpu().clone()
        first_reference = torch.nn.functional.linear_cross_entropy(
            input, weight, target
        ).cpu().clone()
        for _ in range(3):
            torch.testing.assert_close(
                torch.nn.functional.linear_cross_entropy(
                    input, weight, target, options=options
                ).cpu(),
                first_chunked,
            )
            torch.testing.assert_close(
                torch.nn.functional.linear_cross_entropy(
                    input, weight, target
                ).cpu(),
                first_reference,
            )

    def test_linear_cross_entropy_invalid_shapes_raise(self):
        """Invalid dims/shapes raise RuntimeError before any op dispatch."""
        input, weight, target = self._device_batch()
        with self.assertRaisesRegex(RuntimeError, "dimensionality"):
            torch.nn.functional.linear_cross_entropy(
                input.unsqueeze(0), weight, target
            )
        with self.assertRaisesRegex(RuntimeError, "linear_weight"):
            torch.nn.functional.linear_cross_entropy(
                input, weight[0], target
            )
        with self.assertRaisesRegex(RuntimeError, "in_features"):
            torch.nn.functional.linear_cross_entropy(
                input, weight[:, :2], target
            )
        with self.assertRaisesRegex(RuntimeError, "linear_bias"):
            torch.nn.functional.linear_cross_entropy(
                input, weight, target, linear_bias=torch.zeros(3, device=self.device)
            )
        prob_target = torch.softmax(
            _reference_logits(input, weight, None), dim=1
        ).detach()
        with self.assertRaisesRegex(RuntimeError, "ignore_index"):
            torch.nn.functional.linear_cross_entropy(
                input, weight, prob_target, ignore_index=0
            )


if __name__ == "__main__":
    run_tests()
