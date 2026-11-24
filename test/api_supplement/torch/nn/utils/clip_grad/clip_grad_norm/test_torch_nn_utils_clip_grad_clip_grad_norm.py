# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.utils.clip_grad.clip_grad_norm 接口功能正确性
API 名称：torch.nn.utils.clip_grad.clip_grad_norm
API 签名：clip_grad_norm(parameters, max_norm, norm_type=2.0,
         error_if_nonfinite=False, foreach=None) -> torch.Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 参数列表多 Tensor 与单个 Tensor 输入 | 已覆盖（clip_grad_norm_clips_and_returns_total_norm/
clip_grad_norm_single_tensor） |
| 枚举选项 | norm_type=2（默认）/1/inf；error_if_nonfinite=False/True；float32/float16 混合 dtype | 已覆盖
（clip_grad_norm_norm_type_1/clip_grad_norm_norm_type_inf/
clip_grad_norm_error_if_nonfinite_nan/clip_grad_norm_mixed_dtype） |
| 参数类型 | Iterable[Tensor] 与单 Tensor；混合 dtype 梯度按 dtype 分组裁剪 | 已覆盖
（clip_grad_norm_single_tensor/clip_grad_norm_mixed_dtype） |
| 传参与不传参 | 默认 norm_type=2.0、error_if_nonfinite=False 与显式传参 | 已覆盖
（clip_grad_norm_clips_and_returns_total_norm） |
| 等价类/边界值 | 总范数大于 max_norm 触发裁剪、小于 max_norm 不裁剪、空参数列表返回 0、NaN 梯度 | 已覆盖
（clip_grad_norm_no_clip_below_max_norm/clip_grad_norm_empty_parameters/
clip_grad_norm_error_if_nonfinite_nan） |
| 正常传参场景 | 裁剪后梯度等于 max_norm/(total_norm+1e-6) 缩放，返回裁剪前总范数 | 已覆盖
（clip_grad_norm_clips_and_returns_total_norm/clip_grad_norm_cpu_golden） |
| 异常传参场景 | error_if_nonfinite=True 且梯度含 NaN 时抛 RuntimeError；调用按契约发 FutureWarning | 已覆盖
（clip_grad_norm_error_if_nonfinite_nan/clip_grad_norm_deprecation_future_warning） |
| 混合设备类型 | 相同种子梯度下 NPU 与 CPU 的总范数及裁剪后梯度一致（CPU Golden） | 已覆盖
（clip_grad_norm_cpu_golden） |

未覆盖项及原因：
- foreach=True 显式传参未单独成例：clip_grad_norm 将 foreach 透传给 clip_grad_norm_，
  foreach 分组路径已由 mixed_dtype 用例（float32/float16 分组）覆盖默认自适应路径。
- torch.nn.utils.clip_grad.clip_grad_norm 为已废弃入口（内部直接调用 clip_grad_norm_），
  梯度裁剪数值语义与 clip_grad_norm_ 完全一致，废弃状态本身以 FutureWarning 用例看护。

注意：本 API 在 NPU 上对设备参数梯度做真实 foreach 范数与原地裁剪计算；
     返回总范数与裁剪后梯度均以 CPU 按契约公式（clip_coef = max_norm / (total_norm + 1e-6)）
     计算的 Golden 对照，浮点比较使用 torch.testing.assert_close。
"""
import os
import unittest
import warnings

import torch
import torch.nn.utils.clip_grad  # noqa: F401  # ensure torch.nn.utils.clip_grad is importable

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement tests actually run: plain CI
# collection keeps the class-level skipUnless gate and skips device init.
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestClipGradNorm(TestCase):
    """Test cases for torch.nn.utils.clip_grad.clip_grad_norm (NPU/CUDA portable).

    The deprecated entry point delegates to torch.nn.utils.clip_grad_norm_ and
    returns the total norm computed before clipping; when the total norm
    exceeds max_norm every gradient is scaled in place by
    max_norm / (total_norm + 1e-6). Golden values are computed on CPU with
    that documented formula against CPU-seeded gradients.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _device_params(self, shapes, device=None, dtypes=None):
        """Create params on ``device`` with CPU-seeded grads.

        Returns the params and the CPU grad copies used as golden input.
        Parameter values are irrelevant for clipping; only grads matter.
        """
        if device is None:
            device = self.device_name
        torch.manual_seed(42)
        params, grads_cpu = [], []
        for i, shape in enumerate(shapes):
            dtype = dtypes[i] if dtypes else torch.float32
            grad_cpu = torch.randn(*shape, dtype=dtype)
            p = torch.nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=True)
            p = p.to(device)
            p.grad = grad_cpu.clone().to(device)
            params.append(p)
            grads_cpu.append(grad_cpu)
        return params, grads_cpu

    def _cpu_golden(self, grads_cpu, max_norm, norm_type):
        flat = torch.cat([g.reshape(-1) for g in grads_cpu])
        total = torch.linalg.vector_norm(flat, ord=norm_type).item()
        clip_coef = max_norm / (total + 1e-6)
        expected = [g * clip_coef if total > max_norm else g for g in grads_cpu]
        return total, expected

    def _call_api(self, params, max_norm, **kwargs):
        # The entry point is deprecated by contract and emits FutureWarning;
        # record it but keep it from interfering with other assertions.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            total = torch.nn.utils.clip_grad.clip_grad_norm(
                params, max_norm, **kwargs)
        if self.device_name == 'npu':
            torch.npu.synchronize()
        return total, caught

    # ------------------------------------------------------------------
    # Clipping path: total norm above max_norm
    # ------------------------------------------------------------------

    def test_clip_grad_norm_clips_and_returns_total_norm(self):
        """Total norm above max_norm: returns pre-clip norm and scales grads."""
        params, grads_cpu = self._device_params([(4, 3), (5,)])
        total, caught = self._call_api(params, max_norm=0.5)
        self.assertTrue(any(issubclass(w.category, FutureWarning) for w in caught),
                        'deprecated entry must emit FutureWarning')
        golden_total, expected = self._cpu_golden(grads_cpu, 0.5, 2.0)
        self.assertGreater(golden_total, 0.5, 'precondition: norm must exceed max_norm')
        torch.testing.assert_close(total.cpu(), torch.tensor(golden_total),
                                   msg='returned norm must match the pre-clip total norm')
        for p, exp in zip(params, expected):
            torch.testing.assert_close(p.grad.detach().cpu(), exp,
                                       msg='grads must be scaled by max_norm/(total+1e-6)')
        new_total = torch.linalg.vector_norm(
            torch.cat([p.grad.detach().reshape(-1) for p in params])).item()
        self.assertLessEqual(new_total, 0.5 + 1e-5,
                             'post-clip total norm must not exceed max_norm')

    def test_clip_grad_norm_no_clip_below_max_norm(self):
        """Total norm below max_norm: grads are left untouched."""
        params, grads_cpu = self._device_params([(4, 3), (5,)])
        total, _ = self._call_api(params, max_norm=1e6)
        golden_total, _ = self._cpu_golden(grads_cpu, 1e6, 2.0)
        self.assertLess(golden_total, 1e6, 'precondition: norm below max_norm')
        torch.testing.assert_close(total.cpu(), torch.tensor(golden_total),
                                   msg='returned norm must equal the untouched total norm')
        for p, g in zip(params, grads_cpu):
            torch.testing.assert_close(p.grad.detach().cpu(), g,
                                       msg='grads must not change when total norm <= max_norm')

    # ------------------------------------------------------------------
    # norm_type options
    # ------------------------------------------------------------------

    def test_clip_grad_norm_norm_type_1(self):
        """norm_type=1 clips with the L1 total norm."""
        params, grads_cpu = self._device_params([(4, 3), (6,)])
        total, _ = self._call_api(params, max_norm=0.5, norm_type=1)
        golden_total, expected = self._cpu_golden(grads_cpu, 0.5, 1.0)
        torch.testing.assert_close(total.cpu(), torch.tensor(golden_total),
                                   msg='L1 total norm must match the CPU golden')
        for p, exp in zip(params, expected):
            torch.testing.assert_close(p.grad.detach().cpu(), exp,
                                       msg='L1 clipping must scale grads like clip_grad_norm_')

    def test_clip_grad_norm_norm_type_inf(self):
        """norm_type=inf clips with the maximum absolute gradient value."""
        params, grads_cpu = self._device_params([(3,), (4, 2)])
        total, _ = self._call_api(params, max_norm=0.5, norm_type=float('inf'))
        golden_total, expected = self._cpu_golden(grads_cpu, 0.5, float('inf'))
        torch.testing.assert_close(total.cpu(), torch.tensor(golden_total),
                                   msg='inf norm must equal the max absolute grad')
        self.assertEqual(total.item(), max(g.abs().max().item() for g in grads_cpu),
                         'inf norm must equal the exact max abs value')
        for p, exp in zip(params, expected):
            torch.testing.assert_close(p.grad.detach().cpu(), exp,
                                       msg='inf clipping must scale grads like clip_grad_norm_')

    # ------------------------------------------------------------------
    # Input shapes and dtypes
    # ------------------------------------------------------------------

    def test_clip_grad_norm_single_tensor(self):
        """A single Tensor (not an iterable) is accepted as parameters."""
        params, grads_cpu = self._device_params([(3, 4)])
        p = params[0]
        total, _ = self._call_api(p, max_norm=0.5)
        golden_total, expected = self._cpu_golden(grads_cpu, 0.5, 2.0)
        torch.testing.assert_close(total.cpu(), torch.tensor(golden_total))
        torch.testing.assert_close(p.grad.detach().cpu(), expected[0],
                                   msg='single Tensor grads must be clipped the same way')

    def test_clip_grad_norm_mixed_dtype(self):
        """float32 and float16 grads are grouped per dtype and both clipped."""
        params, grads_cpu = self._device_params(
            [(4, 3), (5,)], dtypes=[torch.float32, torch.float16])
        total, _ = self._call_api(params, max_norm=0.3)
        # Golden follows the documented contract: the norm of each individual
        # tensor, combined as if concatenated. The float16 group norm is
        # quantized inside float16 (the dtype of its grads), so each group is
        # normalized at its native dtype precision before combining.
        group_norms = [torch.linalg.vector_norm(g).item() for g in grads_cpu]
        golden_total = torch.linalg.vector_norm(
            torch.tensor(group_norms, dtype=torch.float32)).item()
        clip_coef = 0.3 / (golden_total + 1e-6)
        expected = [g * clip_coef for g in grads_cpu]
        torch.testing.assert_close(total.cpu(), torch.tensor(golden_total),
                                   msg='mixed dtype total norm must match the grouped golden')
        for p, exp in zip(params, expected):
            torch.testing.assert_close(p.grad.detach().cpu(), exp,
                                       msg='each dtype group must be clipped by the shared scale')
        new_total = torch.linalg.vector_norm(
            torch.cat([p.grad.detach().float().reshape(-1) for p in params])).item()
        self.assertLessEqual(new_total, 0.3 + 1e-4,
                             'post-clip total norm must not exceed max_norm')

    # ------------------------------------------------------------------
    # NaN gradients and error_if_nonfinite
    # ------------------------------------------------------------------

    def test_clip_grad_norm_error_if_nonfinite_nan(self):
        """error_if_nonfinite=True raises on NaN grads; default path keeps grads."""
        p = self._device_params([(4,)])[0][0]
        p.grad = torch.full((4,), float('nan')).to(self.device_name)
        with self.assertRaisesRegex(RuntimeError, 'total norm'):
            self._call_api(p, 1.0, error_if_nonfinite=True)
        # Default error_if_nonfinite=False: the total norm is NaN and grads
        # stay NaN (the NaN clip coefficient never gets applied).
        total, _ = self._call_api(p, 1.0)
        self.assertTrue(torch.isnan(total.cpu()),
                        'total norm of NaN grads must be NaN without error_if_nonfinite')
        self.assertTrue(torch.isnan(p.grad.detach()).all(),
                        'NaN grads must stay NaN on the default path')

    # ------------------------------------------------------------------
    # Deprecation contract
    # ------------------------------------------------------------------

    def test_clip_grad_norm_deprecation_future_warning(self):
        """Calling the deprecated entry emits a FutureWarning."""
        p = self._device_params([(3,)])[0][0]
        _, caught = self._call_api(p, 1.0)
        categories = [w.category for w in caught]
        self.assertIn(FutureWarning, categories,
                      'clip_grad_norm is deprecated in favor of clip_grad_norm_')

    # ------------------------------------------------------------------
    # Boundary: empty parameter list
    # ------------------------------------------------------------------

    def test_clip_grad_norm_empty_parameters(self):
        """An empty parameter list returns a zero norm without clipping."""
        total, _ = self._call_api([], 1.0)
        self.assertEqual(total.item(), 0.0,
                         'empty parameters must yield a zero total norm')

    # ------------------------------------------------------------------
    # CPU golden comparison
    # ------------------------------------------------------------------

    def test_clip_grad_norm_cpu_golden(self):
        """Same seeded grads: device total norm and clipped grads match CPU."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(11)
        cpu_p = torch.nn.Parameter(torch.randn(6, 4))
        cpu_p.grad = torch.randn(6, 4)
        dev_p = torch.nn.Parameter(cpu_p.detach().clone().to(self.device_name))
        dev_p.grad = cpu_p.grad.detach().clone().to(self.device_name)
        cpu_total, _ = self._call_api(cpu_p, 0.7)
        dev_total, _ = self._call_api(dev_p, 0.7)
        torch.testing.assert_close(dev_total.cpu(), cpu_total.cpu(),
                                   msg='device total norm must match the CPU golden')
        torch.testing.assert_close(dev_p.grad.detach().cpu(), cpu_p.grad.detach(),
                                   msg='device clipped grads must match the CPU golden')


if __name__ == "__main__":
    run_tests()
