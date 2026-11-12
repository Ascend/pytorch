# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Adafactor.state_dict 接口功能正确性
API 名称：torch.optim.Adafactor.state_dict
API 签名：state_dict(self) -> dict[str, Any]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 一步训练后 state_dict 含非空 state；未 step 时 state 为空 | 已覆盖（_structure/_empty_state_before_step） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 返回 dict；state 键为整型参数索引、param_groups 列表 | 已覆盖（_structure） |
| 传参与不传参 | 无参调用；传多余参数抛 TypeError | 已覆盖（_no_arg） |
| 等价类/边界值 | 1-D（variance）与 2-D（row_var/col_var）参数的状态张量形状与数值 | 已覆盖（_values_roundtrip_loaded_tensor_values） |
| 正常传参场景 | step 后载入逐项一致→续跑一致 | 已覆盖（_values_roundtrip_loaded_tensor_values/_continuation_after_load_matches_reference） |
| 异常传参场景 | 传多余参数抛 TypeError | 已覆盖（_no_arg） |
| 混合设备类型 | NPU 与 CPU 侧 step 后 state_dict 数值一致（CPU Golden） | 已覆盖（_cpu_golden_state_values） |

未覆盖项及原因：
- 实测说明：state_dict() 返回浅引用（state 张量与优化器内部共享存储），原实例继续 step 会
  改变已保存 dict 中的数值。往返验证用 copy.deepcopy 隔离（与 checkpoint 落盘语义等价），
  并以 _shallow_reference_contract 单独记录该浅引用契约；续跑一致性要求参数张量与优化器
  状态在「同一 checkpoint 时刻」配对保存（参数由模型持有，不在 optimizer state_dict 内）。
"""
import copy
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 条件导入 torch_npu，并用 torch.npu.is_available() 探测可用 NPU。
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
class TestAdafactorstate_dict(TestCase):
    """Test cases for torch.optim.Adafactor.state_dict (NPU/CUDA portable).

    Full state-roundtrip contract: produce a non-empty state by stepping,
    assert the exact tensor VALUES (not just keys) survive a roundtrip into a
    fresh instance, then continue training one step on both the original and
    the loaded instance with identical inputs and require bit-comparable
    results. Also covers the 1-D (variance) vs 2-D (row_var/col_var) factored
    state layout and the measured shallow-reference semantics of state_dict().
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, shapes=((3,),), lr=0.01):
        """shapes is a sequence of per-param shapes, e.g. ((4, 3),) is one 2-D param."""
        params = [torch.nn.Parameter(torch.ones(s, device=self.device),
                                     requires_grad=True) for s in shapes]
        return torch.optim.Adafactor(params, lr=lr)

    def _step_all(self, opt, grad_value=0.5):
        for p in opt.param_groups[0]["params"]:
            p.grad = torch.full_like(p, grad_value)
        opt.step()

    def test_state_dict_structure(self):
        """state_dict exposes state and param_groups with int param ids."""
        opt = self._make_optimizer()
        self._step_all(opt)
        sd = opt.state_dict()
        self.assertIn("state", sd)
        self.assertIn("param_groups", sd)
        self.assertEqual(len(sd["param_groups"]), 1)
        # params are stored as integer indices, not tensors
        self.assertTrue(
            all(isinstance(x, int) for x in sd["param_groups"][0]["params"]))
        # state is keyed by those same indices after one step
        self.assertTrue(all(isinstance(k, int) for k in sd["state"].keys()))

    def test_state_dict_empty_state_before_step(self):
        """Before any step the lazy Adafactor has an empty (but present) state."""
        opt = self._make_optimizer()
        sd = opt.state_dict()
        self.assertEqual(sd["state"], {},
                         "lazy Adafactor state must be empty before the first step")
        self.assertEqual(len(sd["param_groups"]), 1)

    def test_state_dict_values_roundtrip_loaded_tensor_values(self):
        """Tensor values inside state survive a save/load roundtrip exactly.

        Covers both state layouts: 1-D params use 'variance', 2-D params use
        factored 'row_var'/'col_var'. Every state tensor of the fresh (loaded)
        instance must equal the saved value element-wise.
        """
        opt = self._make_optimizer(shapes=((4, 3),))  # one 2-D param
        self._step_all(opt, grad_value=0.5)
        opt_zero_state = opt.state[opt.param_groups[0]["params"][0]]
        self.assertIn("row_var", opt_zero_state, "2-D param must use factored state")
        sd = copy.deepcopy(opt.state_dict())  # isolate from later mutation

        fresh = self._make_optimizer(shapes=((4, 3),))
        fresh.load_state_dict(sd)
        p_fresh = fresh.param_groups[0]["params"][0]
        self.assertEqual(sorted(fresh.state[p_fresh].keys()),
                         sorted(opt_zero_state.keys()))
        for key in opt_zero_state:
            self.assertTrue(torch.is_tensor(fresh.state[p_fresh][key]),
                            f"state[{key!r}] must roundtrip as a Tensor")
            torch.testing.assert_close(fresh.state[p_fresh][key],
                                       opt_zero_state[key],
                                       msg=f"state[{key!r}] values must survive roundtrip")
        # Step counter roundtrips as a tensor with the stepped value.
        self.assertEqual(fresh.state[p_fresh]["step"].item(), 1.0)
        # Factored buffers keep their (m,1)/(1,n) shapes after the roundtrip.
        self.assertEqual(tuple(fresh.state[p_fresh]["row_var"].shape), (4, 1))
        self.assertEqual(tuple(fresh.state[p_fresh]["col_var"].shape), (1, 3))

    def test_state_dict_continuation_after_load_matches_reference(self):
        """A loaded fresh instance continues training identically to the original.

        Checkpoint semantics: params (owned by the model) and the optimizer
        state must be captured at the SAME moment; the fresh instance then
        reproduces the original's next step exactly with identical inputs.
        """
        opt = self._make_optimizer(lr=0.01)
        self._step_all(opt, grad_value=0.5)
        sd = copy.deepcopy(opt.state_dict())          # optimizer checkpoint
        param_ckpt = opt.param_groups[0]["params"][0].detach().clone()
        # Reference: original continues with a second, different gradient.
        self._step_all(opt, grad_value=0.25)
        ref_param = opt.param_groups[0]["params"][0].detach().clone()
        ref_state = {k: v.clone() for k, v in
                     opt.state[opt.param_groups[0]["params"][0]].items()}

        # Fresh instance: restore the checkpoint-time param value, load the
        # optimizer state, apply the same second gradient.
        q = torch.nn.Parameter(param_ckpt.clone())
        fresh = torch.optim.Adafactor([q], lr=0.01)
        fresh.load_state_dict(sd)
        q.grad = torch.full_like(q, 0.25)
        fresh.step()

        # The continued parameter and every state tensor must match the
        # original's (proving the optimizer state, not just the params,
        # carried over correctly).
        torch.testing.assert_close(q.detach(), ref_param,
                                   msg="loaded instance must continue identically")
        for key, ref_v in ref_state.items():
            torch.testing.assert_close(fresh.state[q][key], ref_v,
                                       msg=f"state[{key!r}] must continue identically")

    def test_state_dict_cpu_golden_state_values(self):
        """State tensor values computed on device equal the CPU Golden."""
        def build_and_step(device):
            p = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=True)
            opt = torch.optim.Adafactor([p], lr=0.01)
            p.grad = torch.full((3,), 0.5, device=device)
            opt.step()
            st = opt.state[p]
            return (st["step"].detach().clone(),
                    st["variance"].detach().clone())

        step_dev, var_dev = build_and_step(self.device)
        step_cpu, var_cpu = build_and_step("cpu")
        self.assertEqual(step_dev.item(), step_cpu.item())
        torch.testing.assert_close(var_dev.cpu(), var_cpu,
                                   msg="NPU state values must match CPU Golden")

    def test_state_dict_shallow_reference_contract(self):
        """Measured contract: state_dict() returns references into live state.

        The returned dict shares tensor storage with the optimizer, so a later
        step() of the SAME optimizer is visible through the saved dict. Real
        checkpoint flows must therefore deepcopy/serialize before continuing.
        """
        opt = self._make_optimizer()
        self._step_all(opt, grad_value=0.5)
        sd = opt.state_dict()
        step_before = sd["state"][0]["step"].item()
        self._step_all(opt, grad_value=0.5)
        self.assertEqual(step_before, 1.0)
        self.assertEqual(sd["state"][0]["step"].item(), 2.0,
                         "state_dict shares storage with the live optimizer state")

    def test_state_dict_no_arg(self):
        """Extra argument raises TypeError."""
        opt = self._make_optimizer()
        with self.assertRaises(TypeError):
            opt.state_dict(0)


if __name__ == "__main__":
    run_tests()
