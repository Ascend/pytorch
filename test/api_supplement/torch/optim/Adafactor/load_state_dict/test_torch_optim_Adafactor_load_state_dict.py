# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Adafactor.load_state_dict 接口功能正确性
API 名称：torch.optim.Adafactor.load_state_dict
API 签名：load_state_dict(self, state_dict: dict[str, Any]) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 载入非空 state_dict；空 dict / 缺顶层键报 KeyError | 已覆盖（_restores_hyperparams_and_state_values/_missing_top_level_keys_raise） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 入参为 state_dict() 产出的 dict；非 dict 入参报错 | 已覆盖（_non_dict_argument_raises） |
| 传参与不传参 | 必须传 state_dict；实测本版本无 strict 参数（传 strict 抛 TypeError） | 已覆盖（_strict_keyword_not_supported） |
| 等价类/边界值 | 参数组数/组内数不匹配 ValueError；缺state容忍 | 已覆盖（_group_{count,size}_mismatch_raises/_missing_state_entry_tolerated） |
| 正常传参场景 | 加载后超参与 state 恢复；续跑与原实例一致 | 已覆盖（_restores_hyperparams_and_state_values/_continuation_matches_unloaded_twin） |
| 异常传参场景 | 组数/组内数量不匹配 ValueError；缺顶层键 KeyError；非 dict TypeError；strict kwarg TypeError | 已覆盖（四个独立用例） |
| 混合设备类型 | NPU 实例载入 NPU step 产生的 state 并继续训练 | 已覆盖（全部用例在当前设备执行） |

未覆盖项及原因：
- 实测说明：torch 2.11 的 Optimizer.load_state_dict 没有 strict 形参（与 nn.Module 不同），
  传 strict=... 直接 TypeError "got an unexpected keyword argument 'strict'"；任务书中的
  「strict=True 缺 key 报错 / strict=False 容忍」在本版本映射为：
  ① 缺顶层键（'state'/'param_groups'）→ KeyError（实测）；
  ② 缺某参数的 state 条目 → 容忍（load 成功，该参数 state 保持空 dict，后续 step 正常）；
  两条均以实测行为独立成例（_missing_top_level_keys_raise / _missing_state_entry_tolerated）。
- 载入跨设备 state（CPU 存 → NPU 载）属于 Optimizer 通配行为，本文件聚焦同设备往返；
  跨设备恢复在 state_dict CPU Golden 对照类用例中体现数值一致性。
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
class TestAdafactorload_state_dict(TestCase):
    """Test cases for torch.optim.Adafactor.load_state_dict (NPU/CUDA portable).

    Full load contract: exact state VALUE restoration (not just key/structure),
    error taxonomy measured on this torch version (group-count and group-size
    mismatch -> ValueError; missing top-level keys -> KeyError; no 'strict'
    parameter -> TypeError), tolerance for missing per-param state entries,
    and continuation equivalence: a loaded instance must step identically to
    an unloaded twin that saw the same history.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, n_params=1, lr=0.01):
        params = [torch.nn.Parameter(torch.ones(3, device=self.device),
                                     requires_grad=True) for _ in range(n_params)]
        return torch.optim.Adafactor(params, lr=lr)

    def _step_all(self, opt, grad_value=0.5):
        for p in opt.param_groups[0]["params"]:
            p.grad = torch.full_like(p, grad_value)
        opt.step()

    def _stepped_state_dict(self, n_params=1, lr=0.01, grad_value=0.5):
        opt = self._make_optimizer(n_params=n_params, lr=lr)
        self._step_all(opt, grad_value=grad_value)
        # deepcopy isolates the checkpoint from the source optimizer's later
        # steps (state_dict returns live references - measured behavior).
        sd = copy.deepcopy(opt.state_dict())
        return opt, sd

    def test_load_state_dict_restores_hyperparams_and_state_values(self):
        """Loading restores group hyperparameters and exact state tensor values."""
        opt, sd = self._stepped_state_dict(lr=0.07)
        # Mutate the group hyperparameter to prove loading overwrites it.
        opt.param_groups[0]["lr"] = 0.5
        opt.param_groups[0]["beta2_decay"] = -0.1
        # Also wipe the live state: loading must fully repopulate it.
        for p in opt.param_groups[0]["params"]:
            opt.state[p] = {}
        opt.load_state_dict(sd)
        self.assertEqual(opt.param_groups[0]["lr"], 0.07)
        self.assertEqual(opt.param_groups[0]["beta2_decay"], -0.8)
        self.assertEqual(opt.param_groups[0]["eps"], (None, 0.001))
        self.assertEqual(opt.param_groups[0]["d"], 1.0)
        p = opt.param_groups[0]["params"][0]
        # Exact VALUE restoration of every state tensor (measured: step=1.0,
        # variance=[0.25]*3 for grad=0.5 on Adafactor defaults).
        self.assertEqual(opt.state[p]["step"].item(),
                         sd["state"][0]["step"].item())
        torch.testing.assert_close(opt.state[p]["variance"],
                                   sd["state"][0]["variance"],
                                   msg="loaded state values must equal the checkpoint")

    def test_load_state_dict_continuation_matches_unloaded_twin(self):
        """A loaded instance continues training identically to an unloaded twin.

        Golden twin: a second optimizer that saw the same two steps (0.5 then
        0.25) without any load_state_dict roundtrip. The loaded instance (one
        step, checkpoint, fresh load, second step) must produce the same param
        and state values for the second step.
        """
        # Twin: two direct steps.
        twin = self._make_optimizer(lr=0.01)
        self._step_all(twin, grad_value=0.5)
        self._step_all(twin, grad_value=0.25)
        twin_p = twin.param_groups[0]["params"][0]
        twin_param = twin_p.detach().clone()
        twin_state = {k: v.clone() for k, v in twin.state[twin_p].items()}

        # Round-trip instance: step 1 -> checkpoint -> load into a fresh
        # instance (with the checkpoint-time param value) -> step 2.
        opt, sd = self._stepped_state_dict(lr=0.01, grad_value=0.5)
        param_ckpt = opt.param_groups[0]["params"][0].detach().clone()
        q = torch.nn.Parameter(param_ckpt.clone())
        loaded = torch.optim.Adafactor([q], lr=0.01)
        loaded.load_state_dict(sd)
        q.grad = torch.full_like(q, 0.25)
        loaded.step()

        torch.testing.assert_close(q.detach(), twin_param,
                                   msg="loaded continuation must match the unloaded twin param")
        for key, ref_v in twin_state.items():
            torch.testing.assert_close(loaded.state[q][key], ref_v,
                                       msg=f"loaded state[{key!r}] must match the twin")

    def test_load_state_dict_group_count_mismatch_raises(self):
        """A different number of parameter groups raises ValueError."""
        opt, sd = self._stepped_state_dict()
        bad = {"state": sd["state"],
               "param_groups": [sd["param_groups"][0],
                                dict(sd["param_groups"][0])]}
        with self.assertRaisesRegex(ValueError,
                                    "different number of parameter groups"):
            opt.load_state_dict(bad)

    def test_load_state_dict_group_size_mismatch_raises(self):
        """A group whose params list length differs raises ValueError."""
        opt, sd = self._stepped_state_dict()
        bad = {"state": sd["state"], "param_groups": [dict(sd["param_groups"][0])]}
        bad["param_groups"][0]["params"] = [0, 1]  # optimizer group has exactly 1
        with self.assertRaisesRegex(ValueError, "doesn't match the size"):
            opt.load_state_dict(bad)

    def test_load_state_dict_missing_top_level_keys_raise(self):
        """Dicts missing 'state' or 'param_groups' raise KeyError (no strict).

        Measured order: 'param_groups' is read first (its absence raises
        KeyError('param_groups') even when 'state' is also missing), then the
        group-count check runs, then 'state' is read. A single valid group
        without 'state' raises KeyError('state').
        """
        opt = self._make_optimizer()
        with self.assertRaises(KeyError):
            opt.load_state_dict({"state": {}})       # missing 'param_groups'
        with self.assertRaises(KeyError):
            opt.load_state_dict({})                  # both keys missing
        # A well-formed group without 'state' gets past the group-count check
        # and then fails on the missing 'state' key.
        pg_only = {"param_groups": [opt.state_dict()["param_groups"][0]]}
        with self.assertRaises(KeyError):
            opt.load_state_dict(pg_only)

    def test_load_state_dict_missing_state_entry_tolerated(self):
        """A param lacking its saved state entry loads with empty state.

        Measured tolerant path (the local equivalent of the classic
        strict=False leniency): dropping one param's state entry from the
        checkpoint does NOT raise; that param keeps an empty state dict and
        the next step lazily reinitializes it from scratch.
        """
        _, sd = self._stepped_state_dict(n_params=1)
        self.assertIn(0, sd["state"])
        del sd["state"][0]  # the only entry: nothing to restore, load must pass
        fresh = self._make_optimizer(n_params=1)
        fresh.load_state_dict(sd)
        p = fresh.param_groups[0]["params"][0]
        self.assertEqual(len(fresh.state[p]), 0,
                         "missing state entry must load as empty state")
        # Training continues: the lazy init at the next step recreates state.
        p.grad = torch.full_like(p, 0.5)
        fresh.step()
        self.assertIn("step", fresh.state[p])
        self.assertIn("variance", fresh.state[p])
        self.assertEqual(fresh.state[p]["step"].item(), 1.0)

    def test_load_state_dict_strict_keyword_not_supported(self):
        """This torch version exposes no 'strict' kwarg: passing it is a TypeError."""
        _, sd = self._stepped_state_dict()
        fresh = self._make_optimizer()
        # Measured: Optimizer.load_state_dict() got an unexpected keyword
        # argument 'strict' (strict lives on nn.Module, not Optimizer).
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            fresh.load_state_dict(sd, strict=True)

    def test_load_state_dict_non_dict_argument_raises(self):
        """A non-dict argument raises TypeError/AttributeError from dict access."""
        opt = self._make_optimizer()
        with self.assertRaises((TypeError, AttributeError)):
            opt.load_state_dict([("state", {}), ("param_groups", [])])
        with self.assertRaises((TypeError, AttributeError)):
            opt.load_state_dict(None)


if __name__ == "__main__":
    run_tests()
