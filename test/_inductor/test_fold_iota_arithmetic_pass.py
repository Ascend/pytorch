import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    fold_iota_arithmetic_pass,
    _prims_iota_value_range,
    _collect_iota_downcast_closure,
    _hashable_const_key,
    _cse_constant_call,
    _refresh_fake_meta,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_IOTA = torch.ops.prims.iota.default
_FULL = torch.ops.aten.full.default
_SUB = torch.ops.aten.sub.Tensor
_GE_S = torch.ops.aten.ge.Scalar
_GT_S = torch.ops.aten.gt.Scalar
_LT_S = torch.ops.aten.lt.Scalar
_LE_S = torch.ops.aten.le.Scalar
_EQ_S = torch.ops.aten.eq.Scalar
_NE_S = torch.ops.aten.ne.Scalar
_GE_T = torch.ops.aten.ge.Tensor
_GT_T = torch.ops.aten.gt.Tensor


def _build_iota_with_user(length, start=0, step=1, dtype=torch.int64,
                          user_target=torch.ops.aten.add.Tensor, user_rhs_shape=None):
    """构造 iota → transparent_op → closing_op(_to_copy) 的图，
    满足 fold_iota_arithmetic_pass 的闭包检测要求。"""
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    with fm:
        iota_val = torch.ops.prims.iota.default(
            length, start=start, step=step, dtype=dtype, device="cpu", requires_grad=False
        )
        rhs_shape = user_rhs_shape or (length,)
        rhs_val = torch.empty(rhs_shape, dtype=dtype)
    iota = gb.graph.call_function(
        _IOTA,
        args=(length,),
        kwargs={
            "start": start,
            "step": step,
            "dtype": dtype,
            "device": torch.device("cpu"),
            "requires_grad": False,
        },
    )
    iota.meta["val"] = iota_val
    rhs = gb.placeholder("rhs", rhs_val)
    middle = gb.call(user_target, args=(iota, rhs))
    # 最终下游必须是 closing op（如 convert_element_type），否则闭包检测会拒绝。
    closing = gb.call(
        torch.ops.prims.convert_element_type.default,
        args=(middle, torch.float32),
    )
    gb.output(closing)
    return gb.to_module(), iota


class TestFoldIotaArithmeticPass(TestCase):
    # ===== _prims_iota_value_range =====
    def test_iota_value_range_positive_step(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        n = gb.graph.call_function(
            _IOTA, args=(5,),
            kwargs={"start": 3, "step": 2, "dtype": torch.int64,
                    "device": torch.device("cpu"), "requires_grad": False},
        )
        self.assertEqual(_prims_iota_value_range(n), (3, 12))

    def test_iota_value_range_negative_step(self):
        """负 step：序列 [10, 7, 4] → range=(4, 11)。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        n = gb.graph.call_function(
            _IOTA, args=(3,),
            kwargs={"start": 10, "step": -3, "dtype": torch.int64,
                    "device": torch.device("cpu"), "requires_grad": False},
        )
        self.assertEqual(_prims_iota_value_range(n), (4, 11))

    def test_iota_value_range_zero_length(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        n = gb.graph.call_function(
            _IOTA, args=(0,),
            kwargs={"start": 5, "step": 1, "dtype": torch.int64,
                    "device": torch.device("cpu"), "requires_grad": False},
        )
        self.assertEqual(_prims_iota_value_range(n), (5, 5))

    def test_iota_value_range_non_constant(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        n = gb.graph.call_function(
            _IOTA, args=("nonint",),
            kwargs={"start": 0, "step": 1, "dtype": torch.int64},
        )
        self.assertIsNone(_prims_iota_value_range(n))

    def test_iota_value_range_not_iota(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        x = gb.placeholder("x", torch.empty((4,), dtype=torch.float32))
        self.assertIsNone(_prims_iota_value_range(x))

    # ===== iota int64 → int32 downcast =====
    def test_iota_downcast_to_int32(self):
        gm, iota = _build_iota_with_user(length=128)
        fold_iota_arithmetic_pass(gm.graph)
        new_iota = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _IOTA
        )
        self.assertIs(new_iota.kwargs["dtype"], torch.int32)

    def test_iota_out_of_int32_range_no_downcast(self):
        gm, iota = _build_iota_with_user(length=2, start=2**31)
        fold_iota_arithmetic_pass(gm.graph)
        new_iota = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _IOTA
        )
        self.assertIs(new_iota.kwargs["dtype"], torch.int64)

    def test_iota_non_int64_dtype_skipped(self):
        gm, iota = _build_iota_with_user(length=10, dtype=torch.int32)
        fold_iota_arithmetic_pass(gm.graph)
        new_iota = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _IOTA
        )
        self.assertIs(new_iota.kwargs["dtype"], torch.int32)

    # ===== closure collection =====
    def test_collect_closure_only_transparent_then_closing(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            iota_val = torch.ops.prims.iota.default(
                4, start=0, step=1, dtype=torch.int64,
                device="cpu", requires_grad=False,
            )
        iota = gb.graph.call_function(
            _IOTA, args=(4,),
            kwargs={"start": 0, "step": 1, "dtype": torch.int64,
                    "device": torch.device("cpu"), "requires_grad": False},
        )
        iota.meta["val"] = iota_val
        v = gb.call(torch.ops.aten.view.default, args=(iota, [2, 2]))
        cmp = gb.call(_GE_S, args=(v, 1))
        gb.output(cmp)
        ids = _collect_iota_downcast_closure(iota)
        self.assertIsNotNone(ids)
        self.assertIn(id(v), ids)

    def test_collect_closure_rejects_non_transparent_user(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            iota_val = torch.ops.prims.iota.default(
                4, start=0, step=1, dtype=torch.int64,
                device="cpu", requires_grad=False,
            )
        iota = gb.graph.call_function(
            _IOTA, args=(4,),
            kwargs={"start": 0, "step": 1, "dtype": torch.int64,
                    "device": torch.device("cpu"), "requires_grad": False},
        )
        iota.meta["val"] = iota_val
        non_trans = gb.call(torch.ops.aten.relu.default, args=(iota,))
        gb.output(non_trans)
        self.assertIsNone(_collect_iota_downcast_closure(iota))

    # ===== cmp(sub(a,b), 0) → cmp(a,b) =====
    def _build_cmp_sub_zero(self, cmp_target, rhs=0, alpha=1):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            a_fake = torch.empty((4,), dtype=torch.float32)
            b_fake = torch.empty((4,), dtype=torch.float32)
        a = gb.placeholder("a", a_fake)
        b = gb.placeholder("b", b_fake)
        sub_kwargs = {} if alpha == 1 else {"alpha": alpha}
        sub = gb.call(_SUB, args=(a, b), kwargs=sub_kwargs)
        cmp = gb.call(cmp_target, args=(sub, rhs))
        gb.output(cmp)
        return gb.to_module()

    def test_cmp_sub_zero_simplified_ge(self):
        gm = self._build_cmp_sub_zero(_GE_S)
        fold_iota_arithmetic_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _GE_T), 1)
        self.assertEqual(count_target(gm.graph, _GE_S), 0)

    def test_cmp_sub_zero_simplified_gt(self):
        gm = self._build_cmp_sub_zero(_GT_S)
        fold_iota_arithmetic_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _GT_T), 1)

    def test_cmp_non_zero_rhs_skipped(self):
        gm = self._build_cmp_sub_zero(_GE_S, rhs=1)
        fold_iota_arithmetic_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _GE_S), 1)
        self.assertEqual(count_target(gm.graph, _GE_T), 0)

    def test_cmp_sub_with_alpha_skipped(self):
        gm = self._build_cmp_sub_zero(_GE_S, alpha=2)
        fold_iota_arithmetic_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _GE_S), 1)
        self.assertEqual(count_target(gm.graph, _GE_T), 0)

    def test_cmp_lhs_not_sub_skipped(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            a_fake = torch.empty((4,), dtype=torch.float32)
        a = gb.placeholder("a", a_fake)
        cmp = gb.call(_GE_S, args=(a, 0))
        gb.output(cmp)
        gm = gb.to_module()
        fold_iota_arithmetic_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _GE_S), 1)

    # ===== _hashable_const_key & _cse_constant_call =====
    def test_hashable_const_key_basic(self):
        self.assertEqual(_hashable_const_key(3), 3)
        self.assertEqual(
            _hashable_const_key([1, [2, 3]]),
            ("__list__", 1, ("__list__", 2, 3)),
        )
        self.assertEqual(
            _hashable_const_key((1, 2)),
            ("__tuple__", 1, 2),
        )
        self.assertEqual(
            _hashable_const_key({"a": 1}),
            ("__dict__", ("a", 1)),
        )

    def test_cse_constant_call_dedups(self):
        """两个同参数 full 调用应被 CSE 合并。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            f_fake = torch.ops.aten.full.default([4], 0.0, dtype=torch.float32)
        # 确保 placeholder 先于 call_function 节点出现，避免 FX 拓扑告警。
        _ = gb.placeholder("x", torch.empty((4,), dtype=torch.float32))
        f1 = gb.graph.call_function(_FULL, args=([4], 0.0), kwargs={"dtype": torch.float32})
        f1.meta["val"] = f_fake
        f2 = gb.graph.call_function(_FULL, args=([4], 0.0), kwargs={"dtype": torch.float32})
        f2.meta["val"] = f_fake
        add = gb.call(torch.ops.aten.add.Tensor, args=(f1, f2))
        gb.output(add)
        gm = gb.to_module()
        changed = _cse_constant_call(gm.graph, _FULL)
        self.assertTrue(changed)


if __name__ == "__main__":
    run_tests()
