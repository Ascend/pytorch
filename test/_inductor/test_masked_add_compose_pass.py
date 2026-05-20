import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    masked_add_compose_pass,
    _is_zero_tensor_source,
    _strip_logical_not,
    _are_logically_negated_masks,
    _match_masked_zero_where,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_WHERE = torch.ops.aten.where.self
_ADD = torch.ops.aten.add.Tensor
_NOT = torch.ops.aten.logical_not.default
_FULL = torch.ops.aten.full.default
_ZEROS = torch.ops.aten.zeros.default


def _build_masked_add(shape=(4, 4), use_full_zero=True, rhs_other=None,
                      negate_rhs_mask=True, alpha=1):
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    with fm:
        mask_fake = torch.empty(shape, dtype=torch.bool)
        a_fake = torch.empty(shape, dtype=torch.float32)
        b_fake = torch.empty(shape, dtype=torch.float32)
    mask = gb.placeholder("mask", mask_fake)
    a = gb.placeholder("a", a_fake)
    b = gb.placeholder("b", b_fake)
    if use_full_zero:
        zero = gb.graph.call_function(
            _FULL, args=(list(shape), 0), kwargs={"dtype": torch.float32}
        )
        with fm:
            zero.meta["val"] = torch.ops.aten.full.default(
                list(shape), 0, dtype=torch.float32
            )
    else:
        zero = gb.graph.call_function(
            _ZEROS, args=(list(shape),), kwargs={"dtype": torch.float32}
        )
        with fm:
            zero.meta["val"] = torch.ops.aten.zeros.default(
                list(shape), dtype=torch.float32
            )

    not_mask = gb.call(_NOT, args=(mask,)) if negate_rhs_mask else mask
    w_lhs = gb.call(_WHERE, args=(mask, a, zero))
    rhs_other_node = rhs_other if rhs_other is not None else zero
    w_rhs = gb.call(_WHERE, args=(not_mask, b, rhs_other_node))
    if alpha == 1:
        add = gb.call(_ADD, args=(w_lhs, w_rhs))
    else:
        add = gb.call(_ADD, args=(w_lhs, w_rhs), kwargs={"alpha": alpha})
    gb.output(add)
    return gb.to_module(), mask, a, b


class TestMaskedAddComposePass(TestCase):
    # ===== helpers =====
    def test_is_zero_tensor_source_scalar(self):
        self.assertTrue(_is_zero_tensor_source(0))
        self.assertFalse(_is_zero_tensor_source(1))

    def test_is_zero_tensor_source_zeros_call(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        n = gb.graph.call_function(_ZEROS, args=([4],))
        self.assertTrue(_is_zero_tensor_source(n))

    def test_is_zero_tensor_source_full_zero(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        z = gb.graph.call_function(_FULL, args=([4], 0))
        self.assertTrue(_is_zero_tensor_source(z))
        nz = gb.graph.call_function(_FULL, args=([4], 5))
        self.assertFalse(_is_zero_tensor_source(nz))

    def test_strip_logical_not_logical(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            mask_fake = torch.empty((4,), dtype=torch.bool)
        m = gb.placeholder("m", mask_fake)
        nm = gb.call(_NOT, args=(m,))
        inner, neg = _strip_logical_not(nm)
        self.assertTrue(neg)
        self.assertIs(inner, m)

    def test_strip_logical_not_bitwise_on_bool(self):
        """bitwise_not 作用在 bool 张量上时也应被识别为逻辑取反。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            mask_fake = torch.empty((4,), dtype=torch.bool)
        m = gb.placeholder("m", mask_fake)
        nm = gb.call(torch.ops.aten.bitwise_not.default, args=(m,))
        inner, neg = _strip_logical_not(nm)
        self.assertTrue(neg)
        self.assertIs(inner, m)

    def test_strip_logical_not_bitwise_on_int_passthrough(self):
        """bitwise_not 作用在非 bool 张量上时不视作逻辑取反。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            int_fake = torch.empty((4,), dtype=torch.int32)
        x = gb.placeholder("x", int_fake)
        nx = gb.call(torch.ops.aten.bitwise_not.default, args=(x,))
        inner, neg = _strip_logical_not(nx)
        self.assertFalse(neg)
        self.assertIs(inner, nx)

    def test_strip_logical_not_passthrough(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4,), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        inner, neg = _strip_logical_not(x)
        self.assertFalse(neg)
        self.assertIs(inner, x)

    def test_are_logically_negated_masks(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            mask_fake = torch.empty((4,), dtype=torch.bool)
        m = gb.placeholder("m", mask_fake)
        nm = gb.call(_NOT, args=(m,))
        self.assertTrue(_are_logically_negated_masks(m, nm))
        self.assertFalse(_are_logically_negated_masks(m, m))

    def test_match_masked_zero_where_positive(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            m_fake = torch.empty((4,), dtype=torch.bool)
            v_fake = torch.empty((4,), dtype=torch.float32)
        m = gb.placeholder("m", m_fake)
        v = gb.placeholder("v", v_fake)
        w = gb.call(_WHERE, args=(m, v, 0))
        match = _match_masked_zero_where(w)
        self.assertIsNotNone(match)
        self.assertIs(match[0], m)
        self.assertIs(match[1], v)

    def test_match_masked_zero_where_negative(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            m_fake = torch.empty((4,), dtype=torch.bool)
            v_fake = torch.empty((4,), dtype=torch.float32)
        m = gb.placeholder("m", m_fake)
        v = gb.placeholder("v", v_fake)
        # other 不是 0 → 不匹配
        w = gb.call(_WHERE, args=(m, v, v))
        self.assertIsNone(_match_masked_zero_where(w))

    # ===== full pass =====
    def test_masked_add_composed_into_where(self):
        gm, mask, a, b = _build_masked_add()
        masked_add_compose_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _ADD), 0)
        wheres = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _WHERE
        ]
        self.assertEqual(len(wheres), 1)
        self.assertIs(wheres[0].args[0], mask)
        self.assertIs(wheres[0].args[1], a)
        self.assertIs(wheres[0].args[2], b)

    def test_alpha_not_one_skipped(self):
        gm, *_ = _build_masked_add(alpha=2)
        masked_add_compose_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _ADD), 1)

    def test_same_mask_skipped(self):
        """两个 where 用同一 mask（无逻辑取反）→ 不应合成。"""
        gm, *_ = _build_masked_add(negate_rhs_mask=False)
        masked_add_compose_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _ADD), 1)

    def test_non_zero_other_skipped(self):
        """右侧 where 的 other 不是 0 → 不应合成。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            mask_fake = torch.empty((4,), dtype=torch.bool)
            a_fake = torch.empty((4,), dtype=torch.float32)
            b_fake = torch.empty((4,), dtype=torch.float32)
            c_fake = torch.empty((4,), dtype=torch.float32)
        mask = gb.placeholder("mask", mask_fake)
        a = gb.placeholder("a", a_fake)
        b = gb.placeholder("b", b_fake)
        c = gb.placeholder("c", c_fake)
        z = gb.graph.call_function(_ZEROS, args=([4],))
        with fm:
            z.meta["val"] = torch.ops.aten.zeros.default([4])
        not_mask = gb.call(_NOT, args=(mask,))
        w_lhs = gb.call(_WHERE, args=(mask, a, z))
        w_rhs = gb.call(_WHERE, args=(not_mask, b, c))  # non-zero other
        add = gb.call(_ADD, args=(w_lhs, w_rhs))
        gb.output(add)
        gm = gb.to_module()
        masked_add_compose_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _ADD), 1)


if __name__ == "__main__":
    run_tests()
