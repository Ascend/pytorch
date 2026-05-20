import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    bool_cast_mul_to_where_pass,
    _walk_back_view_chain_to_cast,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_MUL = torch.ops.aten.mul.Tensor
_WHERE = torch.ops.aten.where.self
_CAST = torch.ops.prims.convert_element_type.default
_VIEW = torch.ops.aten.view.default
_UNSQUEEZE = torch.ops.aten.unsqueeze.default


def _build_bool_cast_mul(
    shape=(4, 4),
    target_dtype=torch.float32,
    other_dtype=None,
    insert_view=False,
    bool_src=True,
):
    other_dtype = other_dtype or target_dtype
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    with fm:
        mask_fake = torch.empty(shape, dtype=torch.bool if bool_src else torch.int32)
        other_fake = torch.empty(shape, dtype=other_dtype)
    mask = gb.placeholder("mask", mask_fake)
    other = gb.placeholder("other", other_fake)
    cast = gb.call(_CAST, args=(mask, target_dtype))
    after_cast = cast
    if insert_view:
        after_cast = gb.call(_VIEW, args=(cast, list(shape)))
    mul = gb.call(_MUL, args=(after_cast, other))
    gb.output(mul)
    return gb.to_module(), mask, other, cast, mul


class TestBoolCastMulToWherePass(TestCase):
    # ===== _walk_back_view_chain_to_cast =====
    def test_walk_back_direct_cast(self):
        gm, mask, other, cast, mul = _build_bool_cast_mul()
        chain, found = _walk_back_view_chain_to_cast(cast)
        self.assertEqual(chain, [])
        self.assertIs(found, cast)

    def test_walk_back_through_view(self):
        gm, mask, other, cast, mul = _build_bool_cast_mul(insert_view=True)
        view = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _VIEW
        )
        chain, found = _walk_back_view_chain_to_cast(view)
        self.assertEqual(len(chain), 1)
        self.assertIs(found, cast)

    def test_walk_back_no_cast(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4,), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        v = gb.call(_VIEW, args=(x, [4]))
        chain, found = _walk_back_view_chain_to_cast(v)
        self.assertIsNone(found)

    # ===== full pass =====
    def test_basic_rewrite(self):
        gm, mask, other, cast, mul = _build_bool_cast_mul()
        bool_cast_mul_to_where_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _MUL), 0)
        self.assertEqual(count_target(gm.graph, _WHERE), 1)
        w = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _WHERE
        )
        self.assertIs(w.args[0], mask)
        self.assertIs(w.args[1], other)

    def test_with_view_chain(self):
        gm, mask, other, cast, mul = _build_bool_cast_mul(insert_view=True)
        bool_cast_mul_to_where_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _MUL), 0)
        self.assertEqual(count_target(gm.graph, _WHERE), 1)

    def test_non_bool_source_skipped(self):
        gm, *_ = _build_bool_cast_mul(bool_src=False)
        bool_cast_mul_to_where_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _MUL), 1)
        self.assertEqual(count_target(gm.graph, _WHERE), 0)

    def test_dtype_mismatch_skipped(self):
        """cast 的目标 dtype 与 other 的 dtype 不同 → 不应重写。"""
        gm, *_ = _build_bool_cast_mul(target_dtype=torch.float16, other_dtype=torch.float32)
        bool_cast_mul_to_where_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _MUL), 1)
        self.assertEqual(count_target(gm.graph, _WHERE), 0)

    def test_cast_node_with_extra_user_skipped(self):
        """cast 节点有不只一个用户 → 不应重写。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            mask_fake = torch.empty((4,), dtype=torch.bool)
            other_fake = torch.empty((4,), dtype=torch.float32)
        mask = gb.placeholder("mask", mask_fake)
        other = gb.placeholder("other", other_fake)
        cast = gb.call(_CAST, args=(mask, torch.float32))
        mul = gb.call(_MUL, args=(cast, other))
        # 让 cast 同时被另一个算子使用
        relu = gb.call(torch.ops.aten.relu.default, args=(cast,))
        gb.output((mul, relu))
        gm = gb.to_module()
        bool_cast_mul_to_where_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _MUL), 1)
        self.assertEqual(count_target(gm.graph, _WHERE), 0)

    def test_mul_with_scalar_skipped(self):
        """mul 一侧为标量 → 找不到 cast→bool 模式，应跳过。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4,), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        mul = gb.call(_MUL, args=(x, 2.0))
        gb.output(mul)
        gm = gb.to_module()
        bool_cast_mul_to_where_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _MUL), 1)


if __name__ == "__main__":
    run_tests()
