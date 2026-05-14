import torch
import torch.fx as fx
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    cat_to_view_pass,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_CAT = torch.ops.aten.cat.default
_SLICE = torch.ops.aten.slice.Tensor
_ROLL = torch.ops.aten.roll.default


def _build_cat_of_slices(intervals, dim=1, shape=(2, 6, 4), other_parent_idx=None):
    """构造一个 cat([slice_0, slice_1, ...]) 图。

    intervals: List[(start, end)]，按列出顺序填进 cat 的输入。
    other_parent_idx: 若非 None，则该索引位置的 slice 的 parent 替换成另一个 placeholder。
    """
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    with fm:
        parent_fake = torch.empty(shape, dtype=torch.float32)
        other_fake = torch.empty(shape, dtype=torch.float32)
    parent = gb.placeholder("parent", parent_fake)
    other = gb.placeholder("other", other_fake)
    slice_nodes = []
    for i, (s, e) in enumerate(intervals):
        src = other if other_parent_idx == i else parent
        n = gb.call(_SLICE, args=(src, dim, s, e, 1))
        slice_nodes.append(n)
    cat = gb.call(_CAT, args=(slice_nodes, dim))
    gb.output(cat)
    return gb.to_module(), parent, cat


class TestCatToViewPass(TestCase):
    def test_identity_full_cover(self):
        """连续 [0,2)+[2,4)+[4,6) 完整覆盖 dim=1 → cat 应被替换为 parent。"""
        gm, parent, cat = _build_cat_of_slices(
            [(0, 2), (2, 4), (4, 6)], dim=1, shape=(2, 6, 4)
        )
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 0)
        self.assertEqual(count_target(gm.graph, _ROLL), 0)
        out_node = next(n for n in gm.graph.nodes if n.op == "output")
        self.assertIs(out_node.args[0], parent)

    def test_rotation_full_cover(self):
        """[2,4)+[4,6)+[0,2) 完整覆盖但是循环位移 → 应插入 roll 节点。"""
        gm, _, _ = _build_cat_of_slices(
            [(2, 4), (4, 6), (0, 2)], dim=1, shape=(2, 6, 4)
        )
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 0)
        self.assertEqual(count_target(gm.graph, _ROLL), 1)
        roll = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _ROLL
        )
        self.assertEqual(list(roll.args[1]), [-2])
        self.assertEqual(list(roll.args[2]), [1])

    def test_partial_cover_skipped(self):
        """[0,2)+[2,4) 仅部分覆盖 dim=1（size=6）→ 不应折叠。"""
        gm, _, _ = _build_cat_of_slices(
            [(0, 2), (2, 4)], dim=1, shape=(2, 6, 4)
        )
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 1)

    def test_different_parents_skipped(self):
        """不同 parent 的 slice → 不应折叠。"""
        gm, _, _ = _build_cat_of_slices(
            [(0, 3), (3, 6)], dim=1, shape=(2, 6, 4), other_parent_idx=1
        )
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 1)

    def test_step_not_one_skipped(self):
        """slice step != 1 → 不应折叠。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            parent_fake = torch.empty((2, 6), dtype=torch.float32)
        parent = gb.placeholder("parent", parent_fake)
        s0 = gb.call(_SLICE, args=(parent, 1, 0, 6, 2))
        s1 = gb.call(_SLICE, args=(parent, 1, 1, 6, 2))
        cat = gb.call(_CAT, args=([s0, s1], 1))
        gb.output(cat)
        gm = gb.to_module()
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 1)

    def test_non_int_dim_skipped(self):
        """cat dim 非 int（例如 SymInt 之类）→ 直接跳过。"""
        gm, _, cat = _build_cat_of_slices(
            [(0, 3), (3, 6)], dim=1, shape=(2, 6)
        )
        cat.args = (cat.args[0], "bad")
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 1)

    def test_non_cat_node_untouched(self):
        """图中没有 cat 节点 → pass 不修改图。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4, 4), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        add = gb.call(torch.ops.aten.add.Tensor, args=(x, x))
        gb.output(add)
        gm = gb.to_module()
        orig = str(gm.graph)
        cat_to_view_pass(gm.graph)
        self.assertEqual(orig, str(gm.graph))

    def test_negative_dim_normalised(self):
        """dim=-1 应被规范化为 rank-1 并正确进入 identity 路径。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            parent_fake = torch.empty((2, 6), dtype=torch.float32)
        parent = gb.placeholder("parent", parent_fake)
        s0 = gb.call(_SLICE, args=(parent, -1, 0, 3, 1))
        s1 = gb.call(_SLICE, args=(parent, -1, 3, 6, 1))
        cat = gb.call(_CAT, args=([s0, s1], -1))
        gb.output(cat)
        gm = gb.to_module()
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 0)
        out_node = next(n for n in gm.graph.nodes if n.op == "output")
        self.assertIs(out_node.args[0], parent)

    def test_dim_via_kwargs(self):
        """cat 用 kwargs 传 dim 也应能正确处理（默认 dim=0）。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            parent_fake = torch.empty((6, 2), dtype=torch.float32)
        parent = gb.placeholder("parent", parent_fake)
        s0 = gb.call(_SLICE, args=(parent, 0, 0, 3, 1))
        s1 = gb.call(_SLICE, args=(parent, 0, 3, 6, 1))
        # 仅传 inputs 列表，不传 dim → 走 kwargs.get("dim", 0) 分支
        cat = gb.call(_CAT, args=([s0, s1],))
        gb.output(cat)
        gm = gb.to_module()
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 0)

    def test_cat_with_non_slice_input_skipped(self):
        """cat 输入不全是 slice → 不应折叠。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            parent_fake = torch.empty((2, 6), dtype=torch.float32)
        parent = gb.placeholder("parent", parent_fake)
        s0 = gb.call(_SLICE, args=(parent, 1, 0, 3, 1))
        not_a_slice = gb.call(torch.ops.aten.relu.default, args=(parent,))
        s1 = gb.call(_SLICE, args=(not_a_slice, 1, 3, 6, 1))
        cat = gb.call(_CAT, args=([s0, s1], 1))
        gb.output(cat)
        gm = gb.to_module()
        cat_to_view_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAT), 1)


if __name__ == "__main__":
    run_tests()
