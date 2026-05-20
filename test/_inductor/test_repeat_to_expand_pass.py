import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    repeat_to_expand_pass,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_REPEAT = torch.ops.aten.repeat.default
_EXPAND = torch.ops.aten.expand.default
_MUL = torch.ops.aten.mul.Tensor
_RELU = torch.ops.aten.relu.default


def _build(input_shape, repeats, user_target=_MUL, other_shape=None):
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    with fm:
        x_fake = torch.empty(input_shape, dtype=torch.float32)
        other_shape = other_shape or [int(s) * int(r) for s, r in zip(input_shape, repeats)]
        y_fake = torch.empty(other_shape, dtype=torch.float32)
    x = gb.placeholder("x", x_fake)
    y = gb.placeholder("y", y_fake)
    rpt = gb.call(_REPEAT, args=(x, list(repeats)))
    out = gb.call(user_target, args=(rpt, y)) if user_target is not _RELU else gb.call(user_target, args=(rpt,))
    gb.output(out)
    return gb.to_module()


class TestRepeatToExpandPass(TestCase):
    def test_pure_broadcast_repeat_rewritten(self):
        """input_shape=(1,4), repeats=(8,1) → 应替换为 expand."""
        gm = _build((1, 4), (8, 1))
        repeat_to_expand_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _REPEAT), 0)
        self.assertEqual(count_target(gm.graph, _EXPAND), 1)
        exp = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EXPAND
        )
        self.assertEqual(list(exp.args[1]), [8, 4])

    def test_non_broadcast_repeat_skipped(self):
        """非 1 维度也有 repeat>1（实际产生新数据），不能替换。"""
        gm = _build((2, 4), (3, 1))  # 维度 0 上 size=2 且 repeat=3 → 真物理拷贝
        repeat_to_expand_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _REPEAT), 1)
        self.assertEqual(count_target(gm.graph, _EXPAND), 0)

    def test_user_not_broadcast_friendly_skipped(self):
        """下游使用者不属于广播友好集合 → 不应替换。"""
        gm = _build((1, 4), (8, 1), user_target=_RELU)
        repeat_to_expand_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _REPEAT), 1)
        self.assertEqual(count_target(gm.graph, _EXPAND), 0)

    def test_repeats_len_mismatch_skipped(self):
        """repeats 长度 != input 维度数 → 不替换（保留原 repeat）。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((1, 4), dtype=torch.float32)
            y_fake = torch.empty((2, 1, 4), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        y = gb.placeholder("y", y_fake)
        rpt = gb.call(_REPEAT, args=(x, [2, 1, 1]))
        out = gb.call(_MUL, args=(rpt, y))
        gb.output(out)
        gm = gb.to_module()
        repeat_to_expand_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _REPEAT), 1)
        self.assertEqual(count_target(gm.graph, _EXPAND), 0)

    def test_no_users_skipped(self):
        """repeat 节点没有 user（如它就是输出）→ 不替换。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((1, 4), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        rpt = gb.call(_REPEAT, args=(x, [8, 1]))
        gb.output(rpt)
        gm = gb.to_module()
        repeat_to_expand_pass(gm.graph)
        # output 不算 call_function 用户，rpt.users 为空 → 不应替换
        self.assertEqual(count_target(gm.graph, _REPEAT), 1)

    def test_repeats_not_list_skipped(self):
        """repeats 参数不是 list/tuple → 直接跳过。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((1, 4), dtype=torch.float32)
            y_fake = torch.empty((8, 4), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        y = gb.placeholder("y", y_fake)
        rpt = gb.call(_REPEAT, args=(x, "not_a_list"))
        out = gb.call(_MUL, args=(rpt, y))
        gb.output(out)
        gm = gb.to_module()
        repeat_to_expand_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _REPEAT), 1)


if __name__ == "__main__":
    run_tests()
