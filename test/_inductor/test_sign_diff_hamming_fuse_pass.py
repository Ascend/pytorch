import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    sign_diff_hamming_fuse_pass,
    _peel_single_user_relu_sign,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_SIGN = torch.ops.aten.sign.default
_RELU = torch.ops.aten.relu.default
_SUB = torch.ops.aten.sub.Tensor
_ABS = torch.ops.aten.abs.default
_SUM = torch.ops.aten.sum.dim_IntList
_GT = torch.ops.aten.gt.Scalar
_NE = torch.ops.aten.ne.Tensor


def _build_full_pattern(shape=(8,), dim=[0], alpha=1, drop_abs=False, drop_sign_x=False):
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    with fm:
        x_fake = torch.empty(shape, dtype=torch.float32)
        y_fake = torch.empty(shape, dtype=torch.float32)
    x = gb.placeholder("x", x_fake)
    y = gb.placeholder("y", y_fake)
    sx = gb.call(_SIGN, args=(x,)) if not drop_sign_x else x
    sy = gb.call(_SIGN, args=(y,))
    rx = gb.call(_RELU, args=(sx,)) if not drop_sign_x else gb.call(_RELU, args=(sx,))
    ry = gb.call(_RELU, args=(sy,))
    sub_kwargs = {} if alpha == 1 else {"alpha": alpha}
    sub = gb.call(_SUB, args=(rx, ry), kwargs=sub_kwargs)
    upstream = sub if drop_abs else gb.call(_ABS, args=(sub,))
    s = gb.call(_SUM, args=(upstream, dim, False), kwargs={"dtype": torch.int64})
    gb.output(s)
    return gb.to_module(), x, y


class TestSignDiffHammingFusePass(TestCase):
    def test_peel_relu_sign(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4,), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        s = gb.call(_SIGN, args=(x,))
        r = gb.call(_RELU, args=(s,))
        gb.output(r)
        self.assertIs(_peel_single_user_relu_sign(r), x)

    def test_peel_not_relu(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4,), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        self.assertIsNone(_peel_single_user_relu_sign(x))

    def test_peel_sign_with_multiple_users(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4,), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        s = gb.call(_SIGN, args=(x,))
        r1 = gb.call(_RELU, args=(s,))
        r2 = gb.call(_RELU, args=(s,))
        gb.output((r1, r2))
        # sign 节点有两个使用者 → 不匹配
        self.assertIsNone(_peel_single_user_relu_sign(r1))

    def test_full_pattern_fused(self):
        gm, x, y = _build_full_pattern()
        sign_diff_hamming_fuse_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _ABS), 0)
        self.assertEqual(count_target(gm.graph, _SUB), 0)
        self.assertEqual(count_target(gm.graph, _GT), 2)
        self.assertEqual(count_target(gm.graph, _NE), 1)
        new_sum = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _SUM
        )
        self.assertEqual(new_sum.kwargs.get("dtype"), torch.int64)

    def test_missing_abs_not_fused(self):
        gm, *_ = _build_full_pattern(drop_abs=True)
        sign_diff_hamming_fuse_pass(gm.graph)
        # 没有 abs，sum 的输入是 sub，模式不完整 → 不应改写
        self.assertEqual(count_target(gm.graph, _NE), 0)

    def test_missing_sign_chain_not_fused(self):
        gm, *_ = _build_full_pattern(drop_sign_x=True)
        sign_diff_hamming_fuse_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _NE), 0)

    def test_alpha_not_one_skipped(self):
        gm, *_ = _build_full_pattern(alpha=2)
        sign_diff_hamming_fuse_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _NE), 0)

    def test_keepdim_true_still_fused(self):
        """keepdim=True 也应能识别（pass 直接透传 keepdim 给新 sum）。"""
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            x_fake = torch.empty((4, 8), dtype=torch.float32)
            y_fake = torch.empty((4, 8), dtype=torch.float32)
        x = gb.placeholder("x", x_fake)
        y = gb.placeholder("y", y_fake)
        sx = gb.call(_SIGN, args=(x,))
        sy = gb.call(_SIGN, args=(y,))
        rx = gb.call(_RELU, args=(sx,))
        ry = gb.call(_RELU, args=(sy,))
        sub = gb.call(_SUB, args=(rx, ry))
        a = gb.call(_ABS, args=(sub,))
        s = gb.call(_SUM, args=(a, [1], True), kwargs={"dtype": torch.int64})
        gb.output(s)
        gm = gb.to_module()
        sign_diff_hamming_fuse_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _NE), 1)
        new_sum = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _SUM
        )
        self.assertEqual(new_sum.args[2], True)


if __name__ == "__main__":
    run_tests()
