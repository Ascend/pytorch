import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    broadcast_const_mask_compress,
    _extract_const_full_scalar,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_WHERE = torch.ops.aten.where.self
_FULL = torch.ops.aten.full.default
_CAST = torch.ops.prims.convert_element_type.default
_NOT = torch.ops.aten.logical_not.default
_ADD = torch.ops.aten.add.Tensor


def _build_cast_where_full_full(
    shape=(4, 4),
    t_val=1,
    f_val=0,
    target_dtype=torch.float32,
    mask_is_bool=True,
):
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    # 对 bool dtype 的 full，使用 Python bool 字面量，避免某些版本不接受 0/1。
    fill_dtype = torch.float32 if target_dtype == torch.bool else target_dtype
    t_payload = bool(t_val) if target_dtype == torch.bool else t_val
    f_payload = bool(f_val) if target_dtype == torch.bool else f_val
    with fm:
        cond_fake = torch.empty(shape, dtype=torch.bool if mask_is_bool else torch.int32)
        full_t_fake = torch.ops.aten.full.default(list(shape), t_payload, dtype=fill_dtype)
        full_f_fake = torch.ops.aten.full.default(list(shape), f_payload, dtype=fill_dtype)
    cond = gb.placeholder("cond", cond_fake)
    full_t = gb.graph.call_function(
        _FULL, args=(list(shape), t_val), kwargs={"dtype": fill_dtype}
    )
    full_t.meta["val"] = full_t_fake
    full_f = gb.graph.call_function(
        _FULL, args=(list(shape), f_val), kwargs={"dtype": fill_dtype}
    )
    full_f.meta["val"] = full_f_fake
    w = gb.call(_WHERE, args=(cond, full_t, full_f))
    cast = gb.call(_CAST, args=(w, target_dtype))
    out = gb.call(_ADD, args=(cast, cast))
    gb.output(out)
    return gb.to_module(), cond, w, cast


class TestBroadcastConstMaskCompress(TestCase):
    def test_mask_full1_full0_replaces_with_mask(self):
        gm, cond, w, cast = _build_cast_where_full_full(t_val=1, f_val=0)
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _NOT), 0)
        self.assertIs(cast.args[0], cond)

    def test_mask_full0_full1_inserts_logical_not(self):
        gm, cond, w, cast = _build_cast_where_full_full(t_val=0, f_val=1)
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _NOT), 1)
        new_cond = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _NOT
        )
        self.assertIs(cast.args[0], new_cond)

    def test_target_dtype_bool_drops_cast(self):
        gm, cond, w, cast = _build_cast_where_full_full(
            t_val=1, f_val=0, target_dtype=torch.bool
        )
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _CAST), 0)
        out_consumer = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _ADD
        )
        self.assertIs(out_consumer.args[0], cond)

    def test_target_dtype_non_bool_non_01_skipped(self):
        gm, cond, w, cast = _build_cast_where_full_full(t_val=5, f_val=3)
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _WHERE), 1)

    def test_equal_constants_skipped(self):
        gm, cond, w, cast = _build_cast_where_full_full(t_val=1, f_val=1)
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _WHERE), 1)

    def test_mask_not_bool_skipped(self):
        gm, cond, w, cast = _build_cast_where_full_full(
            t_val=1, f_val=0, mask_is_bool=False
        )
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _WHERE), 1)

    def test_where_with_non_full_branch_skipped(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            cond_fake = torch.empty((4,), dtype=torch.bool)
            x_fake = torch.empty((4,), dtype=torch.float32)
        cond = gb.placeholder("cond", cond_fake)
        x = gb.placeholder("x", x_fake)
        full_f = gb.graph.call_function(_FULL, args=([4], 0), kwargs={"dtype": torch.float32})
        with fm:
            full_f.meta["val"] = torch.ops.aten.full.default([4], 0, dtype=torch.float32)
        w = gb.call(_WHERE, args=(cond, x, full_f))
        cast = gb.call(_CAST, args=(w, torch.float32))
        gb.output(cast)
        gm = gb.to_module()
        broadcast_const_mask_compress(gm.graph)
        self.assertEqual(count_target(gm.graph, _WHERE), 1)

    def test_extract_const_full_scalar_helpers(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        f = gb.graph.call_function(_FULL, args=([4], 7), kwargs={"dtype": torch.float32})
        self.assertEqual(_extract_const_full_scalar(f), 7)
        x = gb.placeholder("x", torch.empty((4,), dtype=torch.float32))
        self.assertIsNone(_extract_const_full_scalar(x))
        self.assertIsNone(_extract_const_full_scalar(42))


if __name__ == "__main__":
    run_tests()
