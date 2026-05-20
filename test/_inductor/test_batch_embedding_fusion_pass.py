import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    batch_embedding_fusion_pass,
    _has_default_embedding_args,
    _symbolic_shape_key,
    _weight_node_key,
    _reduce_call_args,
    _detect_reduce_pattern,
    _detect_indices_parent,
)

from _pass_test_utils import (
    GraphBuilder,
    count_target,
    new_fake_mode,
)


_EMBED = torch.ops.aten.embedding.default
_SLICE = torch.ops.aten.slice.Tensor
_SUM = torch.ops.aten.sum.dim_IntList
_PROD = torch.ops.aten.prod.dim_int
_RESHAPE = torch.ops.aten.reshape.default


def _build_batch_embedding(
    n_emb=2,
    V=10,
    D=4,
    L=3,
    keepdim=False,
    use_mixed_reduce=False,
    padding_idx=-1,
    indices_from_single_parent=True,
):
    """构建 n_emb 个 embedding + slice + sum 的图。"""
    fm = new_fake_mode()
    gb = GraphBuilder(fm)
    parent_len = n_emb * L
    with fm:
        w_fake = torch.empty((V, D), dtype=torch.float32)
        idx_fake = torch.empty((parent_len,), dtype=torch.int64).fill_(0)
    weight = gb.placeholder("weight", w_fake)
    parent = gb.placeholder("idx", idx_fake)
    reduces = []
    for i in range(n_emb):
        start = i * L
        end = start + L
        if indices_from_single_parent:
            src = parent
        else:
            # 第二个 embedding 使用一个独立的 placeholder 作为索引父节点
            with fm:
                other_idx = torch.empty((parent_len,), dtype=torch.int64)
            src = gb.placeholder(f"idx_alt_{i}", other_idx)
        sl = gb.call(_SLICE, args=(src, 0, start, end, 1))
        emb_args = (weight, sl)
        if padding_idx != -1:
            emb_args = (weight, sl, padding_idx)
        emb = gb.call(_EMBED, args=emb_args)
        if use_mixed_reduce and i == 1:
            red = gb.call(_PROD, args=(emb, 0, keepdim))
        else:
            red = gb.call(_SUM, args=(emb, [0], keepdim))
        reduces.append(red)
    gb.output(tuple(reduces))
    return gb.to_module(), weight, parent, reduces


class TestBatchEmbeddingFusionPass(TestCase):
    # ===== _has_default_embedding_args =====
    def test_has_default_embedding_args_default(self):
        gm, *_ = _build_batch_embedding(n_emb=1)
        emb = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        )
        self.assertTrue(_has_default_embedding_args(emb))

    def test_has_default_embedding_args_padding_idx(self):
        gm, *_ = _build_batch_embedding(n_emb=1, padding_idx=0)
        emb = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        )
        self.assertFalse(_has_default_embedding_args(emb))

    def test_has_default_embedding_args_kwargs(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        with fm:
            w_fake = torch.empty((4, 4), dtype=torch.float32)
            idx_fake = torch.empty((4,), dtype=torch.int64)
        w = gb.placeholder("w", w_fake)
        idx = gb.placeholder("idx", idx_fake)
        emb = gb.graph.call_function(
            _EMBED, args=(w, idx), kwargs={"scale_grad_by_freq": True}
        )
        self.assertFalse(_has_default_embedding_args(emb))

    # ===== helpers =====
    def test_symbolic_shape_key(self):
        self.assertEqual(_symbolic_shape_key([4, 8]), (4, 8))

    def test_weight_node_key_placeholder(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        ph = gb.placeholder("w", torch.empty((2,), dtype=torch.float32))
        key = _weight_node_key(ph)
        self.assertEqual(key, ("placeholder", "w"))

    def test_weight_node_key_call_function(self):
        fm = new_fake_mode()
        gb = GraphBuilder(fm)
        x = gb.placeholder("x", torch.empty((2,), dtype=torch.float32))
        c = gb.call(torch.ops.aten.relu.default, args=(x,))
        self.assertEqual(_weight_node_key(c), id(c))

    def test_reduce_call_args_dim_list(self):
        self.assertEqual(_reduce_call_args(_SUM, "x", 1), ("x", [1]))

    def test_reduce_call_args_dim_int(self):
        self.assertEqual(_reduce_call_args(_PROD, "x", 1), ("x", 1))

    def test_detect_reduce_pattern_consistent_sum(self):
        gm, *_ = _build_batch_embedding(n_emb=2)
        embs = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        ]
        result = _detect_reduce_pattern(embs, cat_dim=0)
        self.assertIsNotNone(result)
        target, _ = result
        self.assertIs(target, _SUM)

    def test_detect_reduce_pattern_mixed_skipped(self):
        gm, *_ = _build_batch_embedding(n_emb=2, use_mixed_reduce=True)
        embs = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        ]
        self.assertIsNone(_detect_reduce_pattern(embs, cat_dim=0))

    def test_detect_reduce_pattern_keepdim_skipped(self):
        gm, *_ = _build_batch_embedding(n_emb=2, keepdim=True)
        embs = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        ]
        self.assertIsNone(_detect_reduce_pattern(embs, cat_dim=0))

    def test_detect_indices_parent_same_parent(self):
        gm, _, parent, _ = _build_batch_embedding(n_emb=2)
        embs = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        ]
        p, dim = _detect_indices_parent(embs)
        self.assertIs(p, parent)
        self.assertEqual(dim, 0)

    def test_detect_indices_parent_different_parents(self):
        gm, *_ = _build_batch_embedding(n_emb=2, indices_from_single_parent=False)
        embs = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _EMBED
        ]
        p, dim = _detect_indices_parent(embs)
        self.assertIsNone(p)

    # ===== full pass =====
    def test_two_embeddings_fused(self):
        gm, weight, parent, reduces = _build_batch_embedding(n_emb=2, V=10, D=4, L=3)
        n_embed_before = count_target(gm.graph, _EMBED)
        batch_embedding_fusion_pass(gm.graph)
        n_embed_after = count_target(gm.graph, _EMBED)
        self.assertLess(n_embed_after, n_embed_before)

    def test_single_embedding_unchanged(self):
        gm, *_ = _build_batch_embedding(n_emb=1)
        orig = str(gm.graph)
        batch_embedding_fusion_pass(gm.graph)
        self.assertEqual(orig, str(gm.graph))

    def test_non_default_args_skipped(self):
        gm, *_ = _build_batch_embedding(n_emb=2, padding_idx=0)
        n_before = count_target(gm.graph, _EMBED)
        batch_embedding_fusion_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _EMBED), n_before)

    def test_mixed_reduce_skipped(self):
        gm, *_ = _build_batch_embedding(n_emb=2, use_mixed_reduce=True)
        n_before = count_target(gm.graph, _EMBED)
        batch_embedding_fusion_pass(gm.graph)
        self.assertEqual(count_target(gm.graph, _EMBED), n_before)


if __name__ == "__main__":
    run_tests()
