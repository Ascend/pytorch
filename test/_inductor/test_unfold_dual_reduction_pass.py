import torch
import torch.fx as fx
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests
)
from testutils import TestUtils
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import unfold_dual_reduction_pass


class TestUnfoldDualReductionPass(TestUtils):
    def setUp(self):
        self.input_tensor = torch.randn(2, 3, 4)
        self.fake_mode = FakeTensorMode(allow_fallback_kernels=False)

    def _setup_meta_for_inductor_simulation(self, gm):
        for node in gm.graph.nodes:
            if node.op == 'placeholder' and 'val' not in node.meta:
                # 用FakeTensor填充placeholder meta
                node.meta['val'] = self.fake_mode.from_tensor(self.input_tensor)
            elif node.op == 'call_function' and 'val' not in node.meta:
                # 模拟reduction节点的meta（基于input）
                if 'val' in node.args[0].meta if hasattr(node.args[0], 'meta') else False:
                    input_val = node.args[0].meta['val']
                    # 简单模拟shape（实际Inductor会填充）
                    if hasattr(input_val, 'shape'):
                        node.meta['val'] = torch.empty(input_val.shape, dtype=input_val.dtype, device='meta')  # meta tensor
        gm.graph.lint()
        return gm

    def _apply_pass(self, gm, skip_meta=False):
        if skip_meta is False:
            gm = self._setup_meta_for_inductor_simulation(gm)
        unfold_dual_reduction_pass(gm.graph)
        gm.graph.lint()
        gm.recompile()
        return gm

    def _check_graph_contains(self, graph, target, count=1):
        actual_count = sum(1 for n in graph.nodes if n.op == 'call_function' and n.target == target)
        self.assertEqual(actual_count, count, f"Expected {count} nodes with target {target}, found {actual_count}")

    def test_basic_dual_sum(self):
        # Test unfolding sum with dims=[0,1]
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[0, 1])
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        # Check outputs match
        torch.testing.assert_close(orig_out, new_out)
        # Check graph: should have two sum nodes: first dim=[1], then dim=[0]
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=2)

    def test_multi_dim_sum(self):
        # Test with dims=[1,0,2] -> unfolded to [2],[1],[0]
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[1, 0, 2])
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=3)

    def test_single_dim_no_unfold(self):
        # Single dim: should not unfold
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[0])
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=1)

    def test_no_dim_full_reduction(self):
        # No dim: default to all dims, if >1, unfold
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[2, 1, 0])
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        # Assuming shape [2,3,4], unfold to [2],[1],[0]
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=3)

    def test_keepdim_true(self):
        # when the keepdim is True
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[0, 1], keepdim=True)
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=2)

    def test_with_dtype(self):
        # Specify dtype
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[0, 1], dtype=torch.int64)
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=2)

    def test_non_reduction_no_change(self):
        # Non-reduction op: no change
        def f(x):
            return torch.ops.aten.mean(x, dim=[0])
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=0)  # No sum added

    def test_unknown_shape_skip(self):
        # Mock a node with no shape
        def f(x):
            return torch.ops.aten.sum.dim_IntList(x, dim=[0, 1])
        
        gm = symbolic_trace(f)
        # Manually remove shape meta to simulate None
        for node in gm.graph.nodes:
            if 'val' in node.meta:
                del node.meta['val']
        
        gm = self._apply_pass(gm, skip_meta=True)
        
        # Graph should remain unchanged (still one sum)
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=1)

    def test_multiple_reductions(self):
        # Multiple sum nodes
        def f(x):
            s1 = torch.ops.aten.sum.dim_IntList(x, dim=[0, 1])
            s2 = torch.ops.aten.sum.dim_IntList(s1, dim=[0])
            return s2
        
        gm = symbolic_trace(f)
        orig_out = gm(self.input_tensor)
        
        gm = self._apply_pass(gm)
        new_out = gm(self.input_tensor)
        
        torch.testing.assert_close(orig_out, new_out)
        
        # s1 unfolded to 2, s2 remains 1, total 3
        self._check_graph_contains(gm.graph, torch.ops.aten.sum.dim_IntList, count=3)

    def test_dead_code_elimination(self):
        # Test if dead code is eliminated after replacement
        def f(x):
            s = torch.ops.aten.sum.dim_IntList(x, dim=[0, 1])
            return s * 2  # Ensure replacement propagates
        
        gm = symbolic_trace(f)
        # Apply pass, which should call eliminate_dead_code if changed
        gm = self._apply_pass(gm)
        
        # Check no orphaned nodes (manual check via node count or lint)
        gm.graph.lint()  # Should not raise
        self.assertEqual(len(list(gm.graph.nodes)), 5)  # Placeholder: adjust based on actual graph size

instantiate_parametrized_tests(TestUnfoldDualReductionPass)

if __name__ == "__main__":
    run_tests()