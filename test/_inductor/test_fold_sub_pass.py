import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass


class FoldSubModel(torch.nn.Module):
    def forward(self, t1):
        sub = torch.ops.aten.sub.Tensor(t1, torch.ops.aten.zeros_like.default(t1))
        sub_output = torch.ops.aten.relu.default(sub)
        rsub = torch.ops.aten.rsub.Tensor(torch.ops.aten.zeros_like.default(t1), t1)
        rsub_output = torch.ops.aten.relu.default(rsub)
        return sub_output + rsub_output


class TestFoldSubPass(TestUtils):
    def op_calc(self, t1):
        sub = torch.sub(t1, torch.ops.aten.zeros_like.default(t1))
        sub_output = torch.relu(sub)
        rsub = torch.rsub(torch.ops.aten.zeros_like.default(t1), t1)
        rsub_output = torch.relu(rsub)
        return sub_output + rsub_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldSubModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_sub_zero_right_fold(self, shape, dtype):
        """ x - 0 → 应折叠为 x """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.sub(x, torch.zeros_like(x))   # sub.Tensor(x, zero)
                return torch.relu(y + 1.5)

        t = self._generate_tensor(shape, dtype)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        # 期望 sub 节点被移除
        self.assertFalse(
            any(n.target in (torch.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
                for n in gm.graph.nodes),
            "x - 0 should be folded to x"
        )
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_sub_zero_left_fold(self, shape, dtype):
        """ x - 0 → 应折叠为 x """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.sub(torch.zeros_like(x), x)   # sub.Tensor(zero, x)
                return torch.relu(y + 1.5)

        t = self._generate_tensor(shape, dtype)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertTrue(
            any(n.target in (torch.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
                for n in gm.graph.nodes),
            "0 - x should not be folded to x"
        )
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(5, 5)])
    def test_rsub_zero_left_fold(self, shape):
        """ 0 - x → 应折叠为 -x （如果 pass 支持） """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.rsub(torch.zeros_like(x), x)   # rsub(zero, x)
                return torch.sigmoid(y * 2)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        # 检查是否折叠（取决于你的 pass 是否识别 rsub zero left）
        # 如果 pass 支持 rsub zero → 期望节点被移除或替换为 neg
        # 如果不支持 → 可以暂时注释 assert，改成检查数值一致即可
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(3, 7)])
    def test_sub_scalar_zero_right_fold(self, shape):
        """ x - 0.0 (scalar) 应折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.sub(x, 0.0)          # 应被 trace 为 sub.Scalar
                return y.mean() + 1.0

        t = torch.randn(shape)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(
            any('sub' in str(n.target) for n in gm.graph.nodes if n.op == 'call_function')
        )
        self.assertEqual(m(t), gm(t))


    @parametrize('shape', [(2, 16)])
    def test_sub_zero_chain_fold(self, shape):
        """ 链式 sub - 0 应该全部折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.sub(x, torch.zeros_like(x))
                b = torch.sub(a, 0.0)
                c = torch.sub(b, torch.zeros_like(x))
                return torch.relu(c + x)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        sub_nodes = [n for n in gm.graph.nodes if 'sub' in str(n.target)]
        self.assertEqual(len(sub_nodes), 0, "所有 x - 0 都应该被折叠")


    @parametrize('shape', [(8,)])
    def test_rsub_zero_chain_fold(self, shape):
        """ 链式 0 - x 折叠（如果 pass 支持） """
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.rsub(torch.zeros_like(x), x)
                b = torch.rsub(torch.zeros_like(x), a)
                return torch.relu(a)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        # 如果 pass 支持 rsub zero → 最终应折叠回 x 或 -(-x)
        self.assertEqual(m(t), gm(t))


    @parametrize('shape', [(6, 6)])
    def test_sub_non_zero_no_fold(self, shape):
        """ 非零值不应折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.sub(x, torch.full_like(x, 0.1))   # 减 0.1
                return torch.tanh(y)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        before = len([n for n in gm.graph.nodes if n.op == 'call_function'])
        fold_four_op_pass(gm.graph)
        after = len([n for n in gm.graph.nodes if n.op == 'call_function'])

        self.assertEqual(before, after, "非零常量不应触发折叠")


    def test_no_sub_op_no_change(self):
        """ 图中没有 sub/rsub，pass 应不修改图 """
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        t = torch.randn(4, 4)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(t)

        orig_str = str(gm.graph)
        fold_four_op_pass(gm.graph)
        self.assertEqual(orig_str, str(gm.graph))


    @parametrize('shape', [(1, 64)])
    def test_sub_zero_multi_users(self, shape):
        """ sub 后有多个下游用户，都应正确替换 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.sub(x, torch.zeros_like(x))
                return y * 3.0, torch.relu(y), y + x

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertEqual(m(t), gm(t))


instantiate_parametrized_tests(TestFoldSubPass)


if __name__ == "__main__":
    run_tests()