import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass


class FoldMulModel(torch.nn.Module):
    def forward(self, t1):
        add = torch.ops.aten.mul.Tensor(t1, 1)
        add_output = torch.ops.aten.relu.default(add)
        return add_output


class TestFoldMulPass(TestUtils):
    def op_calc(self, t1):
        mul_1 = torch.ops.aten.mul.Tensor(t1, 1)
        mul_output = torch.relu(mul_1)
        return mul_output



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
        model = FoldMulModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 16)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_mul_by_one_right_fold(self, shape, dtype):
        """ x * 1.0 → 应折叠为 x """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.mul(x, 1.0)           # mul.Scalar
                return torch.relu(y + 0.5)

        t = self._generate_tensor(shape, dtype)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(
            any(n.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
                for n in gm.graph.nodes),
            "x * 1.0 should be folded to x"
        )
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(3, 8)])
    def test_mul_by_one_int_fold(self, shape):
        """ x * 1 (int literal) 应折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.mul(x, 1)
                return y.mean()

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(any('mul' in str(n.target) for n in gm.graph.nodes if n.op == 'call_function'))
        self.assertEqual(m(t), gm(t))


    @parametrize('shape', [(5, 5)])
    def test_mul_one_left_fold(self, shape):
        """ 1.0 * x 应折叠为 x """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.mul(1.0, x)           # mul.Scalar(1.0, x)
                return torch.sigmoid(y)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(any('mul' in str(n.target) for n in gm.graph.nodes if n.op == 'call_function'))
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(2, 32)])
    def test_mul_one_chain_fold(self, shape):
        """ 链式 * 1.0 / * 1 应该全部折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.mul(x, 1.0)
                b = torch.mul(a, 1)
                c = torch.mul(b, torch.ones_like(x))
                return torch.relu(c)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        mul_nodes = [n for n in gm.graph.nodes if 'mul' in str(n.target)]
        self.assertEqual(len(mul_nodes), 0, "所有 *1 都应该被折叠")


    @parametrize('shape', [(6, 6)])
    def test_mul_non_one_no_fold(self, shape):
        """ 乘以非 1 值不应折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.mul(x, 2.0)
                return torch.tanh(y + 1.0)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        before = len([n for n in gm.graph.nodes if n.op == 'call_function'])
        fold_four_op_pass(gm.graph)
        after = len([n for n in gm.graph.nodes if n.op == 'call_function'])

        self.assertEqual(before, after, "非 1 常量不应触发折叠")


    @parametrize('shape', [(1, 64)])
    def test_mul_one_multi_users(self, shape):
        """ mul *1 后有多个下游用户，都应正确替换 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.mul(x, 1.0)
                return y * 4.0, torch.relu(y), y + x * 0.5

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertEqual(m(t), gm(t))


    def test_no_mul_op_no_change(self):
        """ 图中没有 mul，pass 应不修改图 """
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) / 2.0 + torch.sub(x, 1.0)

        t = torch.randn(4, 4)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(t)

        orig_str = str(gm.graph)
        fold_four_op_pass(gm.graph)
        self.assertEqual(orig_str, str(gm.graph), "无 mul 节点不应修改图")


instantiate_parametrized_tests(TestFoldMulPass)


if __name__ == "__main__":
    run_tests()