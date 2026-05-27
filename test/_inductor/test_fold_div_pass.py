import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass


class FoldDivModel(torch.nn.Module):
    def forward(self, t1, t2, t3):
        div_1 = torch.ops.aten.div(t1, 1)
        div_output = torch.relu(div_1)
        return div_output


class TestFoldDivPass(TestUtils):
    def op_calc(self, t1):
        div_1 = torch.ops.aten.div(t1, 1)
        div_output = torch.relu(div_1)
        return div_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(2, 4, 8)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        t2 = self._generate_tensor(shape, dtype)
        t3 = self._generate_tensor(shape, dtype)
        model = FoldDivModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2, t3)

        # 应用优化 Pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2, t3)
        inductor_result = graph_module(t1, t2, t3)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 16)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_div_by_one_right_fold(self, shape, dtype):
        """ x / 1.0 → should be x """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.div(x, 1.0)
                return torch.relu(y * 2.0)

        t = self._generate_tensor(shape, dtype)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(
            any(n.target in (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar)
                for n in gm.graph.nodes),
            "x / 1.0 should be folded to x"
        )
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(3, 8)])
    def test_div_by_one_int_fold(self, shape):
        """ x / 1 (int) should be folded """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.div(x, 1)
                return y + x.mean()

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(any('div' in str(n.target) for n in gm.graph.nodes if n.op == 'call_function'))
        self.assertEqual(m(t), gm(t), "x / 1 (int) should be folded")


    @parametrize('shape', [(5, 5)])
    def test_rdiv_one_left_fold(self, shape):
        """ 1.0 / x should be folded """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.div(1.0, x)           # div.Scalar(1.0, x) 或 rdiv
                return torch.sigmoid(y)

        t = self._generate_tensor(shape, 'float32') + 0.1  # 避免除零
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(2, 32)])
    def test_div_one_chain_fold(self, shape):
        """ all / 1.0 should be folded """
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.div(x, 1.0)
                b = torch.div(a, 1)
                c = torch.div(b, torch.ones_like(x))
                return torch.relu(c + x)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        div_nodes = [n for n in gm.graph.nodes if 'div' in str(n.target)]
        self.assertEqual(len(div_nodes), 0, "all / 1.0 should be folded")


    @parametrize('shape', [(6, 6)])
    def test_div_non_one_no_fold(self, shape):
        """ should not fold since div not 1 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.div(x, 2.0)
                return torch.tanh(y)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        before = len([n for n in gm.graph.nodes if n.op == 'call_function'])
        fold_four_op_pass(gm.graph)
        after = len([n for n in gm.graph.nodes if n.op == 'call_function'])

        self.assertEqual(before, after, "should not fold since div not 1")


    @parametrize('shape', [(1, 128)])
    def test_div_one_multi_users(self, shape):
        """ there are multiple usrs after div /1  all of them should be replaced """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.div(x, 1.0)
                return y * 3.0, torch.relu(y), y + x

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertEqual(m(t), gm(t), "there are multiple usrs after div /1  all of them should be replaced")


    def test_no_div_op_no_change(self):
        """ no div node, pass should not modify graph """
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, x) + torch.add(x, 1.0)

        t = torch.randn(4, 4)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(t)

        orig_str = str(gm.graph)
        fold_four_op_pass(gm.graph)
        self.assertEqual(orig_str, str(gm.graph), "no div node, pass should not modify graph")


instantiate_parametrized_tests(TestFoldDivPass)


if __name__ == "__main__":
    run_tests()