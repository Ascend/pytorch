import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_squeeze


class FoldSqueezeModel(torch.nn.Module):
    def forward(self, t1, t2,):
        squeeze_1 = torch.ops.aten.squeeze.default(t1)
        squeeze_2 = torch.ops.aten.squeeze.default(squeeze_1)
        
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(t2, 1)
        squeeze_3 = torch.ops.aten.squeeze.dim(unsqueeze_1, 1)

        return {"squeeze_2": squeeze_2, "squeeze_3": squeeze_3}


class TestFoldSqueezePass(TestUtils):
    def op_calc(self, t1, t2):
        squeeze_1 = torch.squeeze(t1)
        squeeze_2 = torch.squeeze(squeeze_1)
        unsqueeze_1 = torch.unsqueeze(t2, 1)
        squeeze_3 = torch.squeeze(unsqueeze_1, 1)
        return {"squeeze_2": squeeze_2, "squeeze_3": squeeze_3}


    def test_compile_cases(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 1, 1, 4)
        std_result = self.op_calc(t1, t2)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 1, 1, 4)
        model = FoldSqueezeModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2)
        
        # 应用优化 Pass
        fold_squeeze(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2)
        inductor_result = graph_module(t1, t2)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 1, 1, 5)])
    @parametrize('dtype', ['float32'])
    def test_consecutive_squeeze_full_fold(self, shape, dtype):
        """连续 squeeze（无参数）→ 应全部合并"""
        class M(torch.nn.Module):
            def forward(self, x):
                s1 = torch.ops.aten.squeeze.default(x)       # 去掉所有 dim=1
                s2 = torch.ops.aten.squeeze.default(s1)
                return s2 + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_squeeze(gm.graph)
        gm.recompile()

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(squeeze_nodes), 1, "连续 squeeze 应全部合并")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(2, 1, 3, 1)])
    @parametrize('dtype', ['float32'])
    def test_squeeze_then_unsqueeze_match_fold(self, shape, dtype):
        """squeeze → unsqueeze（维度匹配）→ 应折叠回原始输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                u = torch.ops.aten.unsqueeze.default(x, 1)
                s = torch.ops.aten.squeeze.dim(u, 1)
                return s.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        model(x)
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_squeeze(gm.graph)
        gm.recompile()

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        unsqueeze_nodes = [n for n in gm.graph.nodes if 'unsqueeze' in str(n.target)]
        self.assertEqual(len(squeeze_nodes) + len(unsqueeze_nodes), 0, "匹配的 squeeze → unsqueeze 应消除")


    @parametrize('shape', [(2, 1, 3, 1)])
    @parametrize('dtype', ['float32'])
    def test_squeeze_then_unsqueeze_dim_mismatch_no_fold(self, shape, dtype):
        """squeeze → unsqueeze（维度不匹配）→ 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                u = torch.ops.aten.unsqueeze.default(x, 1)
                s = torch.ops.aten.squeeze.dim(u, 2)
                return u.sum()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_squeeze(gm.graph)

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(squeeze_nodes), 2, "维度不匹配不应折叠")


    @parametrize('shape', [(4, 1, 6)])
    @parametrize('dtype', ['float32'])
    def test_squeeze_multi_users_no_fold(self, shape, dtype):
        """squeeze 有多个下游用户 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.squeeze.default(x)
                a = torch.ops.aten.relu.default(s)
                b = s + 1.0
                return a, b

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_squeeze(gm.graph)

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(squeeze_nodes), 1, "多用户时不应折叠 squeeze")


    @parametrize('shape', [(1, 1, 1, 8)])
    def test_consecutive_squeeze_and_unsqueeze_chain(self, shape):
        """多层 squeeze + unsqueeze 组合 → 应尽可能折叠"""
        class M(torch.nn.Module):   
            def forward(self, x):
                u1 = torch.ops.aten.unsqueeze.default(x, 0)
                s1 = torch.ops.aten.squeeze.dim(u1, 0)
                u2 = torch.ops.aten.unsqueeze.default(s1, 1)
                s2 = torch.ops.aten.squeeze.dim(u2, 1)
                return s2.mean()

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_squeeze(gm.graph)
        gm.recompile()

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        unsqueeze_nodes = [n for n in gm.graph.nodes if 'unsqueeze' in str(n.target)]
        # 预期：部分或全部被消除，视 match 函数实现
        self.assertLess(len(squeeze_nodes) + len(unsqueeze_nodes), 4)


    @parametrize('shape', [(4, 5)])
    def test_no_squeeze_unsqueeze_no_change(self, shape):
        """无 squeeze/unsqueeze 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_squeeze(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(2, 1, 3)])
    def test_squeeze_without_shape_info_no_fold(self, shape):
        """输入缺少 shape 信息 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.squeeze.default(x)
                return s.sum()

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 shape 信息
        for node in gm.graph.nodes:
            if "val" in node.meta:
                del node.meta["val"]

        fold_squeeze(gm.graph)

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(squeeze_nodes), 1, "无 shape 信息不应折叠")


instantiate_parametrized_tests(TestFoldSqueezePass)


if __name__ == "__main__":
    run_tests()