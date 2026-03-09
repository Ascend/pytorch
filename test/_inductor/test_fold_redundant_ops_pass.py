import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_redundant_ops


class FoldMultiShapeUnchangeModel(torch.nn.Module):
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):
        embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1)
        view = torch.ops.aten.view.default(embedding, [-1, 1, 64])
        squeeze = torch.ops.aten.squeeze.dim(view, 1)
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, arg3_1)
        view_1 = torch.ops.aten.view.default(embedding_1, [-1, 1, 32])
        squeeze_1 = torch.ops.aten.squeeze.dim(view_1, 1)
        embedding_2 = torch.ops.aten.embedding.default(arg4_1, arg5_1)
        view_2 = torch.ops.aten.view.default(embedding_2, [-1, 1, 16])
        squeeze_2 = torch.ops.aten.squeeze.dim(view_2, 1)
        permute_1 = torch.ops.aten.permute.default(arg6_1, [1, 0])
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0])
        relu = torch.ops.aten.relu.default(arg7_1)
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, relu, permute_2)
        relu_1 = torch.ops.aten.relu.default(addmm_1)
        return {"squeeze": squeeze, "squeeze_1": squeeze_1, "squeeze_2": squeeze_2, "permute_2": permute_2, "relu_1": relu_1}


class TestFoldMultiShapeUnchangePass(TestUtils):
    def op_calc(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):
        embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1)
        view = torch.ops.aten.view.default(embedding, [-1, 1, 64])
        squeeze = torch.ops.aten.squeeze.dim(view, 1)
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, arg3_1)
        view_1 = torch.ops.aten.view.default(embedding_1, [-1, 1, 32])
        squeeze_1 = torch.ops.aten.squeeze.dim(view_1, 1)
        embedding_2 = torch.ops.aten.embedding.default(arg4_1, arg5_1)
        view_2 = torch.ops.aten.view.default(embedding_2, [-1, 1, 16])
        squeeze_2 = torch.ops.aten.squeeze.dim(view_2, 1)
        permute_1 = torch.ops.aten.permute.default(arg6_1, [1, 0])
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0])
        relu = torch.ops.aten.relu.default(arg7_1)
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, relu, permute_2)
        relu_1 = torch.ops.aten.relu.default(addmm_1)
        return {"squeeze": squeeze, "squeeze_1": squeeze_1, "squeeze_2": squeeze_2, "permute_2": permute_2, "relu_1": relu_1}


    def test_compile_cases(self):
        arg0_1 = torch.randn(289094, 64, dtype=torch.float32)
        arg1_1 = torch.randint(0, 289094, (128,), dtype=torch.int64)
        arg2_1 = torch.randn(98, 32, dtype=torch.float32)
        arg3_1 = torch.randint(0, 98, (128,), dtype=torch.int64)
        arg4_1 = torch.randn(14, 16, dtype=torch.float32)
        arg5_1 = torch.randint(0, 14, (128,), dtype=torch.int64)
        arg6_1 = torch.randn(6144, 6144, dtype=torch.float32)
        arg7_1 = torch.randn(128, 6144, dtype=torch.float32)
        arg8_1 = torch.randn(6144, dtype=torch.float32)
        std_result = self.op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        arg0_1 = torch.randn(289094, 64, dtype=torch.float32)
        arg1_1 = torch.randint(0, 289094, (128,), dtype=torch.int64)
        arg2_1 = torch.randn(98, 32, dtype=torch.float32)
        arg3_1 = torch.randint(0, 98, (128,), dtype=torch.int64)
        arg4_1 = torch.randn(14, 16, dtype=torch.float32)
        arg5_1 = torch.randint(0, 14, (128,), dtype=torch.int64)
        arg6_1 = torch.randn(6144, 6144, dtype=torch.float32)
        arg7_1 = torch.randn(128, 6144, dtype=torch.float32)
        arg8_1 = torch.randn(6144, dtype=torch.float32)
        model = FoldMultiShapeUnchangeModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        
        # 应用优化 Pass
        fold_redundant_ops(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)
        inductor_result = graph_module(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32'])
    def test_view_squeeze_dim_fold(self, shape, dtype):
        """view → squeeze(dim=...) → 应折叠回原始输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, [4, 1, 8])
                s = torch.ops.aten.squeeze.dim(v, 1)
                return s + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        model(x)
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_redundant_ops(gm.graph)
        gm.recompile()

        view_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.view.default]
        squeeze_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.squeeze.dim]
        self.assertEqual(len(view_nodes) + len(squeeze_nodes), 0,
                         "view → squeeze(dim) 等价模式应全部折叠")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(2, 3, 5)])
    @parametrize('dtype', ['float32'])
    def test_reshape_squeeze_default_fold(self, shape, dtype):
        """reshape → squeeze.default → 应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                r = torch.ops.aten.reshape.default(x, [2, 3, 1, 5])
                s = torch.ops.aten.squeeze.dim(r, 2)
                return s.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_redundant_ops(gm.graph)
        gm.recompile()

        reshape_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.reshape.default]
        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(reshape_nodes) + len(squeeze_nodes), 0)


    @parametrize('shape', [(4, 5)])
    @parametrize('dtype', ['float32'])
    def test_squeeze_not_after_view_no_fold(self, shape, dtype):
        """squeeze 前面不是 view/reshape → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.squeeze.default(x)
                return s.sum()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_redundant_ops(gm.graph)

        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(squeeze_nodes), 1, "非 view/reshape 前置不应折叠")


    @parametrize('shape', [(2, 3, 1)])
    @parametrize('dtype', ['float32'])
    def test_view_squeeze_shape_mismatch_no_fold(self, shape, dtype):
        """squeeze 后的 shape 与原始不一致 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, [2, 3, 1])
                s = torch.ops.aten.squeeze.dim(v, 2)  # squeeze 后 shape (2,3)，原始是 (2,3,1)
                return s + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_redundant_ops(gm.graph)

        view_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.view.default]
        squeeze_nodes = [n for n in gm.graph.nodes if 'squeeze' in str(n.target)]
        self.assertEqual(len(view_nodes) + len(squeeze_nodes), 2, "shape 不一致不应折叠")


instantiate_parametrized_tests(TestFoldMultiShapeUnchangePass)


if __name__ == "__main__":
    run_tests()