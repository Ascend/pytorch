import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import pad_slice_fold


class FoldPadSliceModel(torch.nn.Module):
    def forward(self, t1):
        inputPad = torch._C._nn.pad(t1, [0, 0, 0, 50], "constant", 0.0)
        inputSlice = inputPad[:, :50]
        output = torch.relu(inputSlice)
        return output


class TestFoldPadSlicePass(TestUtils):
    def op_calc(self, t1):
        inputPad = torch._C._nn.pad(t1, [0, 0, 0, 50], "constant", 0.0)
        inputSlice = inputPad[:, :50]
        output = torch.relu(inputSlice)
        return output


    @parametrize('shape', [(128, 50, 128)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(128, 50, 128)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldPadSliceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        # 应用优化 Pass
        pad_slice_fold(graph_module.graph)
        graph_module.recompile()
        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(64, 32)])
    @parametrize('dtype', ['float16', 'float32'])
    def test_pad_no_slice_user_should_keep_pad(self, shape, dtype):
        """pad 后没有 slice, 直接使用 → 不应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                p = torch._C._nn.pad(x, [0, 0, 0, 20], "constant", 0.0)
                return torch.sigmoid(p)

        t = self._generate_tensor(shape, dtype)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        original_nodes = len(gm.graph.nodes)
        pad_slice_fold(gm.graph)
        gm.recompile()

        self.assertTrue(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))
        self.assertEqual(m(t), gm(t))


    @parametrize('shape', [(4, 20, 128)])
    def test_slice_end_exceed_original_should_not_fold(self, shape):
        """slice 的 end > 原始长度 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                p = torch._C._nn.pad(x, [0, 0, 0, 0, 0, 30], "constant", 0.0)  # pad dim=2
                s = p[:40, :, :]   # 40 > 20+30? 但原始是20 → 应保留 pad
                return s + 1.0

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        pad_slice_fold(gm.graph)
        self.assertTrue(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))


    @parametrize('shape', [(8, 100)])
    def test_slice_with_step_not_one_should_not_fold(self, shape):
        """step ≠ 1 或 None → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                p = torch._C._nn.pad(x, [0, 40], "constant", 0.0)
                s1 = p[:, ::2]          # step=2
                s2 = p[:, 10:60:3]      # step=3
                return s1.mean() + s2.mean()

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        pad_slice_fold(gm.graph)
        self.assertTrue(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))


    @parametrize('shape', [(16, 64, 32)])
    def test_multiple_slices_some_invalid_some_valid(self, shape):
        """有部分 slice 合法，有部分不合法 → 整体不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                p = torch._C._nn.pad(x, [0, 0, 0, 20, 0, 0], "constant", 0.0)
                valid = p[:, :64, :]
                invalid = p[:, :, ::2]
                return valid.sum() - invalid.sum()

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        pad_slice_fold(gm.graph)
        self.assertTrue(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))


    @parametrize('shape', [(1, 128)])
    def test_zero_pad_should_not_fold(self, shape):
        """pad 全为 0 → pad_dim=None 或无实际填充 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                p = torch._C._nn.pad(x, [0, 0, 0, 0], "constant", 0.0)
                return p[:, :100]

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        pad_slice_fold(gm.graph)
        self.assertTrue(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))


    @parametrize('shape', [(4, 16, 64, 128)])
    def test_pad_on_non_last_dim_and_valid_slice(self, shape):
        """在中间维度 pad + 合法切片 → 应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                p = torch._C._nn.pad(x, [0, 0, 0, 0, 0, 40, 0, 0], "constant", 0.0)  # pad dim=1
                s = p[:, :16, :, :]
                return torch.nn.functional.gelu(s)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        pad_slice_fold(gm.graph)
        gm.recompile()

        self.assertFalse(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))
        self.assertEqual(m(t), gm(t))


    def test_no_pad_node_in_graph(self):
        """图中根本没有 pad 节点 → 安全通过"""

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x * 2 + 1, dim=-1)

        t = torch.randn(8, 32)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(t)

        pad_slice_fold(gm.graph)  # 应该不崩溃
        self.assertFalse(any(n.target == torch._C._nn.pad for n in gm.graph.nodes))


instantiate_parametrized_tests(TestFoldPadSlicePass)


if __name__ == "__main__":
    run_tests()