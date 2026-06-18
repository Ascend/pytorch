# Owner(s): ["module: inductor"]
import importlib
from itertools import count
import re
from unittest import skip, skipIf

import torch
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.virtualized import V
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)
from torch.utils._ordered_set import OrderedSet


try:
    importlib.import_module("mfusion")

    HAS_AKG_MFUSION = True
except ModuleNotFoundError:
    HAS_AKG_MFUSION = False

try:
    from torch_npu._inductor.mfusion.graph_fusion import (
        _emit_mfusion_dvm_codegen,
        _find_output_node,
        _restore_output_contract,
        _snapshot_output_contract,
        MFusionPatch,
    )
    from torch_npu._inductor.mfusion.subgraph_registry import Payload
except ImportError:
    _emit_mfusion_dvm_codegen = None
    _find_output_node = None
    _restore_output_contract = None
    _snapshot_output_contract = None
    MFusionPatch = None
    Payload = None

# Gate omits AKG mfusion; keep module importable even if torch_npu mfusion tree is absent.
HAS_MFUSION_STACK = HAS_AKG_MFUSION and MFusionPatch is not None


class _FakeWrapper:
    def __init__(self):
        self.header = IndentedBuffer()
        self.body = IndentedBuffer()
        self.src_to_kernel = {}
        self.imports = set()
        self._suffixes = count()

    def next_kernel_suffix(self):
        return str(next(self._suffixes))

    def writeline(self, line):
        self.body.writeline(line)

    def add_import_once(self, line):
        self.imports.add(line)


class _FakeGraph:
    def try_get_buffer(self, name):
        return None


class _FakeFallbackKernel:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


def _make_add_mul_subgraph():
    graph = Graph()
    meta = torch.empty_strided((2, 4), (4, 1), dtype=torch.float16)
    a = graph.placeholder("arg0")
    a.meta["val"] = meta
    b = graph.placeholder("arg1")
    b.meta["val"] = meta
    add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    add.meta["val"] = meta
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (add, a))
    mul.meta["val"] = meta
    graph.output(mul)
    return GraphModule(torch.nn.Module(), graph)


@skipIf(
    _find_output_node is None
    or _restore_output_contract is None
    or _snapshot_output_contract is None,
    "torch_npu._inductor.mfusion.graph_fusion not available",
)
class TestMfusionOutputContract(TestCase):
    def test_restore_output_contract_preserves_single_output(self):
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        graph.output(add)
        gm = GraphModule(torch.nn.Module(), graph)

        snapshot = _snapshot_output_contract(_find_output_node(gm.graph))

        roundtrip_graph = Graph()
        rt_a = roundtrip_graph.placeholder("a")
        rt_b = roundtrip_graph.placeholder("b")
        rt_add = roundtrip_graph.call_function(torch.ops.aten.add.Tensor, (rt_a, rt_b))
        roundtrip_graph.output(rt_add)
        rt_gm = GraphModule(torch.nn.Module(), roundtrip_graph)

        _restore_output_contract(_find_output_node(rt_gm.graph), snapshot)

        restored_output = _find_output_node(rt_gm.graph)
        self.assertIsNotNone(restored_output)
        self.assertIs(restored_output.args[0], rt_add)

    def test_restore_output_contract_preserves_list_output(self):
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        graph.output([add, b])
        gm = GraphModule(torch.nn.Module(), graph)

        snapshot = _snapshot_output_contract(_find_output_node(gm.graph))

        roundtrip_graph = Graph()
        rt_a = roundtrip_graph.placeholder("a")
        rt_b = roundtrip_graph.placeholder("b")
        rt_add = roundtrip_graph.call_function(torch.ops.aten.add.Tensor, (rt_a, rt_b))
        roundtrip_graph.output((rt_add, rt_b))
        rt_gm = GraphModule(torch.nn.Module(), roundtrip_graph)

        _restore_output_contract(_find_output_node(rt_gm.graph), snapshot)

        restored_output = _find_output_node(rt_gm.graph)
        self.assertIsNotNone(restored_output)
        restored_args = restored_output.args[0]
        self.assertIsInstance(restored_args, list)
        self.assertEqual(len(restored_args), 2)
        self.assertIs(restored_args[0], rt_add)
        self.assertIs(restored_args[1], rt_b)

    def test_restore_output_contract_preserves_none_slots_and_stride_meta(self):
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        output = graph.output((None, add, b))
        output.meta["user_visible_output_idxs"] = [1, 2]
        output.meta["original_output_strides"] = [None, (1, 50257), (1280, 1)]
        gm = GraphModule(torch.nn.Module(), graph)

        snapshot = _snapshot_output_contract(_find_output_node(gm.graph))

        roundtrip_graph = Graph()
        rt_a = roundtrip_graph.placeholder("a")
        rt_b = roundtrip_graph.placeholder("b")
        rt_add = roundtrip_graph.call_function(torch.ops.aten.add.Tensor, (rt_a, rt_b))
        rt_output = roundtrip_graph.output((rt_add, rt_b))
        rt_gm = GraphModule(torch.nn.Module(), roundtrip_graph)

        _restore_output_contract(_find_output_node(rt_gm.graph), snapshot)

        restored_output = _find_output_node(rt_gm.graph)
        self.assertIsNotNone(restored_output)
        restored_args = restored_output.args[0]
        self.assertEqual(len(restored_args), 3)
        self.assertIsNone(restored_args[0])
        self.assertEqual(restored_output.meta["user_visible_output_idxs"], [1, 2])
        self.assertEqual(
            restored_output.meta["original_output_strides"],
            [None, (1, 50257), (1280, 1)],
        )


@skipIf(
    _emit_mfusion_dvm_codegen is None or Payload is None,
    "torch_npu._inductor.mfusion.graph_fusion not available",
)
class TestMfusionDvmCodegen(TestCase):
    def test_dvm_kernel_name_is_descriptive_and_deduped(self):
        wrapper = _FakeWrapper()
        payload0 = Payload(_make_add_mul_subgraph(), is_dynamic=False)
        payload1 = Payload(_make_add_mul_subgraph(), is_dynamic=False)

        with V.set_graph_handler(_FakeGraph()):
            _emit_mfusion_dvm_codegen(
                wrapper,
                _FakeFallbackKernel("buf0"),
                ("arg0", "arg1", "subgraph0"),
                payload0.fx_gm,
                payload0,
            )
            _emit_mfusion_dvm_codegen(
                wrapper,
                _FakeFallbackKernel("buf1"),
                ("arg2", "arg3", "subgraph1"),
                payload1.fx_gm,
                payload1,
            )

        header = wrapper.header.getvalue()
        body = wrapper.body.getvalue()
        definitions = re.findall(r"^def (mfusion_dvm_fused_add_mul_\d+)\(", header, re.M)

        self.assertEqual(len(definitions), 1)
        self.assertEqual(len(wrapper.src_to_kernel), 1)
        kernel_name = definitions[0]
        self.assertIn(f"buf0 = {kernel_name}(arg0, arg1)", body)
        self.assertIn(f"buf1 = {kernel_name}(arg2, arg3)", body)


@skipIf(
    not HAS_MFUSION_STACK,
    "AKG mfusion and torch_npu._inductor.mfusion.graph_fusion not available",
)
class TestMfusionByGraphFusion(TestCase):
    def _compile_and_check(self, model, inputs, *, atol=1e-3, rtol=1e-3):
        expect = model(*inputs)
        with MFusionPatch():
            mfusion_compiled_model = torch.compile(
                model, backend="inductor", fullgraph=True
            )
            result = mfusion_compiled_model(*inputs)
            self.assertEqual(expect, result, atol=atol, rtol=rtol)

    def test_basic_partitioning(self):
        # 不涉及维度broadcast，a, b, c shape完全一致
        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                add = a + b
                mul = add * c
                return mul

        shape = (2, 4)
        a = torch.normal(0, 0.1, size=shape, dtype=torch.float16).npu()
        b = torch.normal(0, 0.1, size=shape, dtype=torch.float16).npu()
        c = torch.normal(0, 0.1, size=shape, dtype=torch.float16).npu()
        model = Model()
        self._compile_and_check(model, (a, b, c), atol=1e-3, rtol=1e-3)

    def test_broadcast_elementwise(self):
        # 覆盖 broadcast + 标量常量 + 简单激活
        class Model(torch.nn.Module):
            def forward(self, a, b):
                # b: (1, 1, C) broadcast to a
                x = a + b
                x = torch.relu(x)
                return x * 0.5

        dtype = torch.float16
        a = torch.randn(64, 8, 128, device="npu", dtype=dtype)
        b = torch.randn(1, 1, 128, device="npu", dtype=dtype)
        self._compile_and_check(Model(), (a, b), atol=1e-3, rtol=1e-3)

    def test_reduction_and_broadcast(self):
        # 覆盖 reduce/keepdim + broadcast + 常量 eps
        class Model(torch.nn.Module):
            def forward(self, x):
                denom = x.sum(dim=-1, keepdim=True) + 1e-3
                y = x / denom
                return y.mean(dim=1)

        dtype = torch.float16
        x = torch.randn(32, 4, 256, device="npu", dtype=dtype)
        # reduce 会引入更大的误差空间，适当放宽
        self._compile_and_check(Model(), (x,), atol=2e-3, rtol=2e-3)

    def test_linear_gelu_with_params(self):
        # 覆盖 get_attr 参数（weight/bias）+ matmul/add + 激活
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 256, bias=True)

            def forward(self, x):
                y = self.linear(x)
                return torch.nn.functional.gelu(y)

        dtype = torch.float16
        torch.manual_seed(0)
        model = Model().eval().to("npu").to(dtype)
        x = torch.randn(64, 128, device="npu", dtype=dtype)
        self._compile_and_check(model, (x,), atol=2e-3, rtol=2e-3)

    @skip("npu inductor bug")
    def test_conv2d_relu_with_params(self):
        # 覆盖 conv2d（参数 get_attr）+ 激活；典型 CV 图结构
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 8, kernel_size=3, stride=1, padding=1, bias=True
                )

            def forward(self, x):
                y = self.conv(x)
                return torch.relu(y + 0.1)

        dtype = torch.float16
        torch.manual_seed(0)
        model = Model().eval().to("npu").to(dtype)
        x = torch.randn(4, 3, 32, 32, device="npu", dtype=dtype)
        self._compile_and_check(model, (x,), atol=3e-3, rtol=3e-3)

    @skip("npu inductor bug")
    def test_chunk_cat_multi_output_and_transpose(self):
        # 覆盖 list/tuple construct/unpack + cat + 多输出 + stride/contiguous 变化
        class Model(torch.nn.Module):
            def forward(self, x):
                # 让输入先变成非 contiguous，触发更多 layout 相关路径
                xt = x.transpose(0, 1)
                a, b = torch.chunk(xt, 2, dim=-1)
                y1 = torch.sin(a + 1.0)
                y2 = torch.cos(b * 2.0)
                y = torch.cat([y1, y2], dim=-1).contiguous()
                # 返回两个输出，覆盖 output tuple
                return y, y.mean(dim=-1)

        dtype = torch.float16
        x = torch.randn(8, 6, 64, device="npu", dtype=dtype)
        self._compile_and_check(Model(), (x,), atol=3e-3, rtol=3e-3)

    def test_mul(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a * b

        a = torch.randn(2, 2, dtype=torch.float16).npu()
        b = torch.randn(2, 2, dtype=torch.float16).npu()
        self._compile_and_check(Model(), (a, b), atol=1e-3, rtol=1e-3)

    def test_mul_mul_fusion(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = a * b
                return c * c

        a = torch.randn(2, 2, dtype=torch.float16).npu()
        b = torch.randn(2, 2, dtype=torch.float16).npu()
        self._compile_and_check(Model(), (a, b), atol=1e-3, rtol=1e-3)


class TestMfusionSymbolFastpath(TestCase):
    def test_is_mfusion_related_ir_detects_recursive_inputs(self):
        from torch_npu._inductor.mfusion.graph_fusion import _is_mfusion_related_ir

        class FakeOpOverload:
            def __init__(self, name):
                self._name = name

        class FakeNode:
            def __init__(self, *, op_name="", inputs=()):
                self.op_overload = FakeOpOverload(op_name) if op_name else None
                self.inputs = inputs

        mfusion_leaf = FakeNode(op_name="mfusion::subgraph_0")
        plain_leaf = FakeNode(op_name="aten::add")
        recursive_node = FakeNode(inputs=(plain_leaf, mfusion_leaf))

        self.assertTrue(_is_mfusion_related_ir(mfusion_leaf))
        self.assertTrue(_is_mfusion_related_ir(recursive_node))
        self.assertFalse(_is_mfusion_related_ir(plain_leaf))

    def test_layout_only_symbol_uses_prefers_output_layout(self):
        from torch_npu._inductor.mfusion.graph_fusion import _layout_only_symbol_uses

        class FakeLayout:
            def __init__(self, symbols=None, *, raise_not_implemented=False):
                self.symbols = OrderedSet(symbols or ())
                self.raise_not_implemented = raise_not_implemented

            def get_free_symbol_uses(self, unbacked_only=False):
                if self.raise_not_implemented:
                    raise NotImplementedError
                return OrderedSet(self.symbols)

        class FakeValue:
            def __init__(self, layout):
                self.layout = layout

        class FakeNode:
            def __init__(self, *, outputs=(), layout=None):
                self.outputs = outputs
                self.layout = layout

        out0 = FakeValue(FakeLayout(("s0",)))
        out1 = FakeValue(FakeLayout(("s1",)))
        node = FakeNode(
            outputs=(out0, out1),
            layout=FakeLayout(("fallback_symbol",)),
        )
        self.assertEqual(_layout_only_symbol_uses(node), OrderedSet(("s0", "s1")))

        fallback_node = FakeNode(
            outputs=(FakeValue(FakeLayout(raise_not_implemented=True)),),
            layout=FakeLayout(("layout_symbol",)),
        )
        self.assertEqual(
            _layout_only_symbol_uses(fallback_node), OrderedSet(("layout_symbol",))
        )


instantiate_parametrized_tests(TestMfusionByGraphFusion)
instantiate_parametrized_tests(TestMfusionSymbolFastpath)

if __name__ == "__main__":
    run_tests()
