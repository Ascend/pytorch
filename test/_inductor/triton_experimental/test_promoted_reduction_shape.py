import os
import re
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor.exc import InductorError
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401


class TestPromotedReductionShape(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("shape", [(5, 5, 5), (5, 7, 9), (17, 5, 19)])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_multi_axis_sum(self, device, dtype, shape):
        from torch_npu._inductor.triton_experimental import config as npu_config
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        original = NPUTritonKernel.codegen_kernel
        original_reduction = NPUTritonKernel.reduction
        reduction_inputs = []
        checked_inputs = []
        promoted = []
        tiled_tail_nodes = []

        def reduction(kernel, dtype, src_dtype, reduction_type, value):
            reduction_inputs.append((kernel, value))
            return original_reduction(kernel, dtype, src_dtype, reduction_type, value)

        def codegen(kernel, *args, **kwargs):
            promoted.extend(getattr(kernel, "_npu_rtree_promoted", {}).values())
            tiles = getattr(kernel, "_npu_rtree_tile", {})
            dynamic = getattr(kernel, "_npu_rtree_node_dynamic", {})
            for tree in kernel.range_trees:
                for node in tree.nodes.values():
                    tile = tiles.get(node.name)
                    if tile and dynamic.get(node.name) and int(node.length) > tile and int(node.length) % tile:
                        tiled_tail_nodes.append(node.name)
            source = original(kernel, *args, **kwargs)
            if not getattr(kernel, "_npu_rtree_promoted", {}):
                return source
            expected_slots = {}
            for tree in kernel.range_trees:
                if tree.tensor_dim is None:
                    continue
                scalar_nodes = getattr(tree, "_npu_scalar_odometer", ()) or ()
                for node in tree.nodes.values():
                    if node.name in tree.tree_node_mapping or node.name in scalar_nodes:
                        continue
                    slot = tree.var_tensor_dims[node.name]
                    if tree.is_reduction:
                        token = f"{node.name}_blk" if dynamic.get(node.name) else f"real_block_{node.name}"
                    else:
                        token = getattr(tree, "node_block_constexpr", {}).get(node.name, f"real_block_{node.name}")
                    self.assertNotIn(slot, expected_slots)
                    expected_slots[slot] = token
            self.assertEqual(sorted(expected_slots), list(range(3)))
            expected_shape = tuple(str(expected_slots[slot]) for slot in range(3))
            for owner, value in reduction_inputs:
                if owner is not kernel:
                    continue
                actual_shape = kernel._npu_assert_shapes.get(value.name)
                self.assertIsNotNone(actual_shape)
                self.assertEqual(tuple(map(str, actual_shape)), expected_shape)
                name = re.escape(value.name)
                condition = rf"tl\.static_assert\([^\n]*len\({name}\.shape\) == 3"
                for axis, token in enumerate(expected_shape):
                    condition += rf"[^\n]*{name}\.shape\[{axis}\] == {re.escape(token)}(?![\w])"
                self.assertRegex(source, condition)
                checked_inputs.append(value.name)
            return source

        def fn(x):
            return x.square().sum(dim=(0, 2))

        torch._dynamo.reset()
        with (
            mock.patch.object(NPUTritonKernel, "codegen_kernel", codegen),
            mock.patch.object(NPUTritonKernel, "reduction", reduction),
            mock.patch.object(npu_config, "rtree_real_block_cap", 64),
            mock.patch.object(npu_config, "rtree_real_block_autotune", False),
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"}), x
            )
        expected = fn(x)
        self.assertEqual(actual, expected)
        self.assertEqual(torch.autograd.grad(actual.sum(), x), torch.autograd.grad(expected.sum(), x))
        self.assertTrue(any(len(nodes) > 1 for nodes in promoted))
        source = "\n".join(codes)
        self.assertTrue(checked_inputs, "expected a promoted reduction input with checked slot assertions")
        if shape == (17, 5, 19):
            self.assertTrue(tiled_tail_nodes)
            for name in tiled_tail_nodes:
                self.assertRegex(source, rf"for {name}inner in range\(")
                self.assertIn(f"{name}mask = {name} <", source)

    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_unknown_source_keeps_original_shape(self, device, dtype):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        original_load = NPUTritonKernel.load
        original_codegen = NPUTritonKernel.codegen_kernel
        unknown = set()
        captured = []

        class SourceCaptured(RuntimeError):
            pass

        def load(kernel, *args, **kwargs):
            result = original_load(kernel, *args, **kwargs)
            result._npu_load_index = None
            unknown.add(result.name)
            return result

        def codegen(kernel, *args, **kwargs):
            consumers = {}
            for name, var in kernel.cse.varname_map.items():
                op = getattr(var, "_npu_shape_op", None)
                if op is not None and any(getattr(arg, "name", None) in unknown for arg in op[1]):
                    unknown.add(name)
                    consumers[name] = var.shape
            source = original_codegen(kernel, *args, **kwargs)
            captured.append((source, consumers, bool(getattr(kernel, "_npu_rtree_promoted", {}))))
            raise SourceCaptured("promoted unknown-source assertions captured")

        def fn(x):
            return x.square().sum(dim=(0, 2))

        x = torch.randn((5, 5, 5), device=device, dtype=dtype)
        torch._dynamo.reset()
        with (
            mock.patch.object(NPUTritonKernel, "load", load),
            mock.patch.object(NPUTritonKernel, "codegen_kernel", codegen),
        ):
            compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
            with self.assertRaisesRegex(InductorError, "promoted unknown-source assertions captured"):
                compiled(x)
        self.assertEqual(len(captured), 1)
        source, consumers, promoted = captured[0]
        self.assertTrue(promoted)
        self.assertTrue(consumers)
        for name, shape in consumers.items():
            self.assertIn(f"len({name}.shape) == {len(shape)}", source)
            for axis, dim in enumerate(shape):
                self.assertIn(f"{name}.shape[{axis}] == {dim}", source)


instantiate_device_type_tests(TestPromotedReductionShape, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
