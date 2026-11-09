import unittest
from types import SimpleNamespace

import sympy
import torch
import torch_npu  # noqa: F401 - register the NPU device backend
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V


class TestComboKernelIndexDtype(unittest.TestCase):
    def _make_subkernel(self, numel):
        from torch_npu._inductor.codegen.npu_kernel_features import NPUKernelFeatures
        from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel
        # Exercise the real index_dtype property without allocating huge tensors
        # or constructing an unrelated scheduling/body codegen pipeline.
        sub = object.__new__(NPUIndexTritonKernel)
        sub.features = NPUKernelFeatures([], numel)
        sub.create_inductor_meta = lambda: {}
        return sub

    def _combo_size_signature(self, subs, numel):
        from unittest.mock import patch
        from torch._inductor.codegen.common import ArgName, SizeArg
        from torch_npu._inductor.codegen import triton_combo_kernel
        kernel = SimpleNamespace(
            sub_kernels=subs, num_warps=1,
            min_x_blocks_sub_kernel=lambda *args: None,
            select_dispatch_strategy=lambda: None,
            dispatch_class=SimpleNamespace(grid_expr=SimpleNamespace(__name__="Grid")),
            get_mutated_args_sub_kernels=lambda: [], combo_grid_meta=lambda: {},
            block_args=[], autotune_subkernel_meta=lambda: [],
        )
        with patch.object(
            triton_combo_kernel, "signature_to_meta",
            wraps=triton_combo_kernel.signature_to_meta,
        ) as signature_to_meta:
            triton_combo_kernel.NPUComboKernel.jit_line(
                kernel, "foreach", {"x": 8}, subs[0],
                [SizeArg("xnumel_0", numel)], [ArgName("xnumel_0")],
            )
        args, kwargs = signature_to_meta.call_args
        return triton_combo_kernel.signature_to_meta(*args, **kwargs)["xnumel_0"]

    def test_combo_actual_index_dtype_boundary(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch_npu._inductor.codegen.triton import patch_triton_scheduling
        # Include the NPU scheduling patch that used to select int32 unconditionally.
        patch_triton_scheduling()
        shape_env = ShapeEnv()
        graph = SimpleNamespace(
            sizevars=SizeVarAllocator(shape_env),
            get_current_device_or_throw=lambda: torch.device("npu:0"),
        )
        with V.set_graph_handler(graph):
            dynamic = shape_env.create_unbacked_symint().node.expr
            for numel, dtype, abi in (
                (sympy.Integer(4096), "tl.int32", "i32"),
                (sympy.Integer(2**31 - 1), "tl.int32", "i32"),
                (sympy.Integer(2**31), "tl.int64", "i64"),
                (sympy.Integer(2**31 + 4096), "tl.int64", "i64"),
                (dynamic, "tl.int64", "i64"),
            ):
                with self.subTest(numel=str(numel)):
                    sub = self._make_subkernel(numel)
                    self.assertEqual(sub.index_dtype, dtype)
                    self.assertEqual(self._combo_size_signature([sub], numel), abi)
            subs = [self._make_subkernel(sympy.Integer(4096)),
                    self._make_subkernel(sympy.Integer(2**31))]
            self.assertEqual(self._combo_size_signature(subs, dynamic), "i64")

    def test_combo_size_dtype_preserves_safe_index_width(self):
        from unittest.mock import patch
        from torch._inductor.codegen.common import ArgName, SizeArg
        from torch_npu._inductor.codegen.triton_combo_kernel import NPUComboKernel
        for dtypes, expected in (
            (("tl.int32", "tl.int32"), "i32"),
            (("tl.int32", "tl.int64"), "i64"),
            (("tl.int64",), "i64"),
        ):
            subs = [SimpleNamespace(index_dtype=dtype, create_inductor_meta=lambda: {},
                                    _disable_grouped_autotune=lambda *args: None)
                    for dtype in dtypes]
            kernel = SimpleNamespace(
                sub_kernels=subs, num_warps=1,
                min_x_blocks_sub_kernel=lambda *args: None,
                select_dispatch_strategy=lambda: None,
                dispatch_class=SimpleNamespace(grid_expr=SimpleNamespace(__name__="Grid")),
                get_mutated_args_sub_kernels=lambda: [], combo_grid_meta=lambda: {},
                block_args=[], autotune_subkernel_meta=lambda: [],
            )
            with (
                V.set_graph_handler(SimpleNamespace(
                    sizevars=SizeVarAllocator(),
                    get_current_device_or_throw=lambda: torch.device("npu:0"))),
                patch("torch_npu._inductor.codegen.triton_combo_kernel.DeviceProperties.create",
                      return_value=SimpleNamespace(type="npu")),
            ):
                code = NPUComboKernel.jit_line(
                    kernel, "foreach", {"x1": 8}, subs[0],
                    [SizeArg("X1BLOCK", sympy.Integer(8))], [ArgName("X1BLOCK")],
                )
            self.assertIn("'X1BLOCK': '" + expected + "'", code)


if __name__ == "__main__":
    unittest.main()
