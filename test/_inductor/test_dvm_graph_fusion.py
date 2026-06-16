from unittest import mock

import torch

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from torch_npu._inductor.dvm.graph_fusion import (
    DvmGraphFusionPatch,
    _dvm_generate_fallback_kernel,
    _fused_metas,
)
from torch_npu._inductor.dvm.graph_build import is_fx_dynamic


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b, c):
        add = a + b
        mul = add * c
        return torch.sum(mul, dim=(0,), keepdim=True) + 1


class MatMulModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b):
        mm = torch.mm(a.t(), b)
        mm = mm.to(torch.float32)
        return mm + 1


class TestDvmByGraphFusion(TestCase):
    @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_basic_partitioning(self, dtype, is_dynamic):
        a = torch.normal(0, 0.1, size=(512, 1024), dtype=dtype).npu()
        b = torch.normal(0, 0.1, size=(512, 1024), dtype=torch.float16).npu()
        c = torch.normal(0, 0.1, size=(1, 1024), dtype=dtype).npu()
        model = TestModule()

        with DvmGraphFusionPatch():
            dvm_compiled_model = torch.compile(
                model, backend="inductor", dynamic=is_dynamic
            )
            with torch.no_grad():
                expect = model(a, b, c)
                result = dvm_compiled_model(a, b, c)
                self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)

    @parametrize("dtype", [torch.float16])
    @parametrize("is_dynamic", [False])
    def test_basic_partitioning_npugraph(self, dtype, is_dynamic):
        a = torch.normal(0, 0.1, size=(512, 1024), dtype=dtype).npu()
        b = torch.normal(0, 0.1, size=(512, 1024), dtype=torch.float16).npu()
        c = torch.normal(0, 0.1, size=(1, 1024), dtype=dtype).npu()
        model = TestModule()
        with DvmGraphFusionPatch():
            dvm_compiled_model = torch.compile(
                model,
                backend="inductor",
                dynamic=is_dynamic,
                options={"triton.cudagraphs": True},
            )
            with torch.no_grad():
                expect = model(a, b, c)
                result = dvm_compiled_model(a, b, c)
                result = dvm_compiled_model(a, b, c)
                self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)

    @parametrize("k,n,m", [(1280, 2560, 640), (1024, 1280, 2560)])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_matmul(self, k, n, m, dtype, is_dynamic):
        a = torch.normal(0, 0.02, size=(k, n), dtype=dtype).npu()
        b = torch.normal(0, 0.02, size=(k, m), dtype=dtype).npu()
        model = MatMulModule()

        with DvmGraphFusionPatch():
            dvm_compiled_model = torch.compile(
                model, backend="inductor", dynamic=is_dynamic
            )
            with torch.no_grad():
                expect = model(a, b)
                result = dvm_compiled_model(a, b)
                self.assertEqual(expect, result, atol=2e-3, rtol=2e-3)


instantiate_parametrized_tests(TestDvmByGraphFusion)


class _AddModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TestDvmFallbackStridePatchGuard(TestCase):
    """Guard adb97b9: skip stride patch in dynamic fused subgraph codegen."""

    def _make_fallback_kernel(self, fused_id=0):
        fallback_kernel = mock.MagicMock()
        fallback_kernel.op_overload._name = "dvm::fused_graph_0"
        fallback_kernel.codegen_args.return_value = ["buf0", fused_id]
        fallback_kernel.codegen_kwargs.return_value = []
        fallback_kernel.get_name.return_value = "buf_out"
        return fallback_kernel

    def _make_codegen_wrapper(self):
        wrapper = mock.MagicMock()
        wrapper.header = mock.MagicMock()
        return wrapper

    def _make_fused_meta(self):
        meta = mock.MagicMock()
        meta.gm = mock.MagicMock()
        meta.name = "dvm_graph_fused_0"
        codegen = mock.MagicMock()
        codegen.cont_flag_input = [True]
        codegen.need_trans_input = [False]
        meta.codegen.return_value = (codegen, "# dvm kernel\n")
        return meta

    @mock.patch(
        "torch_npu._inductor.dvm.graph_fusion.patch_gm_placeholder_strides_from_codegen_args"
    )
    def test_fallback_kernel_stride_patch_guarded_by_is_fx_dynamic(self, mock_patch):
        gm_static = torch.fx.symbolic_trace(_AddModule())
        placeholder = next(n for n in gm_static.graph.nodes if n.op == "placeholder")
        placeholder.meta["val"] = torch.randn(2, 3)
        self.assertFalse(is_fx_dynamic(gm_static))

        batch = torch.export.Dim("batch", min=1, max=1024)
        exported = torch.export.export(
            _AddModule(),
            (torch.randn(2, 3),),
            dynamic_shapes={"x": {0: batch}},
        )
        self.assertTrue(is_fx_dynamic(exported.graph_module))

        meta = self._make_fused_meta()
        fallback_kernel = self._make_fallback_kernel()
        args = fallback_kernel.codegen_args.return_value
        try:
            _fused_metas[0] = meta
            with mock.patch(
                "torch_npu._inductor.dvm.graph_fusion.is_fx_dynamic", return_value=True
            ):
                _dvm_generate_fallback_kernel(
                    self._make_codegen_wrapper(),
                    fallback_kernel,
                    args,
                )
            mock_patch.assert_not_called()

            mock_patch.reset_mock()
            # _dvm_generate_fallback_kernel pops fused_id from _fused_metas; re-seed
            # before exercising the static-shape branch in the same test.
            _fused_metas[0] = meta
            with mock.patch(
                "torch_npu._inductor.dvm.graph_fusion.is_fx_dynamic", return_value=False
            ):
                _dvm_generate_fallback_kernel(
                    self._make_codegen_wrapper(),
                    fallback_kernel,
                    args,
                )
            mock_patch.assert_called_once_with(meta.gm, ["buf0"])
        finally:
            _fused_metas.pop(0, None)


if __name__ == "__main__":
    run_tests()
