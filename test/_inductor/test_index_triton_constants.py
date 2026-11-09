# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import ArgName, SizeArg
from torch._inductor.sizevars import SizeVarAllocator
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_npu._inductor.codegen import triton as triton_codegen
from version_mark import runIfVersion


class _MetadataReady(Exception):
    pass


class TestIndexTritonConstants(TestCase):
    def _codegen_metadata(self, signature, replacements=None, *, uses_attrs_dict=True):
        kernel = object.__new__(triton_codegen.NPUIndexTritonKernel)
        kernel._index_dtype = torch.int32
        kernel.get_size_hints = mock.Mock(return_value={})
        kernel._get_heuristic = mock.Mock(return_value="pointwise")
        kernel.create_inductor_meta = mock.Mock(return_value={})
        kernel._disable_grouped_autotune_if_unsupported = mock.Mock()
        # Stop after the real codegen has constructed the metadata used by TTIR.
        kernel.gen_numel_args = mock.Mock(side_effect=_MetadataReady)
        kernel.args = mock.Mock()
        kernel.args.python_argdefs.return_value = (
            [ArgName(arg.name) for arg in signature], [], signature, []
        )
        sizevars = SizeVarAllocator()
        sizevars.inv_precomputed_replacements.update(replacements or {})
        graph = SimpleNamespace(
            sizevars=sizevars,
            get_current_device_or_throw=lambda: torch.device("npu"),
        )

        with (
            triton_codegen.V.set_graph_handler(graph),
            # Equal-to-one arguments retain i32 in the legacy Triton protocol.
            mock.patch.object(triton_utils, "triton_version_uses_attrs_dict", return_value=uses_attrs_dict),
            mock.patch.object(triton_codegen.DeviceProperties, "create", return_value=None),
            mock.patch.object(triton_codegen.config, "benchmark_kernel", False),
            mock.patch.object(triton_codegen.config, "profile_bandwidth", False),
        ):
            with self.assertRaises(_MetadataReady):
                kernel.codegen_kernel(name="test_kernel")
        return kernel.triton_meta

    @runIfVersion(max="2.13")
    def test_seed_offset_one_has_a_constant_value_before_ttir(self):
        for uses_attrs_dict in (False, True):
            for offset in (0, 1, 2):
                with self.subTest(uses_attrs_dict=uses_attrs_dict, offset=offset):
                    metadata = self._codegen_metadata(
                        [SizeArg("load_seed_offset", sympy.Integer(offset))],
                        uses_attrs_dict=uses_attrs_dict,
                    )
                    expected = {"load_seed_offset": 1} if offset == 1 else {}
                    self.assertEqual(metadata["constants"], expected)
                    expected_type = "constexpr" if uses_attrs_dict and offset == 1 else "i32"
                    self.assertEqual(metadata["signature"]["load_seed_offset"], expected_type)

    @runIfVersion(max="2.13")
    def test_other_static_ones_are_bound_without_specializing_dynamic_args(self):
        for uses_attrs_dict in (False, True):
            with self.subTest(uses_attrs_dict=uses_attrs_dict):
                metadata = self._codegen_metadata([
                    SizeArg("load_seed_offset", sympy.Integer(2)),
                    SizeArg("static_size", sympy.Integer(1)),
                    SizeArg("dynamic_size", sympy.Symbol("s0", integer=True, positive=True)),
                ], uses_attrs_dict=uses_attrs_dict)
                self.assertEqual(metadata["constants"], {"static_size": 1})
                expected_type = "constexpr" if uses_attrs_dict else "i32"
                self.assertEqual(metadata["signature"]["static_size"], expected_type)
                self.assertEqual(metadata["signature"]["load_seed_offset"], "i32")
                self.assertEqual(metadata["signature"]["dynamic_size"], "i32")

    @runIfVersion(max="2.13")
    def test_precomputed_one_is_bound_after_signature_rewrite(self):
        for uses_attrs_dict in (False, True):
            with self.subTest(uses_attrs_dict=uses_attrs_dict):
                symbol = sympy.Symbol("ps0", integer=True, positive=True)
                metadata = self._codegen_metadata(
                    [SizeArg("precomputed_size", symbol)],
                    {symbol: sympy.Integer(1)},
                    uses_attrs_dict=uses_attrs_dict,
                )
                self.assertEqual(metadata["constants"], {"precomputed_size": 1})
                expected_type = "constexpr" if uses_attrs_dict else "i32"
                self.assertEqual(metadata["signature"]["precomputed_size"], expected_type)


if __name__ == "__main__":
    run_tests()
