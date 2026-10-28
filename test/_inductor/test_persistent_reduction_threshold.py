import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.simd import DisableReduction, EnableReduction
from torch._inductor.dependencies import MemoryDep
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V

from torch_npu._inductor.codegen import triton as triton_codegen
from torch_npu._inductor.codegen.npu_kernel_features import NumelList


class TestPersistentReductionThreshold(unittest.TestCase):
    @staticmethod
    def dep(name, index=0):
        return MemoryDep(name, sympy.Integer(index), (), ())

    @staticmethod
    def node(reads, writes=()):
        return SimpleNamespace(read_writes=SimpleNamespace(reads=reads, writes=writes))

    @staticmethod
    def graph(dtypes, unspec=()):
        return SimpleNamespace(
            sizevars=SizeVarAllocator(),
            get_dtype=dtypes.__getitem__,
            is_unspec_arg=lambda name: name in unspec,
        )

    def kernel(self, reduction_numel, nodes):
        kernel = object.__new__(triton_codegen.NPUIndexTritonKernel)
        kernel.inside_reduction = True
        kernel.cooperative_reduction = False
        kernel.features = SimpleNamespace(
            reduction_numel=reduction_numel,
            numel=sympy.Integer(32),
            node_schedule=nodes,
            get_reduction_hint=lambda: ReductionHint.INNER,
        )
        return kernel

    def test_watermark_boundary_and_mixed_dtypes(self):
        names = [f"input{i}" for i in range(65)]
        dtypes = {name: torch.float16 for name in names}
        graph = self.graph(dtypes)
        with (
            V.set_graph_handler(graph),
            patch.object(triton_codegen.npu_config, "is_ascend950", True),
            config.patch({"triton.persistent_reductions": True, "triton.multi_kernel": False}),
        ):
            exact = self.node([self.dep(name) for name in names[:64]])
            over = self.node([self.dep(name) for name in names])
            self.assertTrue(
                self.kernel(NumelList((32, 64)), [exact]).should_use_persistent_reduction()
            )
            self.assertFalse(self.kernel(2048, [over]).should_use_persistent_reduction())

            dtypes[names[0]] = torch.float32
            self.assertFalse(self.kernel(2048, [exact]).should_use_persistent_reduction())

    def test_estimate_ignores_duplicate_internal_scalar_and_epilogue_reads(self):
        source = self.dep("input")
        produced = self.dep("intermediate")
        nodes = [
            self.node([source, source, self.dep("input", 1), self.dep("scalar")], [produced]),
            self.node([source, produced]),
            DisableReduction,
            self.node([self.dep("epilogue")]),
            EnableReduction,
        ]
        graph = self.graph({"input": torch.float32, "scalar": torch.int64}, unspec={"scalar"})
        with V.set_graph_handler(graph):
            self.assertEqual(
                self.kernel(2048, nodes)._estimate_reduction_input_bytes(2048),
                2048 * 2 * 4,
            )

    def test_high_load_threshold_is_950_specific(self):
        names = [f"input{i}" for i in range(81)]
        node = self.node([self.dep(name) for name in names])
        graph = self.graph({name: torch.float32 for name in names})
        with V.set_graph_handler(graph), config.patch(
            {"triton.persistent_reductions": True, "triton.multi_kernel": False}
        ):
            with patch.object(triton_codegen.npu_config, "is_ascend950", True):
                self.assertTrue(self.kernel(1024, [node]).should_use_persistent_reduction())
                self.assertFalse(self.kernel(2048, [node]).should_use_persistent_reduction())
            with patch.object(triton_codegen.npu_config, "is_ascend950", False):
                self.assertFalse(self.kernel(2048, [node]).should_use_persistent_reduction())

    def test_unproven_symbolic_watermark_keeps_existing_threshold(self):
        names = [f"input{i}" for i in range(81)]
        node = self.node([self.dep(name) for name in names])
        graph = self.graph({name: torch.float32 for name in names})
        unknown = sympy.Symbol("s0", positive=True, integer=True)
        kernel = self.kernel(sympy.Min(unknown, 4096), [node])
        with (
            V.set_graph_handler(graph),
            patch.object(triton_codegen.npu_config, "is_ascend950", True),
            config.patch({"triton.persistent_reductions": True, "triton.multi_kernel": False}),
        ):
            with patch.object(
                graph.sizevars,
                "statically_known_leq",
                side_effect=lambda _, limit: limit >= 4096,
            ):
                self.assertTrue(kernel.should_use_persistent_reduction())


if __name__ == "__main__":
    unittest.main()
