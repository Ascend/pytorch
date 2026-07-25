import inspect
from types import SimpleNamespace
from unittest import mock

from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.runtime.triton_heuristics import Grid1D
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.codegen.npu_combined_scheduling import (
    NPUCombinedScheduling,
)
from torch_npu._inductor.codegen.scheduling import (
    NPUNoLinearTritonScheduling,
    NPUTritonScheduling,
)
from torch_npu._inductor.codegen.triton import (
    NPUIndexTritonKernel,
    NPUTritonKernel,
)
from torch_npu._inductor.runtime.triton_heuristics import (
    _create_launcher_grid,
    _remap_fallback_block_subs,
)


class TestSchedulingContract(TestCase):
    def test_combined_scheduling_keeps_pytorch_213_protocol(self):
        self.assertTrue(issubclass(NPUCombinedScheduling, CUDACombinedScheduling))
        self.assertTrue(issubclass(NPUCombinedScheduling, TritonScheduling))
        signature = inspect.signature(
            NPUCombinedScheduling.generate_kernel_code_from_nodes
        )
        self.assertIn("hint_override", signature.parameters)

    def test_index_codegen_is_always_first(self):
        scheduling = object.__new__(NPUCombinedScheduling)
        scheduling._triton_scheduling = mock.Mock()
        scheduling._triton_scheduling.codegen_node.return_value = "index"
        scheduling._nolinear_triton_scheduling = mock.Mock()
        node = mock.Mock()

        self.assertEqual(scheduling.codegen_node(node), "index")
        scheduling._nolinear_triton_scheduling.codegen_node.assert_not_called()

    def test_index_failure_regroups_then_falls_back(self):
        scheduling = object.__new__(NPUCombinedScheduling)
        scheduling._triton_scheduling = mock.Mock()
        scheduling._triton_scheduling.codegen_node.side_effect = RuntimeError("index")
        scheduling._nolinear_triton_scheduling = mock.Mock()
        scheduling._nolinear_triton_scheduling.group_fn.return_value = (64, 1)
        scheduling._nolinear_triton_scheduling.codegen_node.return_value = "fallback"
        snode = SimpleNamespace(group=("npu", "old"), _sizes=[[64], [1]])
        node = mock.Mock()
        node.get_nodes.return_value = [snode]

        self.assertEqual(scheduling.codegen_node(node), "fallback")
        self.assertEqual(snode.group, ("npu", (64, 1)))

    def test_scheduling_kernel_types_are_fixed(self):
        self.assertIs(NPUTritonScheduling.kernel_type, NPUIndexTritonKernel)
        self.assertIs(NPUNoLinearTritonScheduling.kernel_type, NPUTritonKernel)

    def test_only_fallback_configs_are_remapped(self):
        untouched = SimpleNamespace(kwargs={"XBLOCK_SUB": 32})
        fallback = SimpleNamespace(kwargs={"XBLOCK_SUB": 32})
        _remap_fallback_block_subs([untouched], {})
        _remap_fallback_block_subs(
            [fallback], {"requires_no_linear_block_remap": True}
        )
        self.assertEqual(untouched.kwargs, {"XBLOCK_SUB": 32})
        self.assertEqual(fallback.kwargs, {"XBLOCK": 32})

    def test_standard_grid_uses_upstream_grid_factory(self):
        grid = _create_launcher_grid(
            {"grid_type": "Grid1D"},
            {"XBLOCK": 32},
            ["xnumel"],
            (),
        )

        self.assertIsInstance(grid, Grid1D)
        self.assertEqual(grid.eval_slow({"xnumel": 33}), (2, 1, 1))

    def test_fallback_kernel_marks_block_remap_capability(self):
        kernel = object.__new__(NPUTritonKernel)
        kernel.range_trees = [
            SimpleNamespace(is_reduction=False, tensor_dim=0, prefix="x")
        ]
        kernel.inside_reduction = False

        metadata = kernel.add_npu_inductor_meta({})

        self.assertIs(metadata["requires_no_linear_block_remap"], True)

    def test_index_kernel_does_not_emit_fallback_capability(self):
        source = inspect.getsource(NPUIndexTritonKernel.create_inductor_meta)
        self.assertNotIn("requires_no_linear_block_remap", source)
        self.assertNotIn("inductor_" + "ascend_linear_mode", source)


if __name__ == "__main__":
    run_tests()
