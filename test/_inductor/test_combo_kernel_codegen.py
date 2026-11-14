import ast
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import sympy
import torch
import torch_npu  # noqa: F401 - register the NPU device backend
from torch._inductor import config
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.codegen import split_tiling as split_tiling_module
from torch_npu._inductor.codegen import triton_combo_kernel as combo_codegen
from torch_npu._inductor.codegen.npu_combined_scheduling import NPUCombinedScheduling
from torch_npu._inductor.codegen.split_tiling import SplitTiling
from torch_npu._inductor.codegen.triton import (
    IterationRangesRootNPUIndex,
    NPUIndexTritonKernel,
)
from torch_npu._inductor.codegen.triton_combo_kernel import NPUComboKernel
from torch_npu._inductor.runtime.triton_heuristics import SequentialComboKernelGrid


def make_axis(name, length, order=0):
    symbol = sympy.Symbol(name, integer=True)
    return SimpleNamespace(
        name=name,
        prefix=name[0],
        length=sympy.sympify(length),
        expr=symbol,
        symbol=lambda: symbol,
        sorted_order=order,
        is_reduction=name.startswith("r"),
        is_split_axis=False,
        is_tiling_axis=True,
        is_no_loop_axis=False,
    )


class TestComboKernelCodegen(unittest.TestCase):
    def test_dispatch_axis_respects_contiguous_access_layout(self):
        for index_fn, expected in (
            (lambda x1, x2: 380928 * x1 + x2, "x2"),
            (lambda x1, x2: x1 + 2 * x2, "x1"),
        ):
            axes = [make_axis("x1", 2), make_axis("x2", 380928, 1)]
            for axis in axes:
                axis.is_split_axis = True
            x1, x2 = [axis.symbol() for axis in axes]
            member = SimpleNamespace(
                split_axis=axes.copy(),
                sorted_axis=axes,
                persistent_reduction=False,
                no_x_dim=False,
                low_dims=set(),
                load_store_indexing=[index_fn(x1, x2)],
                get_axis_dtype=lambda axis: torch.float32,
            )
            with (
                patch.object(npu_config, "_get_core_nums", return_value=(None, 1, 56)),
                patch.object(combo_codegen, "NPUIndexTritonKernel", return_value=member),
            ):
                actual = NPUComboKernel.create_triton_kernel({}, Mock(), True)
            self.assertEqual([axis.name for axis in actual.split_axis], [expected])

    def test_numel_partition_regroups_large_outliers(self):
        nodes = list(range(6))
        sizes = [3055616, 6111232, 11936, 11936, 11936, 11936]
        kernels = {
            i: SimpleNamespace(sorted_axis=[make_axis("x0", size)])
            for i, size in enumerate(sizes)
        }
        with patch.object(npu_config, "_get_core_nums", return_value=(None, 1, 56)):
            groups = NPUComboKernel._partition_by_numel(nodes, kernels)
        self.assertEqual(groups, [[2, 3, 4, 5], [0, 1]])

    def test_numel_partition_includes_reduction_extent(self):
        nodes = list(range(5))
        shapes = [(256, 1025), (35, 128), (256, 1025), (7, 128), (256, 1025)]
        kernels = {
            i: SimpleNamespace(sorted_axis=[make_axis("x0", x), make_axis("r1", r, 1)])
            for i, (x, r) in enumerate(shapes)
        }
        with patch.object(npu_config, "_get_core_nums", return_value=(None, 1, 56)):
            groups = NPUComboKernel._partition_by_numel(nodes, kernels)
        self.assertEqual(groups, [[3], [0, 1, 2, 4]])

    def test_numel_partition_boundaries_and_recursive_outliers(self):
        cases = (
            ([1, 224, 224], [[0, 1, 2]]),
            ([1, 225, 225], [[0], [1, 2]]),
            ([1, 1000, 1000000, 1000000000, 1000000000], [[0], [1], [2], [3, 4]]),
        )
        with patch.object(npu_config, "_get_core_nums", return_value=(None, 1, 56)):
            for sizes, expected in cases:
                with self.subTest(sizes=sizes):
                    nodes = list(range(len(sizes)))
                    kernels = {
                        i: SimpleNamespace(sorted_axis=[make_axis("x0", size)])
                        for i, size in enumerate(sizes)
                    }
                    self.assertEqual(NPUComboKernel._partition_by_numel(nodes, kernels), expected)

    def test_numel_partition_preserves_dynamic_groups(self):
        nodes = [0, 1, 2]
        sizes = [1, sympy.Symbol("s0", positive=True), 1000000]
        kernels = {
            i: SimpleNamespace(sorted_axis=[make_axis("x0", size)])
            for i, size in enumerate(sizes)
        }
        self.assertEqual(NPUComboKernel._partition_by_numel(nodes, kernels), [nodes])

    def test_sync_removed_outputs_updates_combo_and_subkernels(self):
        combo = object.__new__(NPUComboKernel)
        combo.args = SimpleNamespace(
            output_buffers={"output": object()},
            inplace_buffers={"inplace": object()},
            is_removed=lambda name: name in {"output", "inplace"},
        )
        combo.removed_buffers = set()
        combo.sub_kernels = [SimpleNamespace(removed_buffers=set())]

        combo._sync_removed_outputs()

        expected = {"output", "inplace"}
        self.assertEqual(combo.removed_buffers, expected)
        self.assertEqual(combo.sub_kernels[0].removed_buffers, expected)

    def test_regroup_fused_subkernels(self):
        def node(name, sizes, reduction=False):
            n = SimpleNamespace(
                name=name,
                _sizes=sizes,
                group=("npu", (999, 999)),
                is_reduction=lambda: reduction,
            )
            n.get_nodes = lambda: [n]
            return n
        a = node("reduce", ((7, 208), (113,)), True)
        b = node("epilogue", ((7, 208), ()))
        c = node("next_reduce", ((2,), (60,)), True)
        fused = SimpleNamespace(get_nodes=lambda: [a, b], group=a.group)
        combo = SimpleNamespace(get_subkernel_nodes=lambda: [fused, c])
        backend = SimpleNamespace(
            node_can_linear=lambda: True,
            _nolinear_triton_scheduling=SimpleNamespace(
                group_fn=lambda sizes: tuple(sympy.prod(s) for s in sizes)
            ),
            _triton_scheduling=SimpleNamespace(codegen_combo_kernel=Mock()),
        )
        NPUCombinedScheduling.codegen_combo_kernel(backend, combo)
        self.assertEqual(fused.group, ("npu", (1456, 113)))
        self.assertEqual(c.group, ("npu", (2, 60)))
        self.assertEqual(b.group, ("npu", (1456, 1)))

    def test_lookup_removed_combo_axis(self):
        symbol = sympy.Symbol("x0", integer=True)
        root = SimpleNamespace(
            numel=sympy.Integer(16),
            prefix="x",
            nodes={},
            pid_cache={"pid": 0},
        )
        removed = SimpleNamespace(
            parent=root,
            divisor=sympy.S.One,
            length=sympy.Integer(16),
            is_sub_kernel=False,
            symbol=lambda: symbol,
        )
        root._find_removed_node = lambda divisor, length: removed
        kernel = SimpleNamespace(
            axioms_index_map={},
            range_tree_nodes_removed={symbol: removed},
            iter_vars_count=itertools.count(),
        )
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())),
            V.set_kernel_handler(kernel),
        ):
            result = IterationRangesRootNPUIndex.lookup(root, 1, 16)
        self.assertIs(result, removed)
        self.assertTrue(result.is_sub_kernel)

    def test_y_axis_grid_is_one_dimensional(self):
        grid = SequentialComboKernelGrid(
            inductor_meta={
                "combo_grid_meta": {
                    "split_numels": [15360, 15360, 1],
                    "no_split": [False, False, True],
                    "split_block": "Y0BLOCK",
                }
            }
        )
        grid.generate({"Y0BLOCK": 256})
        self.assertEqual((grid.x_grid, grid.y_grid, grid.z_grid), (121, 1, 1))

    def test_dynamic_grid_uses_axis_argument(self):
        grid = SequentialComboKernelGrid(
            inductor_meta={
                "combo_grid_meta": {
                    "split_numels": ["y0_numel_0", "y0_numel_1"],
                    "no_split": [False, False],
                    "split_block": "Y0BLOCK",
                }
            }
        )
        grid.generate({})
        for size in (31, 257):
            result = eval(
                grid.x_grid,
                {},
                {
                    "y0_numel_0": size,
                    "y0_numel_1": size + 1,
                    "Y0BLOCK": 32,
                },
            )
            self.assertEqual(result, (size + 31) // 32 + (size + 32) // 32)
        self.assertEqual((grid.y_grid, grid.z_grid), (1, 1))

    def test_dynamic_benchmark_arguments_match_normal_call(self):
        s0, s1 = sympy.symbols("s0 s1", integer=True, positive=True)
        sizevars = SizeVarAllocator()
        sizevars.shape_env.var_to_val.update({s0: 31, s1: 257})
        combo = object.__new__(NPUComboKernel)
        combo.dynamic_shape_args = []
        combo.sub_kernels = [
            SimpleNamespace(sorted_axis=[make_axis("x0", s0), make_axis("y1", 16)]),
            SimpleNamespace(sorted_axis=[make_axis("x0", s1), make_axis("y1", 2 * s1 + 1)]),
        ]
        argdefs, signature, call_args, arg_types = [], [], [], []
        with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
            combo.add_numel_to_args(argdefs, signature)
            combo.add_numel_to_call_args("combo", call_args, arg_types)
            benchmark_args = combo.kernel_benchmark_extra_args()
        self.assertEqual(
            [arg.name for arg in argdefs],
            ["x0_numel_0", "x0_numel_1", "y1_numel_1"],
        )
        self.assertEqual([arg.expr for arg in signature], call_args)
        self.assertEqual(call_args, [s0, s1, 2 * s1 + 1])
        self.assertEqual(benchmark_args, ["31", "257", "515"])
        self.assertEqual(len(benchmark_args), len(argdefs))

    def test_static_members_have_no_benchmark_numel_arguments(self):
        combo = object.__new__(NPUComboKernel)
        combo.dynamic_shape_args = []
        combo.sub_kernels = [SimpleNamespace(sorted_axis=[make_axis("x0", 128)])]
        argdefs, signature = [], []
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            combo.add_numel_to_args(argdefs, signature)
            self.assertEqual(combo.kernel_benchmark_extra_args(), [])
        self.assertEqual(argdefs, [])
        self.assertEqual(signature, [])

    def test_combo_disables_grouped_rewrite_without_changing_global_config(self):
        selector = object.__new__(SplitTiling)
        selector.kernel = SimpleNamespace(
            allow_grouped_autotune=False,
            grouped_autotune_meta=object(),
        )
        selector._is_grouped_template_enabled = Mock(return_value=True)
        grouped_meta = object()
        selector._build_grouped_meta = Mock(return_value=grouped_meta)
        with patch.object(npu_config, "enable_symbolic_shape_group_autotune", True):
            selector.apply_grouped_rewrite_if_needed()
            self.assertIsNone(selector.kernel.grouped_autotune_meta)
            selector._is_grouped_template_enabled.assert_not_called()
            selector._build_grouped_meta.assert_not_called()
            self.assertTrue(npu_config.enable_symbolic_shape_group_autotune)
            # The next ordinary kernel must still take the global grouped path.
            selector.kernel = SimpleNamespace(allow_grouped_autotune=True)
            selector.apply_grouped_rewrite_if_needed()
            self.assertIs(selector.kernel.grouped_autotune_meta, grouped_meta)
            selector._build_grouped_meta.assert_called_once_with()

    def test_combo_disables_grouped_split_early_stop(self):
        size = sympy.Symbol("s0", integer=True, positive=True)
        sizevars = SizeVarAllocator()
        sizevars.shape_env.var_to_val[size] = 2

        def selected_axes(allow_grouped):
            axes = [make_axis("y0", size, 0), make_axis("x1", 64, 1), make_axis("r2", 64, 2)]
            selector = object.__new__(SplitTiling)
            selector.kernel = SimpleNamespace(
                allow_grouped_autotune=allow_grouped,
                sorted_axis=axes,
                split_axis=[],
                low_dims={1},
            )
            selector.select_split_axis()
            return [axis.name for axis in selector.kernel.split_axis]
        with (
            V.set_graph_handler(SimpleNamespace(sizevars=sizevars)),
            patch.object(split_tiling_module, "_num_vector_core", return_value=48),
            patch.object(npu_config, "enable_symbolic_shape_group_autotune", True),
        ):
            self.assertEqual(selected_axes(True), ["y0"])
            self.assertEqual(selected_axes(False), ["y0", "x1"])
            self.assertTrue(npu_config.enable_symbolic_shape_group_autotune)
            with patch.object(npu_config, "enable_symbolic_shape_group_autotune", False):
                self.assertEqual(selected_axes(True), ["y0", "x1"])

    def test_combo_metadata_keeps_only_its_actual_runtime_blocks(self):
        # An old member plan must never re-enable grouping or add stale blocks
        # to the Combo launch ABI. This also covers non-first-member mutations.
        metadata = {
            "kernel_name": "combo_test",
            "group_enabled": True,
            "group_template": "pointwise",
            "group_workload": "elementwise",
            "primary_group_axis": "x0",
            "static_split_axes": ("y0",),
            "secondary_runtime_symbolic_axes": ("x0",),
            "group_features": ({"name": "pointwise"},),
            "runtime_block_arg_names": ("Y0BLOCK", "X0BLOCK"),
            "mutated_arg_names": [],
        }
        member = object.__new__(NPUIndexTritonKernel)
        member.create_inductor_meta = lambda: metadata
        other_member = object.__new__(NPUIndexTritonKernel)
        shared_args = SimpleNamespace(
            input_buffers={"buf1": "in_ptr1"}, inplace_buffers={}, output_buffers={}
        )
        member.args = other_member.args = shared_args
        member.mutations = set()
        other_member.mutations = {"buf1"}
        combo = object.__new__(NPUComboKernel)
        combo.per_subkernel_blocks = False
        combo.sub_kernels = [member, other_member]
        combo.block_args = ["Y0BLOCK"]
        combo.dispatch_class = NPUComboKernel.SIMDDispatch
        combo.num_warps = 8
        combo.min_x_blocks_sub_kernel = Mock()
        combo.combo_grid_meta = Mock(return_value={"num_kernels": 2})
        combo.autotune_subkernel_meta = Mock(return_value=[])
        graph = SimpleNamespace(get_current_device_or_throw=lambda: torch.device("npu"))
        with (
            V.set_graph_handler(graph),
            patch.object(combo_codegen.DeviceProperties, "create", return_value=None),
            patch.object(combo_codegen, "config_of", return_value={}),
            patch.object(NPUIndexTritonKernel, "index_dtype", new_callable=PropertyMock,
                         return_value="tl.int32"),
        ):
            combo.jit_line("foreach", {"y0": 128}, member, [], [])
        self.assertFalse(metadata["group_enabled"])
        self.assertIsNone(metadata["group_template"])
        self.assertIsNone(metadata["primary_group_axis"])
        self.assertEqual(metadata["group_features"], ())
        self.assertEqual(metadata["runtime_block_arg_names"], ("Y0BLOCK",))
        self.assertEqual(metadata["mutated_arg_names"], ["in_ptr1"])


@unittest.skipUnless(torch.npu.is_available(), "NPU is required")
class TestComboKernelNPU(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_dynamic_combo_ignores_global_grouped_autotune(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()
        members = []
        combo_metadata = []
        create_member = NPUComboKernel.create_triton_kernel
        generate_jit = NPUComboKernel.jit_line

        def observe_member(*args, **kwargs):
            member = create_member(*args, **kwargs)
            members.append(member)
            return member

        def observe_jit(combo, *args, **kwargs):
            source = generate_jit(combo, *args, **kwargs)
            for line in source.splitlines():
                if line.strip().startswith("inductor_meta="):
                    expression = ast.parse(
                        line.strip().split("=", 1)[1].rstrip(","), mode="eval"
                    ).body
                    # Other fields contain torch.dtype expressions. Inspect
                    # only the literal fields required by the Combo contract.
                    self.assertIsInstance(expression, ast.Dict)
                    combo_metadata.append({
                        key.value: ast.literal_eval(value)
                        for key, value in zip(expression.keys, expression.values)
                        if isinstance(key, ast.Constant)
                        and key.value in ("group_enabled", "combo_grid_meta")
                    })
            return source

        def fn(a, b):
            return a.sin(), b.cos()
        with (
            patch.object(npu_config, "enable_symbolic_shape_group_autotune", True),
            config.patch(combo_kernels=True, benchmark_combo_kernel=False,
                         force_disable_caches=True,
                         combo_kernel_allow_mixed_sizes=2,
                         combo_kernel_foreach_dynamic_shapes=True),
            patch.object(NPUComboKernel, "create_triton_kernel", side_effect=observe_member),
            patch.object(NPUComboKernel, "jit_line", observe_jit),
        ):
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for rows in (31, 257):
                inputs = (torch.rand(rows, device="npu"), torch.rand(rows + 7, device="npu"))
                actual = compiled(*inputs)
                torch.npu.synchronize()
                torch.testing.assert_close(actual, fn(*inputs))
            self.assertTrue(npu_config.enable_symbolic_shape_group_autotune)
        self.assertTrue(members)
        self.assertTrue(all(not member.allow_grouped_autotune for member in members))
        self.assertTrue(all(member.grouped_autotune_meta is None for member in members))
        self.assertTrue(combo_metadata)
        self.assertTrue(all(not meta["group_enabled"] for meta in combo_metadata))
        self.assertTrue(any(meta["combo_grid_meta"]["num_kernels"] > 1 for meta in combo_metadata))


if __name__ == "__main__":
    unittest.main()
