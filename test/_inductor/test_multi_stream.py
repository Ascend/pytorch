import os
import itertools
import torch
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from unittest.mock import patch, Mock, MagicMock
from torch._inductor.virtualized import V
from torch._inductor.codegen.wrapper import WorkspaceArg
from torch_npu._inductor.codegen.catlass.catlass_kernel import CATLASSTemplateKernel
from torch_npu._inductor.codegen.catlass.catlass_scheduling import CATLASSScheduling
from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling
from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel
import torch._inductor.scheduler as sch


class FakeWrapper:
    def __init__(self):
        self.write_triton_header_once = MagicMock()
        self.generate_workspace_allocation = MagicMock()
        self.generate_workspace_deallocation = MagicMock()
        self.generate_kernel_call = MagicMock()


class FakeGraph:
    def __init__(self, cpp_wrapper=False):
        self.wrapper_code = FakeWrapper()
        self.cpp_wrapper = cpp_wrapper
        self.workspace_id = itertools.count()
        self.sizevars = FakeSizeVars()

    def get_current_device_or_throw(self):
        return "npu:0"

    def is_unspec_arg(self, arg):
        return False


class FakeArgs:
    def python_argdefs(self):
        return [], ["input_ptr"], [], []

    def cpp_argdefs(self):
        return [], ["input_ptr"], []


class FakeNode:
    def __init__(self, workspace_size, name):
        self.workspace_size = workspace_size
        self._name = name
        self.last_usage = []

    def get_workspace_size(self):
        return self.workspace_size

    def get_name(self):
        return self._name

    def get_estimated_runtime(self):
        return 1.0

    def get_device(self):
        return None

    def is_extern(self):
        return False

    def is_template(self):
        return False

    def is_foreach(self):
        return False

    def get_buffer_names(self):
        return []

    def get_operation_names(self):
        return []

    def mark_run(self):
        pass
    
    def get_nodes(self):
        return []


class FakeDebugPrinterManager:
    def __init__(self):
        self.set_printer_args = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeWrapperCode:
    def __init__(self):
        self.debug_printer = FakeDebugPrinterManager()


class FakeSizeVars:
    def statically_known_leq(self, a, b):
        return False

    def size_hints(self, x):
        return x

    def simplify(self, x):
        return x


class FakeScheduleGraph:
    def __init__(self):
        self.sizevars = FakeSizeVars()

class FakeKernel:
    def __init__(self):
        self.removed_buffers = set()
        self.args = MagicMock()
        self.args.python_argdefs.return_value = (
            [],
            ["arg0", "arg1"],
            ["sig0", "sig1"],
            [],
        )
        self.get_layout_args = MagicMock(return_value=[1, 2])
        self.call_kernel = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeTemplateBuffer:
    def make_kernel_render(self, ctb, epilogue_nodes=None):
        kernel = FakeKernel()

        def render():
            return "fake_src_code"

        return kernel, render

    def emulate_store_fn(self):
        pass


class FakeCatlassSchedulingNode:
    def __init__(self):
        self.node = FakeTemplateBuffer()
        self.group = (None, (1, 1))

    def mark_run(self):
        pass


class FakeCatlassScheduleGraph:
    def __init__(self):
        self.wrapper_code = FakeWrapperCode()
        self.removed_buffers = set()
        self.sizevars = FakeSizeVars()


class FakeNPUTritonGraph:
    def __init__(self):
        self.removed_buffers = set()
        self.inplaced_to_remove = set()

        self.wrapper_code = MagicMock()
        self.wrapper_code.supports_intermediate_hooks = False


class FakeNPUTritonFeatures:
    def __init__(self, node_schedule):
        self.node_schedule = node_schedule
        self.numel = 1024
        self.reduction_numel = 1

    def scheduler_nodes(self):
        return self.node_schedule


class FakeNPUTritonNode:
    def mark_run(self):
        pass

    def get_name(self):
        return "fake_node"

    @property
    def node(self):
        return None


class FakeNPUTritonKernel:
    def __init__(self):
        self.kernel_name = "fake_kernel"
        self.removed_buffers = set()
        self.inplaced_to_remove = set()

        self.args = MagicMock()
        self.args.live_output_buffers.return_value = set()

    def codegen_kernel(self):
        return "fake_src_code"

    def codegen_nan_check(self):
        pass

    def warn_mix_layout(self, kernel_name):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def call_kernel(self, name, origin_node=None):
        pass


class FakeSchedulerNode:
    def __init__(self, is_reduction=False):
        self._is_reduction = is_reduction
        self.group = ("group_key", (1024, 1))

    def is_reduction(self):
        return self._is_reduction


class FakeFusedNode:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self):
        return self._nodes


class FakeKernelFeatures:
    def __init__(self, node_schedule, numel, rnumel):
        self.node_schedule = node_schedule
        self.numel = numel
        self.rnumel = rnumel


class FakeTritonArgs:
    def python_argdefs(self):
        return (
            None,
            ["a", "b"],   # call_args
            None,
            ["int", "int"]  # arg_types
        )

    @property
    def workspace_args(self):
        return ["ws1", "ws2"]


class TestMultiStreamPass(TestUtils):

    def define_catlass_template_kernel(self):
        kernel = CATLASSTemplateKernel(kernel_name="test_catlass_kernel")
        kernel.args = Mock()
        kernel.args.python_argdefs.return_value = (
            None, ["input", "weight"], None, [torch.float16, torch.float16]
        )
        kernel.args.cpp_argdefs.return_value = (
            None, ["input", "weight"], ["half*", "half*"]
        )
        kernel.get_layout_args = Mock(return_value=["dim0", "dim1", "dim2"])
        return kernel


    def define_catlass_scheduling(self):
        fake_scheduler = MagicMock()
        scheduler = CATLASSScheduling(fake_scheduler)
        scheduler.is_catlass_template = MagicMock(return_value=True)
        scheduler.define_kernel = MagicMock(
            return_value="generated_kernel"
        )
        scheduler.free_buffers_in_scheduler = MagicMock()
        return scheduler


    def define_scheduling(self):
        fake_scheduler = MagicMock()
        scheduling = NPUTritonScheduling(fake_scheduler)
        scheduling.select_tiling = MagicMock(return_value="fake_tiling")
        scheduling.codegen_node_schedule_with_kernel = MagicMock()
        scheduling.make_ttir_for_check = MagicMock()
        scheduling.codegen_comment = MagicMock()
        scheduling.scheduler = MagicMock()
        scheduling.scheduler.free_buffers = MagicMock()
        scheduling.generate_node_schedule = MagicMock(
            return_value=["node_a", "node_b"]
        )
        scheduling.codegen_node_schedule = MagicMock(
            return_value="codegen_result"
        )
        return scheduling

    def define_codegen_node_schedule(self):
        sched = MagicMock()
        kernel = MagicMock()
        kernel.kernel_name = "k0"
        kernel.call_kernel = MagicMock()
        kernel.removed_buffers = set()
        kernel.inplaced_to_remove = set()
        kernel.codegen_kernel.return_value = "src"
        sched.create_kernel_choices.return_value = [kernel]
        sched.select_tiling.return_value = "tiling"
        sched.define_kernel.return_value = ("k0", "src")
        sched.codegen_node_schedule_with_kernel = MagicMock()
        features = MagicMock()
        features.node_schedule = []
        features.numel = 10
        features.reduction_numel = 1
        features.scheduler_nodes.return_value = []
        nodes = []
        origin_node = MagicMock()
        return sched, kernel, features, nodes, origin_node

    @patch("torch_npu._inductor.codegen.catlass.catlass_kernel.is_multi_stream", return_value=False)
    def test_catlass_workspace_single_stream(self, mock_multi_stream):
        """
        case CATLASSTemplateKernel call_kernel:
        1.workspace > 0
        2.single stream
        3.call generate_workspace_allocation/deallocation
        4.kernel_call origin_node=None
        """
        fake_graph = FakeGraph(cpp_wrapper=False)
        node = FakeNode(workspace_size=1024, name="test_01")
        with V.set_graph_handler(fake_graph):
            kernel = self.define_catlass_template_kernel()
            kernel.call_kernel(
                name="test_kernel",
                node=node,
                origin_node="origin",
            )
            wrapper = fake_graph.wrapper_code
            # workspace allocation
            wrapper.generate_workspace_allocation.assert_called_once()
            alloc_args = wrapper.generate_workspace_allocation.call_args[0]
            self.assertIsInstance(alloc_args[0], WorkspaceArg)
            # kernel call
            wrapper.generate_kernel_call.assert_called_once()
            _, kwargs = wrapper.generate_kernel_call.call_args
            self.assertEqual(kwargs["origin_node"], None)
            self.assertEqual(kwargs["triton"], False)
            # workspace deallocation
            wrapper.generate_workspace_deallocation.assert_called_once()


    @patch("torch_npu._inductor.codegen.catlass.catlass_kernel.is_multi_stream", return_value=True)
    def test_catlass_workspace_multi_stream(self, mock_multi_stream):
        """
        case CATLASSTemplateKernel call_kernel:
        1. workspace > 0
        2. multi stream
        3. allocation/deallocation with origin_node
        4. kernel_call with origin_node
        """
        fake_graph = FakeGraph(cpp_wrapper=False)
        node = FakeNode(workspace_size=2048, name="test_02")
        with V.set_graph_handler(fake_graph):   
            kernel = self.define_catlass_template_kernel()
            kernel.call_kernel(
                name="test_kernel",
                node=node,
                origin_node="origin_node_x",
            )
            wrapper = fake_graph.wrapper_code
            # allocation
            wrapper.generate_workspace_allocation.assert_called_once()
            alloc_args = wrapper.generate_workspace_allocation.call_args[0]
            self.assertEqual(alloc_args[1], "origin_node_x")
            # kernel call
            wrapper.generate_kernel_call.assert_called_once()
            call_args, call_kwargs = wrapper.generate_kernel_call.call_args
            self.assertEqual(call_args[0], "test_kernel")
            self.assertEqual(call_args[2], "origin_node_x")
            # deallocation
            wrapper.generate_workspace_deallocation.assert_called_once()
            dealloc_args = wrapper.generate_workspace_deallocation.call_args[0]
            self.assertEqual(dealloc_args[1], "origin_node_x")


    @patch("torch_npu._inductor.codegen.catlass.catlass_scheduling.is_multi_stream", return_value=True)
    def test_catlass_codegen_template_multi_stream(self, mock_multi_stream):
        """
        case CATLASSScheduling codegen_template:
        1. multi stream
        2. kernel.call_kernel(kernel_name, ctb, template_node)
        """
        fake_graph = FakeCatlassScheduleGraph()
        template_node = FakeCatlassSchedulingNode()
        kernel = FakeKernel()

        def fake_make_kernel_render(ctb, epilogue_nodes=None):
            def render():
                return "fake_src"
            return kernel, render
        
        template_node.node.make_kernel_render = fake_make_kernel_render
        with V.set_graph_handler(fake_graph):
            scheduler = self.define_catlass_scheduling()
            scheduler.codegen_template(
                template_node=template_node,
                epilogue_nodes=[],
                prologue_nodes=[],
                only_src_code=False,
            )
            kernel.call_kernel.assert_called_once_with(
                "generated_kernel",
                template_node.node,
                template_node,
            )

    @patch("torch_npu._inductor.codegen.catlass.catlass_scheduling.is_multi_stream", return_value=False)
    def test_catlass_codegen_template_single_stream(self, mock_multi_stream):
        """
        case CATLASSScheduling codegen_template:
        1. single stream
        2. kernel.call_kernel(kernel_name, ctb, None)
        """
        fake_graph = FakeCatlassScheduleGraph()
        template_node = FakeCatlassSchedulingNode()
        kernel = FakeKernel()
        ctb = template_node.node

        def fake_make_kernel_render(ctb, epilogue_nodes=None):
            def render():
                return "fake_src"
            return kernel, render

        ctb.make_kernel_render = fake_make_kernel_render
        with V.set_graph_handler(fake_graph):
            scheduler = self.define_catlass_scheduling()

            scheduler.codegen_template(
                template_node=template_node,
                epilogue_nodes=[],
                prologue_nodes=[],
                only_src_code=False,
            )

            kernel.call_kernel.assert_called_once_with(
                "generated_kernel",
                ctb,
                None,
            )


    @patch("torch_npu._inductor.codegen.scheduling.is_multi_stream", return_value=True)
    @patch("torch_npu._inductor.codegen.scheduling.V")
    def test_codegen_node_schedule_multi_stream(self, mock_V, mock_multi_stream):
        """
        case:
        1. multi stream
        2. final_kernel.call_kernel(..., origin_node=origin_node)
        """
        mock_V.graph.removed_buffers = set()
        mock_V.graph.inplaced_to_remove = set()
        sched, kernel, features, nodes, origin_node = self.define_codegen_node_schedule()
        NPUTritonScheduling.codegen_node_schedule(
            sched,
            features,
            nodes,
            origin_node=origin_node
        )
        kernel.call_kernel.assert_called_once()
        _, kwargs = kernel.call_kernel.call_args
        assert kwargs["name"] == "k0"
        assert kwargs["origin_node"] is origin_node

    
    @patch("torch_npu._inductor.codegen.scheduling.is_multi_stream", return_value=False)
    @patch("torch_npu._inductor.codegen.scheduling.V")
    def test_codegen_node_schedule_single_stream(self, mock_V, mock_multi_stream):
        """
        case:
        1. single stream
        2. final_kernel.call_kernel(..., origin_node=None)
        """
        mock_V.graph.removed_buffers = set()
        mock_V.graph.inplaced_to_remove = set()
        sched, kernel, features, nodes, origin_node = self.define_codegen_node_schedule()
        NPUTritonScheduling.codegen_node_schedule(
            sched,
            features,
            nodes,
            origin_node=origin_node
        )
        kernel.call_kernel.assert_called_once()
        _, kwargs = kernel.call_kernel.call_args
        assert kwargs["name"] == "k0"
        assert kwargs["origin_node"] is None


    @patch("torch_npu._inductor.codegen.scheduling.is_multi_stream", return_value=True)
    def test_codegen_node_multi_stream(
        self,
        mock_multi_stream
    ):
        """
        case:
        1. multi stream
        2. codegen_node_schedule(..., nodes, node)
        """
        scheduler = self.define_scheduling()
        nodes = [
            FakeSchedulerNode(is_reduction=False),
            FakeSchedulerNode(is_reduction=True),
        ]
        fused_node = FakeFusedNode(nodes)
        fake_graph = FakeScheduleGraph()
        with V.set_graph_handler(fake_graph):
            result = scheduler.codegen_node(fused_node)
        self.assertEqual(result, "codegen_result")
        scheduler.codegen_node_schedule.assert_called_once()
        args = scheduler.codegen_node_schedule.call_args[0]
        self.assertIsNotNone(args[0])
        self.assertEqual(args[1], nodes)
        self.assertEqual(args[2], fused_node)
        self.assertEqual(len(args), 3)


    @patch("torch_npu._inductor.codegen.scheduling.is_multi_stream", return_value=False)
    def test_codegen_node_single_stream(
        self,
        mock_multi_stream
    ):
        """
        case:
        1. multi stream
        2. codegen_node_schedule(..., nodes)
        """
        scheduler = self.define_scheduling()
        nodes = [
            FakeSchedulerNode(is_reduction=False),
            FakeSchedulerNode(is_reduction=True),
        ]
        fused_node = FakeFusedNode(nodes)
        fake_graph = FakeScheduleGraph()
        with V.set_graph_handler(fake_graph):
            result = scheduler.codegen_node(fused_node)
        self.assertEqual(result, "codegen_result")
        scheduler.codegen_node_schedule.assert_called_once()
        args = scheduler.codegen_node_schedule.call_args[0]
        self.assertIsNotNone(args[0])
        self.assertEqual(args[1], nodes)
        self.assertEqual(len(args), 2)


    def test_triton_call_kernel_multi_stream(self):
        fake_graph = FakeGraph()
        triton_kernel = MagicMock(spec=NPUIndexTritonKernel)
        triton_kernel.args = FakeTritonArgs()
        triton_kernel.triton_meta = {"meta": 1}
        triton_kernel.add_numel_to_call_args = MagicMock()
        origin_node = MagicMock()
        with patch.object(
            torch_npu._inductor.codegen.triton,
            "is_multi_stream",
            return_value=True,
        ):
            with V.set_graph_handler(fake_graph):
                NPUIndexTritonKernel.call_kernel(
                    triton_kernel,
                    "kernel_a",
                    node=None,
                    origin_node=origin_node,
                )
        wrapper = fake_graph.wrapper_code
        wrapper.write_triton_header_once.assert_called_once()
        triton_kernel.add_numel_to_call_args.assert_called_once()
        numel_args = triton_kernel.add_numel_to_call_args.call_args[0]
        self.assertEqual(numel_args[0], "kernel_a")
        self.assertEqual(numel_args[1], ["a", "b"])
        self.assertEqual(numel_args[2], ["int", "int"])
        self.assertEqual(
            wrapper.generate_workspace_allocation.call_count,
            2,
        )
        wrapper.generate_workspace_allocation.assert_any_call(
            "ws1",
            origin_node,
        )
        wrapper.generate_workspace_allocation.assert_any_call(
            "ws2",
            origin_node,
        )
        wrapper.generate_kernel_call.assert_called_once()
        self.assertEqual(
            wrapper.generate_workspace_deallocation.call_count,
            2,
        )


    def multi_stream_test(
        self,
        arg0_1,
        arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1,
        arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1,
        arg18_1, arg19_1, arg20_1, arg21_1, arg22_1,
        arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1,
        arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1,
        arg35_1, arg36_1, arg37_1, arg38_1, arg39_1
    ):
        slice_2 = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 3)
        sum_1 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg2_1, slice_2), [1])
        slice_4 = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 5)
        sum_2 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg3_1, slice_4), [1])
        slice_6 = torch.ops.aten.slice.Tensor(arg0_1, 1, 5, 6)
        sum_3 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg4_1, slice_6), [1])
        slice_8 = torch.ops.aten.slice.Tensor(arg0_1, 1, 6, 8)
        sum_4 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg5_1, slice_8), [1])
        slice_10 = torch.ops.aten.slice.Tensor(arg0_1, 1, 8, 14)
        sum_5 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg6_1, slice_10), [1])
        slice_12 = torch.ops.aten.slice.Tensor(arg0_1, 1, 14, 15)
        sum_6 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg7_1, slice_12), [1])
        slice_14 = torch.ops.aten.slice.Tensor(arg0_1, 1, 15, 16)
        sum_7 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg8_1, slice_14), [1])
        slice_16 = torch.ops.aten.slice.Tensor(arg0_1, 1, 16, 17)
        sum_8 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg9_1, slice_16), [1])
        slice_18 = torch.ops.aten.slice.Tensor(arg0_1, 1, 17, 18)
        sum_9 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg10_1, slice_18), [1])
        slice_20 = torch.ops.aten.slice.Tensor(arg0_1, 1, 18, 25)
        sum_10 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg11_1, slice_20), [1])
        slice_22 = torch.ops.aten.slice.Tensor(arg0_1, 1, 25, 28)
        sum_11 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg12_1, slice_22), [1])
        slice_24 = torch.ops.aten.slice.Tensor(arg0_1, 1, 28, 36)
        sum_12 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg13_1, slice_24), [1])
        slice_26 = torch.ops.aten.slice.Tensor(arg0_1, 1, 36, 37)
        sum_13 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg14_1, slice_26), [1])
        slice_28 = torch.ops.aten.slice.Tensor(arg0_1, 1, 37, 43)
        sum_14 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg15_1, slice_28), [1])
        slice_30 = torch.ops.aten.slice.Tensor(arg0_1, 1, 43, 52)
        sum_15 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg16_1, slice_30), [1])
        slice_32 = torch.ops.aten.slice.Tensor(arg0_1, 1, 52, 57)
        sum_16 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg17_1, slice_32), [1])
        slice_34 = torch.ops.aten.slice.Tensor(arg0_1, 1, 57, 58)
        sum_17 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg18_1, slice_34), [1])
        slice_36 = torch.ops.aten.slice.Tensor(arg0_1, 1, 58, 59)
        sum_18 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg19_1, slice_36), [1])
        slice_38 = torch.ops.aten.slice.Tensor(arg0_1, 1, 59, 60)
        sum_19 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg20_1, slice_38), [1])
        slice_40 = torch.ops.aten.slice.Tensor(arg0_1, 1, 60, 72)
        sum_20 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg21_1, slice_40), [1])
        slice_42 = torch.ops.aten.slice.Tensor(arg0_1, 1, 72, 172)
        sum_21 = torch.ops.aten.sum.dim_IntList(torch.ops.aten.embedding.default(arg22_1, slice_42), [1])
        cat = torch.ops.aten.cat.default(
            [sum_1, sum_2, sum_3, sum_4, sum_5,
            sum_6, sum_7, sum_8, sum_9, sum_10,
            sum_11, sum_12, sum_13, sum_14, sum_15,
            sum_16, sum_17, sum_18, sum_19, sum_20, sum_21],
            1
        )
        add_relu = torch.ops.aten.add.Tensor(cat, cat)

        # branch A
        a = torch.ops.aten.relu.default(torch.ops.aten.mm.default(arg23_1, arg24_1))
        a = torch.ops.aten.relu.default(torch.ops.aten.mm.default(a, arg25_1))
        a = torch.ops.aten.relu.default(torch.ops.aten.mm.default(a, arg26_1))
        a = torch.ops.aten.relu.default(torch.ops.aten.mm.default(a, arg27_1))
        mm_4 = torch.ops.aten.relu.default(torch.ops.aten.mm.default(a, arg28_1))

        # branch B
        b = torch.ops.aten.relu.default(torch.ops.aten.mm.default(arg29_1, arg30_1))
        b = torch.ops.aten.relu.default(torch.ops.aten.mm.default(b, arg31_1))
        b = torch.ops.aten.relu.default(torch.ops.aten.mm.default(b, arg32_1))
        b = torch.ops.aten.relu.default(torch.ops.aten.mm.default(b, arg33_1))
        mm_8 = torch.ops.aten.relu.default(torch.ops.aten.mm.default(b, arg34_1))

        # merge
        mm_8_t = torch.ops.aten.permute.default(mm_8, [1, 0])
        merge = torch.ops.aten.mm.default(mm_4, mm_8_t)
        merge = torch.ops.aten.mm.default(merge, arg35_1)

        # MAIN
        add = torch.ops.aten.add.Tensor(merge, add_relu)
        mm_15 = torch.ops.aten.mm.default(arg36_1, add)
        relu_15 = torch.ops.aten.relu.default(mm_15)
        add_15 = torch.ops.aten.add.Tensor(arg37_1, relu_15)
        out = torch.ops.aten.addmm.default(
            arg39_1,
            arg38_1,
            add_15,
        )
        return out

    @patch("torch_npu._inductor.fx_passes.parallel_scheduler_pass.is_multi_stream", return_value=True)
    def test_multi_stream_compile_case(self, mock_multi_stream):
        arg0_1  = torch.randint(0, 99, (64, 199), dtype=torch.int64, device="npu")
        arg1_1  = torch.randn(100, 64, device="npu")
        arg2_1  = torch.randn(100, 64, device="npu")
        arg3_1  = torch.randn(100, 64, device="npu")
        arg4_1  = torch.randn(100, 64, device="npu")
        arg5_1  = torch.randn(100, 64, device="npu")
        arg6_1  = torch.randn(100, 64, device="npu")
        arg7_1  = torch.randn(100, 64, device="npu")
        arg8_1  = torch.randn(100, 64, device="npu")
        arg9_1  = torch.randn(100, 64, device="npu")
        arg10_1 = torch.randn(100, 64, device="npu")
        arg11_1 = torch.randn(100, 64, device="npu")
        arg12_1 = torch.randn(100, 64, device="npu")
        arg13_1 = torch.randn(100, 64, device="npu")
        arg14_1 = torch.randn(100, 64, device="npu")
        arg15_1 = torch.randn(100, 64, device="npu")
        arg16_1 = torch.randn(100, 64, device="npu")
        arg17_1 = torch.randn(100, 64, device="npu")
        arg18_1 = torch.randn(100, 64, device="npu")
        arg19_1 = torch.randn(100, 64, device="npu")
        arg20_1 = torch.randn(100, 64, device="npu")
        arg21_1 = torch.randn(100, 64, device="npu")
        arg22_1 = torch.randn(100, 64, device="npu")
        arg23_1 = torch.randn(64, 64, device="npu")
        arg24_1 = torch.randn(64, 64, device="npu")
        arg25_1 = torch.randn(64, 64, device="npu")
        arg26_1 = torch.randn(64, 64, device="npu")
        arg27_1 = torch.randn(64, 64, device="npu")
        arg28_1 = torch.randn(64, 64, device="npu")
        arg29_1 = torch.randn(64, 64, device="npu")
        arg30_1 = torch.randn(64, 64, device="npu")
        arg31_1 = torch.randn(64, 64, device="npu")
        arg32_1 = torch.randn(64, 64, device="npu")
        arg33_1 = torch.randn(64, 64, device="npu")
        arg34_1 = torch.randn(64, 64, device="npu")
        arg35_1 = torch.randn(64, 1344, device="npu")
        arg36_1 = torch.randn(64, 64, device="npu")
        arg37_1 = torch.randn(64, 1344, device="npu")
        arg38_1 = torch.randn(64, 64, device="npu")
        arg39_1 = torch.randn(64, 1344, device="npu")

        std_result = self.multi_stream_test(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.multi_stream_test, backend="inductor")
        inductor_result = compiled_op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)

    def op_calc(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        relu_1 = torch.ops.aten.relu.default(arg0_1)
        mul_1 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        relu_2 = torch.ops.aten.relu.default(mul_1)
        mm_2 = torch.ops.aten.mm.default(relu_1, relu_2)
        relu_3 = torch.ops.aten.relu.default(mm_2)
        mul_3 = torch.ops.aten.mm.default(relu_3, relu_2)
        relu_4 = torch.ops.aten.relu.default(arg3_1)
        relu_5 = torch.ops.aten.relu.default(arg4_1)
        add_1 = torch.ops.aten.add.Tensor(relu_4, relu_5)
        relu_6 = torch.ops.aten.relu.default(arg5_1)
        add_2 = torch.ops.aten.add.Tensor(add_1, relu_6)
        relu_7 = torch.ops.aten.relu.default(add_2)
        silu_1 = torch.ops.aten.silu.default(relu_7)
        add_3 = torch.ops.aten.add.Tensor(mul_3, silu_1)
        slice_1 = torch.ops.aten.slice.Tensor(add_3, dim=0, start=0, end=128, step=1)
        relu_8 = torch.ops.aten.relu.default(slice_1)
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, dim=0, start=0, end=128, step=1)
        relu_9 = torch.ops.aten.relu.default(slice_1)
        add_2 = torch.ops.aten.add.Tensor(relu_8, slice_2)
        add_3 = torch.ops.aten.add.Tensor(slice_2, relu_9)
        mm_3 = torch.ops.aten.mm.default(add_2, add_3)
        relu_10 = torch.ops.aten.relu.default(mm_3)
        return relu_10

    @patch("torch_npu._inductor.fx_passes.parallel_scheduler_pass.is_multi_stream", return_value=False)
    def test_single_stream_compile_case(self, mock_multi_stream):
        arg0_1 = torch.randn(128, 128)
        arg1_1 = torch.randn(128, 64)
        arg2_1 = torch.randn(64, 128)
        arg3_1 = torch.randn(128, 128)
        arg4_1 = torch.randn(128, 128)
        arg5_1 = torch.randn(128, 128)
        std_result = self.op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestMultiStreamPass)


if __name__ == "__main__":
    run_tests()